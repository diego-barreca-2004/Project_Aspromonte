#!/usr/bin/env python3
"""run_colmap.py - Stage 2: frames -> camera poses (Structure-from-Motion) with COLMAP.

Wraps the COLMAP CLI for ORDERED video frames:
  feature_extractor (single shared camera) -> sequential_matcher -> mapper -> image_undistorter
The output workspace is ready for 3D Gaussian Splatting (Inria 3DGS / gsplat / splatfacto):
  <out>/undistorted/images/
  <out>/undistorted/sparse/0/{cameras,images,points3D}.bin

Camera model: pass --camera-model, or --calibration to inject known intrinsics (recommended for
fisheye, where self-calibration of k1..k4 from scratch is less stable). The injected parameters
are scaled to the actual frame resolution, since frames are usually downscaled vs the calibration:
  OPENCV (5-coeff Brown-Conrady)  |  FULL_OPENCV (rational)  |  OPENCV_FISHEYE (Kannala-Brandt)

Requirements: COLMAP >= 3.7 (apt-get install colmap, or a CUDA build) and, ideally, a CUDA GPU.

Devices: feature extraction runs on the GPU by default, matching on the CPU. COLMAP's GPU
matcher (SiftGPU) is often slower than CPU on recent GPUs and under WSL; override per stage with
--extraction-gpu / --matching-gpu, or both at once with --use-gpu.

Usage:
  python3 run_colmap.py --images ./seg01/frames --out ./seg01/colmap \
                        --calibration ./calib_out/calibration_fisheye.json
  python3 run_colmap.py --images ./seg01/frames --out ./seg01/colmap --camera-model OPENCV_FISHEYE
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

CALIB_MODEL_MAP = {"fisheye": "OPENCV_FISHEYE", "rational": "FULL_OPENCV", "standard": "OPENCV"}
N_DIST_COEFFS = {"OPENCV": 4, "FULL_OPENCV": 8, "OPENCV_FISHEYE": 4}   # COLMAP distortion-param counts


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"COLMAP step failed: {cmd[1] if len(cmd) > 1 else cmd}")


# --- Calibration injection ---------------------------------------------------
def _image_size(path):
    """(width, height) of a JPEG or PNG via header parsing; None if unknown."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if head[:8] == b"\x89PNG\r\n\x1a\n":                # PNG IHDR
                return struct.unpack(">II", head[16:24])
            if head[:2] == b"\xff\xd8":                         # JPEG: scan for an SOF marker
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":
                        b = f.read(1)
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if not marker:
                        break
                    if 0xC0 <= marker[0] <= 0xCF and marker[0] not in (0xC4, 0xC8, 0xCC):
                        f.read(3)                               # segment length (2) + sample precision (1)
                        h, w = struct.unpack(">HH", f.read(4))
                        return w, h
                    f.seek(struct.unpack(">H", f.read(2))[0] - 2, 1)
    except (OSError, struct.error):
        pass
    return None


def _first_image(image_dir):
    for f in sorted(os.listdir(image_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(image_dir, f)
    return None


def camera_params_from_calibration(calib_path, image_dir):
    """Build (colmap_model, params_csv) from a calibration JSON, scaled to the frame resolution."""
    with open(calib_path) as fh:
        c = json.load(fh)
    model = CALIB_MODEL_MAP.get(c.get("model"), "OPENCV_FISHEYE")
    K, dist = c["camera_matrix"], list(c["distortion_coefficients"])
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    w_cal, h_cal = c["image_size"]

    size = _image_size(_first_image(image_dir) or "")
    if size and (size[0] != w_cal or size[1] != h_cal):        # frames downscaled vs calibration
        sx, sy = size[0] / w_cal, size[1] / h_cal
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        print(f"  intrinsics scaled to {size[0]}x{size[1]} (calibrated at {w_cal}x{h_cal})")
    elif not size:
        print("  WARNING: could not read frame size; using calibration intrinsics unscaled.")

    n = N_DIST_COEFFS[model]                                    # cv2 coeff order matches COLMAP's
    params = [fx, fy, cx, cy] + (dist + [0.0] * n)[:n]
    return model, ",".join(f"{v:.10g}" for v in params)


def largest_model(sparse_dir):
    """Pick the reconstructed model with the most images (proxy: images.bin size)."""
    models = [os.path.join(sparse_dir, d) for d in os.listdir(sparse_dir)
              if os.path.isfile(os.path.join(sparse_dir, d, "images.bin"))]
    return max(models, key=lambda m: os.path.getsize(os.path.join(m, "images.bin")), default=None)


# --- Pipeline ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="COLMAP SfM for ordered video frames.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="./colmap")
    ap.add_argument("--colmap", default="colmap", help="Path to the colmap executable.")
    ap.add_argument("--camera-model", default="OPENCV_FISHEYE",
                    choices=["OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE", "PINHOLE", "SIMPLE_RADIAL"])
    ap.add_argument("--calibration", default=None,
                    help="Calibration JSON (calibrate_camera.py); injects intrinsics and sets the model.")
    ap.add_argument("--fix-intrinsics", action="store_true",
                    help="Freeze the injected intrinsics during mapping (requires --calibration).")
    ap.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"])
    ap.add_argument("--extraction-gpu", default="1", choices=["0", "1"],
                    help="GPU for feature extraction (default on; large speedup).")
    ap.add_argument("--matching-gpu", default="0", choices=["0", "1"],
                    help="GPU for feature matching (default off: COLMAP's SiftGPU matcher is "
                         "slow on recent GPUs and under WSL, where CPU is usually faster).")
    ap.add_argument("--use-gpu", default=None, choices=["0", "1"],
                    help="Master override: force GPU on/off for BOTH extraction and matching.")
    ap.add_argument("--vocab-tree", default=None,
                    help="Vocab-tree path to enable loop detection (sequential matcher).")
    args = ap.parse_args()

    # --use-gpu, if given, overrides both per-stage flags.
    extraction_gpu = args.use_gpu if args.use_gpu is not None else args.extraction_gpu
    matching_gpu = args.use_gpu if args.use_gpu is not None else args.matching_gpu
    print(f"  devices: extraction={'GPU' if extraction_gpu == '1' else 'CPU'}, "
          f"matching={'GPU' if matching_gpu == '1' else 'CPU'}")

    if shutil.which(args.colmap) is None and not os.path.isfile(args.colmap):
        sys.exit(f"COLMAP not found ('{args.colmap}'). Install it or pass --colmap /path/to/colmap.")
    if not os.path.isdir(args.images):
        sys.exit(f"Images folder not found: {args.images}")
    if args.calibration and not os.path.isfile(args.calibration):
        sys.exit(f"Calibration file not found: {args.calibration}")
    if args.fix_intrinsics and not args.calibration:
        print("NOTE: --fix-intrinsics ignored without --calibration.")

    os.makedirs(args.out, exist_ok=True)
    db = os.path.join(args.out, "database.db")
    sparse = os.path.join(args.out, "sparse")
    undist = os.path.join(args.out, "undistorted")
    os.makedirs(sparse, exist_ok=True)

    # Resolve camera model and, if a calibration is given, the injected intrinsics.
    camera_model, camera_params = args.camera_model, None
    if args.calibration:
        camera_model, camera_params = camera_params_from_calibration(args.calibration, args.images)
        print(f"  injecting calibrated {camera_model}: {camera_params}")

    # 1) Features - single shared camera (one physical camera across all frames).
    feat = [args.colmap, "feature_extractor",
            "--database_path", db, "--image_path", args.images,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", camera_model,
            "--SiftExtraction.use_gpu", extraction_gpu]
    if camera_params:
        feat += ["--ImageReader.camera_params", camera_params]
    run(feat)

    # 2) Matching - sequential for ordered video (fast + correct); exhaustive otherwise.
    if args.matcher == "sequential":
        cmd = [args.colmap, "sequential_matcher", "--database_path", db,
               "--SiftMatching.use_gpu", matching_gpu]
        if args.vocab_tree:
            cmd += ["--SequentialMatching.loop_detection", "1",
                    "--SequentialMatching.vocab_tree_path", args.vocab_tree]
        run(cmd)
    else:
        run([args.colmap, "exhaustive_matcher", "--database_path", db,
             "--SiftMatching.use_gpu", matching_gpu])

    # 3) Sparse reconstruction (mapper) -> sparse/<id>
    mapper = [args.colmap, "mapper",
              "--database_path", db, "--image_path", args.images, "--output_path", sparse]
    if args.calibration and args.fix_intrinsics:
        mapper += ["--Mapper.ba_refine_focal_length", "0",
                   "--Mapper.ba_refine_principal_point", "0",
                   "--Mapper.ba_refine_extra_params", "0"]
    run(mapper)

    model0 = largest_model(sparse)
    if model0 is None:
        sys.exit("Mapper produced no model. Usually too little overlap, motion blur, or too few "
                 "features. Re-capture with more overlap (slower pass / higher frame rate) and "
                 "sharp frames (fast shutter).")
    n_models = sum(os.path.isdir(os.path.join(sparse, d)) for d in os.listdir(sparse))
    if n_models > 1:
        print(f"  NOTE: {n_models} disconnected models; using the largest "
              f"({os.path.basename(model0)}). Fragmentation usually means thin overlap somewhere.")

    # 4) Undistort -> pinhole images + cameras ready for 3DGS
    run([args.colmap, "image_undistorter",
         "--image_path", args.images, "--input_path", model0,
         "--output_path", undist, "--output_type", "COLMAP"])

    # Normalise to <undist>/sparse/0/ (what Inria 3DGS train.py expects).
    u_sparse = os.path.join(undist, "sparse")
    u_sparse0 = os.path.join(u_sparse, "0")
    if os.path.isdir(u_sparse) and not os.path.isdir(u_sparse0):
        os.makedirs(u_sparse0, exist_ok=True)
        for f in ("cameras.bin", "images.bin", "points3D.bin"):
            p = os.path.join(u_sparse, f)
            if os.path.isfile(p):
                shutil.move(p, os.path.join(u_sparse0, f))

    print("\nDone. Workspace ready for 3D Gaussian Splatting:")
    print(f"  images: {os.path.join(undist, 'images')}")
    print(f"  sparse: {u_sparse0}")
    print("\nNext (Inria 3DGS):")
    print(f"  python train.py -s {undist} -m {os.path.join(undist, 'gs_output')}")
    print("nerfstudio/gsplat:")
    print(f"  ns-train splatfacto colmap --data {undist}")


if __name__ == "__main__":
    main()