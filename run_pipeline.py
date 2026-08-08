#!/usr/bin/env python3
"""run_pipeline.py - one-command per-epoch reconstruction for Project Aspromonte.

Runs the audited per-epoch chain end to end, driven by a shared config so every epoch is
reconstructed IDENTICALLY (like-with-like), with QC gates that HALT - with a clear message -
at the points that have historically failed silently. It stops at fused_utm.ply (and the UTM
splat) per epoch; the two-epoch M3C2 comparison stays manual on purpose.

Stages:  ingest -> sfm -> geo -> dense -> georef [-> splat]

  ingest   ingest_gopro.py     video           -> frames/ + gps.csv
  sfm      run_colmap.py       frames          -> colmap sparse (+ undistorted workspace)
  geo      geo_align.py        sparse + gps    -> geo_transform.txt
  dense    COLMAP MVS          undistorted     -> dense/fused.ply
  georef   georef_cloud.py     fused + Sim3    -> dense/fused_utm.ply
  splat    3DGS + georef_splat undistorted     -> UTM splat + view.ply   (if splat=true)

QC gates (halt on failure):
  ingest -> frame longest side == longest_side (JPEG header read, no full decode)
            (catches epochs ingested at different resolutions - the like-with-like killer:
             epoch 2 froze fusion because it ran at native 5312 px vs the 1600 px baseline)
  sfm    -> registered images >= min_images AND registered/frames >= min_reg_ratio
            (catches the spurious 2-frame fragment / thin overlap)
  geo    -> inlier ratio >= min_geo_inlier_ratio AND residual <= max_geo_residual
            (the first traffic light; a bad capture shows up here, before the long MVS)
  dense  -> fused points >= min_dense_points
            (catches the 0-fused-points Blackwell codegen failure)
  georef -> UTM centroid inside the configured box AND vertical extent <= max_vertical_extent
            (catches floater-inflated clouds and gross mislocation)

Each stage skips when its output already exists (resume); force re-run with --force.
Run a sub-range with --from/--to. Preview without executing with --dry-run.

Usage:
  python3 run_pipeline.py --config pipeline.conf --video ep2.mp4 --workdir seg01_ep2
  python3 run_pipeline.py --config pipeline.conf --video ep2.mp4 --workdir seg01_ep2 --from dense
"""
import argparse
import glob
import os
import re
import shutil
import struct
import subprocess
import sys

STAGES = ["ingest", "sfm", "geo", "dense", "georef", "splat"]

DEFAULTS = {
    "colmap": "colmap", "python": "python3", "epsg": "32633",
    "ingest_script": "ingest_gopro.py", "sfm_script": "run_colmap.py",
    "geo_script": "geo_align.py", "cloud_script": "georef_cloud.py",
    "splat_script": "georef_splat.py",
    "calibration": "", "dtm": "",
    # reconstruction contract (identical for every epoch)
    # longest_side is DELIBERATELY absent: contract parameter, NO default - a silent
    # full-resolution ingest is exactly how epoch 2 broke (5312 px vs the 1600 px baseline).
    "every_sec": "0.2", "jpeg_quality": "95", "matcher": "sequential",
    "max_image_size": "1000", "sor_iters": "2", "sor_k": "20", "sor_std": "2.0",
    # splat
    "splat": "false", "gs_train": "", "gs_iterations": "30000",
    # gs_extra_args: execution strategy, NOT part of the numeric contract (e.g. --data_device
    # cpu keeps ~600 frames in system RAM instead of oversubscribing 12 GB of VRAM).
    "gs_extra_args": "",
    # QC gate thresholds
    "min_images": "50", "min_reg_ratio": "0.80",
    "min_geo_inlier_ratio": "0.70", "max_geo_residual": "0.60",
    "min_dense_points": "100000", "max_vertical_extent": "200",
    "utm_e_min": "-1e18", "utm_e_max": "1e18",
    "utm_n_min": "-1e18", "utm_n_max": "1e18",
}


# --- config ------------------------------------------------------------------
def load_config(path):
    cfg = dict(DEFAULTS)
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            parts = line.strip().split(None, 1)
            cfg[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return cfg


def cfg_f(cfg, k):
    return float(cfg[k])


def cfg_i(cfg, k):
    return int(float(cfg[k]))


def cfg_bool(cfg, k):
    return str(cfg.get(k, "")).lower() in ("1", "true", "yes", "on")


# --- small readers (lightweight; no full-file loads) -------------------------
IMG_EXT = (".jpg", ".jpeg", ".png")


def n_frames(d):
    return sum(1 for f in os.listdir(d) if f.lower().endswith(IMG_EXT)) if os.path.isdir(d) else 0


def jpeg_dims(path):
    """(width, height) from the JPEG SOF header; None if unparseable.
    Header-only read - never decodes the bitmap, so it is safe on arbitrarily large frames."""
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            b = f.read(1)
            if not b:
                return None
            if b != b"\xff":
                continue
            marker = f.read(1)
            while marker == b"\xff":                 # padding fill bytes
                marker = f.read(1)
            if not marker:
                return None
            m = marker[0]
            if 0xD0 <= m <= 0xD9 or m == 0x01:       # RST0-7 / SOI / EOI / TEM: no payload
                continue
            if m == 0xDA:                            # SOS: SOF always precedes it; give up
                return None
            size = f.read(2)
            if len(size) < 2:
                return None
            if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):   # SOF0-15 minus DHT/JPG/DAC
                sof = f.read(5)
                if len(sof) < 5:
                    return None
                h, w = struct.unpack(">HH", sof[1:5])
                return (w, h)
            f.seek(max(0, struct.unpack(">H", size)[0] - 2), 1)


def imagesbin_count(model_dir):
    """Registered-image count from a COLMAP images.bin header (first uint64)."""
    p = os.path.join(model_dir, "images.bin")
    if not os.path.isfile(p):
        return 0
    with open(p, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def ply_vertex_count(path):
    """Vertex count from a PLY header (ASCII header even in binary PLYs)."""
    if not os.path.isfile(path):
        return 0
    with open(path, "rb") as f:
        for _ in range(60):
            line = f.readline()
            if not line or line.strip() == b"end_header":
                break
            m = re.match(rb"element\s+vertex\s+(\d+)", line.strip())
            if m:
                return int(m.group(1))
    return 0


def done(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


# --- command runner ----------------------------------------------------------
def run(cmd, capture=False):
    print("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=capture, text=True)
    if capture and r.stdout:
        sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.exit(f"FAILED: {cmd[0]} {cmd[1] if len(cmd) > 1 else ''} (exit {r.returncode}). "
                 "Fix the cause above, then re-run - completed stages will be skipped.")
    return r.stdout if capture else ""


def halt(stage, msg):
    sys.exit(f"\nQC GATE FAILED [{stage}]: {msg}\nStopping before wasting downstream compute.")


# --- paths -------------------------------------------------------------------
class P:
    def __init__(self, wd):
        self.wd = wd
        self.frames = os.path.join(wd, "frames")
        self.gps = os.path.join(wd, "gps.csv")
        self.colmap = os.path.join(wd, "colmap")
        self.undist = os.path.join(self.colmap, "undistorted")
        self.model0 = os.path.join(self.undist, "sparse", "0")
        self.undist_images = os.path.join(self.undist, "images")
        self.geo = os.path.join(self.colmap, "geo_transform.txt")
        self.dense = os.path.join(self.colmap, "dense")
        self.pm_cfg = os.path.join(self.dense, "stereo", "patch-match.cfg")
        self.fused = os.path.join(self.dense, "fused.ply")
        self.fused_utm = os.path.join(self.dense, "fused_utm.ply")
        self.gs_out = os.path.join(wd, "gs_output")
        self.splat_utm = os.path.join(wd, "point_cloud_utm.ply")

    def trained_ply(self, iters):
        return os.path.join(self.gs_out, "point_cloud", f"iteration_{iters}", "point_cloud.ply")


# --- stages ------------------------------------------------------------------
def gate_ingest_dims(cfg, p):
    """Frames must match the contract longest_side. Runs on fresh AND skipped ingest, so a
    resume over pre-existing frames is checked too (the exact epoch-2 failure mode)."""
    frames = sorted(f for f in os.listdir(p.frames) if f.lower().endswith(IMG_EXT))
    if not frames:
        halt("ingest", "no frames found to check.")
    expected = cfg_i(cfg, "longest_side")
    for name in {frames[0], frames[len(frames) // 2], frames[-1]}:  # one writer => uniform dims
        dims = jpeg_dims(os.path.join(p.frames, name))
        if dims is None:
            halt("ingest", f"cannot read JPEG dimensions from {name}.")
        if max(dims) != expected:
            halt("ingest", f"CONTRACT VIOLATION: {name} is {dims[0]}x{dims[1]} (longest side "
                           f"{max(dims)} px, contract longest_side = {expected} px). Epochs "
                           f"ingested at different resolutions are not comparable "
                           f"(like-with-like). Quarantine this workdir and re-ingest with "
                           f"--longest-side {expected}.")
    print(f"  gate ingest OK: {len(frames)} frames at longest side {expected} px "
          f"(checked first/middle/last).")


def stage_ingest(cfg, p, args):
    try:
        if cfg_i(cfg, "longest_side") <= 0:
            raise ValueError
    except (KeyError, ValueError):
        sys.exit("longest_side is missing/invalid in the config: it is a CONTRACT parameter "
                 "with NO default (epoch-1 baseline = 1600). Add 'longest_side 1600' to "
                 "pipeline.conf - a silent default is exactly how epoch 2 broke.")
    if done(p.gps) and n_frames(p.frames) > 0 and not args.force:
        print(f"  skip ingest (frames + gps.csv exist: {n_frames(p.frames)} frames)")
    else:
        run([cfg["python"], cfg["ingest_script"], "--video", args.video, "--out", p.wd,
             "--every-sec", cfg["every_sec"], "--jpeg-quality", cfg["jpeg_quality"],
             "--longest-side", cfg["longest_side"]])
        if n_frames(p.frames) == 0:
            halt("ingest", "no frames were extracted (check the video path / codec).")
    gate_ingest_dims(cfg, p)


def stage_sfm(cfg, p, args):
    if not (done(p.gps) and n_frames(p.frames) > 0):
        sys.exit("sfm needs frames + gps.csv from ingest; run --from ingest.")
    if done(os.path.join(p.model0, "images.bin")) and not args.force:
        print("  skip sfm (undistorted/sparse/0 exists)")
    else:
        cmd = [cfg["python"], cfg["sfm_script"], "--images", p.frames, "--out", p.colmap,
               "--matcher", cfg["matcher"]]
        if cfg["calibration"]:
            cmd += ["--calibration", cfg["calibration"]]
        else:
            cmd += ["--camera-model", "OPENCV_FISHEYE"]
        run(cmd)
    reg, nf = imagesbin_count(p.model0), n_frames(p.frames)
    ratio = reg / nf if nf else 0.0
    if reg < cfg_i(cfg, "min_images") or ratio < cfg_f(cfg, "min_reg_ratio"):
        halt("sfm", f"only {reg}/{nf} images registered (ratio {ratio:.2f}; need "
                    f">= {cfg['min_images']} and >= {cfg['min_reg_ratio']}). Thin overlap / "
                    f"motion blur, or a spurious small model. Re-capture with more overlap.")
    print(f"  gate sfm OK: {reg}/{nf} images registered (ratio {ratio:.2f}).")


def stage_geo(cfg, p, args):
    if not done(os.path.join(p.model0, "images.bin")):
        sys.exit("geo needs the SfM model; run --from sfm.")
    if done(p.geo) and not args.force:
        print("  skip geo (geo_transform.txt exists)")
    else:
        run([cfg["python"], cfg["geo_script"], "--gps", p.gps, "--images", p.undist_images,
             "--model", p.model0, "--out", p.geo, "--epsg", cfg["epsg"]])
    fit = None
    with open(p.geo) as f:
        for line in f:
            m = re.search(r"([\d]+)\s*/\s*([\d]+)\s+inliers.*?residual median\s+([\d.]+)", line)
            if m:
                fit = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
                break
    if fit is None:
        halt("geo", "could not read the fit line from geo_transform.txt.")
    ninl, ntot, resid = fit
    ratio = ninl / ntot if ntot else 0.0
    if ratio < cfg_f(cfg, "min_geo_inlier_ratio") or resid > cfg_f(cfg, "max_geo_residual"):
        halt("geo", f"Sim3 fit weak: {ninl}/{ntot} inliers (ratio {ratio:.2f}), residual "
                    f"{resid:.3f} m (need ratio >= {cfg['min_geo_inlier_ratio']}, residual "
                    f"<= {cfg['max_geo_residual']} m). GPS/SfM disagree - likely a bad capture.")
    print(f"  gate geo OK: {ninl}/{ntot} inliers (ratio {ratio:.2f}), residual {resid:.3f} m.")


def stage_dense(cfg, p, args):
    if not done(os.path.join(p.model0, "images.bin")):
        sys.exit("dense needs the undistorted workspace; run --from sfm.")
    if done(p.fused) and not args.force:
        print("  skip dense (fused.ply exists)")
    else:
        # (a) build the stereo workspace (identity undistort on the already-pinhole model)
        if not done(p.pm_cfg):
            run([cfg["colmap"], "image_undistorter", "--image_path", p.undist_images,
                 "--input_path", p.model0, "--output_path", p.dense, "--output_type", "COLMAP"])
        # (b) PatchMatch (always geom+filter per the contract; COLMAP self-skips done views)
        run([cfg["colmap"], "patch_match_stereo", "--workspace_path", p.dense,
             "--workspace_format", "COLMAP", "--PatchMatchStereo.gpu_index", "0",
             "--PatchMatchStereo.max_image_size", cfg["max_image_size"],
             "--PatchMatchStereo.geom_consistency", "true",
             "--PatchMatchStereo.filter", "true"])
        # (c) fuse
        run([cfg["colmap"], "stereo_fusion", "--workspace_path", p.dense,
             "--workspace_format", "COLMAP", "--input_type", "geometric",
             "--output_path", p.fused])
    npts = ply_vertex_count(p.fused)
    if npts < cfg_i(cfg, "min_dense_points"):
        halt("dense", f"only {npts} fused points (need >= {cfg['min_dense_points']}). On "
                      "Blackwell this is the arch-120 codegen bug - rebuild COLMAP with "
                      "-DCMAKE_CUDA_ARCHITECTURES=89 (see BUILD_COLMAP_CUDA.md).")
    print(f"  gate dense OK: {npts} fused points.")


def stage_georef(cfg, p, args):
    if not done(p.fused):
        sys.exit("georef needs fused.ply; run --from dense.")
    if not done(p.geo):
        sys.exit("georef needs geo_transform.txt; run --from geo.")
    out = ""
    if done(p.fused_utm) and not args.force:
        print("  skip georef (fused_utm.ply exists)")
        return
    out = run([cfg["python"], cfg["cloud_script"], p.fused, p.geo, p.fused_utm,
               "--sor-iters", cfg["sor_iters"], "--sor-k", cfg["sor_k"],
               "--sor-std", cfg["sor_std"]], capture=True)
    mc = re.search(r"E=([\-\d.]+)\s+N=([\-\d.]+)", out)
    me = re.search(r"UTM bbox:.*?extent=\[([^\]]+)\]", out)   # anchored: raw/clean COLMAP
    #                                                          # bboxes also print extent=[
    if not mc or not me:
        print("  gate georef: could not parse centroid/extent from output; skipping numeric check.")
        return
    E, N = float(mc.group(1)), float(mc.group(2))
    nums = re.findall(r"[\d.]+", me.group(1))
    zext = float(nums[2]) if len(nums) >= 3 else 0.0
    if not (cfg_f(cfg, "utm_e_min") <= E <= cfg_f(cfg, "utm_e_max") and
            cfg_f(cfg, "utm_n_min") <= N <= cfg_f(cfg, "utm_n_max")):
        halt("georef", f"UTM centroid E={E:.1f} N={N:.1f} outside the configured box - "
                       "wrong CRS or a bad Sim3.")
    if zext > cfg_f(cfg, "max_vertical_extent"):
        halt("georef", f"vertical extent {zext:.1f} m > {cfg['max_vertical_extent']} m - "
                       "floaters survived SOR; tighten --sor-std or inspect the cloud.")
    print(f"  gate georef OK: centroid E={E:.1f} N={N:.1f}, vertical extent {zext:.1f} m.")


def stage_splat(cfg, p, args):
    if not cfg_bool(cfg, "splat"):
        print("  skip splat (splat=false in config)")
        return
    if not cfg["gs_train"]:
        sys.exit("splat=true but gs_train (path to 3DGS train.py) is not set in the config.")
    if not done(os.path.join(p.model0, "images.bin")):
        sys.exit("splat needs the undistorted workspace; run --from sfm.")
    if not done(p.geo):
        sys.exit("splat needs geo_transform.txt; run --from geo.")
    iters = cfg["gs_iterations"]
    trained = p.trained_ply(iters)
    # (a) train the 3DGS splat
    if done(trained) and not args.force:
        print(f"  skip 3DGS training ({os.path.relpath(trained, p.wd)} exists)")
    else:
        run([cfg["python"], cfg["gs_train"], "-s", p.undist, "-m", p.gs_out,
             "--iterations", iters] + cfg["gs_extra_args"].split())
    if not done(trained):
        halt("splat", f"3DGS training did not produce {trained} "
                      "(check the train.py path / iterations).")
    # (b) georeference the splat
    if done(p.splat_utm) and not args.force:
        print("  skip georef_splat (UTM splat exists)")
    else:
        cmd = [cfg["python"], cfg["splat_script"], "--ply", trained, "--transform", p.geo,
               "--out", p.splat_utm]
        if cfg["dtm"]:
            cmd += ["--dtm", cfg["dtm"]]
        run(cmd)
    print(f"  splat OK: {p.splat_utm}")


STAGE_FN = {"ingest": stage_ingest, "sfm": stage_sfm, "geo": stage_geo,
            "dense": stage_dense, "georef": stage_georef, "splat": stage_splat}


# --- main --------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="One-command per-epoch reconstruction (Aspromonte).")
    ap.add_argument("--config", required=True, help="pipeline.conf (shared contract + paths)")
    ap.add_argument("--video", help="input video for this epoch (required unless starting past ingest)")
    ap.add_argument("--workdir", required=True, help="per-epoch output dir (use a NEW dir per epoch)")
    ap.add_argument("--from", dest="frm", choices=STAGES, default="ingest")
    ap.add_argument("--to", dest="to", choices=STAGES, default="splat")
    ap.add_argument("--force", action="store_true", help="re-run stages even if outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    p = P(args.workdir)
    i0, i1 = STAGES.index(args.frm), STAGES.index(args.to)
    if i0 > i1:
        sys.exit("--from is after --to.")
    todo = STAGES[i0:i1 + 1]
    if args.video is None and i0 == 0:
        sys.exit("--video is required when the range includes 'ingest'.")

    print(f"Pipeline: {args.workdir}  |  stages: {' -> '.join(todo)}"
          f"  |  splat={cfg_bool(cfg, 'splat')}")
    if args.dry_run:
        for s in todo:
            print(f"  would run: {s}")
        return

    os.makedirs(args.workdir, exist_ok=True)
    for s in todo:
        print(f"\n=== {s} ===")
        STAGE_FN[s](cfg, p, args)

    print(f"\nDone: {args.workdir}")
    if i1 >= STAGES.index("georef"):
        print(f"  dense UTM cloud: {p.fused_utm}")
    print("  Next (manual): run M3C2 between this epoch's fused_utm.ply and the other epoch's.")


if __name__ == "__main__":
    main()