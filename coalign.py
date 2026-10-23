#!/usr/bin/env python3
"""coalign.py - one Structure-from-Motion model for several passes over the same trail.

Each pass reconstructed alone keeps its own non-rigid drift, and a single rigid ICP between the
clouds cannot remove it: the leftover drift shows up as false changes. Reconstructing the passes
in one bundle adjustment, tied by matches between them on stable ground, removes it at the source
(Time-SIFT: Feurer and Vinatier 2018; Cook and Dietze 2019).

Steps:
  1. link the frames of each pass into <out>/images/<label>/ (one camera per folder: same GoPro,
     intrinsics refined per pass);
  2. extract SIFT features with the calibrated intrinsics;
  3. build the pair list: consecutive frames inside each pass (as sequential_matcher does) plus,
     across passes, every frame pair whose GPS positions are within --radius metres;
  4. match the listed pairs and verify them geometrically (matches_importer);
  5. run the mapper and report, per pass, the registered images and the 3D points seen by more
     than one pass (the tie points that hold the passes together);
  6. georeference the joint model once: one Sim3 fitted to the camera centres of every pass
     against that pass's own GPS (geo_align.py), written to <out>/geo_transform.txt;
  7. split the joint model per pass into <out>/passes/<label>/, laid out like a pass folder
     (frames, gps.csv, colmap/sparse/0, colmap/geo_transform.txt), so every pass keeps the
     common poses and the common georeference;
  8. dense reconstruction per pass with those poses (image_undistorter, patch_match_stereo,
     stereo_fusion, as run_pipeline.py) and georef_cloud.py to colmap/dense/fused_utm.ply.
     The clouds are then compared without ICP (measure_change.py --ref-seg <out>/passes/<a>
     --cmp-seg <out>/passes/<b>).

Each step writes a marker in <out>/ and is skipped on a re-run; delete the marker to redo it.
The source frames are only linked, never modified.

Usage:
  python3 coalign.py --pass ep2=~/Progetto_Aspromonte/seg01_ep2 \
                     --pass ep3=~/Progetto_Aspromonte/seg01_ep3 \
                     --out ~/Progetto_Aspromonte/coalign/seg01_ep2_ep3 \
                     --calibration ./calib_out/calibration_fisheye.json
"""
import argparse
import os
import shutil
import sqlite3
import struct
import sys

import numpy as np

from geo_align import fit_transform, frame_time, gps_world, load_gps, read_camera_centres, write_transform
from run_colmap import camera_params_from_calibration, largest_model, run

IMG_EXT = (".jpg", ".jpeg", ".png")
PAIR_ID_BASE = 2147483647                                  # COLMAP's pair_id encoding


def pass_frames(seg_dir):
    frames = os.path.join(seg_dir, "frames")
    names = sorted(f for f in os.listdir(frames) if f.lower().endswith(IMG_EXT))
    if not names:
        sys.exit(f"No frames in {frames}")
    return frames, names


def pass_positions(seg_dir, names, epsg):
    """Easting/northing of each frame, interpolated from the pass's gps.csv; NaN without a time."""
    from pyproj import Transformer
    t_s, lat, lon, _ = load_gps(os.path.join(seg_dir, "gps.csv"))
    ft = np.array([frame_time(n) if frame_time(n) is not None else np.nan for n in names])
    project = Transformer.from_crs(4326, epsg, always_xy=True).transform
    e, n = project(np.interp(ft, t_s, lon), np.interp(ft, t_s, lat))
    xy = np.column_stack([e, n])
    xy[np.isnan(ft)] = np.nan
    return xy


def build_pairs(passes, overlap, radius):
    """Within-pass pairs (i, i+1..i+overlap) and cross-pass pairs closer than radius."""
    from scipy.spatial import cKDTree
    pairs, n_seq, n_cross = [], 0, 0
    for label, names, _ in passes:
        for i in range(len(names)):
            for j in range(i + 1, min(i + 1 + overlap, len(names))):
                pairs.append((f"{label}/{names[i]}", f"{label}/{names[j]}"))
                n_seq += 1
    for a in range(len(passes)):
        for b in range(a + 1, len(passes)):
            la, na, xa = passes[a]
            lb, nb, xb = passes[b]
            ok_b = np.flatnonzero(np.isfinite(xb).all(1))
            tree = cKDTree(xb[ok_b])
            for i in np.flatnonzero(np.isfinite(xa).all(1)):
                for k in tree.query_ball_point(xa[i], radius):
                    pairs.append((f"{la}/{na[i]}", f"{lb}/{nb[ok_b[k]]}"))
                    n_cross += 1
    return pairs, n_seq, n_cross


def read_image_names(model_dir):
    names = {}
    with open(os.path.join(model_dir, "images.bin"), "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            iid = struct.unpack("<i", f.read(4))[0]
            f.seek(7 * 8 + 4, 1)                           # qvec, tvec, camera_id
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
            names[iid] = name.decode()
    return names


def read_point_tracks(model_dir):
    """(reprojection error, track image ids) for each 3D point."""
    out = []
    with open(os.path.join(model_dir, "points3D.bin"), "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            f.seek(8 + 24 + 3, 1)                          # id, xyz, rgb
            err, n = struct.unpack("<dQ", f.read(16))
            track = np.frombuffer(f.read(8 * n), dtype="<i4")[0::2]
            out.append((err, track))
    return out


def georeference(model, segs, epsg, path):
    """One Sim3 for all passes: every registered camera centre against its own pass's GPS."""
    centres = read_camera_centres(model)
    world = {}
    for label, seg in segs:
        names = [n for n in centres if n.startswith(label + "/")]
        world.update(gps_world(os.path.join(seg, "gps.csv"), names, epsg))
    c, R, t, res, n_common = fit_transform(centres, world)
    write_transform(path, epsg, c, R, t, res, n_common)
    return (f"georeference: {len(res)}/{n_common} frames of all passes kept, GPS residual median "
            f"{np.median(res):.2f} m, scale {c:.5f}")


def strip_prefix(model_dir, prefix):
    """Rewrite images.bin with the pass prefix dropped from every image name."""
    path = os.path.join(model_dir, "images.bin")
    with open(path, "rb") as f:
        src = f.read()
    out, pos = bytearray(src[:8]), 8
    for _ in range(struct.unpack_from("<Q", src)[0]):
        end = src.index(b"\0", pos + 64)                   # id, qvec, tvec, camera_id: 64 bytes
        name = src[pos + 64:end].decode()
        stop = end + 1 + 8 + 24 * struct.unpack_from("<Q", src, end + 1)[0]
        out += src[pos:pos + 64] + name.removeprefix(prefix).encode() + src[end:stop]
        pos = stop
    with open(path, "wb") as f:
        f.write(out)


def split_pass(model, label, dst, colmap):
    """The joint model restricted to one pass (image_deleter), with plain frame names."""
    tmp = dst + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    others = os.path.join(tmp, "others.txt")
    with open(others, "w") as f:
        f.writelines(n + "\n" for n in read_image_names(model).values() if not n.startswith(label + "/"))
    run([colmap, "image_deleter", "--input_path", model, "--output_path", tmp,
         "--image_names_path", others])
    os.remove(others)
    strip_prefix(tmp, label + "/")
    os.replace(tmp, dst)


def ply_vertex_count(path):
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.startswith(b"end_header"):
                return 0
    return 0


def report(db, sparse, labels):
    lines = []
    con = sqlite3.connect(db)
    img_label = {iid: name.split("/")[0] for iid, name in con.execute("SELECT image_id, name FROM images")}
    cross = [rows for pid, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0")
             if img_label[pid // PAIR_ID_BASE] != img_label[pid % PAIR_ID_BASE]]
    con.close()
    cross = np.array(cross, dtype=float)
    lines.append(f"cross-pass verified pairs: {len(cross)}"
                 + (f" | inliers median {np.median(cross):.0f}, P90 {np.percentile(cross, 90):.0f}"
                    if len(cross) else ""))
    n_models = sum(os.path.isdir(os.path.join(sparse, d)) for d in os.listdir(sparse))
    model = largest_model(sparse)
    if model is None:
        lines.append("mapper: no model")
        return lines
    names = read_image_names(model)
    lines.append(f"models: {n_models} | largest: {os.path.basename(model)}")
    for lab in labels:
        n_reg = sum(1 for v in names.values() if v.startswith(lab + "/"))
        n_tot = sum(1 for v in img_label.values() if v == lab)
        lines.append(f"  {lab}: {n_reg}/{n_tot} images registered")
    pts = read_point_tracks(model)
    tie = [e for e, tr in pts if len({names[i].split('/')[0] for i in tr}) > 1]
    errs = np.array([e for e, _ in pts])
    lines.append(f"3D points: {len(pts)} | seen by more than one pass: {len(tie)}"
                 f" | mean reprojection error {errs.mean():.3f} px (tie points {np.mean(tie):.3f} px)"
                 if tie else f"3D points: {len(pts)} | seen by more than one pass: 0")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="passes", action="append", required=True, metavar="LABEL=SEG_DIR",
                    help="a pass: label and folder holding frames/ and gps.csv (repeat, at least 2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", required=True, help="calibration JSON (calibrate_camera.py)")
    ap.add_argument("--colmap", default="colmap")
    ap.add_argument("--epsg", type=int, default=32633)
    ap.add_argument("--overlap", type=int, default=10, help="within-pass neighbours per frame")
    ap.add_argument("--radius", type=float, default=6.0,
                    help="cross-pass pairs: maximum GPS distance between the two frames [m]")
    ap.add_argument("--matching-gpu", default="0", choices=["0", "1"])
    ap.add_argument("--max-image-size", default="1000",
                    help="PatchMatch image size [px] (pipeline.conf max_image_size)")
    ap.add_argument("--sor-iters", default="2", help="georef_cloud.py floater removal, as pipeline.conf")
    ap.add_argument("--sor-k", default="20")
    ap.add_argument("--sor-std", default="2.0")
    args = ap.parse_args()

    if len(args.passes) < 2:
        sys.exit("Need at least two --pass.")
    out = os.path.abspath(os.path.expanduser(args.out))
    images, db, sparse = (os.path.join(out, d) for d in ("images", "database.db", "sparse"))
    os.makedirs(sparse, exist_ok=True)
    done = lambda step: os.path.join(out, f".done_{step}")

    passes, labels, segs, first_frames = [], [], [], None
    for spec in args.passes:
        label, _, seg = spec.partition("=")
        seg = os.path.abspath(os.path.expanduser(seg))
        frames, names = pass_frames(seg)
        first_frames = first_frames or frames
        os.makedirs(os.path.join(images, label), exist_ok=True)
        for n in names:
            link = os.path.join(images, label, n)
            if not os.path.lexists(link):
                os.symlink(os.path.join(frames, n), link)
        passes.append((label, names, pass_positions(seg, names, args.epsg)))
        labels.append(label)
        segs.append((label, seg))
        print(f"  pass {label}: {len(names)} frames from {frames}")

    if not os.path.exists(done("extract")):
        model, params = camera_params_from_calibration(args.calibration, first_frames)
        run([args.colmap, "feature_extractor", "--database_path", db, "--image_path", images,
             "--ImageReader.single_camera_per_folder", "1",
             "--ImageReader.camera_model", model, "--ImageReader.camera_params", params,
             "--SiftExtraction.use_gpu", "1"])
        open(done("extract"), "w").close()

    if not os.path.exists(done("match")):
        pairs, n_seq, n_cross = build_pairs(passes, args.overlap, args.radius)
        print(f"  pairs: {n_seq} within passes + {n_cross} across passes (GPS radius {args.radius:g} m)")
        pair_file = os.path.join(out, "pairs.txt")
        with open(pair_file, "w") as f:
            f.writelines(f"{a} {b}\n" for a, b in pairs)
        run([args.colmap, "matches_importer", "--database_path", db, "--match_list_path", pair_file,
             "--match_type", "pairs", "--SiftMatching.use_gpu", args.matching_gpu])
        open(done("match"), "w").close()

    if not os.path.exists(done("map")):
        run([args.colmap, "mapper", "--database_path", db, "--image_path", images,
             "--output_path", sparse])
        open(done("map"), "w").close()

    lines = report(db, sparse, labels)
    model = largest_model(sparse)
    if model is None:
        sys.exit("\n".join(lines))

    geo = os.path.join(out, "geo_transform.txt")
    if not os.path.exists(done("georef")):
        print("  " + georeference(model, segs, args.epsg, geo))
        open(done("georef"), "w").close()
    lines.append(open(geo).readlines()[1].strip("# \n"))

    here = os.path.dirname(os.path.abspath(__file__))
    for label, seg in segs:
        pdir = os.path.join(out, "passes", label)
        cm, dense = os.path.join(pdir, "colmap"), os.path.join(pdir, "colmap", "dense")
        sub, fused = os.path.join(cm, "sparse", "0"), os.path.join(dense, "fused.ply")
        os.makedirs(os.path.join(cm, "sparse"), exist_ok=True)
        for name in ("frames", "gps.csv"):
            if not os.path.lexists(os.path.join(pdir, name)):
                os.symlink(os.path.join(seg, name), os.path.join(pdir, name))
        if not os.path.isdir(sub):
            split_pass(model, label, sub, args.colmap)
        shutil.copyfile(geo, os.path.join(cm, "geo_transform.txt"))
        if not os.path.exists(fused):
            if not os.path.exists(os.path.join(dense, "patch-match.cfg")):
                run([args.colmap, "image_undistorter", "--image_path", os.path.join(seg, "frames"),
                     "--input_path", sub, "--output_path", dense, "--output_type", "COLMAP"])
            run([args.colmap, "patch_match_stereo", "--workspace_path", dense,
                 "--workspace_format", "COLMAP", "--PatchMatchStereo.gpu_index", "0",
                 "--PatchMatchStereo.max_image_size", args.max_image_size,
                 "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.filter", "true"])
            run([args.colmap, "stereo_fusion", "--workspace_path", dense, "--workspace_format", "COLMAP",
                 "--input_type", "geometric", "--output_path", fused])
        fused_utm = os.path.join(dense, "fused_utm.ply")
        if not os.path.exists(fused_utm):
            run([sys.executable, os.path.join(here, "georef_cloud.py"), fused,
                 os.path.join(cm, "geo_transform.txt"), fused_utm, "--sor-iters", args.sor_iters,
                 "--sor-k", args.sor_k, "--sor-std", args.sor_std])
        lines.append(f"  {label}: {ply_vertex_count(fused)} fused points -> {fused_utm}")

    with open(os.path.join(out, "coalign_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
