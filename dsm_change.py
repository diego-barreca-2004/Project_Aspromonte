#!/usr/bin/env python3
"""dsm_change.py - compact-object change detection on the trail by DSM differencing.

M3C2 (run_m3c2.py) measures distributed surface change along local normals. Small
rigid objects on a near-horizontal trail (tens of centimetres) are found more
reliably by differencing two gridded surface models:

  1. keep the points of both epochs inside a corridor around the GPS track;
  2. grid each epoch at --cell and take the --pct height percentile per cell;
  3. keep only cells where the surface the change is measured against is smooth
     bare ground: the P90-P10 height spread in the cell is below --max-rough. This
     rejects vegetation, whose height changes between epochs for benign reasons;
  4. subtract, remove the median offset, and estimate the noise as the NMAD;
  5. report connected blobs higher than max(--min-height, --k x NMAD) whose area
     lies in [--min-area, --max-area]: "appeared" (dz > 0) and "removed" (dz < 0).

Usage (epoch 3 against epoch 2, with the same registration as the M3C2 run):

  python3 dsm_change.py --ref seg01_ep2/colmap/dense/fused_utm.ply \\
      --cmp seg01_ep3/colmap/dense/fused_utm.ply \\
      --icp m3c2_ep3_vs_ep2_v2/icp_auto.txt --cc-shift -558997 -4214492 -123 \\
      --track seg01_ep2/gps.csv --out dsm_ep3_vs_ep2

Outputs in --out: dsm_report.txt, dsm_detections.csv, dsm_map.png.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

from run_m3c2 import nmad, parse_cc_matrices, read_ply_xyz


def grid_stats(pts, origin, cell, shape, pct, min_pts):
    """Per-cell height percentile and P90-P10 spread; NaN where < min_pts points."""
    nx, ny = shape
    ix = np.floor((pts[:, 0] - origin[0]) / cell).astype(np.int64)
    iy = np.floor((pts[:, 1] - origin[1]) / cell).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    key, z = ix[inside] * ny + iy[inside], pts[inside, 2]
    order = np.lexsort((z, key))
    key, z = key[order], z[order]
    cells, start, count = np.unique(key, return_index=True, return_counts=True)
    keep = count >= min_pts
    cells, start, count = cells[keep], start[keep], count[keep]

    def quantile(q):
        return z[start + np.floor(q * (count - 1)).astype(np.int64)]

    height = np.full(nx * ny, np.nan)
    spread = np.full(nx * ny, np.nan)
    height[cells] = quantile(pct / 100.0)
    spread[cells] = quantile(0.9) - quantile(0.1)
    return height.reshape(shape), spread.reshape(shape)


def read_track(paths, epsg, trim_ends=0.0):
    """Track vertices in UTM; with trim_ends, drop the first and last trim_ends metres
    of each file (sequence ends are seen from few views and drift the most)."""
    from pyproj import Transformer
    trafo = Transformer.from_crs(4326, epsg, always_xy=True)
    parts = []
    for path in paths:
        tp = np.asarray([trafo.transform(float(r["lon"]), float(r["lat"]))
                         for r in csv.DictReader(open(path)) if r.get("lat") and r.get("lon")],
                        float).reshape(-1, 2)
        if trim_ends > 0 and len(tp):
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(tp, axis=0).T))]
            tp = tp[(s >= trim_ends) & (s <= s[-1] - trim_ends)]
        parts.append(tp)
    tp = np.vstack(parts)
    if not len(tp):
        sys.exit("--track: no usable lat/lon rows found (or --trim-ends too large).")
    return tp


def main(argv=None):
    ap = argparse.ArgumentParser(description="DSM-difference object detection on the trail.")
    ap.add_argument("--ref", required=True, help="reference epoch PLY (older)")
    ap.add_argument("--cmp", required=True, help="compared epoch PLY (newer)")
    ap.add_argument("--icp", help="CloudCompare ICP matrix txt (applied to --cmp)")
    ap.add_argument("--matrix-index", type=int, default=0,
                    help="which matrix to use if the file contains several (default 0 = first)")
    ap.add_argument("--cc-shift", nargs=3, type=float, metavar=("SX", "SY", "SZ"),
                    help="CloudCompare Global Shift the matrix was estimated in "
                         "(REQUIRED with --icp)")
    ap.add_argument("--track", action="append", required=True, metavar="GPS_CSV",
                    help="gps.csv of an epoch (repeatable); defines the trail corridor")
    ap.add_argument("--track-epsg", type=int, default=32633,
                    help="UTM EPSG for --track conversion (default 32633)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--corridor", type=float, default=1.0,
                    help="half-width of the trail corridor around the track [m]")
    ap.add_argument("--trim-ends", type=float, default=0.0,
                    help="exclude the first and last metres of each track (weakly "
                         "constrained sequence ends) [m]")
    ap.add_argument("--cell", type=float, default=0.05, help="DSM cell size [m]")
    ap.add_argument("--pct", type=float, default=90.0,
                    help="height percentile per cell (upper surface, robust to floaters)")
    ap.add_argument("--min-pts", type=int, default=3, help="minimum points per valid cell")
    ap.add_argument("--max-rough", type=float, default=0.05,
                    help="max P90-P10 height spread [m] for a cell to count as bare ground")
    ap.add_argument("--k", type=float, default=4.0, help="detection threshold in NMADs")
    ap.add_argument("--min-height", type=float, default=0.10,
                    help="minimum |dz| of a detection [m], whatever the noise")
    ap.add_argument("--min-area", type=float, default=0.04,
                    help="minimum blob area [m^2] (0.04 = a 20 x 20 cm object)")
    ap.add_argument("--max-area", type=float, default=1.0, help="maximum blob area [m^2]")
    ap.add_argument("--sat", type=float, default=0.3, help="map colour saturation [m]")
    args = ap.parse_args(argv)
    if args.icp and args.cc_shift is None:
        sys.exit("--cc-shift is REQUIRED with --icp: the matrix lives in the shifted frame.")

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    report = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    log(f"# dsm_change.py  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("# cmd: " + " ".join(sys.argv if argv is None else ["dsm_change.py"] + list(argv)))

    # --- load, local frame, registration ---
    log("\n=== load ===")
    ref, cmp_ = read_ply_xyz(args.ref), read_ply_xyz(args.cmp)
    shift = (np.array(args.cc_shift, float) if args.cc_shift is not None
             else -np.floor(ref.min(axis=0)))
    ref_l, cmp_l = ref + shift, cmp_ + shift
    log(f"  ref: {args.ref}  ({len(ref):,} pts)")
    log(f"  cmp: {args.cmp}  ({len(cmp_):,} pts)")
    if args.icp:
        mats = parse_cc_matrices(args.icp)
        if not mats:
            sys.exit(f"No 4x4 matrix found in {args.icp}.")
        M = mats[args.matrix_index]
        cmp_l = (M[:3, :3] @ cmp_l.T).T + M[:3, 3]
        log(f"  registration: matrix [{args.matrix_index}] from {args.icp} applied to cmp")

    # --- corridor ---
    from scipy.spatial import cKDTree
    track = cKDTree(read_track(args.track, args.track_epsg, args.trim_ends) + shift[:2])
    ref_l = ref_l[track.query(ref_l[:, :2])[0] <= args.corridor]
    cmp_l = cmp_l[track.query(cmp_l[:, :2])[0] <= args.corridor]
    if args.trim_ends:
        log(f"  track ends trimmed: {args.trim_ends:g} m at each end")
    log(f"  corridor <= {args.corridor:g} m: ref {len(ref_l):,} pts, cmp {len(cmp_l):,} pts")
    if len(ref_l) < 1000 or len(cmp_l) < 1000:
        sys.exit("Corridor holds <1000 points per epoch: check --track / --corridor.")

    # --- DSMs ---
    log("\n=== DSM difference ===")
    both = np.vstack([ref_l[:, :2], cmp_l[:, :2]])
    origin = both.min(axis=0)
    shape = tuple((np.ceil((both.max(axis=0) - origin) / args.cell) + 1).astype(int))
    h_ref, s_ref = grid_stats(ref_l, origin, args.cell, shape, args.pct, args.min_pts)
    h_cmp, s_cmp = grid_stats(cmp_l, origin, args.cell, shape, args.pct, args.min_pts)
    dz = h_cmp - h_ref
    valid = np.isfinite(dz)
    smooth = valid & (s_ref <= args.max_rough) & (s_cmp <= args.max_rough)
    if smooth.sum() < 100:
        sys.exit("Fewer than 100 bare-ground cells: check --max-rough / --cell.")
    offset = float(np.median(dz[smooth]))
    dz = dz - offset
    noise = float(nmad(dz[smooth]))
    thr = max(args.min_height, args.k * noise)
    log(f"  grid {shape[0]} x {shape[1]} cells of {args.cell} m | percentile {args.pct:g}")
    log(f"  cells with both epochs: {valid.sum():,} | bare ground in both: {smooth.sum():,}")
    log(f"  median offset removed: {offset:+.4f} m | NMAD on bare ground: {noise:.4f} m")
    log(f"  detection threshold: max({args.min_height} m, {args.k:g} x NMAD) = {thr:.3f} m")

    # --- blobs ---
    from scipy import ndimage
    rows = []
    for kind, sign, rough in (("appeared", 1, s_ref), ("removed", -1, s_cmp)):
        # the change is measured against bare ground: the reference surface for an
        # object that appeared, the compared surface for one that was removed
        mask = valid & (rough <= args.max_rough) & (sign * dz > thr)
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            ii, jj = np.nonzero(lab == k)
            area = len(ii) * args.cell ** 2
            if not args.min_area <= area <= args.max_area:
                continue
            vals = dz[ii, jj]
            rows.append(dict(
                kind=kind,
                E=origin[0] + (ii.mean() + 0.5) * args.cell - shift[0],
                N=origin[1] + (jj.mean() + 0.5) * args.cell - shift[1],
                area_m2=area, dz_median=float(np.median(vals)),
                dz_peak=float(vals[np.argmax(sign * vals)]), n_cells=len(ii)))
    rows.sort(key=lambda r: -r["area_m2"] * abs(r["dz_median"]))
    log(f"\n=== detections ({len(rows)}) ===")
    log(f"  {'#':>3} {'kind':>8} {'E':>12} {'N':>13} {'area m2':>8} {'dz med':>8} {'dz peak':>8}")
    for i, r in enumerate(rows):
        log(f"  {i:>3} {r['kind']:>8} {r['E']:12.2f} {r['N']:13.2f} {r['area_m2']:8.3f} "
            f"{r['dz_median']:+8.3f} {r['dz_peak']:+8.3f}")

    # --- outputs ---
    with open(os.path.join(args.out, "dsm_detections.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "E", "N", "area_m2", "dz_median",
                                          "dz_peak", "n_cells"])
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = (origin[0] - shift[0], origin[0] - shift[0] + shape[0] * args.cell,
           origin[1] - shift[1], origin[1] - shift[1] + shape[1] * args.cell)
    fig, ax = plt.subplots(figsize=(16, 6))
    shown = np.where(smooth, dz, np.nan)
    im = ax.imshow(shown.T, origin="lower", extent=ext, cmap="RdBu_r",
                   vmin=-args.sat, vmax=args.sat, interpolation="nearest")
    for i, r in enumerate(rows):
        ax.plot(r["E"], r["N"], "o", mfc="none", mec="k", ms=14, mew=1.5)
        ax.annotate(str(i), (r["E"], r["N"]), xytext=(6, 6), textcoords="offset points")
    ax.set_title(f"DSM difference on bare ground (NMAD {noise * 100:.1f} cm); "
                 f"detections above {thr * 100:.0f} cm circled")
    ax.set_xlabel("E [m]")
    ax.set_ylabel("N [m]")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="dz cmp - ref [m]")
    fig.savefig(os.path.join(args.out, "dsm_map.png"), dpi=110, bbox_inches="tight")
    log(f"\n  wrote {args.out}/dsm_detections.csv, dsm_map.png, dsm_report.txt")
    log(f"Done in {time.time() - t0:.1f} s.")
    with open(os.path.join(args.out, "dsm_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    return rows, noise


if __name__ == "__main__":
    main()
