#!/usr/bin/env python3
"""run_m3c2.py - scripted M3C2 change detection between two georeferenced epochs.

Replaces the interactive CloudCompare M3C2 workflow with a reproducible script:

  inputs   two fused_utm.ply (absolute UTM, float64) straight from the per-epoch
           pipeline, untouched on disk, plus the fine-registration (ICP) matrix
           exported from CloudCompare.
  steps    apply ICP to the compared epoch -> crop both clouds to their mutual
           XY footprint -> voxel-pick core points on the reference -> M3C2
           (py4dgeo, the reference implementation by the 3DGeo group) -> stats
           on user-defined stable patches.
  outputs  <out>/m3c2_core.ply          core points + scalar fields (opens in
                                        CloudCompare with SFs preloaded)
           <out>/m3c2_map.png           top view: distance map + significant-only map
           <out>/m3c2_patches_hist.png  per-patch histograms (if --patch given)
           <out>/m3c2_patch_stats.csv   per-patch mean/median/std/NMAD/sig-fraction
           <out>/m3c2_report.txt        every parameter, matrix, count and stat

Conventions (chosen to match the CloudCompare qM3C2 dialog):
  * --normal-d / --proj-d are DIAMETERS like the CC dialog (D and d).
    py4dgeo wants radii; the conversion is done here and echoed in the report.
  * Sign: reference = epoch 1 (older), compared = epoch 2 (newer), normals
    oriented +Z  =>  positive = material ADDED, negative = material REMOVED.
  * The ICP matrix is the CC "Applied transformation matrix" (the one with the
    SMALL translation), which lives in CloudCompare's Global-Shift frame, so
    --cc-shift (the shift CC printed at load time) is REQUIRED with --icp.

Registration error: run once with --reg-error 0, measure mean/std on stable
patches, then re-run with --reg-error <measured> so the significant-change
flag is honest. M3C2 stays outside run_pipeline.py on purpose: it is a
cross-epoch comparison, the orchestrator is per-epoch.

Usage:
  python3 run_m3c2.py --ref seg01/colmap/dense/fused_utm.ply \
                      --cmp seg01_ep2/colmap/dense/fused_utm.ply \
                      --icp icp_ep2_to_ep1.txt --cc-shift -559000 -4214000 0 \
                      --out m3c2_out \
                      [--patch E,N,r ...] [--reg-error 0.0]
  # or estimate the registration here and restrict to the trajectory corridor:
  python3 run_m3c2.py --ref ... --cmp ... --auto-icp \\
                      --track seg01/gps.csv --track seg01_ep2/gps.csv --max-track-dist 10 \\
                      --out m3c2_out
"""
import argparse
import csv
import os
import re
import sys
import time

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    sys.exit("Missing dependency: pip install plyfile")


# --- PLY I/O -------------------------------------------------------------------
def read_ply_xyz(path):
    """(N,3) float64 xyz from a PLY vertex element; extra properties are ignored."""
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    ply = PlyData.read(path)
    if "vertex" not in [el.name for el in ply.elements]:
        sys.exit(f"{path}: no 'vertex' element in PLY.")
    v = ply["vertex"]
    names = v.data.dtype.names
    if not all(c in names for c in ("x", "y", "z")):
        sys.exit(f"{path}: vertex element lacks x/y/z (has: {names}).")
    return np.ascontiguousarray(
        np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64))


def write_core_ply(path, pts_global, dist, lod, sig, unc, track_dist=None):
    """Core points + results as a PLY CloudCompare opens with the SFs preloaded."""
    n = len(pts_global)
    fields = [
        ("x", "f8"), ("y", "f8"), ("z", "f8"),
        ("scalar_M3C2_distance", "f8"),
        ("scalar_distance_uncertainty", "f8"),
        ("scalar_significant", "u1"),
        ("scalar_spread1", "f4"), ("scalar_spread2", "f4"),
        ("scalar_Npoints1", "f4"), ("scalar_Npoints2", "f4"),
    ]
    if track_dist is not None:
        fields.append(("scalar_dist_to_track", "f4"))
    arr = np.empty(n, dtype=fields)
    if track_dist is not None:
        arr["scalar_dist_to_track"] = track_dist
    arr["x"], arr["y"], arr["z"] = pts_global.T
    arr["scalar_M3C2_distance"] = dist
    arr["scalar_distance_uncertainty"] = lod
    arr["scalar_significant"] = sig.astype(np.uint8)
    arr["scalar_spread1"] = unc["spread1"]
    arr["scalar_spread2"] = unc["spread2"]
    arr["scalar_Npoints1"] = unc["num_samples1"]
    arr["scalar_Npoints2"] = unc["num_samples2"]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


# --- CloudCompare ICP matrix ---------------------------------------------------
def parse_cc_matrices(path):
    """All 4x4 matrices found in a text file, tolerant to pasted console noise.

    A matrix is 4 consecutive lines with exactly 4 floats each after stripping
    bracketed console timestamps; the last row must be ~[0,0,0,1]."""
    if not os.path.isfile(path):
        sys.exit(f"ICP matrix file not found: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            line = re.sub(r"\[[^\]]*\]", " ", line)          # strip [21:04:23] etc.
            floats = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", line)
            rows.append([float(x) for x in floats] if len(floats) == 4 else None)
    mats = []
    for i in range(len(rows) - 3):
        block = rows[i:i + 4]
        if all(r is not None for r in block):
            m = np.array(block)
            if np.allclose(m[3], [0, 0, 0, 1], atol=1e-9):
                mats.append(m)
    return mats


# --- geometry helpers ----------------------------------------------------------
def mutual_footprint_crop(a, b, cell, dilate):
    """Masks keeping each cloud only where the OTHER one also has XY coverage
    (occupancy grids dilated by `dilate` cells). Kills far clusters/floaters."""
    from scipy import ndimage
    mn = np.minimum(a[:, :2].min(0), b[:, :2].min(0)) - cell
    ia = np.floor((a[:, :2] - mn) / cell).astype(np.int64)
    ib = np.floor((b[:, :2] - mn) / cell).astype(np.int64)
    shape = (int(max(ia[:, 0].max(), ib[:, 0].max())) + 2,
             int(max(ia[:, 1].max(), ib[:, 1].max())) + 2)
    occ_a = np.zeros(shape, bool)
    occ_b = np.zeros(shape, bool)
    occ_a[ia[:, 0], ia[:, 1]] = True
    occ_b[ib[:, 0], ib[:, 1]] = True
    occ_a = ndimage.binary_dilation(occ_a, iterations=dilate)
    occ_b = ndimage.binary_dilation(occ_b, iterations=dilate)
    return occ_b[ia[:, 0], ia[:, 1]], occ_a[ib[:, 0], ib[:, 1]]


def voxel_pick(pts, spacing):
    """Indices of one existing point per (spacing)^3 voxel (spatial subsample)."""
    ijk = np.floor((pts - pts.min(0)) / spacing).astype(np.int64)
    key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
    _, idx = np.unique(key, return_index=True)
    return np.sort(idx)


def trimmed_p2plane_icp(ref_pts, ref_nrm, mov, iters=60, trim=0.7, max_pair=1.0):
    """Trimmed point-to-plane ICP refine: at each iteration keeps only the `trim`
    fraction of correspondences with the smallest |plane residual|, so changed
    vegetation / real change cannot bias the solution (same idea as CloudCompare's
    'final overlap'). Returns (R, t) with p' = R p + t. Validated on synthetic
    trail fixtures where the untrimmed cascade leaves a 4-6 cm vegetation bias."""
    from scipy.spatial import cKDTree
    tree = cKDTree(ref_pts)
    P = mov.copy()
    for _ in range(iters):
        d, j = tree.query(P, workers=-1)
        m = d < max_pair
        p, q, n = P[m], ref_pts[j[m]], ref_nrm[j[m]]
        r = np.einsum("ij,ij->i", n, p - q)
        keep = np.abs(r) <= np.quantile(np.abs(r), trim)
        p, n, r = p[keep], n[keep], r[keep]
        cc = p.mean(0)
        A = np.c_[np.cross(p - cc, n), n]
        x, *_ = np.linalg.lstsq(A, -r, rcond=None)
        w, dt = x[:3], x[3:]
        th = float(np.linalg.norm(w))
        if th > 1e-12:
            k = w / th
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            dR = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        else:
            dR = np.eye(3)
        P = (dR @ (P - cc).T).T + cc + dt
        if th < 1e-8 and np.linalg.norm(dt) < 1e-6:
            break
    cm, cp = mov.mean(0), P.mean(0)
    H = (mov - cm).T @ (P - cp)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cp - R @ cm


def nmad(x):
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))


# --- main ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Scripted M3C2 between two epochs.")
    ap.add_argument("--ref", required=True, help="reference epoch PLY (older, epoch 1)")
    ap.add_argument("--cmp", required=True, help="compared epoch PLY (newer, epoch 2)")
    ap.add_argument("--icp", help="CloudCompare ICP matrix txt (applied to --cmp)")
    ap.add_argument("--matrix-index", type=int, default=0,
                    help="which matrix to use if the file contains several (default 0 = first)")
    ap.add_argument("--cc-shift", nargs=3, type=float, metavar=("SX", "SY", "SZ"),
                    help="CloudCompare Global Shift at load time (REQUIRED with --icp); "
                         "e.g. -559000 -4214000 0")
    ap.add_argument("--out", required=True, help="output directory")
    # contract-style parameters, CloudCompare DIAMETER convention
    ap.add_argument("--normal-d", type=float, default=1.0, help="normal scale D, diameter [m]")
    ap.add_argument("--proj-d", type=float, default=0.25, help="projection scale d, diameter [m]")
    ap.add_argument("--max-depth", type=float, default=3.0, help="max search depth along normal [m]")
    ap.add_argument("--core-spacing", type=float, default=0.10, help="core-point voxel spacing [m]")
    ap.add_argument("--reg-error", type=float, default=0.0,
                    help="registration error folded into the LoD (2nd pass; measure it first)")
    ap.add_argument("--crop-cell", type=float, default=1.0, help="footprint-crop grid cell [m]")
    ap.add_argument("--crop-dilate", type=int, default=2, help="footprint dilation [cells]")
    ap.add_argument("--no-crop", action="store_true", help="skip the mutual-footprint crop")
    ap.add_argument("--patch", action="append", default=[], metavar="E,N,r",
                    help="stable patch: UTM easting,northing,radius [m]; repeatable")
    ap.add_argument("--sat", type=float, default=0.5, help="map colour saturation [m]")
    ap.add_argument("--auto-icp", action="store_true",
                    help="estimate the fine registration here (py4dgeo ICP: point-to-point "
                         "then TRIMMED point-to-plane refine) instead of using a CloudCompare matrix")
    ap.add_argument("--track", action="append", default=[], metavar="GPS_CSV",
                    help="gps.csv of an epoch (repeatable); enables distance-to-trajectory")
    ap.add_argument("--track-epsg", type=int, default=32633,
                    help="UTM EPSG for --track conversion (default 32633)")
    ap.add_argument("--icp-trim", type=float, default=0.7,
                    help="auto-ICP stage-2 trimming fraction (keep best X of "
                         "correspondences; lower = more robust to vegetation)")
    ap.add_argument("--max-track-dist", type=float,
                    help="mask core points farther than this [m] from the trajectory "
                         "(domain-of-validity corridor; requires --track)")
    args = ap.parse_args(argv)
    if args.icp and args.auto_icp:
        sys.exit("--icp and --auto-icp are mutually exclusive.")
    if args.max_track_dist and not args.track:
        sys.exit("--max-track-dist requires --track.")

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    report = []

    def log(msg=""):
        print(msg)
        report.append(str(msg))

    log(f"# run_m3c2.py  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"# cmd: {' '.join(sys.argv)}")

    # --- load ---
    log("\n=== load ===")
    ref_g = read_ply_xyz(args.ref)
    cmp_g = read_ply_xyz(args.cmp)
    log(f"  ref: {args.ref}  ({len(ref_g):,} pts)")
    log(f"  cmp: {args.cmp}  ({len(cmp_g):,} pts)")

    # --- frames: work in the CloudCompare shifted (local) frame ---
    if args.icp and args.cc_shift is None:
        sys.exit("--cc-shift is REQUIRED with --icp: the CC matrix lives in the "
                 "Global-Shift frame (the 'Translation: (...)' CC printed at load).")
    shift = np.array(args.cc_shift, float) if args.cc_shift is not None \
        else -np.round(ref_g.mean(0))
    ref_l = ref_g + shift
    cmp_l = cmp_g + shift
    log(f"  local frame shift: {shift.tolist()}"
        + ("" if args.cc_shift is not None else "  (auto)"))

    # --- ICP ---
    log("\n=== registration ===")
    if args.icp:
        mats = parse_cc_matrices(args.icp)
        if not mats:
            sys.exit(f"No 4x4 matrix found in {args.icp}.")
        if len(mats) > 1:
            log(f"  WARNING: {len(mats)} matrices found in {args.icp}; using index "
                f"{args.matrix_index}. CC prints the shifted-frame 'Applied "
                f"transformation matrix' FIRST - that is the right one.")
        M = mats[args.matrix_index]
        R, t = M[:3, :3], M[:3, 3]
        det = float(np.linalg.det(R))
        scale = float(np.cbrt(abs(det)))
        log(f"  matrix [{args.matrix_index}] from {args.icp}:")
        for r in M:
            log("    " + "  ".join(f"{v: .9f}" for v in r))
        log(f"  |det R|^(1/3) = {scale:.9f} (must be ~1: rigid, no scale)")
        if abs(scale - 1.0) > 1e-3:
            sys.exit("Matrix is not rigid (scale != 1). Wrong matrix? Aborting.")
        cmp_l = (R @ cmp_l.T).T + t
        rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        log(f"  applied to cmp: rotation {rot_deg:.3f} deg, "
            f"|t_local| = {np.linalg.norm(t):.3f} m")
    else:
        log("  auto-ICP after the crop." if args.auto_icp else
            "  no --icp given: assuming clouds are already co-registered.")

    # --- crop ---
    log("\n=== mutual-footprint crop ===")
    if args.no_crop:
        log("  skipped (--no-crop)")
    else:
        ka, kb = mutual_footprint_crop(ref_l, cmp_l, args.crop_cell, args.crop_dilate)
        log(f"  ref: kept {ka.sum():,} / {len(ka):,} ({100 * ka.mean():.1f}%)")
        log(f"  cmp: kept {kb.sum():,} / {len(kb):,} ({100 * kb.mean():.1f}%)")
        ref_l, cmp_l = ref_l[ka], cmp_l[kb]
        if len(ref_l) < 1000 or len(cmp_l) < 1000:
            sys.exit("Crop left <1000 points: no real overlap between the clouds?")

    # --- auto ICP (estimated here, on the cropped clouds) ---
    if args.auto_icp:
        log("\n=== auto-ICP (py4dgeo point-to-point, then trimmed point-to-plane refine) ===")
        import py4dgeo

        def _as_Rb(tr):
            A = tr.affine_transformation
            rp = np.asarray(tr.reduction_point, float)
            R = A[:3, :3]
            return R, A[:3, 3] + rp - R @ rp

        sub_r = ref_l[voxel_pick(ref_l, 0.15)]
        sub_c = cmp_l[voxel_pick(cmp_l, 0.15)]
        log(f"  estimating on subsampled clouds: {len(sub_r):,} / {len(sub_c):,} pts")
        e_r = py4dgeo.Epoch(np.ascontiguousarray(sub_r))
        tr1 = py4dgeo.iterative_closest_point(
            e_r, py4dgeo.Epoch(np.ascontiguousarray(sub_c)), max_iterations=100)
        R1, b1 = _as_Rb(tr1)
        sub_c1 = (R1 @ sub_c.T).T + b1
        e_r.calculate_normals(radius=args.normal_d / 2.0)
        log(f"  stage 2: TRIMMED point-to-plane (keep best {args.icp_trim:.0%} - "
            f"vegetation/real change cannot pull the solution)")
        R2, b2 = trimmed_p2plane_icp(sub_r, e_r.normals, sub_c1, trim=args.icp_trim)
        R, b = R2 @ R1, R2 @ b1 + b2
        cmp_l = (R @ cmp_l.T).T + b
        rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        log(f"  auto-ICP result: rotation {rot_deg:.3f} deg, "
            f"|t_local| = {np.linalg.norm(b):.3f} m")
        if rot_deg > 10 or np.linalg.norm(b) > 20:
            log("  WARNING: large auto-ICP correction - sanity-check (stable-patch means) "
                "before trusting the map.")
        icp_out = os.path.join(args.out, "icp_auto.txt")
        with open(icp_out, "w") as f:
            f.write(f"# auto-ICP; local frame; shift {shift.tolist()}\n")
            for i in range(3):
                f.write(" ".join(f"{v:.12f}" for v in [*R[i], b[i]]) + "\n")
            f.write("0 0 0 1\n")
        log(f"  wrote {icp_out}  (reusable: --icp {icp_out} --cc-shift "
            f"{shift[0]:g} {shift[1]:g} {shift[2]:g})")

    # --- core points ---
    log("\n=== core points ===")
    core_idx = voxel_pick(ref_l, args.core_spacing)
    core = np.ascontiguousarray(ref_l[core_idx])
    log(f"  {len(core):,} core points (voxel {args.core_spacing} m on reference)")
    if len(core) < 1000:
        sys.exit("Fewer than 1000 core points: check inputs / crop.")

    # --- distance to trajectory / corridor mask ---
    track_dist = None
    if args.track:
        log("\n=== trajectory corridor ===")
        from pyproj import Transformer
        trafo = Transformer.from_crs(4326, args.track_epsg, always_xy=True)
        tp = []
        for path in args.track:
            for r in csv.DictReader(open(path)):
                if r.get("lat") and r.get("lon"):
                    tp.append(trafo.transform(float(r["lon"]), float(r["lat"])))
        if not tp:
            sys.exit("--track: no usable lat/lon rows found.")
        tp = np.asarray(tp, float) + shift[:2]
        log(f"  trajectory: {len(tp):,} fixes from {len(args.track)} file(s)")
        from scipy.spatial import cKDTree
        track_dist, _ = cKDTree(tp).query(core[:, :2])
        if args.max_track_dist:
            keep = track_dist <= args.max_track_dist
            log(f"  corridor mask <= {args.max_track_dist:g} m: kept {keep.sum():,} / "
                f"{len(keep):,} core points")
            core = np.ascontiguousarray(core[keep])
            track_dist = track_dist[keep]
            if len(core) < 1000:
                sys.exit("Corridor mask left <1000 core points: check --max-track-dist.")

    # --- M3C2 ---
    log("\n=== M3C2 (py4dgeo) ===")
    import py4dgeo
    log(f"  py4dgeo {py4dgeo.__version__} | numpy {np.__version__}")
    log(f"  normal scale  D = {args.normal_d} m diameter -> radius {args.normal_d / 2}")
    log(f"  projection    d = {args.proj_d} m diameter -> radius {args.proj_d / 2}")
    log(f"  max depth       = {args.max_depth} m | reg. error = {args.reg_error} m")
    log(f"  normals: on core points from reference, oriented +Z")
    e1 = py4dgeo.Epoch(np.ascontiguousarray(ref_l))
    e2 = py4dgeo.Epoch(np.ascontiguousarray(cmp_l))
    m3c2 = py4dgeo.M3C2(
        epochs=(e1, e2), corepoints=core,
        normal_radii=[args.normal_d / 2.0],
        cyl_radius=args.proj_d / 2.0,
        max_distance=args.max_depth,
        registration_error=args.reg_error,
        orientation_vector=np.array([0.0, 0.0, 1.0]),
    )
    t1 = time.time()
    dist, unc = m3c2.run()
    lod = unc["lodetection"]
    valid = np.isfinite(dist)
    sig = np.zeros(len(dist), bool)
    sig[valid] = np.abs(dist[valid]) > lod[valid]
    log(f"  computed in {time.time() - t1:.1f} s")
    log(f"  valid distances: {valid.sum():,} / {len(dist):,} ({100 * valid.mean():.1f}%)")
    if valid.mean() < 0.5:
        log("  WARNING: <50% valid - poor overlap or too-small max depth?")
    med, rob = float(np.nanmedian(dist)), float(nmad(dist))
    log(f"  global median = {med:+.4f} m | NMAD = {rob:.4f} m | "
        f"significant: {sig.sum():,} ({100 * sig.mean():.1f}%)")
    if abs(med) > 0.05:
        log("  WARNING: |global median| > 5 cm - residual registration bias likely; "
            "check stable-patch means.")

    # --- patches ---
    patch_rows = []
    if args.patch:
        log("\n=== stable patches ===")
        log(f"  {'patch':>5} {'E':>12} {'N':>13} {'r':>5} {'n_valid':>8} "
            f"{'mean':>9} {'median':>9} {'std':>8} {'NMAD':>8} {'sig%':>6}")
        for i, spec in enumerate(args.patch):
            try:
                E, N, r = (float(v) for v in spec.split(","))
            except ValueError:
                sys.exit(f"Bad --patch '{spec}': expected E,N,r")
            c_local = np.array([E, N, 0.0]) + shift
            m = (np.hypot(core[:, 0] - c_local[0], core[:, 1] - c_local[1]) <= r) & valid
            d = dist[m]
            row = dict(patch=i, E=E, N=N, r=r, n_valid=int(m.sum()))
            if m.sum() < 10:
                log(f"  {i:>5}  ->  only {m.sum()} valid cores: patch skipped "
                    f"(off-cloud coordinates?)")
                row.update(mean=np.nan, median=np.nan, std=np.nan,
                           nmad=np.nan, sig_frac=np.nan)
            else:
                row.update(mean=float(np.mean(d)), median=float(np.median(d)),
                           std=float(np.std(d)), nmad=float(nmad(d)),
                           sig_frac=float(sig[m].mean()))
                log(f"  {i:>5} {E:>12.2f} {N:>13.2f} {r:>5.2f} {row['n_valid']:>8} "
                    f"{row['mean']:>+9.4f} {row['median']:>+9.4f} {row['std']:>8.4f} "
                    f"{row['nmad']:>8.4f} {100 * row['sig_frac']:>5.1f}%")
            patch_rows.append(row)
        stable = [r for r in patch_rows if r["n_valid"] >= 10]
        if stable:
            meds = np.array([r["median"] for r in stable])
            log(f"  -> patch-median spread (bias): {meds.min():+.4f} .. {meds.max():+.4f} m")
            log(f"  -> suggested --reg-error for pass 2: "
                f"{float(np.sqrt(np.mean(meds ** 2))):.4f} m "
                f"(rms of patch MEDIANS - robust to vegetation tails; use unimodal, "
                f"bare-ground patches only)")
        csv_path = os.path.join(args.out, "m3c2_patch_stats.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(patch_rows[0].keys()))
            w.writeheader()
            w.writerows(patch_rows)
        log(f"  wrote {csv_path}")

    # --- outputs ---
    log("\n=== outputs ===")
    core_g = core - shift
    ply_path = os.path.join(args.out, "m3c2_core.ply")
    write_core_ply(ply_path, core_g, dist, lod, sig, unc, track_dist)
    log(f"  wrote {ply_path}  (opens in CloudCompare with SFs preloaded)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = core_g[:, :2].max(0) - core_g[:, :2].min(0)
    h = float(np.clip(9 * ext[1] / max(ext[0], 1e-9), 4, 18))
    fig, axes = plt.subplots(1, 2, figsize=(18, h), sharex=True, sharey=True)
    for ax in axes:
        ax.set_aspect("equal")
        ax.scatter(core_g[~valid, 0], core_g[~valid, 1], s=0.4, c="0.75",
                   lw=0, rasterized=True)
    sc = axes[0].scatter(core_g[valid, 0], core_g[valid, 1], s=0.4, c=dist[valid],
                         cmap="bwr", vmin=-args.sat, vmax=args.sat, lw=0, rasterized=True)
    axes[0].set_title(f"M3C2 distance [m]  (sat \u00b1{args.sat})")
    fig.colorbar(sc, ax=axes[0], shrink=0.7)
    ns = valid & ~sig
    axes[1].scatter(core_g[ns, 0], core_g[ns, 1], s=0.4, c="0.88", lw=0, rasterized=True)
    sc2 = axes[1].scatter(core_g[sig, 0], core_g[sig, 1], s=0.6, c=dist[sig],
                          cmap="bwr", vmin=-args.sat, vmax=args.sat, lw=0, rasterized=True)
    axes[1].set_title(f"significant only (|d| > LoD95, reg={args.reg_error} m)")
    fig.colorbar(sc2, ax=axes[1], shrink=0.7)
    for row in patch_rows:
        for ax in axes:
            ax.add_patch(plt.Circle((row["E"], row["N"]), row["r"], fill=False,
                                    ec="lime", lw=1.2))
        axes[0].annotate(str(row["patch"]), (row["E"], row["N"]),
                         color="lime", fontsize=9, weight="bold")
    for ax in axes:
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=7)
    map_path = os.path.join(args.out, "m3c2_map.png")
    fig.tight_layout()
    fig.savefig(map_path, dpi=180)
    plt.close(fig)
    log(f"  wrote {map_path}")

    if patch_rows:
        ok_rows = [r for r in patch_rows if r["n_valid"] >= 10]
        if ok_rows:
            cols = min(3, len(ok_rows))
            rows_n = int(np.ceil(len(ok_rows) / cols))
            fig, axs = plt.subplots(rows_n, cols, figsize=(5 * cols, 3.4 * rows_n),
                                    squeeze=False)
            for ax, row in zip(axs.ravel(), ok_rows):
                c_local = np.array([row["E"], row["N"], 0.0]) + shift
                m = (np.hypot(core[:, 0] - c_local[0],
                              core[:, 1] - c_local[1]) <= row["r"]) & valid
                ax.hist(dist[m], bins=40, color="steelblue")
                ax.axvline(0, color="k", lw=0.8)
                ax.set_title(f"patch {row['patch']}: \u03bc={row['mean']:+.3f}  "
                             f"\u03c3={row['std']:.3f}  NMAD={row['nmad']:.3f}",
                             fontsize=9)
            for ax in axs.ravel()[len(ok_rows):]:
                ax.axis("off")
            hist_path = os.path.join(args.out, "m3c2_patches_hist.png")
            fig.tight_layout()
            fig.savefig(hist_path, dpi=160)
            plt.close(fig)
            log(f"  wrote {hist_path}")

    log(f"\nDone in {time.time() - t0:.1f} s.")
    with open(os.path.join(args.out, "m3c2_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"  wrote {os.path.join(args.out, 'm3c2_report.txt')}")


if __name__ == "__main__":
    main()