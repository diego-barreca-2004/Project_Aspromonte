#!/usr/bin/env python3
"""measure_change.py - measure surface change between two passes with a per-cell noise model
and an a-contrario threshold, with no fixed area, height or roughness rule.

dsm_change.py keeps fixed rules (1 m corridor, roughness mask, minimum height, area limits)
and ranks blobs by area x |dz|. Here:

  1. analysis area: cells seen by at least 2 cameras of EACH pass within --max-range, from
     the poses (dense/sparse of each pass), instead of a fixed corridor;
  2. DSM per pass (height percentile per cell), dz = cmp - ref, median offset removed;
  3. noise per cell (Hugonnet et al. 2022): robust model of log|dz| as a sum of binned
     functions of point counts, height spread, views per point (fused.ply.vis), angular
     spread of the camera rays (small at the track ends), local roughness, slope and track
     distance, calibrated so that NMAD(dz / sigma) = 1;
  4. heavy tails: z = dz / sigma is not Gaussian (branches over the trail, floaters and mixed
     cells give |z| of tens or hundreds), and one such cell would decide a whole window mean.
     So z goes through the empirical distribution of the pair itself to normal scores,
     u = Phi^-1(F(z)): exactly N(0,1) per cell, and a single cell can never reach the
     threshold, whatever its z;
  5. spatial correlation of u: robust (Dowd) variogram up to 20 m, nugget plus two spherical
     models (MVS smoothing; residual drift), giving the effective sample size of a window
     (Rolstad et al. 2009);
  6. a-contrario test (Desolneux, Moisan, Morel): square windows of 7 sizes (side k) centred
     on every cell, contrast C = mean(u in the window) - mean(u in the ring out to 3k). The
     weights sum to zero, so Var(C) follows from the variogram alone even when the residual
     drift makes the field non-stationary (intrinsic random functions, Matheron): no drift is
     subtracted from the data, and a compact change is judged against its own surroundings.
     T = C / sd(C). On N1 one variogram does not fit every surface: the NMAD of T grows
     from 0.9 on smooth ground to 1.8 in vegetation at 0.1 m, and windows with missing cells
     have heavier tails. So T is re-centred and re-scaled by its median and NMAD within
     classes of window sigma (quintiles) and fill (empirical null, Efron 2004).
     NFA = N_tests * P(N(0,1) >= T). A change is stated where
     NFA < --epsilon * track km, so --epsilon is the expected number of false statements per
     km. Significant windows of one sign merge into one change; rank by -log10 NFA.

The only exposed parameter is --epsilon (false statements per km). Outputs in --out:
changes.csv (kind, E, N, area_m2, dz_median, dz_peak, n_cells, log10_nfa, scale_m, score),
noise_model.json, change_report.txt, change_map.png.

  python3 measure_change.py --ref-seg seg01_ep2 --cmp-seg seg01_ep3 \\
      --icp change_detection/m3c2_ep3_vs_ep2_v2/icp_auto.txt --cc-shift -558997 -4214492 -123 \\
      --out change_detection/v2/nfa_ep3_vs_ep2
"""
import argparse
import csv
import json
import os
import struct
import sys
import time

import numpy as np
from plyfile import PlyData
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
from scipy.stats import norm

from dsm_change import grid_stats, read_track
from georef_cloud import parse_geo_transform
from run_m3c2 import nmad, parse_cc_matrices

SCALES = (2, 3, 5, 9, 15, 25, 40)  # window sides in cells: 0.1 to 2.0 m at 5 cm


# --- inputs -------------------------------------------------------------------
def read_pinhole_model(model_dir):
    """Undistorted cameras (PINHOLE) and world-to-camera poses of a dense/sparse model."""
    cams = {}
    with open(os.path.join(model_dir, "cameras.bin"), "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            if mid != 1:
                sys.exit(f"{model_dir}: camera model {mid} is not PINHOLE (1)")
            cams[cid] = (w, h, np.array(struct.unpack("<4d", f.read(32))))
    poses = []
    with open(os.path.join(model_dir, "images.bin"), "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            _, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<i7di", f.read(64))
            while f.read(1) != b"\0":
                pass
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
            w, x, y, z = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            poses.append((R, np.array([tx, ty, tz]), cams[cid]))
    return poses


def read_views(path):
    """Number of images that see each fused point (COLMAP fused.ply.vis)."""
    raw = np.fromfile(path, dtype=np.uint32, offset=8)
    n = struct.unpack("<Q", open(path, "rb").read(8))[0]
    counts = np.empty(n, np.uint32)
    i = 0
    for k in range(n):
        c = raw[i]
        counts[k] = c
        i += c + 1
    return counts


def load_pass(seg, ply=None, cache=None):
    """UTM points of a pass (fused_utm.ply, or --*-ply) with their number of views.

    A `views` vertex property is used when present (synthetic clouds). Otherwise the points
    are matched to the raw fused.ply (georeferenced the same way) to read fused.ply.vis:
    fused_utm.ply is fused.ply after SOR, so the order alone does not match."""
    dense = os.path.join(seg, "colmap", "dense")
    v = PlyData.read(ply or os.path.join(dense, "fused_utm.ply"))["vertex"]
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    if "views" in v.data.dtype.names:
        return xyz, np.asarray(v["views"], np.float64)
    key = os.path.join(cache, os.path.basename(os.path.normpath(seg)) + "_views.npy") if cache else None
    if key and os.path.isfile(key):
        views = np.load(key)
        if len(views) == len(xyz):
            return xyz, views
    r = PlyData.read(os.path.join(dense, "fused.ply"))["vertex"]
    raw = np.column_stack([r["x"], r["y"], r["z"]]).astype(np.float64)
    s, R, t, _ = parse_geo_transform(os.path.join(seg, "colmap", "geo_transform.txt"))
    d, j = cKDTree(s * raw @ R.T + t).query(xyz, distance_upper_bound=0.01)
    if np.mean(np.isfinite(d)) < 0.999:
        sys.exit(f"{seg}: fused_utm.ply does not match fused.ply + geo_transform.txt")
    views = np.where(np.isfinite(d), read_views(os.path.join(dense, "fused.ply.vis"))[np.minimum(j, len(raw) - 1)], 0).astype(np.float64)
    if key:
        os.makedirs(cache, exist_ok=True)
        np.save(key, views)
    return xyz, views


def to_frame(p, M, shift, inverse=False):
    """Apply (or undo) a CloudCompare matrix estimated in the shifted frame."""
    if M is None:
        return p
    if inverse:
        M = np.linalg.inv(M)
    return (p + shift) @ M[:3, :3].T + M[:3, 3] - shift


# --- grid ---------------------------------------------------------------------
def cell_mean(pts, val, origin, cell, shape):
    ix = np.floor((pts[:, 0] - origin[0]) / cell).astype(np.int64)
    iy = np.floor((pts[:, 1] - origin[1]) / cell).astype(np.int64)
    ok = (ix >= 0) & (ix < shape[0]) & (iy >= 0) & (iy < shape[1])
    key = ix[ok] * shape[1] + iy[ok]
    n = np.bincount(key, minlength=shape[0] * shape[1])
    s = np.bincount(key, val[ok], minlength=shape[0] * shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return n.reshape(shape), (s / n).reshape(shape)


def visibility(cx, cy, cz, seg, M, shift, max_range):
    """Number of cameras of a pass that see each cell centre (in front, inside the image,
    within max_range metres), and the angular spread of their rays in degrees,
    sqrt(2 (1 - |mean unit ray|)): small where all cameras look from one side, as at the ends
    of the track, so depth is poorly triangulated. Cells are in the reference UTM frame."""
    s, R, t, _ = parse_geo_transform(os.path.join(seg, "colmap", "geo_transform.txt"))
    poses = read_pinhole_model(os.path.join(seg, "colmap", "dense", "sparse"))
    P = to_frame(np.c_[cx, cy, cz], M, shift, inverse=True)  # into the pass's UTM frame
    Pc = (P - t) @ R / s  # into its COLMAP frame
    centres = np.array([-Rc.T @ tc for Rc, tc, _ in poses])
    tree = cKDTree(P[:, :2])
    count = np.zeros(len(P), np.int32)
    rays = np.zeros((len(P), 3))
    for (Rc, tc, (w, h, (fx, fy, px, py))), c in zip(poses, centres):
        cu = s * c @ R.T + t
        idx = np.array(tree.query_ball_point(cu[:2], max_range), np.int64)
        if not len(idx):
            continue
        X = Pc[idx] @ Rc.T + tc
        front = X[:, 2] > 0
        u = fx * X[:, 0] / np.where(front, X[:, 2], 1) + px
        v = fy * X[:, 1] / np.where(front, X[:, 2], 1) + py
        d = P[idx] - cu
        rng = np.linalg.norm(d, axis=1)
        seen = front & (rng <= max_range) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        count[idx[seen]] += 1
        rays[idx[seen]] += d[seen] / rng[seen, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.linalg.norm(rays, axis=1) / count
    spread = np.degrees(np.sqrt(2 * np.clip(1 - r, 0, None)))
    return count, np.where(count >= 2, spread, 0.0)


# --- noise model ----------------------------------------------------------------
def fit_sigma(dz, X, names, n_bins=10, iters=6):
    """sigma per cell from binned medians of log|dz| (additive in log, backfitting)."""
    y = np.log(np.maximum(np.abs(dz), 1e-5))
    edges = [np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1))) for x in X.T]
    idx = [np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2) for e, x in zip(edges, X.T)]
    f = [np.zeros(len(e) - 1) for e in edges]
    c = np.median(y)
    for _ in range(iters):
        for k in range(len(f)):
            r = y - c - sum(f[j][idx[j]] for j in range(len(f)) if j != k)
            f[k] = np.array([np.median(r[idx[k] == b]) if np.any(idx[k] == b) else 0.0
                             for b in range(len(f[k]))])
            f[k] -= np.median(f[k][idx[k]])
    model = dict(c=float(c), terms=[dict(name=nm, centres=[float(np.median(x[i == b])) if np.any(i == b) else float(e[b])
                                                          for b in range(len(e) - 1)],
                                         values=fk.tolist())
                                    for nm, x, i, e, fk in zip(names, X.T, idx, edges, f)])
    sig = predict_sigma(model, X)
    model["scale"] = float(nmad(dz / sig))
    return model


def predict_sigma(model, X):
    logs = model["c"] + sum(np.interp(x, t["centres"], t["values"]) for x, t in zip(X.T, model["terms"]))
    return np.exp(logs) / 0.6745 * model.get("scale", 1.0)


# --- spatial correlation --------------------------------------------------------
def sph(h, a):
    h = np.minimum(h / a, 1.0)
    return 1.5 * h - 0.5 * h ** 3


def variogram(z, cell):
    """Dowd variogram of z along the grid axes and diagonals, fitted with a nugget plus
    two spherical models. Returns (lags, gamma, params)."""
    lags, gam = [], []
    for d in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100, 150, 200, 300, 400):
        diffs = []
        for sx, sy in ((d, 0), (0, d), (d, d)):
            a, b = z[sx:, sy:], z[:z.shape[0] - sx, :z.shape[1] - sy]
            m = np.isfinite(a) & np.isfinite(b)
            diffs.append(np.abs(a[m] - b[m]))
        diff = np.concatenate(diffs)
        if len(diff) > 1000:
            lags.append(d * cell * (1 + (np.sqrt(2) - 1) / 3))
            gam.append(2.198 * np.median(diff) ** 2 / 2)
    lags, gam = np.array(lags), np.array(gam)

    def model(h, n0, s1, a1, s2, a2):
        return n0 + s1 * sph(h, a1) + s2 * sph(h, a2)

    p, _ = optimize.curve_fit(model, lags, gam, p0=[0.3, 0.4, 0.15, 0.3, 1.5],
                              bounds=([0, 0, cell, 0, 0.3], [2, 2, 0.6, 2, 50]), maxfev=20000)
    return lags, gam, p


def contrast_sd(k, cell, p):
    """Standard deviation of mean(window k x k) - mean(ring out to 3k x 3k) on a full grid:
    Var = -sum over offsets of (kernel autocorrelation x variogram), valid because the
    kernel weights sum to zero."""
    from scipy.signal import fftconvolve
    n0, s1, a1, s2, a2 = p
    K = 3 * k
    ker = np.full((K, K), -1.0 / (K * K - k * k))
    ker[k:2 * k, k:2 * k] = 1.0 / (k * k)
    R = fftconvolve(ker, ker[::-1, ::-1])
    d = np.arange(-(K - 1), K)
    dx, dy = np.meshgrid(d, d, indexing="ij")
    h = np.hypot(dx, dy) * cell
    g = np.where(h == 0, 0.0, n0 + s1 * sph(h, a1) + s2 * sph(h, a2))
    return float(np.sqrt(-np.sum(R * g)))


def empirical_null(T, ok, wsig, full, n_strata=5, min_n=1000):
    """Empirical null by central matching (Efron 2004), per class of window sigma (quintiles)
    and fill (window and ring complete or not): T -> (T - median) / NMAD within each class.
    One variogram cannot hold for both smooth ground (neighbouring cells strongly correlated)
    and vegetation (nearly independent cells), nor for windows with missing cells; changes are
    rare, so the median and NMAD of each class describe its null. Returns T and the classes."""
    out = np.zeros_like(T)
    edges = np.quantile(wsig[ok], np.linspace(0, 1, n_strata + 1))
    cls = np.clip(np.searchsorted(edges, wsig, side="right") - 1, 0, n_strata - 1) * 2 + full
    pooled = (float(np.median(T[ok])), float(nmad(T[ok])))
    rows = []
    for c in range(2 * n_strata):
        m = ok & (cls == c)
        if not m.any():
            continue
        loc, sc = (float(np.median(T[m])), float(nmad(T[m]))) if m.sum() >= min_n else pooled
        out[m] = (T[m] - loc) / sc
        rows.append(dict(sigma=[float(edges[c // 2]), float(edges[c // 2 + 1])], full=bool(c % 2),
                         n=int(m.sum()), loc=loc, scale=sc))
    return out, rows


# --- main -----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-cell noise model and a-contrario change measurement.")
    ap.add_argument("--ref-seg", required=True, help="reference pass folder (older)")
    ap.add_argument("--cmp-seg", required=True, help="compared pass folder (newer)")
    ap.add_argument("--ref-ply", help="reference cloud (default <ref-seg>/colmap/dense/fused_utm.ply)")
    ap.add_argument("--cmp-ply", help="compared cloud (default <cmp-seg>/colmap/dense/fused_utm.ply)")
    ap.add_argument("--icp", help="CloudCompare matrix, cmp to ref (none after co-alignment)")
    ap.add_argument("--cc-shift", nargs=3, type=float, metavar=("SX", "SY", "SZ"))
    ap.add_argument("--track", help="gps.csv of the reference pass (default <ref-seg>/gps.csv)")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="expected false change statements per km of track")
    ap.add_argument("--max-range", type=float, default=6.0,
                    help="camera-to-cell distance for a cell to count as seen [m]")
    ap.add_argument("--cell", type=float, default=0.05, help="DSM cell size [m]")
    ap.add_argument("--pct", type=float, default=90.0, help="height percentile per cell")
    ap.add_argument("--min-pts", type=int, default=3, help="minimum points per valid cell")
    ap.add_argument("--cache", help="folder for the views-per-point cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.icp and args.cc_shift is None:
        sys.exit("--cc-shift is REQUIRED with --icp: the matrix lives in the shifted frame.")

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    report = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    log(f"# measure_change.py  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("# cmd: " + " ".join(sys.argv if argv is None else ["measure_change.py"] + list(argv)))
    M = np.asarray(parse_cc_matrices(args.icp)[0]) if args.icp else None
    shift = np.array(args.cc_shift or [0, 0, 0], float)

    # --- clouds, restricted to max-range of the track ---
    track = read_track([args.track or os.path.join(args.ref_seg, "gps.csv")], 32633)
    km = float(np.hypot(*np.diff(track, axis=0).T).sum()) / 1000
    ttree = cKDTree(track)
    ref, v_ref = load_pass(args.ref_seg, args.ref_ply, args.cache)
    cmp_, v_cmp = load_pass(args.cmp_seg, args.cmp_ply, args.cache)
    cmp_ = to_frame(cmp_, M, shift)
    keep_r = ttree.query(ref[:, :2])[0] <= args.max_range
    keep_c = ttree.query(cmp_[:, :2])[0] <= args.max_range
    ref, v_ref, cmp_, v_cmp = ref[keep_r], v_ref[keep_r], cmp_[keep_c], v_cmp[keep_c]
    log(f"\n=== load ===\n  ref {args.ref_seg}: {len(ref):,} pts | cmp {args.cmp_seg}: {len(cmp_):,} pts "
        f"(within {args.max_range:g} m of the track, {km * 1000:.1f} m long)")

    # --- DSMs and per-cell predictors ---
    both = np.vstack([ref[:, :2], cmp_[:, :2]])
    origin = both.min(axis=0)
    shape = tuple((np.ceil((both.max(axis=0) - origin) / args.cell) + 1).astype(int))
    h_ref, s_ref = grid_stats(ref, origin, args.cell, shape, args.pct, args.min_pts)
    h_cmp, s_cmp = grid_stats(cmp_, origin, args.cell, shape, args.pct, args.min_pts)
    n_ref, w_ref = cell_mean(ref, v_ref, origin, args.cell, shape)
    n_cmp, w_cmp = cell_mean(cmp_, v_cmp, origin, args.cell, shape)
    gx, gy = np.meshgrid(origin[0] + (np.arange(shape[0]) + 0.5) * args.cell,
                         origin[1] + (np.arange(shape[1]) + 0.5) * args.cell, indexing="ij")

    # --- analysis area: cells seen by >= 2 cameras of each pass ---
    hz = np.where(np.isfinite(h_ref), h_ref, h_cmp)
    data = np.isfinite(hz)
    seen, ray = {}, {}
    for name, seg, m in (("ref", args.ref_seg, None), ("cmp", args.cmp_seg, M)):
        cnt, spr = np.zeros(shape, np.int32), np.zeros(shape)
        cnt[data], spr[data] = visibility(gx[data], gy[data], hz[data], seg, m, shift, args.max_range)
        seen[name], ray[name] = cnt, spr
    observed = (seen["ref"] >= 2) & (seen["cmp"] >= 2)
    valid = observed & np.isfinite(h_ref) & np.isfinite(h_cmp)
    empty_one = observed & (np.isfinite(h_ref) != np.isfinite(h_cmp))
    log(f"\n=== analysis area ===\n  grid {shape[0]} x {shape[1]} cells of {args.cell} m")
    log(f"  seen by >= 2 cameras of each pass: {observed.sum():,} cells; with data in both: "
        f"{valid.sum():,}; empty in one pass (for the lost-surface channel): {empty_one.sum():,}")

    # --- noise model ---
    dz = h_cmp - h_ref
    offset = float(np.median(dz[valid]))
    dz = dz - offset
    smooth = ndimage.uniform_filter(np.where(np.isfinite(h_ref), h_ref, 0), 3)
    wts = ndimage.uniform_filter(np.isfinite(h_ref).astype(float), 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        rough = np.abs(h_ref - smooth / wts)
    sx, sy = np.gradient(np.where(np.isfinite(h_ref), h_ref, np.nanmedian(h_ref)), args.cell)
    names = ["points (min)", "spread (max) m", "views (min)", "ray spread (min) deg", "roughness m",
             "slope", "track dist m"]
    X = np.column_stack([np.minimum(n_ref, n_cmp)[valid], np.maximum(s_ref, s_cmp)[valid],
                         np.minimum(w_ref, w_cmp)[valid], np.minimum(ray["ref"], ray["cmp"])[valid],
                         rough[valid], np.hypot(sx, sy)[valid],
                         ttree.query(np.c_[gx[valid], gy[valid]])[0]])
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    model = fit_sigma(dz[valid], X, names)
    sigma = np.full(shape, np.nan)
    sigma[valid] = predict_sigma(model, X)
    z = dz / sigma
    log(f"\n=== noise model ===\n  median offset removed: {offset:+.4f} m | NMAD dz: {nmad(dz[valid]):.4f} m "
        f"| sigma per cell P10/P50/P90: " + "/".join(f"{q:.3f}" for q in np.percentile(sigma[valid], [10, 50, 90])) + " m")
    log("  standardisation, NMAD of z per decile of each predictor (should all be ~1):")
    for k, nm in enumerate(names):
        e = np.unique(np.quantile(X[:, k], np.linspace(0, 1, 11)))
        b = np.clip(np.searchsorted(e, X[:, k], side="right") - 1, 0, len(e) - 2)
        log(f"    {nm:>15}: " + " ".join(f"{nmad(z[valid][b == i]):.2f}" for i in range(len(e) - 1)))
    zv = z[valid]
    u = np.full(shape, np.nan)
    u[valid] = norm.ppf((np.argsort(np.argsort(zv)) + 0.5) / len(zv))
    log(f"  tails of z: P(|z| > 5) = {np.mean(np.abs(zv) > 5):.4f} (Gaussian 5.7e-07), "
        f"P(|z| > 20) = {np.mean(np.abs(zv) > 20):.4f}; z -> normal scores u")
    lags, gam, vp = variogram(u, args.cell)
    log(f"  variogram of u: nugget {vp[0]:.2f}, sph {vp[1]:.2f} range {vp[2]:.2f} m, "
        f"sph {vp[3]:.2f} range {vp[4]:.2f} m")

    # --- a-contrario windows ---
    zf = np.where(valid, u, 0.0)
    vf = valid.astype(float)
    lsig = np.where(valid, np.log(sigma), 0.0)
    best = {k: np.full(shape, np.inf) for k in ("appeared", "removed")}  # min log10 NFA per cell
    stats, null = [], {}
    n_tests = 0
    for k in SCALES:
        S, C = (ndimage.uniform_filter(a, k, mode="constant") * k * k for a in (zf, vf))
        S3, C3 = (ndimage.uniform_filter(a, 3 * k, mode="constant") * 9 * k * k for a in (zf, vf))
        ok = valid & (C >= 0.5 * k * k) & (C3 - C >= 0.5 * 8 * k * k)
        n_tests += 2 * int(ok.sum())
        sd = contrast_sd(k, args.cell, vp)
        with np.errstate(invalid="ignore", divide="ignore"):
            T = np.where(ok, (S / C - (S3 - S) / (C3 - C)) / sd, 0.0)
            wsig = np.exp(ndimage.uniform_filter(lsig, k, mode="constant") * k * k / C)
        full = (np.rint(C) == k * k) & (np.rint(C3 - C) == 8 * k * k)
        T, null[str(k)] = empirical_null(T, ok, wsig, full.astype(int))
        stats.append((k, sd, ok, T))
    log_n = np.log10(n_tests)
    thr = np.log10(args.epsilon * km)
    log(f"\n=== a-contrario test ===\n  {n_tests:,} tests ({len(SCALES)} window sizes "
        f"{SCALES[0] * args.cell:g}-{SCALES[-1] * args.cell:g} m, both signs); "
        f"statement when NFA < {args.epsilon:g}/km x {km:.3f} km = {10 ** thr:.3f}, "
        f"i.e. T > {norm.isf(10 ** thr / n_tests):.2f}")
    log("  empirical null, NMAD of T per class (sigma quintile; partial / complete windows):")
    for k, sd, ok, T in stats:
        rows = null[str(k)]
        log(f"    {k * args.cell:5.2f} m: " + " ".join(
            f"{r['scale']:.2f}" + ("" if r["full"] else "p") for r in rows))
    log("  null check per window side after it: sd of T, P(|T| > 3, 4, 5) "
        "(Gaussian 2.7e-03, 6.3e-05, 5.7e-07)")
    for k, sd, ok, T in stats:
        t = T[ok]
        log(f"    {k * args.cell:5.2f} m: sd {t.std():.2f} | "
            + " ".join(f"{np.mean(np.abs(t) > c):.1e}" for c in (3, 4, 5)))
    for k, sd, ok, T in stats:
        for kind, sign in (("appeared", 1), ("removed", -1)):
            lnfa = np.where(ok, log_n + norm.logsf(sign * T) / np.log(10), np.inf)
            sig = lnfa < thr
            if not sig.any():
                continue
            # every cell of a significant window carries the window's NFA
            lmin = -ndimage.maximum_filter(np.where(sig, -lnfa, -np.inf), k, mode="constant",
                                            cval=-np.inf)
            best[kind] = np.minimum(best[kind], lmin)
            scale_hit = best[kind] == lmin
            best[kind + "_k"] = np.where(scale_hit & np.isfinite(lmin), k, best.get(kind + "_k", np.zeros(shape)))

    rows = []
    for kind, sign in (("appeared", 1), ("removed", -1)):
        lab, n = ndimage.label(np.isfinite(best[kind]), structure=np.ones((3, 3)))
        for c in range(1, n + 1):
            comp = lab == c
            core = comp & valid & (sign * z > 2)
            if not core.any():
                core = comp & valid
            ii, jj = np.nonzero(core)
            vals = dz[ii, jj]
            lmin = float(best[kind][comp].min())
            rows.append(dict(kind=kind, E=float(gx[ii, jj].mean()), N=float(gy[ii, jj].mean()),
                             area_m2=len(ii) * args.cell ** 2, dz_median=float(np.median(vals)),
                             dz_peak=float(vals[np.argmax(sign * vals)]), n_cells=len(ii),
                             log10_nfa=lmin,
                             scale_m=float(best[kind + "_k"][comp][np.argmin(best[kind][comp])] * args.cell),
                             score=-lmin))
    rows.sort(key=lambda r: -r["score"])
    log(f"\n=== changes ({len(rows)}) ===")
    log(f"  {'#':>3} {'kind':>8} {'E':>12} {'N':>13} {'area m2':>8} {'dz med':>8} {'dz peak':>8} {'log10 NFA':>9}")
    for i, r in enumerate(rows):
        log(f"  {i + 1:>3} {r['kind']:>8} {r['E']:12.2f} {r['N']:13.2f} {r['area_m2']:8.3f} "
            f"{r['dz_median']:+8.3f} {r['dz_peak']:+8.3f} {r['log10_nfa']:9.2f}")

    # --- outputs ---
    fields = ["kind", "E", "N", "area_m2", "dz_median", "dz_peak", "n_cells", "log10_nfa", "scale_m", "score"]
    with open(os.path.join(args.out, "changes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(model, offset=offset, variogram=dict(lags=lags.tolist(), gamma=gam.tolist(),
                                                         params=vp.tolist()),
                   contrast_sd={str(k): sd for k, sd, _, _ in stats}, empirical_null=null,
                   n_tests=n_tests, track_km=km),
              open(os.path.join(args.out, "noise_model.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = (origin[0], origin[0] + shape[0] * args.cell, origin[1], origin[1] + shape[1] * args.cell)
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(np.where(valid, z, np.nan).T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-5, vmax=5)
    ax.imshow(np.where(observed & ~valid, 1.0, np.nan).T, origin="lower", extent=ext, cmap="Greys",
              vmin=0, vmax=2, alpha=0.6)
    for i, r in enumerate(rows):
        ax.plot(r["E"], r["N"], "o", mfc="none", mec="k", ms=10)
        ax.annotate(str(i + 1), (r["E"], r["N"]), fontsize=7)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="z = dz / sigma (grey: seen but empty in one pass)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "change_map.png"), dpi=150)
    log(f"\n  done in {time.time() - t0:.0f} s -> {args.out}")
    open(os.path.join(args.out, "change_report.txt"), "w").write("\n".join(report) + "\n")
    return rows


if __name__ == "__main__":
    main()
