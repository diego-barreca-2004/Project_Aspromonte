#!/usr/bin/env python3
"""georef_splat.py - one-shot georeferencing of a 3DGS splat: Sim3 + ICP-to-DTM.

Single, self-contained Binary 1. Takes a trained 3D Gaussian Splatting point_cloud.ply and:
  1. applies the Sim3 from geo_align.py (geo_transform.txt) -> true UTM,
  2. (optional) refines it onto a bare-earth LiDAR DTM by rigid point-to-plane ICP,
     run entirely in Python,
  3. composes the two into ONE transform and applies it a single time to every Gaussian
     attribute (positions, rotation quaternions, log-scales, spherical harmonics),
  4. writes the georeferenced (refined) splat and reports the residual to the DTM.

No CloudCompare, no global-shift bookkeeping: the ICP runs in a locally-centred float64
frame, so the ~4.2 M UTM magnitudes never wreck the conditioning. The per-attribute splat
transform (SH resampled on Inria's basis, Hamilton quaternion product, log-space scales) is
implemented here directly - this script has no dependency on apply_transform_ply.py.

ICP aligns the splat's GROUND to the DTM, so it is well-conditioned where the terrain has
relief; on dead-flat ground some degrees of freedom are intrinsically loose (true of any
cloud-to-DTM ICP) - the surface residual it prints is the figure that matters. Open terrain
like seg01 is the ideal first case; under dense canopy widen the ground filter
(--ground-band), since the splat sees the canopy while the DTM is bare earth.

Note: a 3DGS .ply stores positions as float32, which quantises absolute UTM northings
(~4.2 M) to ~0.5 m. The ICP is computed in float64 and the reported residual is exact; the
stored file inherits the format's float32 floor (viewers / CloudCompare handle the
magnitudes via a global shift on load). Pass --recenter to also shift the output splat to a
local origin (writing a .offset.txt sidecar): this both removes the float32 quantisation and
lets it render in WebGL viewers like SuperSplat, which cannot draw absolute UTM magnitudes.

Requires: numpy, rasterio, plyfile   (pip install numpy rasterio plyfile)

Usage:
  # Sim3 + ICP refinement (recommended)
  python3 georef_splat.py \
      --ply ./seg01/gs_output/point_cloud/iteration_30000/point_cloud.ply \
      --transform ./seg01/colmap/geo_transform.txt \
      --dtm aspromonte_dtm_utm33n.tif \
      --out ./seg01/gs_output/point_cloud_utm_icp.ply

  # Sim3 only (no DTM): just georeference the splat, no refinement
  python3 georef_splat.py --ply ... --transform ... --out point_cloud_utm.ply
"""
import argparse
import sys
import numpy as np


# --- Spherical-harmonic rotation (Inria SH convention, sh_utils.py) ----------
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def sh_basis(deg, d):
    """Inria SH basis values at directions d (N,3) -> (N,(deg+1)^2), matching eval_sh."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    out = [np.full_like(x, C0)]
    if deg >= 1:
        out += [-C1 * y, C1 * z, -C1 * x]
    if deg >= 2:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        out += [C2[0]*xy, C2[1]*yz, C2[2]*(2*zz-xx-yy), C2[3]*xz, C2[4]*(xx-yy)]
    if deg >= 3:
        out += [C3[0]*y*(3*xx-yy), C3[1]*xy*z, C3[2]*y*(4*zz-xx-yy),
                C3[3]*z*(2*zz-3*xx-3*yy), C3[4]*x*(4*zz-xx-yy),
                C3[5]*z*(xx-yy), C3[6]*x*(xx-3*yy)]
    return np.stack(out, axis=1)


def sh_rotation_matrix(R, deg, n=300):
    """(deg+1)^2 square matrix M with sh' = M @ sh under a world rotation R."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2*i/n)
    th = np.pi * (1 + 5**0.5) * i
    d = np.stack([np.sin(phi)*np.cos(th), np.sin(phi)*np.sin(th), np.cos(phi)], 1)  # Fibonacci
    B = sh_basis(deg, d)
    B_rot = sh_basis(deg, d @ R)            # rows of (d @ R) equal (R^T d): gives Y_k(R^T d)
    return np.linalg.pinv(B) @ B_rot


# --- Quaternion helpers ------------------------------------------------------
def mat2quat(R):
    """Proper rotation matrix -> unit quaternion (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def quat_mul(q1, q2):
    """Hamilton product (w,x,y,z); q1 (4,) applied on the left of q2 (N,4) -> (N,4)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=1)


# --- transform file I/O ------------------------------------------------------
def read_transform(path):
    """Parse geo_transform.txt -> (scale c, rotation R (3,3), translation t (3,))."""
    c = R = t = None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        if key == "scale": c = float(vals[0])
        elif key == "R":   R = np.array(list(map(float, vals))).reshape(3, 3)
        elif key == "t":   t = np.array(list(map(float, vals)))
    if c is None or R is None or t is None:
        sys.exit("transform file must contain 'scale', 'R' (9 values) and 't' (3 values).")
    return c, R, t


def read_epsg(path):
    for line in open(path):
        if line.strip().startswith("epsg"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


# --- SO(3) helpers -----------------------------------------------------------
def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def expmap(w):
    """Rodrigues: so(3) vector -> rotation matrix."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def compose_sim3(c, R1, t1, R2, t2):
    """Compose rigid (R2,t2) AFTER similarity (c,R1,t1): x -> R2(c R1 x + t1) + t2."""
    return c, R2 @ R1, R2 @ t1 + t2


# --- point-to-plane ICP (correspondence supplied by the caller) --------------
def icp_point_to_plane(P0, correspond, origin, iters=60, k_mad=3.0, tol=1e-7):
    """Rigid fit mapping points P0 onto a surface via linearised point-to-plane ICP.

    P0:         (M,3) source points in global coordinates.
    correspond: callable(P_global) -> (q (M,3), n (M,3) unit normals, valid (M,) bool),
                the nearest surface point and its normal for each source point.
    origin:     (3,) centring origin; the linear system is built in (P - origin) to keep
                the p x n terms well-conditioned at large UTM magnitudes.
    Returns (R, t_global, info) with the global map x -> R x + t_global.
    """
    O = np.asarray(origin, float)
    R = np.eye(3)
    t = np.zeros(3)                       # accumulated, centred frame
    P = P0.astype(float).copy()
    info = {"iters": 0, "inliers": 0, "rms0": None, "rms": None, "median_abs": None}
    for it in range(iters):
        q, n, valid = correspond(P)
        d = np.einsum("ij,ij->i", P - q, n)
        v = valid & np.isfinite(d)
        if v.sum() < 10:
            break
        dv = d[v]
        med, mad = np.median(dv), np.median(np.abs(dv - np.median(dv))) + 1e-9
        keep = v.copy()
        keep[v] = np.abs(dv - med) <= k_mad * 1.4826 * mad
        if keep.sum() < 10:
            keep = v
        r = np.einsum("ij,ij->i", P[keep] - q[keep], n[keep])
        if info["rms0"] is None:
            info["rms0"] = float(np.sqrt(np.mean(r ** 2)))
        A = np.hstack([np.cross(P[keep] - O, n[keep]), n[keep]])   # (m,6): [p x n | n]
        x, *_ = np.linalg.lstsq(A, -r, rcond=None)
        w, dt = x[:3], x[3:]
        dR = expmap(w)
        R, t = dR @ R, dR @ t + dt
        P = (dR @ (P - O).T).T + O + dt                            # increment about O
        info["iters"], info["inliers"] = it + 1, int(keep.sum())
        if np.linalg.norm(w) < tol and np.linalg.norm(dt) < 1e-5:
            break
    q, n, valid = correspond(P)
    d = np.einsum("ij,ij->i", P - q, n)
    v = valid & np.isfinite(d)
    if v.any():
        info["rms"] = float(np.sqrt(np.mean(d[v] ** 2)))
        info["median_abs"] = float(np.median(np.abs(d[v])))
    t_global = t + (np.eye(3) - R) @ O                             # centred -> global
    return R, t_global, info


# --- DTM I/O and correspondence ----------------------------------------------
def load_dtm(path):
    """Read a DTM GeoTIFF -> (Z with NaN nodata, unit normals grid, Affine, CRS)."""
    import rasterio
    with rasterio.open(path) as ds:
        Z = ds.read(1).astype(np.float64)
        nodata, T, crs = ds.nodata, ds.transform, ds.crs
    if nodata is not None:
        Z[Z == nodata] = np.nan
    px, py = abs(T.a), abs(T.e)
    gy, gx = np.gradient(Z, py, px)                       # d/d(row=south), d/d(col=east)
    nrm = np.stack([-gx, gy, np.ones_like(Z)], axis=-1)   # normal of z=f(E,N), +up
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
    return Z, nrm, T, crs


def nearest_cell(T, Z, E, N):
    """Indices + validity of the DTM cell containing each (E,N)."""
    inv = ~T
    H, W = Z.shape
    col = inv.a * E + inv.b * N + inv.c
    row = inv.d * E + inv.e * N + inv.f
    ci, ri = np.floor(col).astype(int), np.floor(row).astype(int)
    ok = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    cic, ric = np.clip(ci, 0, W - 1), np.clip(ri, 0, H - 1)
    ok &= np.isfinite(Z[ric, cic])
    return ric, cic, ok


def dtm_correspondence(Z, nrm, T):
    """Build correspond(P) -> (q, n, valid) using nearest-cell point-to-plane matching."""
    def correspond(P):
        ri, ci, ok = nearest_cell(T, Z, P[:, 0], P[:, 1])
        cE = T.a * (ci + 0.5) + T.b * (ri + 0.5) + T.c
        cN = T.d * (ci + 0.5) + T.e * (ri + 0.5) + T.f
        q = np.stack([cE, cN, np.where(ok, Z[ri, ci], 0.0)], axis=1)
        return q, nrm[ri, ci], ok
    return correspond


# --- per-attribute splat transform -------------------------------------------
def apply_to_vertex(v, names, c, R, t, N):
    """Apply Sim3 (c,R,t) to every Gaussian attribute of a 3DGS vertex array, in place."""
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    xyz = c * (R @ xyz.T).T + t
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # rotation quaternions:  q' = quat(R) (x) normalize(q)
    if all(k in names for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
        q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        qp = quat_mul(mat2quat(R), q)
        qp /= np.linalg.norm(qp, axis=1, keepdims=True) + 1e-12
        v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = qp[:, 0], qp[:, 1], qp[:, 2], qp[:, 3]

    # scales (log-space):  + ln(c)
    lnc = float(np.log(c))
    for k in ("scale_0", "scale_1", "scale_2"):
        if k in names:
            v[k] = v[k] + lnc

    # spherical harmonics: DC unchanged, higher orders rotated by R
    rest = sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    if rest and all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        per = len(rest) // 3                       # coeffs per colour channel (excl. DC)
        deg = int(round((per + 1) ** 0.5)) - 1     # per = (deg+1)^2 - 1
        M = sh_rotation_matrix(R, deg)
        dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64)
        rmat = np.stack([v[n] for n in rest], 1).astype(np.float64).reshape(N, 3, per)
        for ch in range(3):
            coeffs = np.concatenate([dc[:, ch:ch + 1], rmat[:, ch, :]], 1)   # (N, per+1)
            new = coeffs @ M.T
            v["f_dc_%d" % ch] = new[:, 0]
            rmat[:, ch, :] = new[:, 1:]
        flat = rmat.reshape(N, 3 * per)
        for i, n in enumerate(rest):
            v[n] = flat[:, i]


def main():
    ap = argparse.ArgumentParser(description="Apply Sim3 (+ optional ICP-to-DTM) to a 3DGS .ply.")
    ap.add_argument("--ply", required=True, help="input 3DGS point_cloud.ply")
    ap.add_argument("--transform", required=True, help="geo_transform.txt from geo_align.py")
    ap.add_argument("--out", required=True, help="output georeferenced .ply")
    ap.add_argument("--dtm", default=None, help="bare-earth DTM GeoTIFF (same CRS); enables ICP")
    ap.add_argument("--ground-band", type=float, default=1.5,
                    help="half-width (m) of the height-above-DTM band used to pick ground points")
    ap.add_argument("--icp-iters", type=int, default=60)
    ap.add_argument("--recenter", action="store_true",
                    help="shift the output to a local origin (+ .offset.txt) so it renders in "
                         "WebGL viewers (SuperSplat) and avoids float32 quantisation")
    args = ap.parse_args()

    from plyfile import PlyData, PlyElement

    c, R1, t1 = read_transform(args.transform)
    ply = PlyData.read(args.ply)
    v = ply["vertex"].data
    names = v.dtype.names
    Nv = len(v)
    X1 = c * (R1 @ np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64).T).T + t1

    R2, t2 = np.eye(3), np.zeros(3)
    if args.dtm:
        Z, nrm, T, crs = load_dtm(args.dtm)
        epsg_t = read_epsg(args.transform)
        if crs is not None and epsg_t is not None and crs.to_epsg() not in (None, epsg_t):
            print(f"  WARNING: DTM CRS EPSG:{crs.to_epsg()} != transform EPSG:{epsg_t}. "
                  "Reproject the DTM (dtm_merge_reproject.py) to match.")

        ri, ci, ok = nearest_cell(T, Z, X1[:, 0], X1[:, 1])
        res = np.full(Nv, np.nan)
        res[ok] = X1[ok, 2] - Z[ri[ok], ci[ok]]
        fin = np.isfinite(res)
        if fin.sum() < 50:
            sys.exit("Splat and DTM barely overlap (check CRS / extents). Cannot refine.")
        gm = fin & (np.abs(res - np.median(res[fin])) <= args.ground_band)
        if gm.sum() < 50:
            sys.exit(f"Only {int(gm.sum())} ground points; widen --ground-band.")

        Pg = X1[gm]
        O = np.floor(Pg.mean(0))
        R2, t2, info = icp_point_to_plane(Pg, dtm_correspondence(Z, nrm, T), O, iters=args.icp_iters)

        ang = np.degrees(np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1)))
        print(f"  ground points: {int(gm.sum())} / {Nv}")
        print(f"  ICP: {info['iters']} iters, {info['inliers']} inliers, "
              f"rotation {ang:.3f} deg, translation {np.linalg.norm(t2):.3f} m")
        if info["rms0"] and info["rms"]:
            print(f"  ground-to-DTM residual: median {info['median_abs']:.3f} m  "
                  f"(RMS {info['rms0']:.3f} -> {info['rms']:.3f} m)")

    c_c, R_c, t_c = compose_sim3(c, R1, t1, R2, t2)
    apply_to_vertex(v, names, c_c, R_c, t_c, Nv)

    if args.recenter:
        pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        off = np.floor(pos.mean(0))
        pos -= off
        v["x"], v["y"], v["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
        epsg_t = read_epsg(args.transform)
        with open(args.out + ".offset.txt", "w") as f:
            f.write(f"# local splat: world{f' (EPSG:{epsg_t})' if epsg_t else ''} = local + offset\n")
            f.write("offset " + " ".join(f"{x:.3f}" for x in off) + "\n")
        print(f"  recentered by {off.tolist()} -> {args.out}.offset.txt")

    PlyData([PlyElement.describe(v, "vertex")], text=False, byte_order="<").write(args.out)
    print(f"Wrote {args.out}  ({Nv} Gaussians; scale {c_c:.5f}; "
          f"{'Sim3 + ICP' if args.dtm else 'Sim3 only'}{'; recentered' if args.recenter else ''}).")


if __name__ == "__main__":
    main()