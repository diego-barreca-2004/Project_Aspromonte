#!/usr/bin/env python3
"""
georef_cloud.py -- Georeference a COLMAP dense point cloud into absolute UTM.

Track 2, Phase 1. Takes the COLMAP-frame fused.ply, removes MVS floaters with an
iterative statistical outlier removal (SOR), applies the Sim3 stored in
geo_transform.txt (X_utm = scale * R @ X + t), and writes an absolute-UTM cloud.

Two correctness points baked in:
  * Output x/y/z are float64. UTM coordinates are ~4.2e6; float32 would quantise
    them to ~0.25 m, destroying sub-metre change detection.
  * SOR runs with fixed, documented parameters. Use the SAME values for epoch 2,
    or M3C2 measures filtering differences instead of terrain change.

Usage:
    python3 georef_cloud.py \
        ./seg01/colmap/dense/fused.ply \
        ./seg01/colmap/geo_transform.txt \
        ./seg01/colmap/dense/fused_utm.ply
        
    python3 georef_cloud.py \
        ./seg01/colmap/dense/fused.ply \
        ./seg01/colmap/geo_transform.txt \
        ./seg01/colmap/dense/fused_utm.ply \
        --sor-iters 2 --sor-k 20 --sor-std 2.0
    # --sor-iters 0 disables floater removal
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement


def parse_geo_transform(path):
    """Read scale, R (3x3), t (3,), epsg from a geo_transform.txt."""
    scale = R = t = epsg = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            key = p[0].lower()
            if key == "scale":
                scale = float(p[1])
            elif key == "r":
                R = np.array(list(map(float, p[1:10])), dtype=np.float64).reshape(3, 3)
            elif key == "t":
                t = np.array(list(map(float, p[1:4])), dtype=np.float64)
            elif key == "epsg":
                epsg = p[1]
    if scale is None or R is None or t is None:
        raise ValueError("geo_transform.txt must define scale, R and t")
    return scale, R, t, epsg


def sor_mask(xyz, k=20, std_ratio=2.0):
    """Statistical outlier removal. Returns a boolean keep-mask.
    Invariant to similarity transforms, so it is applied in the COLMAP frame."""
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1, workers=-1)   # k+1: first neighbour is self
    mean_d = d[:, 1:].mean(axis=1)
    mu, sigma = mean_d.mean(), mean_d.std()
    return mean_d <= mu + std_ratio * sigma


def bbox_line(xyz):
    lo, hi = xyz.min(0), xyz.max(0)
    return f"min={np.round(lo,3)}  max={np.round(hi,3)}  extent={np.round(hi-lo,3)}"


def main():
    ap = argparse.ArgumentParser(
        description="Georeference a COLMAP dense cloud into absolute UTM.")
    ap.add_argument("input_ply")
    ap.add_argument("geo_transform")
    ap.add_argument("output_ply")
    ap.add_argument("--sor-iters", type=int, default=2, help="SOR passes (0 = off)")
    ap.add_argument("--sor-k", type=int, default=20, help="neighbours per point")
    ap.add_argument("--sor-std", type=float, default=2.0, help="std-dev multiplier")
    args = ap.parse_args()

    scale, R, t, epsg = parse_geo_transform(args.geo_transform)
    print(f"Sim3: scale={scale:.6f}  EPSG:{epsg}")
    print(f"t = {t}")

    ply = PlyData.read(args.input_ply)
    vtx = ply["vertex"]
    names = list(vtx.data.dtype.names)
    xyz = np.column_stack([vtx["x"], vtx["y"], vtx["z"]]).astype(np.float64)
    has_normals = all(n in names for n in ("nx", "ny", "nz"))
    print(f"\nLoaded {len(xyz):,} points ({'with' if has_normals else 'no'} normals)")
    print(f"COLMAP bbox (raw):   {bbox_line(xyz)}")

    # --- floater removal (COLMAP frame) ---
    mask = np.ones(len(xyz), dtype=bool)
    for i in range(args.sor_iters):
        m = sor_mask(xyz[mask], k=args.sor_k, std_ratio=args.sor_std)
        idx = np.where(mask)[0]
        mask[idx[~m]] = False
        print(f"SOR pass {i+1}: kept {mask.sum():,} / {len(xyz):,} "
              f"({100*mask.sum()/len(xyz):.1f}%)")
    if args.sor_iters > 0:
        print(f"COLMAP bbox (clean): {bbox_line(xyz[mask])}")

    # --- apply Sim3: X_utm = scale * R @ X + t ---
    xyz_utm = scale * (xyz[mask] @ R.T) + t

    # --- build output; x/y/z promoted to float64, everything else preserved ---
    kept = vtx.data[mask]
    out_dtype = [(n, "f8") if n in ("x", "y", "z") else (n, kept.dtype[n]) for n in names]
    out = np.empty(len(kept), dtype=out_dtype)
    for n in names:
        out[n] = kept[n]
    out["x"], out["y"], out["z"] = xyz_utm[:, 0], xyz_utm[:, 1], xyz_utm[:, 2]
    if has_normals:
        nrm = np.column_stack([kept["nx"], kept["ny"], kept["nz"]]).astype(np.float64) @ R.T
        norm = np.linalg.norm(nrm, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        nrm /= norm
        out["nx"], out["ny"], out["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]

    PlyData([PlyElement.describe(out, "vertex")], text=False).write(args.output_ply)

    # --- report ---
    c = xyz_utm.mean(0)
    print(f"\nUTM bbox:   {bbox_line(xyz_utm)}")
    print(f"UTM centroid (box center): E={c[0]:.2f}  N={c[1]:.2f}  Z={c[2]:.2f}")
    print(f"Wrote {len(out):,} points -> {args.output_ply}  (x/y/z as float64)")


if __name__ == "__main__":
    main()