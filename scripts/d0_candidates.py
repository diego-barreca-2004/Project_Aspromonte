#!/usr/bin/env python3
"""d0_candidates.py - why are the DSM candidates there? Views, roughness and dipoles.

For every candidate of a dsm_change.py run, measured on both dense clouds within --radius
of its centre (the raw fused.ply, georeferenced with geo_transform.txt and, for the compared
epoch, the run's ICP; SOR is not applied so the points keep their fused.ply.vis entries):

  - points and mean number of views per point (fused.ply.vis), in each epoch;
  - P90 - P10 height spread, in each epoch;
  - distance from the track and position along it;
  - distance to the nearest candidate of the opposite kind (dipole partner).

The same views and spread are measured at --background random track positions, so a
candidate can be compared with the corridor as a whole (percentile columns).

With --probe LABEL:E:N:Z (positions in the compared epoch frame, as in objects.json; Z is
needed because the ICP rotation mixes height into E and N) it also
reports point density and heights inside --probe-radius of each position against the ring
0.5-1.0 m around it, in both epochs: a hole in one epoch shows as a low density ratio.

  python3 scripts/d0_candidates.py --ref seg01_ep2 --cmp seg01_ep3 \\
      --icp change_detection/m3c2_ep3_vs_ep2_v2/icp_auto.txt --cc-shift -558997 -4214492 -123 \\
      --candidates change_detection/dsm_ep3_vs_ep2/dsm_detections.csv --pair ep3/ep2 \\
      --out change_detection/v2/d0_candidates.csv
"""
import argparse
import csv
import os
import struct
import sys

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm_change import read_track  # noqa: E402
from georef_cloud import parse_geo_transform  # noqa: E402
from run_m3c2 import parse_cc_matrices  # noqa: E402


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


def load_epoch(seg, M=None, shift=None):
    dense = os.path.join(seg, "colmap", "dense")
    v = PlyData.read(os.path.join(dense, "fused.ply"))["vertex"]
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    views = read_views(os.path.join(dense, "fused.ply.vis"))
    if len(views) != len(xyz):
        sys.exit(f"{seg}: {len(xyz)} points but {len(views)} vis entries")
    scale, R, t, _ = parse_geo_transform(os.path.join(seg, "colmap", "geo_transform.txt"))
    xyz = scale * xyz @ R.T + t
    if M is not None:
        xyz = (xyz + shift) @ M[:3, :3].T + M[:3, 3] - shift
    return xyz, views


def disk_stats(tree, xyz, views, xy, r):
    idx = tree.query_ball_point(xy, r)
    if len(idx) < 3:
        return len(idx), np.nan, np.nan
    z = xyz[idx, 2]
    return len(idx), float(views[idx].mean()), float(np.percentile(z, 90) - np.percentile(z, 10))


def main():
    ap = argparse.ArgumentParser(description="D0 diagnostics of DSM candidates.")
    ap.add_argument("--ref", required=True, help="reference epoch folder (seg01_ep2)")
    ap.add_argument("--cmp", required=True, help="compared epoch folder (seg01_ep3)")
    ap.add_argument("--icp", required=True, help="the run's CloudCompare matrix (cmp to ref)")
    ap.add_argument("--cc-shift", nargs=3, type=float, required=True)
    ap.add_argument("--candidates", required=True, help="dsm_detections.csv of the run")
    ap.add_argument("--track", help="gps.csv of the reference epoch (default <ref>/gps.csv)")
    ap.add_argument("--pair", required=True, help="name written in the pair column")
    ap.add_argument("--radius", type=float, default=0.25, help="measurement disk [m]")
    ap.add_argument("--background", type=int, default=2000)
    ap.add_argument("--probe", action="append", default=[], metavar="LABEL:E:N:Z")
    ap.add_argument("--probe-radius", type=float, default=0.15)
    ap.add_argument("--out", required=True, help="CSV, appended to if it exists")
    args = ap.parse_args()

    M = np.asarray(parse_cc_matrices(args.icp)[0])
    shift = np.array(args.cc_shift, float)
    track = read_track([args.track or os.path.join(args.ref, "gps.csv")], 32633)
    s_track = np.r_[0.0, np.cumsum(np.hypot(*np.diff(track, axis=0).T))]
    ttree = cKDTree(track)
    epochs = {}
    for name, seg, m in (("ref", args.ref, None), ("cmp", args.cmp, M)):
        xyz, views = load_epoch(seg, m, shift)
        keep = ttree.query(xyz[:, :2])[0] <= 2.0
        xyz, views = xyz[keep], views[keep]
        epochs[name] = (cKDTree(xyz[:, :2]), xyz, views)
        print(f"{name}: {seg}, {keep.sum():,} points within 2 m of the track")

    rng = np.random.default_rng(0)
    bg = {}
    pick = rng.integers(0, len(track), args.background)
    for name, (tree, xyz, views) in epochs.items():
        st = np.array([disk_stats(tree, xyz, views, track[i], args.radius) for i in pick])
        bg[name] = st[np.isfinite(st[:, 1])]

    cands = list(csv.DictReader(open(args.candidates)))
    xy = np.array([[float(c["E"]), float(c["N"])] for c in cands])
    rows = []
    for i, c in enumerate(cands):
        d, j = ttree.query(xy[i])
        opp = [k for k, o in enumerate(cands) if o["kind"] != c["kind"]]
        dip = min((np.hypot(*(xy[k] - xy[i])) for k in opp), default=np.nan)
        row = dict(pair=args.pair, rank=i + 1, kind=c["kind"], E=xy[i, 0], N=xy[i, 1],
                   area_m2=float(c["area_m2"]), dz_median=float(c["dz_median"]),
                   along_m=s_track[j], track_dist_m=d, dipole_m=dip)
        for name, (tree, xyz, views) in epochs.items():
            n, v, sp = disk_stats(tree, xyz, views, xy[i], args.radius)
            row[f"{name}_pts"], row[f"{name}_views"], row[f"{name}_spread_m"] = n, v, sp
            row[f"{name}_views_pct"] = float((bg[name][:, 1] < v).mean() * 100) if np.isfinite(v) else np.nan
            row[f"{name}_spread_pct"] = float((bg[name][:, 2] < sp).mean() * 100) if np.isfinite(sp) else np.nan
        rows.append(row)

    new = not os.path.isfile(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)

    print(f"\n{args.pair}: {len(rows)} candidates -> {args.out}")
    for name in epochs:
        b = bg[name]
        print(f"  background {name}: views median {np.median(b[:, 1]):.1f}, "
              f"spread median {np.median(b[:, 2]) * 100:.1f} cm ({len(b)} disks)")
    for name in epochs:
        vp = np.array([r[f"{name}_views_pct"] for r in rows])
        sp = np.array([r[f"{name}_spread_pct"] for r in rows])
        print(f"  {name}: candidates below background P10 of views {np.mean(vp < 10) * 100:.0f} %, "
              f"above P90 of spread {np.mean(sp > 90) * 100:.0f} %")
    dip = np.array([r["dipole_m"] for r in rows])
    far = np.array([r["track_dist_m"] for r in rows])
    print(f"  dipole partner within 1 m: {np.mean(dip <= 1.0) * 100:.0f} % | "
          f"track distance > 0.7 m: {np.mean(far > 0.7) * 100:.0f} %")

    for spec in args.probe:
        label, *enz = spec.split(":")
        p = np.array(enz, float)
        p = (M[:3, :3] @ (p + shift) + M[:3, 3] - shift)[:2]
        print(f"\n  probe {label} at {p[0]:.2f} {p[1]:.2f} (ref frame), r = {args.probe_radius:g} m")
        for name, (tree, xyz, views) in epochs.items():
            inner = tree.query_ball_point(p, args.probe_radius)
            ring = [k for k in tree.query_ball_point(p, 1.0) if np.hypot(*(xyz[k, :2] - p)) >= 0.5]
            a_in, a_ring = np.pi * args.probe_radius ** 2, np.pi * (1.0 - 0.25)
            ratio = (len(inner) / a_in) / max(len(ring) / a_ring, 1e-9)
            zr = np.median(xyz[ring, 2]) if ring else np.nan
            top = np.percentile(xyz[inner, 2], 90) - zr if len(inner) >= 3 else np.nan
            print(f"    {name}: {len(inner)} pts inside, density ratio {ratio:.2f}, "
                  f"P90 height above ring median {top * 100:+.1f} cm")


if __name__ == "__main__":
    main()
