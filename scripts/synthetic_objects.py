#!/usr/bin/env python3
"""synthetic_objects.py - insert synthetic objects into a dense cloud and measure recall.

One run: about 40 boxes, cylinders and domes are placed on the trail of the compared epoch,
one per --slot metres of track (so they stay >= 3 m apart), away from real objects and
excluded zones. For each object:

  - the local ground is a plane through the points within 0.6 m of its centre, refitted
    after dropping residuals beyond 2.5 NMAD (grass and stones would lift a plain least-squares
    plane and leave the object floating); the point density per m^2 and the noise (NMAD of
    the kept residuals) are taken there;
  - the ground points under the footprint are removed;
  - surface points are sampled at that density on every face, with that noise along the
    normal, and kept only where the normal faces at least 2 cameras of the pass within 8 m.

Heights cycle through --heights; footprint areas are log-uniform in --area. A dome higher
than its base radius becomes a cylinder.

Then the detector runs on (ref, modified cmp) and cd_eval.py scores its candidates against
the inserted objects; candidates already found without insertions (run_null/, made once)
never count as detections. With --runs N this repeats with seeds 0..N-1 and writes the recall by
height and area to <out>/recall.csv.

  python3 scripts/synthetic_objects.py --ref seg01/colmap/dense/fused_utm.ply \\
      --cmp-seg seg01_ep2 --icp change_detection/icp_ep2_to_ep1.txt \\
      --cc-shift -559000 -4214000 0 --track seg01/gps.csv \\
      --exclude 558957.5 4214479.3 3 --runs 80 --out change_detection/v2/synthetic/n1_dsm
"""
import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import cd_eval  # noqa: E402
import dsm_change  # noqa: E402
from object_ground_truth import read_images, read_sim3  # noqa: E402
from run_colmap import largest_model  # noqa: E402
from run_m3c2 import nmad  # noqa: E402


def camera_centres(seg):
    """UTM centres of the registered cameras of a pass."""
    model = largest_model(os.path.join(seg, "colmap", "sparse"))
    s, R, t = read_sim3(os.path.join(seg, "colmap", "geo_transform.txt"))
    c = np.array([-Rc.T @ tc for Rc, tc, _ in read_images(model).values()])
    return s * c @ R.T + t


def surface(shape, h, area, rng, density):
    """Points and normals on the visible faces of an object standing on z = 0 at the origin."""
    def n_pts(a):
        return rng.poisson(density * a)

    pts, nrm = [], []
    if shape == "box":
        aspect = rng.uniform(1, 2)
        a, b = np.sqrt(area * aspect), np.sqrt(area / aspect)
        k = n_pts(a * b)
        pts.append(np.c_[rng.uniform(-a / 2, a / 2, k), rng.uniform(-b / 2, b / 2, k), np.full(k, h)])
        nrm.append(np.tile([0, 0, 1.0], (k, 1)))
        for sx, sy, w in ((1, 0, b), (-1, 0, b), (0, 1, a), (0, -1, a)):
            k = n_pts(w * h)
            u, z = rng.uniform(-w / 2, w / 2, k), rng.uniform(0, h, k)
            x = sx * a / 2 + (sy != 0) * u
            y = sy * b / 2 + (sx != 0) * u
            pts.append(np.c_[x, y, z])
            nrm.append(np.tile([sx, sy, 0.0], (k, 1)))
        yaw = rng.uniform(0, np.pi)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        P, N = np.vstack(pts) @ Rz.T, np.vstack(nrm) @ Rz.T
        inside = lambda xy: np.all(np.abs(xy @ Rz[:2, :2]) <= [a / 2, b / 2], axis=1)  # noqa: E731
        return P, N, inside
    r = np.sqrt(area / np.pi)
    if shape == "cylinder":
        k = n_pts(np.pi * r * r)
        rho, phi = r * np.sqrt(rng.uniform(0, 1, k)), rng.uniform(0, 2 * np.pi, k)
        pts.append(np.c_[rho * np.cos(phi), rho * np.sin(phi), np.full(k, h)])
        nrm.append(np.tile([0, 0, 1.0], (k, 1)))
        k = n_pts(2 * np.pi * r * h)
        phi = rng.uniform(0, 2 * np.pi, k)
        pts.append(np.c_[r * np.cos(phi), r * np.sin(phi), rng.uniform(0, h, k)])
        nrm.append(np.c_[np.cos(phi), np.sin(phi), np.zeros(k)])
    else:  # dome: spherical cap of base radius r and height h <= r, uniform on its surface
        R = (r * r + h * h) / (2 * h)
        k = n_pts(2 * np.pi * R * h)
        u, phi = rng.uniform(R - h, R, k), rng.uniform(0, 2 * np.pi, k)
        rho = np.sqrt(R * R - u * u)
        n = np.c_[rho * np.cos(phi), rho * np.sin(phi), u] / R
        pts.append(n * R - [0, 0, R - h])
        nrm.append(n)
    return np.vstack(pts), np.vstack(nrm), lambda xy: np.hypot(*xy.T) <= r


def insert(cloud, cams, track, rng, args, avoid):
    """Add objects to cloud (N x 3, UTM). Returns the new cloud and the object list."""
    tree = cKDTree(cloud[:, :2])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(track, axis=0).T))]
    keep = np.ones(len(cloud), bool)
    added, objects = [], {}
    heights = np.array(args.heights)
    for i, s0 in enumerate(np.arange(2.0, s[-1] - 2.0, args.slot)):
        si = s0 + rng.uniform(0, args.slot - 3.0)  # adjacent objects stay >= 3 m apart
        j = min(np.searchsorted(s, si), len(track) - 2)
        d = track[j + 1] - track[j]
        d /= np.linalg.norm(d) or 1.0
        c = track[j] + rng.uniform(-args.lateral, args.lateral) * np.array([-d[1], d[0]])
        if any(np.hypot(*(c - a[:2])) < a[2] for a in avoid):
            continue
        g = cloud[tree.query_ball_point(c, 0.6)]
        if len(g) < 50:
            continue
        A = np.c_[g[:, 0] - c[0], g[:, 1] - c[1], np.ones(len(g))]
        kept = np.ones(len(g), bool)
        for _ in range(5):
            coef = np.linalg.lstsq(A[kept], g[kept, 2], rcond=None)[0]
            res = g[:, 2] - A @ coef
            kept = np.abs(res) <= 2.5 * max(nmad(res[kept]), 0.005)
        density = len(g) / (np.pi * 0.6 ** 2)
        noise = float(nmad(res[kept]))
        h = float(heights[i % len(heights)])
        area = float(np.exp(rng.uniform(*np.log(args.area))))
        shape = ["box", "cylinder", "dome"][rng.integers(3)]
        if shape == "dome" and h > np.sqrt(area / np.pi):
            shape = "cylinder"
        P, N, inside = surface(shape, h, area, rng, density)
        P = P + N * rng.normal(0, noise, (len(P), 1))
        z0 = coef[2]
        P += [c[0], c[1], z0]
        near = np.array(tree.query_ball_point(c, 1.0), int)
        under = near[inside(cloud[near, :2] - c) & (cloud[near, 2] < z0 + h + 0.1)]
        keep[under] = False
        rel = cams[np.hypot(*(cams[:, :2] - c).T) <= args.cam_range]
        facing = ((rel[None, :, :] - P[:, None, :]) * N[:, None, :]).sum(axis=2) > 0
        P = P[facing.sum(axis=1) >= 2]
        added.append(P)
        objects[f"syn{len(objects):02d}"] = dict(E=float(c[0]), N=float(c[1]), z=float(z0),
                                                 height_m=h, area_m2=area, shape=shape,
                                                 n_points=len(P), kind="appeared")
    return np.vstack([cloud[keep]] + added), objects


def write_xyz(path, xyz):
    v = np.empty(len(xyz), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    v["x"], v["y"], v["z"] = xyz.T
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(path)


def recall_table(results, heights, area_bins):
    rows = []
    for h in heights:
        for lo, hi in zip(area_bins[:-1], area_bins[1:]):
            sel = [r for r in results if r["height_m"] == h and lo <= r["area_m2"] < hi]
            if sel:
                rows.append(dict(height_m=h, area_lo=lo, area_hi=hi, n=len(sel),
                                 recall=sum(r["rank"] is not None for r in sel) / len(sel)))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Synthetic objects: insertion and recall.")
    ap.add_argument("--ref", required=True, help="reference epoch fused_utm.ply")
    ap.add_argument("--cmp-seg", required=True, help="compared epoch folder (objects go here)")
    ap.add_argument("--icp", required=True)
    ap.add_argument("--cc-shift", nargs=3, required=True)
    ap.add_argument("--track", required=True, help="gps.csv of the reference epoch")
    ap.add_argument("--gt", help="real objects JSON: no synthetic object within 3 m")
    ap.add_argument("--exclude", nargs=3, type=float, action="append", default=[],
                    metavar=("E", "N", "R"), help="zone kept free and out of the false-alarm count")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--slot", type=float, default=3.5, help="track metres per object (>= 3)")
    ap.add_argument("--lateral", type=float, default=0.5, help="max offset from the track [m]")
    ap.add_argument("--heights", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60])
    ap.add_argument("--area", type=float, nargs=2, default=[0.02, 0.4])
    ap.add_argument("--cam-range", type=float, default=8.0)
    ap.add_argument("--detector-args", default="",
                    help="extra dsm_change.py arguments, as one string")
    ap.add_argument("--keep-ply", action="store_true", help="keep the modified clouds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.slot < 3.0:
        sys.exit("--slot must be >= 3 m: objects closer than 3 m merge in the candidates.")

    os.makedirs(args.out, exist_ok=True)
    cmp_ply = os.path.join(args.cmp_seg, "colmap", "dense", "fused_utm.ply")
    cloud = dsm_change.read_ply_xyz(cmp_ply)
    cams = camera_centres(args.cmp_seg)
    track = dsm_change.read_track([args.track], 32633)
    M = np.asarray(dsm_change.parse_cc_matrices(args.icp)[0])
    shift = np.array(args.cc_shift, float)
    # track and excluded zones are in the ref frame, objects go into the cmp frame; the
    # height is needed because the ICP rotation mixes it into E and N
    Minv = np.linalg.inv(M)
    ztree = cKDTree(cloud[:, :2])

    def to_cmp(xy):
        p = np.c_[xy, cloud[ztree.query(xy)[1], 2]] + shift
        return (p @ Minv[:3, :3].T + Minv[:3, 3] - shift)[:, :2]

    avoid = [np.r_[to_cmp(np.array([[e, n]]))[0], r + 1.0] for e, n, r in args.exclude]
    if args.gt:
        for lab, o in json.load(open(args.gt)).items():
            if not lab.startswith("_"):
                avoid.append(np.array([o["E"], o["N"], 3.0]))
    track_cmp = to_cmp(track)

    null_dir = os.path.join(args.out, "run_null")
    null_csv = os.path.join(null_dir, "dsm_detections.csv")
    if not os.path.isfile(null_csv):
        dsm_change.main(["--ref", args.ref, "--cmp", cmp_ply, "--icp", args.icp,
                         "--cc-shift", *args.cc_shift, "--track", args.track,
                         "--out", null_dir] + args.detector_args.split())
    results = []
    for seed in range(args.first_seed, args.first_seed + args.runs):
        run = os.path.join(args.out, f"run_{seed:03d}")
        os.makedirs(run, exist_ok=True)
        rng = np.random.default_rng(seed)
        mod, objects = insert(cloud, cams, track_cmp, rng, args, avoid)
        ply = os.path.join(run, "cmp_synthetic.ply")
        write_xyz(ply, mod)
        json.dump(objects, open(os.path.join(run, "synthetic.json"), "w"), indent=1)
        dsm_change.main(["--ref", args.ref, "--cmp", ply, "--icp", args.icp,
                         "--cc-shift", *args.cc_shift, "--track", args.track,
                         "--out", run] + args.detector_args.split())
        ev = ["--candidates", os.path.join(run, "dsm_detections.csv"), "--track", args.track,
              "--synthetic", os.path.join(run, "synthetic.json"), "--icp", args.icp,
              "--cc-shift", *args.cc_shift, "--label", f"seed {seed}",
              "--null-candidates", null_csv,
              "--out", os.path.join(run, "eval.json")]
        for e, n, r in args.exclude:
            ev += ["--exclude", str(e), str(n), str(r)]
        if args.gt:
            ev += ["--gt", args.gt]
        res = cd_eval.main(ev)
        for lab, o in objects.items():
            results.append(dict(seed=seed, label=lab, **{k: o[k] for k in ("height_m", "area_m2", "shape", "n_points")},
                                rank=res["objects"][lab]["rank"],
                                height_meas_m=res["objects"][lab].get("height_m")))
        if not args.keep_ply:
            os.remove(ply)

    with open(os.path.join(args.out, "objects.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    bins = [0.02, 0.04, 0.08, 0.16, 0.41]
    table = recall_table(results, args.heights, bins)
    with open(os.path.join(args.out, "recall.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"\n=== recall by height (rows) and footprint area (columns), {len(results)} objects ===")
    print("  h m  " + "".join(f"{lo:>6.2f}-{hi:<5.2f}" for lo, hi in zip(bins[:-1], bins[1:])))
    for h in args.heights:
        cells = {(r["area_lo"]): r for r in table if r["height_m"] == h}
        print(f"  {h:.2f} " + "".join(f"{cells[lo]['recall']:>6.2f} n={cells[lo]['n']:<3d}"
                                       if lo in cells else f"{'-':>12}" for lo in bins[:-1]))
    shutil.copy(__file__, os.path.join(args.out, "synthetic_objects.py.used"))


if __name__ == "__main__":
    main()
