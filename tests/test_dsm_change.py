#!/usr/bin/env python3
"""test_dsm_change.py - validation of dsm_change.py on a synthetic trail with known changes.

Fixture: a 20 x 4 m sloping trail (1 cm noise per epoch) with a GPS track along its
centre line, and four known cases:
  - a 0.30 m box that APPEARS in the compared epoch      -> must be detected
  - a 0.30 m box that is REMOVED in the compared epoch   -> must be detected
  - a flat 1.3 cm sheet (plasterboard) that appears      -> must NOT be detected
  - a vegetation patch, randomly 0-0.5 m high in both    -> must be rejected as rough

Run:  python3 tests/test_dsm_change.py
"""
import csv
import os
import shutil
import sys

import numpy as np
from plyfile import PlyData, PlyElement
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # dsm_change lives in the repository root
T = os.path.join(HERE, "testdata_dsm")
E0, N0 = 559000.0, 4214000.0
APPEAR, REMOVE, SHEET = (5.0, 0.0), (15.0, 0.5), (10.0, 0.0)
VEG = (17.0, 19.0, -1.0, 0.0)  # xmin, xmax, ymin, ymax


def ground(rng, n):
    x, y = rng.uniform(0, 20, n), rng.uniform(-2, 2, n)
    return x, y, 100.0 + 0.02 * x + rng.normal(0, 0.01, n)


def in_box(x, y, c, wx, wy):
    return (np.abs(x - c[0]) < wx / 2) & (np.abs(y - c[1]) < wy / 2)


def epoch(seed, appear, sheet, remove):
    rng = np.random.default_rng(seed)
    x, y, z = ground(rng, 288_000)  # ~3600 pts/m2, ~9 per 5 cm cell
    if appear:
        z = z + 0.30 * in_box(x, y, APPEAR, 0.3, 0.3)
    if sheet:
        z = z + 0.013 * in_box(x, y, SHEET, 0.6, 0.3)
    if remove:
        z = z + 0.30 * in_box(x, y, REMOVE, 0.3, 0.3)
    veg = (x > VEG[0]) & (x < VEG[1]) & (y > VEG[2]) & (y < VEG[3])
    z[veg] += rng.uniform(0, 0.5, veg.sum())
    return np.column_stack([x + E0, y + N0, z])


def write_ply(path, pts):
    v = np.empty(len(pts), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    v["x"], v["y"], v["z"] = pts.T
    PlyData([PlyElement.describe(v, "vertex")]).write(path)


def build_fixture():
    os.makedirs(T)
    write_ply(os.path.join(T, "ref.ply"), epoch(1, appear=False, sheet=False, remove=True))
    write_ply(os.path.join(T, "cmp.ply"), epoch(2, appear=True, sheet=True, remove=False))
    inv = Transformer.from_crs(32633, 4326, always_xy=True)
    with open(os.path.join(T, "track.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "lat", "lon"])
        for i, x in enumerate(np.arange(0, 20.01, 0.1)):
            lon, lat = inv.transform(E0 + x, N0)
            w.writerow([i / 10, f"{lat:.9f}", f"{lon:.9f}"])


def main():
    if os.path.isdir(T):
        shutil.rmtree(T)
    build_fixture()
    import dsm_change
    out = os.path.join(T, "out")
    rows, noise = dsm_change.main([
        "--ref", os.path.join(T, "ref.ply"), "--cmp", os.path.join(T, "cmp.ply"),
        "--track", os.path.join(T, "track.csv"), "--corridor", "1.5", "--out", out])

    checks = []

    def ok(name, cond, detail=""):
        checks.append((name, bool(cond)))
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    def near(c, r):
        return [d for d in rows if np.hypot(d["E"] - E0 - c[0], d["N"] - N0 - c[1]) < r]

    print("== assertions ==")
    app, rem = near(APPEAR, 0.15), near(REMOVE, 0.15)
    ok("appeared box detected once", len(app) == 1 and app[0]["kind"] == "appeared")
    ok("appeared box height ~ +0.30 m", app and 0.25 < app[0]["dz_median"] < 0.35,
       f"({app[0]['dz_median']:+.3f})" if app else "")
    ok("appeared box area ~ 0.09 m2", app and 0.05 < app[0]["area_m2"] < 0.15,
       f"({app[0]['area_m2']:.3f})" if app else "")
    ok("removed box detected once", len(rem) == 1 and rem[0]["kind"] == "removed")
    ok("removed box height ~ -0.30 m", rem and -0.35 < rem[0]["dz_median"] < -0.25,
       f"({rem[0]['dz_median']:+.3f})" if rem else "")
    ok("1.3 cm sheet stays below the detection threshold", not near(SHEET, 0.6))
    ok("vegetation patch rejected as rough",
       not [d for d in rows if VEG[0] - 0.2 < d["E"] - E0 < VEG[1] + 0.2
            and VEG[2] - 0.2 < d["N"] - N0 < VEG[3] + 0.2])
    ok("no other detections", len(rows) == 2, f"({len(rows)} total)")
    ok("bare-ground noise estimate is plausible", 0.005 < noise < 0.03, f"(NMAD {noise:.4f})")
    ok("outputs written", all(os.path.isfile(os.path.join(out, f)) for f in
                              ("dsm_detections.csv", "dsm_map.png", "dsm_report.txt")))

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
