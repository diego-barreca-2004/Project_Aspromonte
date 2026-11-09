#!/usr/bin/env python3
"""test_synthetic_objects.py - the evaluation must use the detector arguments of the run.

Fixture: synthetic_objects.main() on a flat cloud along a straight 40 m track, with the detector
(dsm_change.main) replaced by a stand-in that writes no candidates and logs its arguments;
cd_eval.py is the real one. Checks:
  - --trim-ends in --detector-args also shortens the track length used for false alarms per km;
  - run_null is reused with the same detector arguments and recomputed with other ones, or when
    it carries no record of its arguments.

Run:  python3 tests/test_synthetic_objects.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import dsm_change  # noqa: E402
import synthetic_objects  # noqa: E402

checks = []
null_calls = []


def ok(name, cond, extra=""):
    checks.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")


def fake_detector(argv):
    out = argv[argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    if os.path.basename(out) == "run_null":
        null_calls.append(list(argv))
    with open(os.path.join(out, "dsm_detections.csv"), "w") as f:
        f.write("kind,E,N,dz_peak,score\n")


def main():
    from pyproj import Transformer
    T = tempfile.mkdtemp(prefix="test_synthetic_objects_")
    e0, n0 = 559000.0, 4214000.0
    to_ll = Transformer.from_crs(32633, 4326, always_xy=True).transform

    track = os.path.join(T, "gps.csv")
    with open(track, "w") as f:
        f.write("lat,lon\n")
        for s in np.arange(0, 40.5, 1.0):
            lon, lat = to_ll(e0 + s, n0)
            f.write(f"{lat:.9f},{lon:.9f}\n")
    length = float(np.hypot(*np.diff(dsm_change.read_track([track], 32633), axis=0).T).sum())

    icp = os.path.join(T, "icp.txt")
    with open(icp, "w") as f:
        f.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    x, y = np.meshgrid(np.arange(e0 - 2, e0 + 42, 0.05), np.arange(n0 - 2, n0 + 2, 0.05))
    cloud = np.c_[x.ravel(), y.ravel(), np.full(x.size, 100.0)]
    seg = os.path.join(T, "cmp")
    os.makedirs(os.path.join(seg, "colmap", "dense"))
    ply = os.path.join(seg, "colmap", "dense", "fused_utm.ply")
    synthetic_objects.write_xyz(ply, cloud)

    s = np.arange(0, 41, 1.0)
    synthetic_objects.camera_centres = lambda _: np.c_[e0 + s, np.full(s.size, n0 + 1.0), np.full(s.size, 101.5)]
    dsm_change.main = fake_detector
    out = os.path.join(T, "out")

    def run_main(detector_args):
        null_calls.clear()
        sys.argv = ["synthetic_objects.py", "--ref", ply, "--cmp-seg", seg, "--icp", icp,
                    "--cc-shift", "0", "0", "0", "--track", track, "--runs", "1", "--out", out,
                    "--detector-args", detector_args]
        synthetic_objects.main()
        return json.load(open(os.path.join(out, "run_000", "eval.json")))

    print("--trim-ends through --detector-args:")
    ev = run_main("--trim-ends 5")
    expected = dsm_change.read_track([track], 32633, 5.0)
    expected = float(np.hypot(*np.diff(expected, axis=0).T).sum())
    ok("detector called with it", "--trim-ends" in null_calls[0])
    ok("track length trimmed in the evaluation", abs(ev["track_km"] * 1000 - expected) < 1e-6,
       f"({ev['track_km'] * 1000:.1f} m, full track {length:.1f} m, trimmed {expected:.1f} m)")

    print("run_null:")
    ok("computed on the first run", len(null_calls) == 1)
    run_main("--trim-ends 5")
    ok("reused with the same detector arguments", null_calls == [])
    run_main("--trim-ends 5 --k 5")
    ok("recomputed with other detector arguments", len(null_calls) == 1 and "--k" in null_calls[0])
    run_main("--trim-ends 5 --k 5")
    ok("reused again with those", null_calls == [])
    os.remove(os.path.join(out, "run_null", "detector_args.json"))
    run_main("--trim-ends 5 --k 5")
    ok("recomputed when it has no record of its arguments", len(null_calls) == 1)

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    shutil.rmtree(T)
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
