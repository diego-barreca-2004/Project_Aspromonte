#!/usr/bin/env python3
"""test_cd_eval.py - validation of cd_eval.py on a hand-made candidate list.

Fixture: a 200 m straight GPS track, one real object (hidden unless --reveal-objects), one
synthetic object, one excluded zone, and five candidates ranked by a score column:
  - two on the real object (one of the wrong kind)   -> no false alarm, rank 2
  - one on the synthetic object                      -> no false alarm, rank 3
  - one in the excluded zone                         -> no false alarm
  - one elsewhere                                    -> the only false alarm
With a null run that already holds the candidate on the real object, that candidate becomes
a false alarm and the object is missed.

Run:  python3 tests/test_cd_eval.py
"""
import csv
import json
import os
import shutil
import sys

from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # cd_eval lives in the repository root
T = os.path.join(HERE, "testdata_cd_eval")
E0, N0 = 559000.0, 4214000.0


def build_fixture():
    os.makedirs(T)
    inv = Transformer.from_crs(32633, 4326, always_xy=True)
    with open(os.path.join(T, "gps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "lat", "lon"])
        for i in range(201):
            lon, lat = inv.transform(E0 + i, N0)
            w.writerow([i, f"{lat:.9f}", f"{lon:.9f}"])
    json.dump({"bucket": {"E": E0 + 50, "N": N0, "z": 100, "height_m": 0.3},
               "_gt_interval": {"bucket": {"height_m": [0.265, 0.278]}}},
              open(os.path.join(T, "gt.json"), "w"))
    json.dump({"box1": {"E": E0 + 120, "N": N0, "height_m": 0.2}},
              open(os.path.join(T, "synth.json"), "w"))
    rows = [("removed", 50.1, 0.0, -0.2, 9), ("appeared", 50.2, 0.1, 0.29, 8),
            ("appeared", 120.3, 0.0, 0.5, 7), ("appeared", 160.0, 0.0, 0.2, 6),
            ("appeared", 90.0, 0.0, 0.2, 5)]
    with open(os.path.join(T, "cands.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "E", "N", "dz_peak", "score"])
        for kind, x, y, dz, s in rows:
            w.writerow([kind, E0 + x, N0 + y, dz, s])
    with open(os.path.join(T, "null.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "E", "N"])
        w.writerow(["appeared", E0 + 50.25, N0 + 0.1])


def main():
    if os.path.isdir(T):
        shutil.rmtree(T)
    build_fixture()
    import cd_eval

    base = ["--candidates", os.path.join(T, "cands.csv"), "--track", os.path.join(T, "gps.csv"),
            "--gt", os.path.join(T, "gt.json"), "--synthetic", os.path.join(T, "synth.json"),
            "--exclude", str(E0 + 160), str(N0), "1"]
    hidden = cd_eval.main(base)
    shown = cd_eval.main(base + ["--reveal-objects", "--out", os.path.join(T, "out.json")])
    nulled = cd_eval.main(base + ["--reveal-objects", "--null-candidates", os.path.join(T, "null.csv")])

    checks = []

    def ok(name, cond, detail=""):
        checks.append((name, bool(cond)))
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    print()
    ok("track length 200 m", abs(hidden["track_km"] - 0.2) < 1e-3, f"({hidden['track_km']:.4f} km)")
    ok("one false alarm", hidden["false_alarms"] == 1, f"({hidden['false_alarms']})")
    ok("5 false alarms per km", abs(hidden["false_per_km"] - 5) < 0.05)
    ok("real object hidden without --reveal-objects", "bucket" not in hidden["objects"])
    ok("synthetic object always shown, rank 3", hidden["objects"]["box1"]["rank"] == 3)
    b = shown["objects"]["bucket"]
    ok("real object rank 2 (wrong-kind candidate skipped)", b["rank"] == 2, f"({b['rank']})")
    ok("height 0.29 m inside 0.265-0.278 +/- 0.03", b["height_ok"] is True)
    ok("FROC reaches recall 1 with one false alarm",
       shown["froc"][-1][1] == 1.0 and abs(shown["froc"][-1][0] - 5) < 0.05)
    ok("null run: pre-existing candidate is a false alarm", nulled["false_alarms"] == 2)
    ok("null run: object reached only by it is missed", nulled["objects"]["bucket"]["rank"] is None)
    ok("metrics JSON written", os.path.isfile(os.path.join(T, "out.json")))

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    shutil.rmtree(T)
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
