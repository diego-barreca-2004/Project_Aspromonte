#!/usr/bin/env python3
"""test_undistort_pinhole.py - the script must run on the installed requirements, without PyYAML.

Fixture: 3 synthetic 800x700 frames and a fisheye calibration at 1600x1400. PyYAML is blocked for
the import, since requirements.txt does not install it. Checks:
  - the script runs and writes one undistorted image per frame;
  - intrinsics.yaml holds the layout VidMap reads with yaml.safe_load (integer camera id 1,
    params, images: all), with the calibration scaled to the frames;
  - if PyYAML happens to be installed: the text is what yaml.safe_dump writes, and it loads back.

Run:  python3 tests/test_undistort_pinhole.py
"""
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    import yaml as real_yaml
except ImportError:
    real_yaml = None
sys.modules["yaml"] = None                             # any "import yaml" now raises ImportError

checks = []


def ok(name, cond, extra=""):
    checks.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")


def main():
    T = tempfile.mkdtemp(prefix="test_undistort_pinhole_")
    frames, out, calib = os.path.join(T, "frames"), os.path.join(T, "out"), os.path.join(T, "calib.json")
    os.makedirs(frames)
    for k in range(3):
        cv2.imwrite(os.path.join(frames, f"frame_{k}.jpg"), np.full((700, 800, 3), 100, np.uint8))
    with open(calib, "w") as f:
        json.dump(dict(model="fisheye", image_size=[1600, 1400], distortion_coefficients=[0.01, 0, 0, 0],
                       camera_matrix=[[1400.5, 0, 800.25], [0, 1402.0, 700.75], [0, 0, 1]]), f)

    import undistort_pinhole
    sys.argv = ["undistort_pinhole.py", "--frames", frames, "--out", out, "--calib", calib]
    undistort_pinhole.main()

    ok("one image per frame", len(os.listdir(os.path.join(out, "images"))) == 3)
    text = open(os.path.join(out, "intrinsics.yaml")).read()
    expected = ("1:\n  params:\n  - 700.25\n  - 701.0\n  - 400.125\n  - 350.375\n  images: all\n")
    ok("intrinsics.yaml in the layout VidMap reads, scaled to the frames", text == expected, f"({text!r})")
    if real_yaml is not None:
        values = {1: {"params": [700.25, 701.0, 400.125, 350.375], "images": "all"}}
        ok("same text as yaml.safe_dump", text == real_yaml.safe_dump(values, sort_keys=False))
        ok("loads back with yaml.safe_load", real_yaml.safe_load(text) == values)
    else:
        print("  (PyYAML not installed: the comparison with yaml.safe_dump is skipped)")

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    shutil.rmtree(T)
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
