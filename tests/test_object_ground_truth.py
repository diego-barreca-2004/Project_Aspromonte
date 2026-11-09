#!/usr/bin/env python3
"""test_object_ground_truth.py - going back from an early frame must stop at the first frame.

Fixture: a pass of 20 frames with a dark disc on grey and known camera positions, so that
measure() finds the disc in any frame. Checks that with the default 8 frames and step 2:
  - from frame 3 it uses frames 3 and 1 only, and no negative index wraps to the end of the pass;
  - from frame 18 it uses the 8 frames 18, 16, ..., 4.

Run:  python3 tests/test_object_ground_truth.py
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from object_ground_truth import measure  # noqa: E402

checks = []


def ok(name, cond, extra=""):
    checks.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")


def main():
    T = tempfile.mkdtemp(prefix="test_object_ground_truth_")
    os.makedirs(os.path.join(T, "frames"))
    names = [f"frame_{k:04d}.png" for k in range(20)]
    px = lambda k: 120 + 3 * k                         # parallax: the disc moves 3 px per frame
    for k, n in enumerate(names):
        img = np.full((300, 400, 3), 128, np.uint8)
        cv2.circle(img, (px(k), 150), 20, (20, 20, 20), -1)
        cv2.imwrite(os.path.join(T, "frames", n), img)
    cams = {1: (5, np.array([300.0, 300.0, 200.0, 150.0, 0.0, 0.0, 0.0, 0.0]))}  # OPENCV_FISHEYE
    imgs = {n: (np.eye(3), np.array([-0.5 * k, 0.0, 0.0]), 1) for k, n in enumerate(names)}
    sim = (1.0, np.eye(3), np.zeros(3))

    def used(start):
        r = measure(T, None, cams, imgs, sim, "disc", "dark", start, (float(px(start)), 150.0), 8, 2, 120)
        return [f["frame"] for f in r["frames"]]

    print("from frame 3 (fewer than 8 frames before it):")
    got = used(3)
    ok("frames 3 and 1 only", got == [names[3], names[1]], f"({got})")

    print("from frame 18:")
    got = used(18)
    ok("the 8 frames going back", got == [names[k] for k in range(18, 2, -2)], f"({len(got)} frames)")

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    shutil.rmtree(T)
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
