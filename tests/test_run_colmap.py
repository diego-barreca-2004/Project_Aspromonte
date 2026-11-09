#!/usr/bin/env python3
"""test_run_colmap.py - a re-run into the same --out must start from a new database and sparse/.

Fixture: run_colmap.main() with COLMAP replaced by a stand-in. Like COLMAP, its feature extractor
keeps the camera of a database it finds, and its mapper adds to the sparse folder. Checks:
  - the second run's calibration reaches the database;
  - an older, larger sparse/7 is not picked as the largest model.

Run:  python3 tests/test_run_colmap.py
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import run_colmap  # noqa: E402

checks = []
state = {"undistorted_from": None}


def ok(name, cond, extra=""):
    checks.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")


def fake_run(cmd):
    tool = cmd[1]
    opt = lambda key: cmd[cmd.index(key) + 1]
    if tool == "feature_extractor":
        db = opt("--database_path")
        if not os.path.exists(db):                   # an existing database keeps its camera
            with open(db, "w") as f:
                f.write(opt("--ImageReader.camera_params"))
    elif tool == "mapper":
        model = os.path.join(opt("--output_path"), "0")
        os.makedirs(model, exist_ok=True)
        with open(os.path.join(model, "images.bin"), "wb") as f:
            f.write(b"\0" * 100)
    elif tool == "image_undistorter":
        state["undistorted_from"] = opt("--input_path")


def main():
    T = tempfile.mkdtemp(prefix="test_run_colmap_")
    images, out, calib = os.path.join(T, "frames"), os.path.join(T, "colmap"), os.path.join(T, "calib.json")
    os.makedirs(images)
    open(os.path.join(images, "frame_0.jpg"), "w").close()
    binary = os.path.join(T, "colmap_binary")
    open(binary, "w").close()
    run_colmap.run = fake_run

    def run_main(fx):
        with open(calib, "w") as f:
            json.dump(dict(model="fisheye", image_size=[1600, 1400], distortion_coefficients=[0, 0, 0, 0],
                           camera_matrix=[[fx, 0, 800], [0, fx, 700], [0, 0, 1]]), f)
        sys.argv = ["run_colmap.py", "--images", images, "--out", out, "--calibration", calib,
                    "--colmap", binary]
        run_colmap.main()

    print("first run:")
    run_main(700.0)
    db = os.path.join(out, "database.db")
    ok("calibration in the database", open(db).read().startswith("700"))

    print("re-run with a new calibration and an older, larger model on disk:")
    stale = os.path.join(out, "sparse", "7")
    os.makedirs(stale)
    with open(os.path.join(stale, "images.bin"), "wb") as f:
        f.write(b"\0" * 5000)
    run_main(900.0)
    ok("new calibration in the database", open(db).read().startswith("900"), f"({open(db).read()[:30]})")
    ok("older larger model not picked", state["undistorted_from"] == os.path.join(out, "sparse", "0"),
       f"({os.path.relpath(state['undistorted_from'], out)})")
    ok("older model removed", not os.path.exists(stale))

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    shutil.rmtree(T)
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
