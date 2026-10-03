#!/usr/bin/env python3
"""gopro_gravity.py - per-frame gravity direction from a GoPro video, in COLMAP camera axes.

HERO11/12/13 record a GRAV stream (exiftool tag GravityVector) in the GPMF telemetry: one unit
vector per video frame (~30 Hz), the direction of gravity in the camera frame. On the
HERO13 used here it is already expressed in COLMAP's camera convention (x right, y down,
z forward): compared against the camera rotations of the existing epoch-1 model, the
identity mapping agrees to 1.5 deg median, and every other signed axis permutation is
worse by more than 5 deg (checked on all three seg01 epochs). Check this again on another
camera model before trusting it.

This script
  1. reads GravityVector (and SampleTime / SampleDuration per one-second GPMF document),
  2. interpolates it at each extracted frame's capture time (from the frame file name, as
     written by ingest_gopro.py: frame_NNNNNN_tSSSSS.SS.jpg),
  3. writes gravity.csv (name, t_s, gx, gy, gz), and
  4. optionally (--database) writes gravity-only pose priors into a COLMAP >= 4.2 database,
     which the global mapper uses when rotation averaging runs with use_gravity.

--database modifies the database in place. Point it at a COPY (compare_mappers.py does so).

Requires: numpy; exiftool >= 12.x on PATH; pycolmap >= 4.2 for --database.

Usage:
  python3 gopro_gravity.py --video Attempt.MP4 --frames ./seg01/frames --out ./seg01/gravity.csv
  python3 gopro_gravity.py --video Attempt.MP4 --frames ./seg01/frames \
                           --out ./work/gravity.csv --database ./work/database_gravity.db
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

TS_RE = re.compile(r"_t(\d+)[._](\d+)")   # frame_000093_t00018.60.jpg -> 18.6


def frame_time(name):
    m = TS_RE.search(name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def read_gravity_stream(video):
    """(t (N,), g (N,3)) from the GPMF GravityVector stream, sorted by time."""
    if shutil.which("exiftool") is None:
        sys.exit("exiftool not found on PATH.")
    raw = subprocess.run(
        ["exiftool", "-ee", "-n", "-b", "-j", "-G3", "-api", "largefilesupport=1",
         "-SampleTime", "-SampleDuration", "-GravityVector", video],
        capture_output=True, text=True, check=True).stdout
    docs = {}
    for key, val in json.loads(raw)[0].items():
        doc, _, tag = key.partition(":")
        docs.setdefault(doc, {})[tag] = val
    ts, gs = [], []
    for tags in docs.values():
        if "GravityVector" not in tags or "SampleTime" not in tags:
            continue
        g = np.array(str(tags["GravityVector"]).split(), float).reshape(-1, 3)
        t0, dur = float(tags["SampleTime"]), float(tags.get("SampleDuration", 1.0))
        ts.append(t0 + dur * np.arange(len(g)) / len(g))   # samples evenly spread in the doc
        gs.append(g)
    if not gs:
        sys.exit(f"No GravityVector in {video} (needs a HERO11 or newer, and exiftool that "
                 "decodes the GRAV stream).")
    t, g = np.concatenate(ts), np.concatenate(gs)
    order = np.argsort(t)
    return t[order], g[order]


def gravity_at(t_query, t, g):
    """Linear interpolation of the gravity stream at t_query, renormalised to unit length."""
    t_query = np.atleast_1d(np.asarray(t_query, float))
    out = np.stack([np.interp(t_query, t, g[:, i]) for i in range(3)], axis=1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def frame_gravity(video, names):
    """{frame name: unit gravity (3,)} for every name carrying a capture time."""
    t, g = read_gravity_stream(video)
    named = [(n, frame_time(n)) for n in names]
    named = [(n, ft) for n, ft in named if ft is not None and t[0] - 0.5 <= ft <= t[-1] + 0.5]
    if not named:
        return {}
    vals = gravity_at([ft for _, ft in named], t, g)
    return {n: v for (n, _), v in zip(named, vals)}


def write_csv(path, grav, times):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "t_s", "gx", "gy", "gz"])
        for n in sorted(grav):
            w.writerow([n, f"{times[n]:.3f}"] + [f"{x:.6f}" for x in grav[n]])


def write_database_priors(database, grav):
    """Write one gravity-only PosePrior per image found in `grav`; returns the count."""
    import pycolmap
    db = pycolmap.Database.open(database)
    try:
        if db.num_pose_priors():
            sys.exit(f"{database} already has {db.num_pose_priors()} pose priors; use a fresh copy.")
        n = 0
        for image in db.read_all_images():
            g = grav.get(image.name)
            if g is None:
                continue
            prior = pycolmap.PosePrior()
            prior.corr_data_id = pycolmap.data_t(
                pycolmap.sensor_t(pycolmap.SensorType.CAMERA, image.camera_id), image.image_id)
            prior.gravity = np.asarray(g, float)
            db.write_pose_prior(prior)
            n += 1
    finally:
        db.close()
    return n


def main():
    ap = argparse.ArgumentParser(description="Per-frame GoPro gravity (COLMAP camera axes).")
    ap.add_argument("--video", required=True, help="GoPro .MP4 the frames were extracted from")
    ap.add_argument("--frames", required=True, help="frame folder from ingest_gopro.py")
    ap.add_argument("--out", required=True, help="output gravity.csv")
    ap.add_argument("--database", default=None,
                    help="COLMAP >= 4.2 database COPY to receive gravity-only pose priors")
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.frames) if n.lower().endswith((".jpg", ".png")))
    grav = frame_gravity(args.video, names)
    if not grav:
        sys.exit("No frame time overlaps the gravity stream.")
    write_csv(args.out, grav, {n: frame_time(n) for n in grav})
    print(f"Wrote {len(grav)}/{len(names)} frame gravity vectors to {args.out}")
    if args.database:
        print(f"Wrote {write_database_priors(args.database, grav)} gravity priors to {args.database}")


if __name__ == "__main__":
    main()
