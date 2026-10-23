#!/usr/bin/env python3
"""test_coalign.py - validation of the steps that turn the joint model into per-pass folders.

Fixture: two passes of 120 frames each along a curving 120 m walk, cameras in a model frame that
differs from UTM by a known similarity (scale 0.085, arbitrary rotation, offset). Each pass has
its own GPS log, offset by +0.8 m and -0.8 m east (two days, two GPS biases) with 0.3 m noise.
Checks:
  - georeference fits one Sim3 to both passes and recovers the scale and the rotation; the GPS
    biases of the two days average out instead of moving one pass against the other;
  - strip_prefix drops the pass label from the image names and leaves poses and 2D points
    byte-identical;
  - camera_track (measure_change.py) returns the pass's camera centres in UTM, in frame order,
    without a misregistered camera placed 60 m off the walk.

Run:  python3 tests/test_coalign.py
"""
import os
import shutil
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # scripts live in the repository root

from coalign import georeference, read_image_names, strip_prefix  # noqa: E402
from geo_align import qvec2rotmat  # noqa: E402
from measure_change import camera_track  # noqa: E402

checks = []


def ok(name, cond, extra=""):
    checks.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")


def rot(axis, deg):
    a = np.radians(deg)
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def write_images_bin(path, images):
    """images: list of (name, qvec, tvec, points2D array (n, 3) of x, y, point3D_id)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for i, (name, q, t, pts) in enumerate(images, 1):
            f.write(struct.pack("<i4d3di", i, *q, *t, 1))
            f.write(name.encode() + b"\0")
            f.write(struct.pack("<Q", len(pts)))
            for x, y, pid in pts:
                f.write(struct.pack("<ddq", x, y, int(pid)))


def write_cameras_bin(path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ4d", 1, 1, 1600, 900, 800.0, 800.0, 800.0, 450.0))  # PINHOLE


def rotmat2qvec(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


def main():
    from pyproj import Transformer
    rng = np.random.default_rng(3)
    T = tempfile.mkdtemp(prefix="test_coalign_")

    # walk in UTM (EPSG:32633), and the model frame: X_utm = s R X_model + t
    s_true, R_true = 1 / 0.085, rot([0.3, -0.5, 0.8], 71)
    t_true = np.array([558990.0, 4214480.0, 120.0])
    u = np.linspace(0, 1, 120)
    walk = np.column_stack([558950 + 120 * u, 4214470 + 15 * np.sin(3 * u), 115 + 12 * u])
    to_ll = Transformer.from_crs(32633, 4326, always_xy=True).transform

    model, segs, images = os.path.join(T, "model"), [], []
    os.makedirs(model)
    write_cameras_bin(os.path.join(model, "cameras.bin"))
    for label, bias in (("ep2", 0.8), ("ep3", -0.8)):
        seg = os.path.join(T, label)
        os.makedirs(seg)
        centres = walk + rng.normal(0, 0.05, walk.shape)       # the two walks differ a little
        times = np.arange(len(walk)) * 0.4
        with open(os.path.join(seg, "gps.csv"), "w") as f:
            f.write("t_s,lat,lon,alt\n")
            for tt, c in zip(times, centres):
                g = c + np.array([bias, 0, 0]) + rng.normal(0, 0.3, 3)
                lon, lat = to_ll(g[0], g[1])
                f.write(f"{tt:.2f},{lat:.9f},{lon:.9f},{g[2]:.3f}\n")
        for k, (tt, c) in enumerate(zip(times, centres)):
            Rc = rot(rng.normal(size=3), rng.uniform(0, 180))   # world-to-camera in the model frame
            C_model = R_true.T @ (c - t_true) / s_true
            pts = np.column_stack([rng.uniform(0, 1600, 3), rng.uniform(0, 900, 3), rng.integers(-1, 9, 3)])
            images.append((f"{label}/frame_{k:06d}_t{int(tt):05d}.{round(100 * (tt % 1)):02d}.jpg",
                           rotmat2qvec(Rc), -Rc @ C_model, pts))
        segs.append((label, seg))
    write_images_bin(os.path.join(model, "images.bin"), images)

    print("georeference (one Sim3 for both passes):")
    geo = os.path.join(T, "geo_transform.txt")
    line = georeference(model, segs, 32633, geo)
    print("   ", line)
    d = {l.split()[0]: l.split()[1:] for l in open(geo) if l.strip() and not l.startswith("#")}
    s, R, t = float(d["scale"][0]), np.array(d["R"], float).reshape(3, 3), np.array(d["t"], float)
    ok("scale within 0.5 %", abs(s / s_true - 1) < 5e-3, f"({s:.4f} vs {s_true:.4f})")
    ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    ok("rotation within 1 deg", ang < 1.0, f"({ang:.2f} deg)")
    mapped = {lab: np.array([s * R @ (-qvec2rotmat(q).T @ tv) + t for n, q, tv, _ in images
                             if n.startswith(lab + "/")]) for lab, _ in segs}
    true = {lab: np.array([R_true @ (-qvec2rotmat(q).T @ tv) * s_true + t_true for n, q, tv, _ in images
                           if n.startswith(lab + "/")]) for lab, _ in segs}
    e2 = np.median(mapped["ep2"][:, 0] - true["ep2"][:, 0])
    e3 = np.median(mapped["ep3"][:, 0] - true["ep3"][:, 0])
    ok("same offset for both passes (GPS biases do not split them)", abs(e2 - e3) < 0.02,
       f"(ep2 {e2:+.3f} m, ep3 {e3:+.3f} m)")
    ok("absolute east offset is the mean bias, under 0.2 m", abs(e2) < 0.2)

    print("strip_prefix:")
    sub = os.path.join(T, "ep3_model")
    os.makedirs(sub)
    ep3 = [im for im in images if im[0].startswith("ep3/")]
    write_images_bin(os.path.join(sub, "images.bin"), ep3)
    before = open(os.path.join(sub, "images.bin"), "rb").read()
    strip_prefix(sub, "ep3/")
    after = open(os.path.join(sub, "images.bin"), "rb").read()
    names = read_image_names(sub)
    ok("names without the pass label", all(not n.startswith("ep3/") for n in names.values())
       and names[1] == ep3[0][0][4:], f"({names[1]})")
    ok("file shorter by exactly the dropped prefixes", len(before) - len(after) == 4 * len(ep3))
    ok("poses and 2D points unchanged", before.replace(b"ep3/", b"") == after)

    print("camera_track:")
    seg = os.path.join(T, "pass")
    dense = os.path.join(seg, "colmap", "dense", "sparse")
    os.makedirs(dense)
    shuffled = [ep3[i] for i in rng.permutation(len(ep3))]
    q, tv = ep3[50][1], ep3[50][2]                       # a misregistered camera, 60 m off the walk
    shuffled.append(("ep3/frame_000050_t00020.10.jpg", q, tv + qvec2rotmat(q) @ np.array([60 / s_true, 0, 0]),
                     ep3[50][3]))
    write_images_bin(os.path.join(dense, "images.bin"), [(n[4:], q, tv, p) for n, q, tv, p in shuffled])
    shutil.copyfile(geo, os.path.join(seg, "colmap", "geo_transform.txt"))
    track = camera_track(seg)
    ok("one vertex per camera, in frame order, misregistered camera dropped", len(track) == len(ep3)
       and np.allclose(track, mapped["ep3"][:, :2]), f"({len(track)} of {len(shuffled)})")

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    shutil.rmtree(T)
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
