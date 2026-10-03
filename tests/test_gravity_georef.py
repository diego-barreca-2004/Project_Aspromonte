#!/usr/bin/env python3
"""test_gravity_georef.py - validation of the gravity helpers and of the two georeferencings.

Fixture: 300 cameras along a gently curving 150 m walk on a 10 % slope, a model frame that
differs from UTM by a known similarity (scale 0.085, arbitrary rotation, offset), per-frame
gravity in camera axes with 0.5 deg noise, and GPS with 0.5 m horizontal / 1.5 m vertical
noise plus a slow altitude drift. Checks:
  - frame-name parsing and gravity-stream interpolation (gopro_gravity.py);
  - gravity_in_model recovers the model's down direction;
  - the gravity-levelled 4-DoF fit recovers scale and rotation and leaves no tilt;
  - the GPS-only Sim3 on the same data is tilted by more than a degree (the roll about a
    straight track is poorly constrained), which is the effect compare_mappers.py measures.

Run:  python3 tests/test_gravity_georef.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # scripts live in the repository root

from compare_mappers import (DOWN, angle_deg, fit_georef, gravity_in_model,  # noqa: E402
                             rotation_angle, rotation_between)
from gopro_gravity import frame_time, gravity_at  # noqa: E402


def rot(axis, deg):
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def look_rotation(forward):
    """cam_from_world for a camera looking along `forward` (world, Z up), image y down."""
    z = forward / np.linalg.norm(forward)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def build(seed=0):
    rng = np.random.default_rng(seed)
    n = 300
    s = np.linspace(0, 150, n)
    W = np.stack([559000 + 0.9 * s, 4214000 + 0.3 * s + 3.0 * np.sin(s / 25),
                  120 + 0.1 * s], 1)                          # true camera centres (UTM)
    fwd = np.gradient(W, axis=0)
    R_true = rot(np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81]), 70.0)
    c_true, t_true = 1 / 0.085, W.mean(0) + np.array([3.0, -2.0, 1.0])
    C = (R_true.T @ (W - t_true).T).T / c_true                # model centres
    Rcw = []                                                  # cam_from_model
    g_cam = []
    for i in range(n):
        Rw = look_rotation(fwd[i]) @ rot(np.array([1.0, 0, 0]), rng.normal(0, 3))
        Rcw.append(Rw @ R_true)                               # x_cam = Rw x_world
        g = Rw @ DOWN
        axis = rng.normal(size=3)
        g_cam.append(rot(axis / np.linalg.norm(axis), rng.normal(0, 0.5)) @ g)
    gps = W + np.column_stack([rng.normal(0, 0.5, n), rng.normal(0, 0.5, n),
                               rng.normal(0, 1.0, n) + 2.0 * np.sin(s / 40)])
    return C, np.array(Rcw), np.array(g_cam), gps, c_true, R_true, W


def main():
    checks = []

    def ok(name, cond, detail=""):
        checks.append((name, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    print("gopro_gravity helpers")
    ok("frame time parsed", frame_time("frame_000093_t00018.60.jpg") == 18.6)
    ok("name without time gives None", frame_time("frame_000093.jpg") is None)
    t = np.array([0.0, 1.0])
    g = np.array([[0, 1.0, 0], [0, 0, 1.0]])
    gi = gravity_at([0.5], t, g)[0]
    ok("interpolated gravity is unit length", abs(np.linalg.norm(gi) - 1) < 1e-9)
    ok("interpolated gravity is halfway", angle_deg(gi, np.array([0, 1.0, 1.0])) < 1e-6)

    print("geometry helpers")
    a, b = np.array([0.2, -0.4, 0.9]), np.array([-0.7, 0.1, 0.3])
    Rab = rotation_between(a, b)
    ok("rotation_between maps a onto b", angle_deg(Rab @ a, b) < 1e-6)
    ok("rotation_between is a rotation", abs(np.linalg.det(Rab) - 1) < 1e-9)

    print("georeferencing on a synthetic walk")
    C, Rcw, g_cam, gps, c_true, R_true, W = build()
    g_model, spread = gravity_in_model(Rcw, g_cam)
    ok("model gravity recovered", angle_deg(g_model, R_true.T @ DOWN) < 0.2,
       f"({angle_deg(g_model, R_true.T @ DOWN):.3f} deg)")
    ok("gravity spread reflects the 0.5 deg noise", 0.2 < np.median(spread) < 1.5,
       f"(median {np.median(spread):.2f} deg)")
    fits = {}
    for kind in ("gravity", "sim3"):
        c, R, t, n_in, med = fit_georef(C, gps, g_model, kind)
        fits[kind] = dict(scale_err=abs(c / c_true - 1), rot_err=rotation_angle(R @ R_true.T),
                          tilt=angle_deg(R.T @ DOWN, R_true.T @ DOWN), n_in=n_in, med=med)
        f = fits[kind]
        print(f"    {kind:8s} scale err {f['scale_err'] * 100:.2f} %  rotation err "
              f"{f['rot_err']:.2f} deg  tilt {f['tilt']:.2f} deg  inliers {n_in}  "
              f"GPS residual {med:.2f} m")
    ok("gravity fit: scale within 1 %", fits["gravity"]["scale_err"] < 0.01)
    ok("gravity fit: rotation within 0.5 deg", fits["gravity"]["rot_err"] < 0.5)
    ok("gravity fit: tilt within 0.2 deg", fits["gravity"]["tilt"] < 0.2)
    ok("GPS-only Sim3 is tilted by more than 1 deg", fits["sim3"]["tilt"] > 1.0)

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
