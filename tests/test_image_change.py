#!/usr/bin/env python3
"""test_image_change.py - validation of the image geometry and change cue of image_change.py.

Synthetic views, no data on disk except a temporary depth map. Checks:
  - COLMAP depth maps: write_depth / read_depth round trip, 0 read as NaN;
  - render_depth: a dense plane at 2 m gives depth 2 over its whole footprint, and a plane behind
    it does not show through (z-buffer keeps the nearest point);
  - warp: a view warped into itself through the true depth returns the same image, and a nearer
    surface in the source depth map hides every pixel;
  - photometric_cue: a gain and offset between the images (exposure) leave the cue near 0, while a
    dark patch in one image stands out by more than 10 times the rest;
  - render_cylinders: an upright cylinder seen from above covers the disc of its radius at the
    depth of its top, seen from the side its height and width in pixels follow f H / distance,
    and a nearer surface hides it;
  - fields / window_stats: the score is the mean over the views of their mean cue, and the class
    feature of a window is the mean log of the views per cell (2 views where 2 see it, 20 where 20).

Run:  python3 tests/test_image_change.py
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # scripts live in the repository root

import image_change  # noqa: E402
from image_change import (fields, photometric_cue, pixel_rays, read_depth, render_cylinders,  # noqa: E402
                          render_depth, warp, window_stats, write_depth)

checks = []
K = np.array([280.0, 280.0, 500.0, 380.0])
SHAPE = (760, 1000)
GEO = (1.0, np.eye(3), np.zeros(3))  # UTM = COLMAP frame


def ok(name, cond, extra=""):
    checks.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")


def plane(z, half=(2.0, 1.5), step=0.002):
    x, y = np.meshgrid(np.arange(-half[0], half[0], step), np.arange(-half[1], half[1], step))
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def look(C, R):
    """World-to-camera pose of a camera at C with rotation R (rows: camera axes in the world)."""
    return R, -R @ C


def main():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.1, 3, SHAPE).astype(np.float32)
    d[::7, ::5] = np.nan
    with tempfile.TemporaryDirectory() as tmp:
        write_depth(os.path.join(tmp, "a.bin"), d)
        back = read_depth(os.path.join(tmp, "a.bin"))
    ok("depth map round trip, missing depth as NaN",
       back.shape == SHAPE and np.array_equal(np.isnan(back), np.isnan(d)) and np.allclose(back[~np.isnan(d)], d[~np.isnan(d)]))

    pts = np.vstack([plane(2.0), plane(3.0, (3.0, 2.2))])
    dr = render_depth(pts, K, SHAPE)
    foot = np.zeros(SHAPE, bool)
    foot[380 - 205:380 + 205, 500 - 275:500 + 275] = True  # inside the 2 m plane's footprint (210 x 280 px)
    ok("render_depth: plane at 2 m everywhere on its footprint, the 3 m plane hidden",
       np.all(np.abs(dr[foot] - 2.0) < 1e-6), f"(max |d - 2| {np.nanmax(np.abs(dr[foot] - 2.0)):.2e})")
    floaters = np.array([[0.0, -1.9, 1.0], [0.3, -1.9, 1.0]])  # isolated points, 84 px apart
    dr = render_depth(np.vstack([plane(2.0, (0.5, 0.5)), floaters]), K, SHAPE)
    ok("render_depth: isolated points (MVS floaters) are not rendered, the plane is",
       np.isnan(dr[:180]).all() and np.isfinite(dr[380 - 60:380 + 60, 500 - 60:500 + 60]).all())

    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    img = np.stack([0.5 + 0.4 * np.sin(xx / 9.0) * np.cos(yy / 7.0), 0.5 + 0.3 * np.cos(xx / 13.0),
                    0.4 + 0.3 * np.sin(yy / 11.0)], -1).astype(np.float32)
    R, t = np.eye(3), np.zeros(3)
    Xw = pixel_rays(K, SHAPE) * 2.0
    depth = np.full(SHAPE, 2.0, np.float32)
    wimg, seen = warp(img, depth, K, R, t, Xw)
    inner = np.zeros(SHAPE, bool)
    inner[2:-2, 2:-2] = True
    ok("warp of a view into itself returns the same image",
       seen[inner].all() and np.abs(wimg - img)[inner].max() < 1e-3, f"(max diff {np.abs(wimg - img)[inner].max():.1e})")
    _, seen = warp(img, np.full(SHAPE, 1.0, np.float32), K, R, t, Xw)
    ok("warp: a nearer surface in the source depth map hides every pixel", not seen.any())

    allok = np.ones(SHAPE, bool)
    cue = photometric_cue(img, 0.5 * img + 0.2, allok)
    ok("photometric_cue: gain and offset between the images leave the cue near 0", np.median(cue) < 0.01,
       f"(median {np.median(cue):.4f})")
    dark = img.copy()
    dark[300:320, 400:410] = 0.05
    cue = photometric_cue(dark, img, allok)
    rest = np.median(cue[np.abs(yy - 310) + np.abs(xx - 405) > 60])
    ok("photometric_cue: a dark 10 x 20 px patch stands out by > 10 times the rest",
       np.median(cue[302:318, 402:408]) > 10 * max(rest, 1e-4), f"(patch {np.median(cue[302:318, 402:408]):.3f}, rest {rest:.4f})")

    none = np.full(SHAPE, np.nan)
    cyl = [(0.0, 0.0, 0.0, 0.2, 0.5, (0.1, 0.1, 0.1))]
    R, t = look(np.array([0.0, 0.0, 2.0]), np.diag([1.0, -1.0, -1.0]))  # looking straight down
    z, alpha, colour = render_cylinders(cyl, R, t, K, SHAPE, GEO, none)
    area = np.pi * (280 * 0.1 / 1.5) ** 2
    ok("cylinder from above: disc of f r / 1.5 m, depth of its top",
       abs(alpha.sum() / area - 1) < 0.03 and abs(z[380, 500] - 1.5) < 1e-9 and colour[380, 500].max() < 0.11,
       f"(area {alpha.sum():.0f} px vs {area:.0f}, depth {z[380, 500]:.3f})")
    R, t = look(np.array([3.0, 0.0, 0.25]), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]))
    z, alpha, _ = render_cylinders(cyl, R, t, K, SHAPE, GEO, none)
    rows, cols = np.nonzero(alpha >= 0.5)
    hgt, wid = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    ok("cylinder from the side: height and width in pixels follow f H / distance",
       abs(hgt - 280 * 0.5 / 2.9) <= 2 and abs(wid - 280 * 0.2 / 3.0) <= 2,
       f"(height {hgt} px vs {280 * 0.5 / 2.9:.1f}, width {wid} px vs {280 * 0.2 / 3.0:.1f})")
    _, alpha, _ = render_cylinders(cyl, R, t, K, SHAPE, GEO, np.full(SHAPE, 1.0))
    ok("cylinder behind a nearer surface is hidden", alpha.sum() == 0)

    image_change._G["shape"] = (40, 40)
    every, few = np.arange(1600), np.arange(1600).reshape(40, 40)[10:20, 10:20].ravel()
    votes = [(every[~np.isin(every, few)], np.full(1500, 0.1 + 0.01 * (i % 2), np.float32), np.zeros(1500, np.float32))
             for i in range(18)] + [(every, np.full(1600, 0.1 + 0.01 * (i % 2), np.float32), np.zeros(1600, np.float32))
                                    for i in range(2)] + [None]
    f = fields(votes)
    k, _, _, fv = window_stats(f)[0]
    ok("fields: score is the mean over the views; window class is the mean log views per cell",
       np.allclose(f["score"], 0.105) and f["views"][15, 15] == 2 and f["views"][0, 0] == 20
       and abs(fv[15, 15] - np.log(2)) < 1e-9 and abs(fv[30, 30] - np.log(20)) < 1e-9 and k == 2,
       f"(class feature {fv[15, 15]:.3f} vs log 2 = {np.log(2):.3f}, {fv[30, 30]:.3f} vs log 20 = {np.log(20):.3f})")

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
