#!/usr/bin/env python3
"""image_change.py - measure change between two co-aligned passes in the images, through the
geometry of the reference pass, so that a new object needs no reconstruction of its own.

measure_change.py compares heights. An object one or two DSM cells wide (the dark cylinder of
ep3, 7.4 cm) stays below the noise of the MVS at 1000 px there, while in the frames it covers
about a hundred pixels in tens of views. Label-free scene change detection renders the
reference scene into each view of the new pass and compares the images (3DGS-CD, arXiv
2411.03706; MV3DCD, CVPR 2025, arXiv 2412.03911; SCaR-3D, arXiv 2512.22830; online SCD with
multi-view fusion, arXiv 2511.12370). Here the reference scene is the MVS of the co-alignment
(coalign.py: one COLMAP frame for all passes), not a trained radiance field:

  1. each view of the compared pass (cmp) gets the depth of the reference cloud (ref
     fused.ply): z-buffer at half the depth-map size, holes filled from the nearest surface;
  2. the N_SOURCES ref views nearest to it (camera centres, optical axes within MAX_ANGLE
     degrees) are warped into it through that depth. A pixel is dropped where the ref view's
     own depth map shows a nearer surface (hidden there);
  3. change cue per pixel: the photometric error of 3DGS used by the online method,
     (1 - LAMBDA) L1 + LAMBDA (1 - SSIM), after a gain and offset per channel; the minimum over
     the ref views, so a pixel hidden or blurred in one of them is not a change;
  4. each pixel goes to the ground grid at the nearer of the two surfaces, the cmp depth map or
     the ref render: the object, where there is one (nearer in cmp for an object that appeared,
     in ref for one removed, which also gives the kind). Per 5 cm UTM cell the score is the mean,
     over the cmp views that see the cell, of each view's mean cue there: one vote per view;
  5. a-contrario windows on the score, as in measure_change.py step 7: contrast D = window mean
     minus ring mean (window sides SCALES), declustered generalised Pareto tails per window side
     and class of views (CLASSES: the window mean of log views per cell, i.e. how precise its
     mean score is), fitted on a null pair (N1) and frozen. NFA = N_tests * rate *
     P(peak excess >= D - u0); a change is stated where NFA < --epsilon * track km.

The constants of steps 1-4 come from the methods above and are not tuned on any pair. The
structure of step 5 was chosen on N1 (ep2 vs ep1) and synthetic cylinders drawn in its ep2 frames
(5-30 cm wide, 5-30 cm high, random colour; tails fitted on one half of the track, objects on the
other), never on the true objects: classes of views over classes of image texture (texture5 at
5/km stated 28 false/km, views5 7/km at the same recall) and the mean of the views over their
median or the mean over pixels (more recall for 5-7.5 cm). On the same-pass pairs ep1/ep1 and
ep2/ep2 (ref = cmp, quieter than N1) the N1 tails state no false change up to 100/km: valid, but
conservative on a quieter pair. Neither D divided by the pair's NMAD (21 false/km at 1/km on
ep1/ep1 with the N1 tails) nor classes of local background, ring mean or ring standard deviation
of the score (7 and 14 false/km at 1/km on N1, less recall), carried them better. Outputs
in --out: changes.csv (the columns of measure_change.py; dz from the two DSMs, 90th percentile
per cell, median offset removed), fields.npz, null_tails.json (this pair's own tails, the null
of other pairs when the pair is N1), change_report.txt, change_map.png.

  python3 image_change.py --ref-seg coalign/seg01_ep1_ep2_ep3/passes/ep2 \\
      --cmp-seg coalign/seg01_ep1_ep2_ep3/passes/ep3 \\
      --null-tails change_detection/v2/img/co3_n1_ep2_vs_ep1/null_tails.json \\
      --out change_detection/v2/img/co3_ep3_vs_ep2
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from functools import lru_cache

import cv2
import numpy as np
from plyfile import PlyData
from scipy import ndimage

from dsm_change import grid_stats
from georef_cloud import parse_geo_transform
from measure_change import SCALES, camera_track, fit_null_tails, read_pinhole_model, tail_log10p
from run_m3c2 import nmad

CLASSES = "quintiles of the window mean of log(views per cell)"

N_SOURCES = 3     # ref views per cmp view
MAX_ANGLE = 30.0  # degrees between the optical axes of a cmp view and its ref views
LAMBDA = 0.2      # weight of 1 - SSIM in the 3DGS photometric error
DEPTH_TOL = 0.05  # relative depth difference for one surface to be nearer than another


# --- inputs -------------------------------------------------------------------
def read_depth(path):
    """COLMAP depth map (.bin): text header "width&height&channels&", then float32 in column
    order (one plane per channel). Missing depth (0) becomes NaN."""
    with open(path, "rb") as f:
        head = b""
        while head.count(b"&") < 3:
            head += f.read(1)
        w, h, c = map(int, head[:-1].split(b"&"))
        d = np.fromfile(f, np.float32, w * h * c).reshape(c, h, w)[0]
    return np.where(d > 0, d, np.nan)


def write_depth(path, d):
    """Inverse of read_depth (NaN written as 0)."""
    h, w = d.shape
    with open(path, "wb") as f:
        f.write(f"{w}&{h}&1&".encode())
        np.nan_to_num(d, nan=0.0).astype(np.float32).tofile(f)


def load_views(seg):
    """Views of a pass's dense model sorted by name: (name, R, t, (W, H, fx, fy, cx, cy)) with
    world-to-camera poses, plus arrays of camera centres and optical axes (world frame)."""
    poses = read_pinhole_model(os.path.join(seg, "colmap", "dense", "sparse"), names=True)
    views = [(name, R, t, (w, h, *p)) for R, t, (w, h, p), name in sorted(poses, key=lambda v: v[3])]
    return views, np.array([-R.T @ t for _, R, t, _ in views]), np.array([R[2] for _, R, _, _ in views])


def scale_K(cam, shape):
    """Intrinsics (fx, fy, cx, cy) of an undistorted camera at the size of its depth map."""
    w0, h0, fx, fy, cx, cy = cam
    sx, sy = shape[1] / w0, shape[0] / h0
    return np.array([fx * sx, fy * sy, cx * sx, cy * sy])


def load_image(dense, name, shape):
    """Undistorted frame as RGB float32 in [0, 1], area-resampled to the depth-map size."""
    img = cv2.imread(os.path.join(dense, "images", name), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


# --- geometry -----------------------------------------------------------------
def pixel_rays(K, shape):
    """Camera-frame rays (x, y, 1) through the pixel centres (COLMAP: pixel i covers [i, i + 1))."""
    h, w = shape
    x, y = np.meshgrid((np.arange(w) + 0.5 - K[2]) / K[0], (np.arange(h) + 0.5 - K[3]) / K[1])
    return np.stack([x, y, np.ones_like(x)], -1)


def render_depth(Xc, K, shape, factors=(2, 4, 8)):
    """Depth of camera-frame points in a view of `shape` (h, w): z-buffers (nearest point per
    pixel) at 1/2, 1/4 and 1/8 of the size, each upsampled by repetition; a pixel takes the finest
    level that has a surface there. The near ground, where the cloud is sparser than the pixels,
    gets its depth from the coarser levels and the rest stays sharp. A surface needs support: on
    every level a pixel with fewer than 2 of its 8 neighbours filled is dropped, so the floaters of
    the MVS in the sky never render. NaN where there is no surface."""
    h, w = shape
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u0, v0 = K[0] * Xc[:, 0] / z + K[2], K[1] * Xc[:, 1] / z + K[3]
    d = np.full(shape, np.inf)
    for f in factors:
        hl, wl = -(-h // f), -(-w // f)
        u, v = np.floor(u0 / f), np.floor(v0 / f)
        m = (z > 0) & (u >= 0) & (u < wl) & (v >= 0) & (v < hl)
        key, zm = (v[m] * wl + u[m]).astype(np.int64), z[m]
        o = np.lexsort((zm, key))
        key, zm = key[o], zm[o]
        first = np.r_[True, key[1:] != key[:-1]]  # the nearest point of each pixel
        buf = np.full(hl * wl, np.inf)
        buf[key[first]] = zm[first]
        buf = buf.reshape(hl, wl)
        filled = np.isfinite(buf).astype(float)
        buf[np.rint(ndimage.uniform_filter(filled, 3, mode="constant") * 9 - filled) < 2] = np.inf
        up = np.repeat(np.repeat(buf, f, 0), f, 1)[:h, :w]
        d = np.where(np.isfinite(d), d, up)
    return np.where(np.isfinite(d), d, np.nan)


def warp(src, src_depth, K, R, t, Xw):
    """Sample a ref view (image and its own depth map, same size) at the projection of the world
    points Xw (h, w, 3). Returns the warped image and the mask of the points the view sees: in
    front, inside the image and not behind a nearer surface of its depth map."""
    Y = Xw @ R.T + t
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0] * Y[..., 0] / Y[..., 2] + K[2]
        v = K[1] * Y[..., 1] / Y[..., 2] + K[3]
    h, w = src_depth.shape
    ok = (Y[..., 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    sd = src_depth[np.where(ok, v, 0).astype(int), np.where(ok, u, 0).astype(int)]
    with np.errstate(invalid="ignore"):
        ok &= ~(sd < Y[..., 2] * (1 - DEPTH_TOL))
    mu, mv = (np.where(ok, a - 0.5, -1).astype(np.float32) for a in (u, v))
    return cv2.remap(src, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), ok


def nearest_views(c, a, C, A, n=N_SOURCES, max_angle=MAX_ANGLE):
    """Indices of the n views nearest to centre c among those whose optical axis is within
    max_angle degrees of a."""
    cand = np.nonzero(A @ a >= np.cos(np.radians(max_angle)))[0]
    return cand[np.argsort(np.linalg.norm(C[cand] - c, axis=1))[:n]]


def source_views(name, c, a):
    """The N_SOURCES ref views nearest to the cmp view name (centre c, axis a). In a pair of a
    pass with itself (a same-pass null) the view of the same name is left out: it would compare
    the frame with itself."""
    G = _G
    near = nearest_views(c, a, G["ref_C"], G["ref_A"], N_SOURCES + G["self_pair"])
    return [j for j in near if not (G["self_pair"] and G["ref"][j][0] == name)][:N_SOURCES]


# --- change cue ---------------------------------------------------------------
def ssim(a, b):
    """SSIM map of two grey images in [0, 1], Gaussian window of sigma 1.5 (Wang et al. 2004)."""
    def g(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = g(a), g(b)
    va, vb, cov = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (2 * ma * mb + c1) * (2 * cov + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


def photometric_cue(img, warped, ok):
    """(1 - LAMBDA) L1 + LAMBDA (1 - SSIM) between img and warped (RGB in [0, 1]) after a gain
    and offset per channel from the median and NMAD of the pixels seen in both (the exposure
    changes from frame to frame); NaN where not ok. Unseen pixels take the image's own values
    before the SSIM, so they do not leak into the windows of their neighbours."""
    if ok.sum() < 100:
        return np.full(ok.shape, np.nan, np.float32)
    w2 = img.copy()
    for c in range(3):
        a, b = img[..., c][ok], warped[..., c][ok]
        sb = nmad(b)
        gain = nmad(a) / sb if sb > 0 else 1.0
        w2[..., c][ok] = (b - np.median(b)) * gain + np.median(a)
    l1 = np.abs(img - w2).mean(-1)
    s = ssim(img.mean(-1), w2.mean(-1))
    return np.where(ok, (1 - LAMBDA) * l1 + LAMBDA * (1 - s), np.nan).astype(np.float32)


# --- synthetic objects (for the null pair only) --------------------------------
def render_cylinders(objs, R, t, K, shape, geo, d_scene, ss=2):
    """Upright cylinders (E, N, Z0, diameter, height, rgb) in UTM, seen from one view by ray
    casting at ss x ss samples per pixel, hidden behind nearer surfaces of d_scene. Returns their
    depth (COLMAP units, NaN where none), coverage alpha and colour (Lambert shading, light from
    above, ambient 0.5)."""
    s, Rg, tg = geo
    h, w = shape
    Cu = s * Rg @ (-R.T @ t) + tg
    M = Rg @ R.T  # camera ray (x, y, 1) -> UTM direction; a point at depth z is Cu + s z M ray
    light = np.array([0.3, 0.2, 1.0]) / np.linalg.norm([0.3, 0.2, 1.0])
    zb = np.full((h * ss, w * ss), np.inf)
    col = np.zeros((h * ss, w * ss, 3))
    for E, N, Z0, dia, H, rgb in objs:
        r = dia / 2
        box = np.array([[E + a * r, N + b * r, Z0 + c * H] for a in (-1, 1) for b in (-1, 1) for c in (0, 1)])
        Xc = ((box - tg) @ Rg / s) @ R.T + t
        if np.any(Xc[:, 2] <= 0):
            continue
        u, v = K[0] * Xc[:, 0] / Xc[:, 2] + K[2], K[1] * Xc[:, 1] / Xc[:, 2] + K[3]
        u0, u1 = (int(np.clip(x, 0, w)) for x in (np.floor(u.min()) - 1, np.ceil(u.max()) + 1))
        v0, v1 = (int(np.clip(x, 0, h)) for x in (np.floor(v.min()) - 1, np.ceil(v.max()) + 1))
        if u1 <= u0 or v1 <= v0:
            continue
        X, Y = np.meshgrid((np.arange(u0 * ss, u1 * ss) + 0.5) / ss, (np.arange(v0 * ss, v1 * ss) + 0.5) / ss)
        d = np.stack([(X - K[2]) / K[0], (Y - K[3]) / K[1], np.ones_like(X)], -1) @ M.T
        px, py = Cu[0] - E, Cu[1] - N
        qa = d[..., 0] ** 2 + d[..., 1] ** 2
        qb = 2 * (px * d[..., 0] + py * d[..., 1])
        disc = qb ** 2 - 4 * qa * (px * px + py * py - r * r)
        with np.errstate(invalid="ignore", divide="ignore"):
            ls = (-qb - np.sqrt(disc)) / (2 * qa)
            zs = Cu[2] + ls * d[..., 2]
            side = (disc >= 0) & (ls > 0) & (zs >= Z0) & (zs <= Z0 + H)
            lt = (Z0 + H - Cu[2]) / d[..., 2]
            top = (lt > 0) & ((px + lt * d[..., 0]) ** 2 + (py + lt * d[..., 1]) ** 2 <= r * r)
        lam = np.where(side, ls, np.inf)
        is_top = top & (lt < lam)
        lam = np.where(is_top, lt, lam)
        with np.errstate(invalid="ignore"):  # rays that miss: lam = inf, normal NaN, never drawn
            nrm = np.stack([(px + lam * d[..., 0]) / r, (py + lam * d[..., 1]) / r, np.zeros_like(lam)], -1)
            nrm[is_top] = (0, 0, 1)
            shade = 0.5 + 0.5 * np.clip(nrm @ light, 0, None)
        sl = (slice(v0 * ss, v1 * ss), slice(u0 * ss, u1 * ss))
        z = lam / s
        nearer = np.isfinite(z) & (z < zb[sl])
        zb[sl] = np.where(nearer, z, zb[sl])
        col[sl] = np.where(nearer[..., None], np.asarray(rgb) * shade[..., None], col[sl])
    ds = np.repeat(np.repeat(d_scene, ss, 0), ss, 1)
    with np.errstate(invalid="ignore"):
        hit = np.isfinite(zb) & ~(zb > ds * (1 + DEPTH_TOL))
    n = hit.reshape(h, ss, w, ss).sum((1, 3))
    alpha = n / ss ** 2
    colour = (col * hit[..., None]).reshape(h, ss, w, ss, 3).sum((1, 3)) / np.maximum(n, 1)[..., None]
    zpix = np.where(hit, zb, np.inf).reshape(h, ss, w, ss).min((1, 3))
    return np.where(np.isfinite(zpix), zpix, np.nan), alpha, colour


# --- one cmp view -------------------------------------------------------------
_G = {}  # read-only state shared with the worker processes (fork)


@lru_cache(maxsize=24)
def _ref_view(j):
    name = _G["ref"][j][0]
    path = os.path.join(_G["ref_dense"], "stereo", "depth_maps", name + ".geometric.bin")
    if not os.path.isfile(path):
        return None
    d = read_depth(path)
    return load_image(_G["ref_dense"], name, d.shape), d


def view_votes(i):
    """One cmp view: change cue per pixel against its ref views, carried to the grid. Returns the
    cells it sees with the view's mean cue and mean signed cue there (+ where cmp is nearer, - where
    ref is); None when no ref view sees it; False with synthetic objects and
    skip_empty when none is drawn in this view (its clean votes stand)."""
    G = _G
    name, R, t, cam = G["cmp"][i]
    path = os.path.join(G["cmp_dense"], "stereo", "depth_maps", name + ".geometric.bin")
    if not os.path.isfile(path):
        return None
    d_cmp = read_depth(path)
    shape = d_cmp.shape
    K = scale_K(cam, shape)
    s = G["geo"][0]
    c = -R.T @ t
    P = G["ref_pts"]
    near = np.einsum("ij,ij->i", P - c, P - c) <= (G["max_range"] / s) ** 2
    d_ref = render_depth(P[near] @ R.T + t, K, shape)
    img = load_image(G["cmp_dense"], name, shape)
    if G.get("objects"):
        zo, alpha, colour = render_cylinders(G["objects"], R, t, K, shape, G["geo"], np.fmin(d_cmp, d_ref))
        if not (alpha > 0).any() and G.get("skip_empty"):
            return False
        img = (alpha[..., None] * colour + (1 - alpha[..., None]) * img).astype(np.float32)
        if G["objects_in_depth"]:
            d_cmp = np.where(alpha >= 0.5, zo, d_cmp)
    rays = pixel_rays(K, shape)
    Xw = (rays * d_ref[..., None] - t) @ R
    cues = []
    for j in source_views(name, c, R[2]):
        src = _ref_view(j)
        if src is None:
            continue
        Ij, dj = src
        _, Rj, tj, camj = G["ref"][j]
        warped, ok = warp(Ij, dj, scale_K(camj, dj.shape), Rj, tj, Xw)
        cues.append(photometric_cue(img, warped, ok & np.isfinite(d_ref)))
    if not cues:
        return None
    cue = np.fmin.reduce(np.stack(cues))
    d_loc = np.fmin(d_cmp, d_ref)
    with np.errstate(invalid="ignore"):
        sgn = (d_cmp < d_ref * (1 - DEPTH_TOL)).astype(np.float32) - (d_ref < d_cmp * (1 - DEPTH_TOL))
    dist = s * d_loc * np.linalg.norm(rays, axis=-1)
    with np.errstate(invalid="ignore"):
        m = np.isfinite(cue) & (dist <= G["max_range"])
    _, Rg, tg = G["geo"]
    utm = s * ((rays[m] * d_loc[m, None] - t) @ R) @ Rg.T + tg
    ix = np.floor((utm[:, 0] - G["origin"][0]) / G["cell"]).astype(np.int64)
    iy = np.floor((utm[:, 1] - G["origin"][1]) / G["cell"]).astype(np.int64)
    nx, ny = G["shape"]
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cells, inv = np.unique(ix[inside] * ny + iy[inside], return_inverse=True)
    n = np.bincount(inv)
    cm, sm = (np.bincount(inv, a[m][inside]) / n for a in (cue, sgn * cue))
    return cells, cm.astype(np.float32), sm.astype(np.float32)


# --- the pair -----------------------------------------------------------------
def prepare(ref_seg, cmp_seg, max_range=6.0, cell=0.05, track=None):
    """Shared state of a pair in _G: views, ref cloud (COLMAP frame), common georeference, grid
    around the track (default: the ref camera centres) out to max_range."""
    geo = parse_geo_transform(os.path.join(cmp_seg, "colmap", "geo_transform.txt"))[:3]
    geo_r = parse_geo_transform(os.path.join(ref_seg, "colmap", "geo_transform.txt"))[:3]
    if not all(np.allclose(a, b) for a, b in zip(geo, geo_r)):
        sys.exit("the passes have different geo_transform.txt: co-align them first (coalign.py)")
    ref, ref_C, ref_A = load_views(ref_seg)
    cmp_, _, _ = load_views(cmp_seg)
    v = PlyData.read(os.path.join(ref_seg, "colmap", "dense", "fused.ply"))["vertex"]
    track = camera_track(ref_seg) if track is None else track
    origin = track.min(axis=0) - max_range
    shape = tuple((np.ceil((track.max(axis=0) - track.min(axis=0) + 2 * max_range) / cell) + 1).astype(int))
    _G.clear()
    _G.update(ref=ref, ref_C=ref_C, ref_A=ref_A, cmp=cmp_, geo=geo, max_range=max_range, cell=cell,
              self_pair=os.path.realpath(ref_seg) == os.path.realpath(cmp_seg),
              ref_pts=np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32),
              ref_dense=os.path.join(ref_seg, "colmap", "dense"),
              cmp_dense=os.path.join(cmp_seg, "colmap", "dense"),
              origin=origin, shape=shape, track=track,
              km=float(np.hypot(*np.diff(track, axis=0).T).sum()) / 1000)
    _ref_view.cache_clear()


def votes(workers, objects=None, base=None, in_depth=True):
    """view_votes over all cmp views, in parallel. With objects, views where none is drawn keep
    their votes from base (a clean run); in_depth False leaves them out of the cmp depth maps (the
    MVS missed them: their pixels are placed on the ref surface behind them)."""
    _G.update(objects=objects, skip_empty=base is not None, objects_in_depth=in_depth)
    with mp.get_context("fork").Pool(workers) as pool:
        out = pool.map(view_votes, range(len(_G["cmp"])), chunksize=8)
    return [b if o is False else o for o, b in zip(out, base or [None] * len(out))]


def fields(contribs):
    """Grid fields from the per-view votes: score (mean over views of the view's mean cue),
    views per cell and mean signed cue."""
    size = int(np.prod(_G["shape"]))
    acc = np.zeros((3, size))
    for v in contribs:
        if v is None:
            continue
        cells, cm, sm = v
        acc[0, cells] += 1
        acc[1, cells] += cm
        acc[2, cells] += sm
    n = acc[0].reshape(_G["shape"])
    with np.errstate(invalid="ignore", divide="ignore"):
        score, signed = (a.reshape(_G["shape"]) / n for a in acc[1:])
    return dict(score=score, views=n, signed=signed, valid=n >= 2)


def window_stats(f):
    """Per window side k: contrast D (window mean minus ring mean of the score, ring out to
    3k x 3k), windows with at least half of window and ring seen, window mean of log views per
    cell (the class of its null tail: a mean over 2 views is far noisier than one over 20)."""
    valid = f["valid"]
    sf, vf = np.where(valid, f["score"], 0.0), valid.astype(float)
    lt = np.where(valid, np.log(np.maximum(f["views"], 1)), 0.0)
    out = []
    for k in SCALES:
        S, C = (ndimage.uniform_filter(a, k, mode="constant") * k * k for a in (sf, vf))
        S3, C3 = (ndimage.uniform_filter(a, 3 * k, mode="constant") * 9 * k * k for a in (sf, vf))
        ok = valid & (C >= 0.5 * k * k) & (C3 - C >= 0.5 * 8 * k * k)
        with np.errstate(invalid="ignore", divide="ignore"):
            D = np.where(ok, S / C - (S3 - S) / (C3 - C), 0.0)
            wt = np.where(ok, ndimage.uniform_filter(lt, k, mode="constant") * k * k / C, 0.0)
        out.append((k, D, ok, wt))
    return out


def nfa_map(windows, tails):
    """Minimum log10 NFA per cell (every cell of a window carries the window's NFA) and the window
    side that gives it; N_tests counts the windows of all sides (one sign: a change raises the
    score whatever its kind)."""
    n_tests = sum(int(ok.sum()) for _, _, ok, _ in windows)
    best = np.full(windows[0][1].shape, np.inf)
    best_k = np.zeros(best.shape)
    for k, D, ok, wt in windows:
        lnfa = np.where(ok, np.log10(n_tests) + tail_log10p(D, ok, wt, tails["scales"][str(k)], "appeared"), np.inf)
        lmin = -ndimage.maximum_filter(-lnfa, k, mode="constant", cval=-np.inf)
        best_k = np.where(lmin < best, k, best_k)
        best = np.minimum(best, lmin)
    return best, best_k, n_tests


def main(argv=None):
    ap = argparse.ArgumentParser(description="Change measured in the images of two co-aligned passes.")
    ap.add_argument("--ref-seg", required=True, help="reference pass folder (older), e.g. coalign/.../passes/ep2")
    ap.add_argument("--cmp-seg", required=True, help="compared pass folder (newer)")
    ap.add_argument("--epsilon", type=float, default=1.0, help="expected false change statements per km of track")
    ap.add_argument("--max-range", type=float, default=6.0, help="camera-to-point distance kept [m]")
    ap.add_argument("--cell", type=float, default=0.05, help="grid cell [m]")
    ap.add_argument("--null-tails", help="null_tails.json of a run on a null pair (N1); without it, "
                    "the pair's own tails (inflated by its own changes: conservative)")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    report = []

    def log(msg=""):
        print(msg, flush=True)
        report.append(msg)

    log(f"# image_change.py  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("# cmd: " + " ".join(sys.argv if argv is None else ["image_change.py"] + list(argv)))
    prepare(args.ref_seg, args.cmp_seg, args.max_range, args.cell)
    G = _G
    log(f"\n=== pair ===\n  ref {args.ref_seg}: {len(G['ref'])} views, {len(G['ref_pts']):,} points | "
        f"cmp {args.cmp_seg}: {len(G['cmp'])} views | track {G['km'] * 1000:.1f} m | grid "
        f"{G['shape'][0]} x {G['shape'][1]} cells of {args.cell} m")
    contribs = votes(args.workers)
    f = fields(contribs)
    valid = f["valid"]
    log(f"  views with ref views: {sum(c is not None for c in contribs)}/{len(contribs)} | cells seen by >= 2 "
        f"views: {valid.sum():,} | score P50/P90/P99: "
        + "/".join(f"{q:.3f}" for q in np.percentile(f["score"][valid], [50, 90, 99]))
        + f" | {time.time() - t0:.0f} s")

    windows = window_stats(f)
    own = fit_null_tails(windows)
    own.update(track_km=G["km"], source=os.path.abspath(args.out), classes=CLASSES)
    json.dump(own, open(os.path.join(args.out, "null_tails.json"), "w"), indent=1)
    tails = json.load(open(args.null_tails)) if args.null_tails else own
    if tails.get("classes") != CLASSES:
        sys.exit(f"{args.null_tails}: tails in classes '{tails.get('classes')}', this version uses "
                 f"'{CLASSES}': run image_change.py on the null pair again")
    best, best_k, n_tests = nfa_map(windows, tails)
    thr = np.log10(args.epsilon * G["km"])
    log(f"\n=== a-contrario test ===\n  {n_tests:,} tests ({len(SCALES)} window sides "
        f"{SCALES[0] * args.cell:g}-{SCALES[-1] * args.cell:g} m); tails of "
        + (f"{tails['source']} ({tails['track_km']:.3f} km)" if args.null_tails else "this pair (no --null-tails)")
        + f"; statement when NFA < {args.epsilon:g}/km x {G['km']:.3f} km")

    # heights from the two DSMs, for the size of what is found
    h = {}
    for name, seg in (("ref", args.ref_seg), ("cmp", args.cmp_seg)):
        v = PlyData.read(os.path.join(seg, "colmap", "dense", "fused_utm.ply"))["vertex"]
        h[name] = grid_stats(np.column_stack([v["x"], v["y"], v["z"]]), G["origin"], args.cell, G["shape"], 90.0, 3)[0]
    dz = h["cmp"] - h["ref"]
    both = valid & np.isfinite(dz)
    dz -= np.median(dz[both])
    gx, gy = np.meshgrid(G["origin"][0] + (np.arange(G["shape"][0]) + 0.5) * args.cell,
                         G["origin"][1] + (np.arange(G["shape"][1]) + 0.5) * args.cell, indexing="ij")
    lab, n = ndimage.label(best < thr, structure=np.ones((3, 3)))
    base = float(np.median(f["score"][valid]))
    rows = []
    for c in range(1, n + 1):
        comp = (lab == c) & valid
        if not comp.any():
            continue
        wgt = np.clip(f["score"][comp] - base, 0, None)
        wgt = wgt if wgt.sum() > 0 else np.ones(comp.sum())
        sign = 1 if np.nansum(f["signed"][comp]) >= 0 else -1
        zz = dz[comp & both]
        lmin = float(best[comp].min())
        rows.append(dict(kind="appeared" if sign > 0 else "removed",
                         E=float(np.average(gx[comp], weights=wgt)), N=float(np.average(gy[comp], weights=wgt)),
                         area_m2=int(comp.sum()) * args.cell ** 2,
                         dz_median=float(np.median(zz)) if len(zz) else float("nan"),
                         dz_peak=float(zz[np.argmax(sign * zz)]) if len(zz) else float("nan"),
                         n_cells=int(comp.sum()), log10_nfa=lmin,
                         scale_m=float(best_k[comp][np.argmin(best[comp])] * args.cell), score=-lmin))
    rows.sort(key=lambda r: -r["score"])
    log(f"\n=== changes ({len(rows)}) ===")
    log(f"  {'#':>3} {'kind':>8} {'E':>12} {'N':>13} {'area m2':>8} {'dz med':>8} {'dz peak':>8} {'log10 NFA':>9}")
    for i, r in enumerate(rows):
        log(f"  {i + 1:>3} {r['kind']:>8} {r['E']:12.2f} {r['N']:13.2f} {r['area_m2']:8.3f} "
            f"{r['dz_median']:+8.3f} {r['dz_peak']:+8.3f} {r['log10_nfa']:9.2f}")

    fields_out = ["kind", "E", "N", "area_m2", "dz_median", "dz_peak", "n_cells", "log10_nfa", "scale_m", "score"]
    with open(os.path.join(args.out, "changes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields_out)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(args.out, "fields.npz"), origin=G["origin"], cell=args.cell,
                        km=G["km"], **{k: np.asarray(v, np.float32) for k, v in f.items()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = (G["origin"][0], G["origin"][0] + G["shape"][0] * args.cell,
           G["origin"][1], G["origin"][1] + G["shape"][1] * args.cell)
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(np.where(valid, f["score"], np.nan).T, origin="lower", extent=ext, cmap="magma",
                   vmin=0, vmax=float(np.percentile(f["score"][valid], 99.5)))
    for i, r in enumerate(rows):
        ax.plot(r["E"], r["N"], "o", mfc="none", mec="c", ms=10)
        ax.annotate(str(i + 1), (r["E"], r["N"]), fontsize=7, color="c")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="image change score (mean cue over views)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "change_map.png"), dpi=150)
    log(f"\n  done in {time.time() - t0:.0f} s -> {args.out}")
    open(os.path.join(args.out, "change_report.txt"), "w").write("\n".join(report) + "\n")
    return rows


if __name__ == "__main__":
    main()
