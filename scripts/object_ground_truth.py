#!/usr/bin/env python3
"""object_ground_truth.py - position and size of the placed objects, measured in the video frames.

The objects are gone, so their ground truth comes from two independent sources that must agree:
the sizes of the standard products they are (bucket, bike tool bottle, size-5 volleyball) and
this measurement, which uses only the frames and the SfM poses, never the dense cloud being
evaluated.

For each object, starting from a frame and a rough pixel position read off the video:
  1. follow it back through earlier frames, segmenting it by colour in a window around its
     previous position (white, dark or saturated);
  2. turn the blob centre into a ray with the fisheye model and the camera pose, and triangulate
     the object centre from all rays (least squares), in UTM through geo_transform.txt;
  3. per frame, distance d to the object and depression angle t of the line of sight:
     diameter D = horizontal angular width x d;
     height H = (vertical angular extent x d - D sin t) / cos t, since an upright cylinder seen
     from above spans H cos t plus its top ellipse, D sin t (for a sphere, H = vertical extent x d).
The median over frames is reported, with the spread.

Usage:
  python3 scripts/object_ground_truth.py --seg seg01_ep3 \
      --object bucket:white:336:704:1190 --object ball:saturated:175:448:1120:sphere \
      --object cylinder:dark:64:848:1106 --out gt.json
"""
import argparse
import json
import os
import struct
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_colmap import largest_model  # noqa: E402

CAMERA_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}


def read_cameras(model_dir):
    cams = {}
    with open(os.path.join(model_dir, "cameras.bin"), "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            cams[cid] = (mid, np.array(struct.unpack(f"<{CAMERA_NPARAMS[mid]}d",
                                                     f.read(8 * CAMERA_NPARAMS[mid]))))
    return cams


def read_images(model_dir):
    out = {}
    with open(os.path.join(model_dir, "images.bin"), "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            _, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<i7di", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
            w, x, y, z = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            out[name.decode()] = (R, np.array([tx, ty, tz]), cid)
    return out


def read_sim3(path):
    d = {l.split()[0]: l.split()[1:] for l in open(path) if l.strip() and not l.startswith("#")}
    return float(d["scale"][0]), np.array(d["R"], float).reshape(3, 3), np.array(d["t"], float)


def segment(bgr, kind):
    if kind == "white":       # neutral and clearly brighter than the trail around it (low light)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(int)
        m = ((lab[..., 0] > np.median(lab[..., 0]) + 18) & (lab[..., 2] < 136)
             & (abs(lab[..., 1] - 128) < 8))
    elif kind == "dark":
        m = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2] < 70
    elif kind == "saturated":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = (hsv[..., 1] > 120) & (hsv[..., 2] > 90)
    else:
        sys.exit(f"Unknown colour rule '{kind}'")
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_blob(img, kind, guess, half, max_half=480):
    """Blob nearest to guess; the window doubles while the blob touches its edge (clipped)."""
    x0, y0 = max(0, int(guess[0]) - half), max(0, int(guess[1]) - half)
    x1, y1 = min(img.shape[1], int(guess[0]) + half), min(img.shape[0], int(guess[1]) + half)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(segment(img[y0:y1, x0:x1], kind))
    best = None
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 25:
            continue
        dist = np.hypot(cent[k, 0] + x0 - guess[0], cent[k, 1] + y0 - guess[1])
        if best is None or dist < best[0]:
            best = (dist, k)
    if best is None:
        return None
    x, y, w, h = stats[best[1], :4]
    clipped = ((x == 0 and x0 > 0) or (y == 0 and y0 > 0)
               or (x + w == x1 - x0 and x1 < img.shape[1]) or (y + h == y1 - y0 and y1 < img.shape[0]))
    if clipped and 2 * half <= max_half:
        return find_blob(img, kind, guess, 2 * half, max_half)
    ys, xs = np.nonzero(lab == best[1])
    return xs + x0, ys + y0


def rays(px, params, R, t, sim):
    """Unit world (UTM) directions of pixels, and the camera centre in UTM."""
    K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
    n = cv2.fisheye.undistortPoints(np.asarray(px, float).reshape(-1, 1, 2), K, params[4:8]).reshape(-1, 2)
    d = (sim[1] @ R.T @ np.column_stack([n, np.ones(len(n))]).T).T
    return d / np.linalg.norm(d, axis=1, keepdims=True), sim[0] * sim[1] @ (-R.T @ t) + sim[2]


def triangulate(centres, dirs):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(centres, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def angle(u, v):
    return float(np.arccos(np.clip(u @ v, -1, 1)))


def measure(seg, model, cams, imgs, sim, label, kind, start, guess, n_frames, step, half,
            shape="cylinder", debug=None):
    frames = sorted(os.listdir(os.path.join(seg, "frames")))
    obs = []
    for i in range(start, start - n_frames * step, -step):
        name = frames[i]
        if name not in imgs:
            continue
        blob = find_blob(cv2.imread(os.path.join(seg, "frames", name)), kind, guess, half)
        if blob is None:
            print(f"  {label}: lost at {name}")
            break
        xs, ys = blob
        cx, cy = xs.mean(), ys.mean()
        guess = (cx, cy)
        if debug is not None:
            img = cv2.imread(os.path.join(seg, "frames", name))
            m = np.zeros(img.shape[:2], np.uint8)
            m[ys, xs] = 255
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (0, 0, 255), 1)
            r = max(60, int(0.7 * max(np.ptp(xs), np.ptp(ys))))
            x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
            debug.append(cv2.resize(img[y0:int(cy) + r, x0:int(cx) + r], (180, 180)))
        row = ys == np.bincount(ys - ys.min()).argmax() + ys.min()      # widest row
        col = np.abs(xs - cx) < 2
        pts = [(cx, cy), (xs[row].min(), ys[row][0]), (xs[row].max() + 1, ys[row][0]),
               (cx, ys[col].min()), (cx, ys[col].max() + 1)]
        R, t, cid = imgs[name]
        dirs, C = rays(pts, cams[cid][1], R, t, sim)
        obs.append(dict(frame=name, px=[round(cx, 1), round(cy, 1)], C=C, dirs=dirs, n_px=len(xs)))
    if len(obs) < 2:
        sys.exit(f"{label}: found in {len(obs)} frame(s), need at least 2")
    X = triangulate([o["C"] for o in obs], [o["dirs"][0] for o in obs])
    D, H = [], []
    for o in obs:
        d = np.linalg.norm(X - o["C"])
        tilt = np.arcsin(np.clip((o["C"][2] - X[2]) / d, -1, 1))
        o["dist"] = d
        D.append(angle(o["dirs"][1], o["dirs"][2]) * d)
        if shape == "sphere":
            H.append(angle(o["dirs"][3], o["dirs"][4]) * d)
        else:
            H.append((angle(o["dirs"][3], o["dirs"][4]) * d - D[-1] * np.sin(tilt)) / np.cos(tilt))
    miss = [float(np.linalg.norm(np.cross(X - o["C"], o["dirs"][0]))) for o in obs]
    return dict(E=float(X[0]), N=float(X[1]), z=float(X[2]), n_frames=len(obs),
                ray_miss_median_m=float(np.median(miss)),
                diameter_m=float(np.median(D)), diameter_spread_m=float(np.ptp(D)),
                height_m=float(np.median(H)), height_spread_m=float(np.ptp(H)),
                frames=[dict(frame=o["frame"], px=o["px"], dist_m=round(float(o["dist"]), 2),
                             n_px=o["n_px"], D_m=round(float(d_), 3), H_m=round(float(h_), 3))
                        for o, d_, h_ in zip(obs, D, H)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", required=True, help="pass folder with frames/ and colmap/")
    ap.add_argument("--object", action="append", required=True, metavar="LABEL:RULE:FRAME:X:Y[:SHAPE]",
                    help="RULE white|dark|saturated; FRAME index and rough pixel X,Y where it is seen; "
                         "SHAPE cylinder (default) or sphere")
    ap.add_argument("--frames", type=int, default=8, help="frames used per object")
    ap.add_argument("--step", type=int, default=2, help="frame step going back from FRAME")
    ap.add_argument("--window", type=int, default=120, help="search half-window [px]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", help="optional JPEG: one row of blob outlines per object, to check by eye")
    args = ap.parse_args()

    seg = os.path.abspath(os.path.expanduser(args.seg))
    model = largest_model(os.path.join(seg, "colmap", "sparse"))
    cams, imgs = read_cameras(model), read_images(model)
    sim = read_sim3(os.path.join(seg, "colmap", "geo_transform.txt"))
    res, rows = {}, []
    for spec in args.object:
        label, kind, f, x, y, *shape = spec.split(":")
        tiles = [] if args.debug else None
        res[label] = measure(seg, model, cams, imgs, sim, label, kind, int(f), (float(x), float(y)),
                             args.frames, args.step, args.window, (shape or ["cylinder"])[0], tiles)
        if tiles:
            rows.append(np.hstack(tiles + [np.zeros_like(tiles[0])] * (args.frames - len(tiles))))
        r = res[label]
        print(f"  {label}: E {r['E']:.2f} N {r['N']:.2f} z {r['z']:.2f} | D {100 * r['diameter_m']:.1f} cm "
              f"(spread {100 * r['diameter_spread_m']:.1f}) | H {100 * r['height_m']:.1f} cm "
              f"(spread {100 * r['height_spread_m']:.1f}) | {r['n_frames']} frames, "
              f"ray miss {100 * r['ray_miss_median_m']:.1f} cm")
    res["_source"] = dict(seg=seg, model=model, geo_transform=os.path.join(seg, "colmap", "geo_transform.txt"))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if rows:
        cv2.imwrite(args.debug, np.vstack(rows))


if __name__ == "__main__":
    main()
