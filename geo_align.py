#!/usr/bin/env python3
"""geo_align.py - Stage 4: georeference a COLMAP/3DGS reconstruction from GoPro GPS.

Computes the similarity transform (scale + rotation + translation) that maps the
reconstruction into real-world coordinates, by matching each frame's COLMAP camera centre
to its GPS position from the gps.csv produced by ingest_gopro.py.

Why not `colmap model_aligner`? Its robust estimator is unstable on near-linear walking
trajectories (it returned ~68 m residuals on data where a sub-metre fit exists). This does
the alignment directly with a robust Umeyama fit (iterative outlier rejection), which is
both stable here and verifiable (it reports the residual and how many frames it kept).

Steps:
  1. read each frame's capture time from its filename and interpolate lat/lon/alt from gps.csv,
  2. project to a metric CRS (default UTM 33N / EPSG:32633, correct for Calabria),
  3. read the COLMAP camera centres from <model>/images.bin,
  4. robust Umeyama fit (SfM -> world), rejecting outlier frames,
  5. write the transform to <out> (scale, rotation, translation) and report the residual.

GoPro GPS gives metre-level ABSOLUTE accuracy; the residual reported here measures how well
the SfM and GPS tracks agree in shape. For sub-metre absolute registration, reproject a LiDAR
DTM to the same CRS (dtm_merge_reproject.py) and refine with ICP (CloudCompare).

Requires: numpy, pyproj   (pip install numpy pyproj)

Usage:
  python3 geo_align.py --gps ./seg01/gps.csv \
                       --images ./seg01/colmap/undistorted/images \
                       --model  ./seg01/colmap/undistorted/sparse/0 \
                       --out    ./seg01/colmap/geo_transform.txt
"""
import argparse
import csv
import os
import re
import struct
import sys

TS_RE = re.compile(r"_t(\d+)[._](\d+)")   # frame_000093_t00018_62.jpg -> 18.62


def frame_time(name):
    m = TS_RE.search(name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def load_gps(path):
    import numpy as np
    t, lat, lon, alt = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t.append(float(row["t_s"])); lat.append(float(row["lat"]))
                lon.append(float(row["lon"])); alt.append(float(row.get("alt") or 0.0))
            except (KeyError, ValueError):
                continue
    if not t:
        sys.exit("No usable rows in gps.csv (need columns t_s,lat,lon,alt).")
    o = np.argsort(t)
    return np.array(t)[o], np.array(lat)[o], np.array(lon)[o], np.array(alt)[o]


def qvec2rotmat(q):
    import numpy as np
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def read_camera_centres(model_dir):
    """Camera centres C = -R^T t per image name, parsed from COLMAP images.bin."""
    import numpy as np
    path = os.path.join(model_dir, "images.bin")
    if not os.path.isfile(path):
        sys.exit(f"images.bin not found in {model_dir} (point --model at a COLMAP sparse model).")
    centres = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)                                  # image_id
            q = struct.unpack("<4d", f.read(32))
            t = np.array(struct.unpack("<3d", f.read(24)))
            f.read(4)                                  # camera_id
            name = b""
            while (ch := f.read(1)) != b"\x00":
                name += ch
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(npts * 24)                          # skip 2D point records
            centres[name.decode()] = -qvec2rotmat(q).T @ t
    return centres


def umeyama(X, Y):
    """Similarity (scale c, rotation R, translation t) mapping X -> Y, least squares."""
    import numpy as np
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd((Yc.T @ Xc) / len(X))
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    c = np.trace(np.diag(D) @ S) / ((Xc ** 2).sum() / len(X))
    return c, R, my - c * R @ mx


def robust_umeyama(S, Y, iters=6, k=3.0, min_keep=10):
    """Umeyama with iterative rejection of frames whose residual exceeds k*median."""
    import numpy as np
    idx = np.arange(len(S))
    c, R, t = umeyama(S, Y)
    for _ in range(iters):
        res = np.linalg.norm((c * (R @ S.T).T + t) - Y, axis=1)
        keep = np.where(res <= np.median(res[idx]) * k + 1e-6)[0]
        if len(keep) < min_keep or len(keep) == len(idx):
            break
        idx = keep
        c, R, t = umeyama(S[idx], Y[idx])
    return c, R, t, idx


def main():
    ap = argparse.ArgumentParser(description="Georeference a COLMAP model from GoPro GPS.")
    ap.add_argument("--gps", required=True, help="gps.csv from ingest_gopro.py")
    ap.add_argument("--images", required=True, help="COLMAP image folder (names match the model).")
    ap.add_argument("--model", required=True, help="COLMAP sparse model folder (with images.bin).")
    ap.add_argument("--out", default="./geo_transform.txt")
    ap.add_argument("--epsg", type=int, default=32633, help="Target metric CRS (default UTM 33N).")
    args = ap.parse_args()

    import numpy as np
    from pyproj import Transformer

    if not os.path.isfile(args.gps):
        sys.exit(f"gps.csv not found: {args.gps}")
    if not os.path.isdir(args.images):
        sys.exit(f"Image folder not found: {args.images}")

    # 1-2) per-frame world coordinates: interpolate GPS at each frame time, project to the CRS.
    t_s, lat, lon, alt = load_gps(args.gps)
    t0, t1 = float(t_s[0]), float(t_s[-1])
    project = Transformer.from_crs(4326, args.epsg, always_xy=True).transform
    world = {}
    for name in os.listdir(args.images):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        ft = frame_time(name)
        if ft is None or ft < t0 - 0.5 or ft > t1 + 0.5:
            continue
        e, n = project(float(np.interp(ft, t_s, lon)), float(np.interp(ft, t_s, lat)))
        world[name] = np.array([e, n, float(np.interp(ft, t_s, alt))])

    # 3) SfM camera centres from the COLMAP model.
    centres = read_camera_centres(args.model)

    common = sorted(set(world) & set(centres))
    if len(common) < 10:
        sys.exit(f"Only {len(common)} frames matched between GPS and the model - cannot align.")
    S = np.array([centres[n] for n in common])
    W = np.array([world[n] for n in common])

    # 4) robust fit in a locally-centred frame (raw UTM magnitudes would wreck the SVD).
    origin = np.floor(W.min(0))
    c, R, t, idx = robust_umeyama(S, W - origin)
    t = t + origin                                     # back to true CRS coordinates

    res = np.linalg.norm((c * (R @ S.T).T + t) - W, axis=1)[idx]
    print(f"Aligned {len(common)} frames; kept {len(idx)} inliers "
          f"(rejected {len(common) - len(idx)}).")
    print(f"Residual (inliers): mean {res.mean():.2f}  median {np.median(res):.2f}  "
          f"max {res.max():.2f} m   |   scale {c:.5f}")

    # 5) write the transform: X_world = scale * R @ X_sfm + t
    with open(args.out, "w") as f:
        f.write(f"# Sim3  X_world = scale * R @ X_sfm + t   (EPSG:{args.epsg})\n")
        f.write(f"# fit: {len(idx)}/{len(common)} inliers, residual median {np.median(res):.3f} m\n")
        f.write(f"epsg {args.epsg}\n")
        f.write(f"scale {c:.10g}\n")
        f.write("R " + " ".join(f"{v:.10g}" for v in R.ravel()) + "\n")
        f.write("t " + " ".join(f"{v:.6f}" for v in t) + "\n")
    print(f"\nWrote transform to {args.out}")
    print(f"The reconstruction is now in EPSG:{args.epsg}. Reproject your DTM to the same CRS "
          "(dtm_merge_reproject.py), then apply this transform to the 3DGS .ply and refine with "
          "ICP (CloudCompare) on the ground points.")


if __name__ == "__main__":
    main()