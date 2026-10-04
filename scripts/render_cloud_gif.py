"""Orbit GIF of a georeferenced dense point cloud (COLMAP MVS, UTM).

Points are projected with a pinhole camera that circles the cloud centre and looks at it;
a per-pixel z-buffer keeps the nearest point, drawn as a small square. Optional crops: a
corridor around the GPS track and sky-coloured outliers. Reads only, apart from the GIF.

    python3 scripts/render_cloud_gif.py --ply seg01_ep2/colmap/dense/fused_utm.ply \
        --track seg01_ep2/gps.csv --drop-sky --radius 150 --height-m 80 --frames 60 \
        --out orbit.gif
"""
import argparse
import math
import os

import numpy as np
from plyfile import PlyData
from PIL import Image
from pyproj import Transformer
from scipy.spatial import cKDTree


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """World-to-camera rotation (rows: x right, y down, z forward)."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def render(P, rgb, eye, target, w, h, fov, size):
    R = look_at(eye, target)
    Q = (P - eye) @ R.T
    keep = Q[:, 2] > 1.0
    Q, c = Q[keep], rgb[keep]
    f = 0.5 * w / math.tan(math.radians(fov) / 2)
    u = np.round(f * Q[:, 0] / Q[:, 2] + w / 2).astype(int)
    v = np.round(f * Q[:, 1] / Q[:, 2] + h / 2).astype(int)
    img = np.full((h, w, 3), 255, np.uint8)
    zbuf = np.full((h, w), np.inf)
    order = np.argsort(-Q[:, 2])                      # far to near: near points overwrite
    u, v, z, c = u[order], v[order], Q[order, 2], c[order]
    for du in range(size):
        for dv in range(size):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            pix = vv[ok] * w + uu[ok]
            # last occurrence of each pixel = nearest point
            last = len(pix) - 1 - np.unique(pix[::-1], return_index=True)[1]
            p, zz, cc = pix[last], z[ok][last], c[ok][last]
            better = zz < zbuf.flat[p]
            zbuf.flat[p[better]] = zz[better]
            img.reshape(-1, 3)[p[better]] = cc[better]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=560)
    ap.add_argument('--height', type=int, default=340)
    ap.add_argument('--fov', type=float, default=45.0)
    ap.add_argument('--radius', type=float, default=110.0, help='orbit radius [m]')
    ap.add_argument('--height-m', type=float, default=60.0, help='camera height above centre [m]')
    ap.add_argument('--sweep', type=float, default=360.0, help='orbit angle [deg]')
    ap.add_argument('--frames', type=int, default=72)
    ap.add_argument('--fps', type=float, default=12.0)
    ap.add_argument('--size', type=int, default=2, help='point size [px]')
    ap.add_argument('--max-points', type=int, default=3000000)
    ap.add_argument('--colors', type=int, default=192)
    ap.add_argument('--track', help='GPS csv (lat, lon): keep points near the walked path')
    ap.add_argument('--max-track-dist', type=float, default=15.0, help='[m], horizontal')
    ap.add_argument('--epsg', type=int, default=32633, help='UTM EPSG of the cloud (default 32633)')
    ap.add_argument('--drop-sky', action='store_true', help='drop sky-coloured outliers')
    args = ap.parse_args()

    v = PlyData.read(args.ply)['vertex'].data
    P = np.stack([v['x'], v['y'], v['z']], 1)
    rgb = np.stack([v['red'], v['green'], v['blue']], 1).astype(np.uint8)
    if args.track:
        rows = np.genfromtxt(args.track, delimiter=',', names=True)
        tr = Transformer.from_crs(4326, args.epsg, always_xy=True)
        T = np.stack(tr.transform(rows['lon'], rows['lat']), 1)
        d, _ = cKDTree(T).query(P[:, :2])
        P, rgb = P[d < args.max_track_dist], rgb[d < args.max_track_dist]
    if args.drop_sky:
        # sky-coloured points matched onto the terrain near the horizon (MVS outliers)
        r, g, b = (rgb[:, i].astype(int) for i in range(3))
        sky = (b > r + 15) & (b > g)
        print(f'dropping {sky.sum():,} sky-coloured points')
        P, rgb = P[~sky], rgb[~sky]
    if len(P) > args.max_points:
        sel = np.random.default_rng(0).choice(len(P), args.max_points, replace=False)
        P, rgb = P[sel], rgb[sel]
    centre = np.median(P, 0)
    P = P - centre                                     # local metres, z up
    print(f'{len(P):,} points, extent {np.ptp(P, 0).round(1)} m')
    frames = []
    for k in range(args.frames):
        a = math.radians(args.sweep * k / args.frames)
        eye = np.array([args.radius * math.cos(a), args.radius * math.sin(a), args.height_m])
        frames.append(render(P, rgb, eye, np.zeros(3), args.width, args.height, args.fov,
                             args.size))
    imgs = [Image.fromarray(f).quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT)
            for f in frames]
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:], duration=round(1000 / args.fps),
                 loop=0, optimize=True)
    print('wrote', args.out, f'{os.path.getsize(args.out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
