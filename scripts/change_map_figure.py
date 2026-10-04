"""Figure: DSM difference between two epochs along the trail, with a zoom on one detection.

Recomputes the dsm_change.py difference grid (default parameters) and draws
  top    - bare-ground DSM difference along the corridor, all candidates (grey), the chosen
           detection (red) and the camera positions where other objects were passed
           (triangles), e.g. objects the MVS did not reconstruct;
  bottom - 5 x 5 m zooms: the detection on bare-ground cells, on all cells, and the first
           passed object on all cells.
Reads only, apart from the output image. The README figure assets/change_map.png comes from

    python3 scripts/change_map_figure.py --ref-epoch /data/aspromonte/seg01_ep2 \
        --cmp-epoch /data/aspromonte/seg01_ep3 \
        --icp /data/aspromonte/m3c2_ep3_vs_ep2_v2/icp_auto.txt \
        --cc-shift -558997 -4214492 -123 \
        --detections /data/aspromonte/dsm_ep3_vs_ep2/dsm_detections.csv --detection 24 \
        --pass cylinder:66 --pass ball:177 --out assets/change_map.png
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsm_change import grid_stats, read_track  # noqa: E402
from geo_align import read_camera_centres  # noqa: E402
from georef_cloud import parse_geo_transform  # noqa: E402
from run_m3c2 import nmad, parse_cc_matrices, read_ply_xyz  # noqa: E402

# dsm_change.py defaults
CELL, PCT, MINPTS, ROUGH, CORR = 0.05, 90.0, 3, 0.05, 1.0
SAT = 0.3                                              # colour saturation [m]
ZOOM = 2.5                                             # zoom half-width [m]


def passed_positions(epoch, passes, M, shift):
    """Camera centre at the given frames of `epoch`, in UTM after the epoch-to-reference ICP."""
    s, R, t, _ = parse_geo_transform(os.path.join(epoch, 'colmap', 'geo_transform.txt'))
    centres = {int(re.search(r'frame_(\d+)', n).group(1)): C for n, C in
               read_camera_centres(os.path.join(epoch, 'colmap', 'undistorted', 'sparse', '0')).items()}
    out = {}
    for name, frame in passes:
        g = s * R @ centres[frame] + t + shift
        out[name] = (frame, M[:3, :3] @ g + M[:3, 3] - shift)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ref-epoch', required=True,
                    help='reference epoch folder (colmap/dense/fused_utm.ply, gps.csv)')
    ap.add_argument('--cmp-epoch', required=True, help='compared epoch folder')
    ap.add_argument('--icp', required=True, help='ICP matrix of cmp onto ref (run_m3c2.py)')
    ap.add_argument('--cc-shift', nargs=3, type=float, required=True, metavar=('SX', 'SY', 'SZ'),
                    help='local shift the ICP matrix was estimated in')
    ap.add_argument('--detections', required=True, help='dsm_detections.csv of dsm_change.py')
    ap.add_argument('--detection', type=int, required=True,
                    help='row of the highlighted detection in the CSV (0-based)')
    ap.add_argument('--detection-name', default='bucket')
    ap.add_argument('--pass', dest='passes', action='append', default=[], metavar='NAME:FRAME',
                    help='object passed at this frame of the compared epoch (repeatable)')
    ap.add_argument('--ref-label', default='epoch 2')
    ap.add_argument('--cmp-label', default='epoch 3')
    ap.add_argument('--epsg', type=int, default=32633)
    ap.add_argument('--out', required=True, help='output PNG')
    args = ap.parse_args()

    shift = np.array(args.cc_shift)
    M = np.asarray(parse_cc_matrices(args.icp)[0])
    ref = read_ply_xyz(os.path.join(args.ref_epoch, 'colmap', 'dense', 'fused_utm.ply')) + shift
    cmp_ = read_ply_xyz(os.path.join(args.cmp_epoch, 'colmap', 'dense', 'fused_utm.ply')) + shift
    cmp_ = cmp_ @ M[:3, :3].T + M[:3, 3]
    gps = [os.path.join(args.ref_epoch, 'gps.csv')]
    trk = read_track(gps, args.epsg, 0.0)
    track = cKDTree(trk + shift[:2])
    ref = ref[track.query(ref[:, :2])[0] <= CORR]
    cmp_ = cmp_[track.query(cmp_[:, :2])[0] <= CORR]
    both = np.vstack([ref[:, :2], cmp_[:, :2]])
    origin = both.min(axis=0)
    shape = tuple((np.ceil((both.max(axis=0) - origin) / CELL) + 1).astype(int))
    h_ref, s_ref = grid_stats(ref, origin, CELL, shape, PCT, MINPTS)
    h_cmp, s_cmp = grid_stats(cmp_, origin, CELL, shape, PCT, MINPTS)
    dz = h_cmp - h_ref
    valid = np.isfinite(dz)
    smooth = valid & (s_ref <= ROUGH) & (s_cmp <= ROUGH)
    dz = dz - np.median(dz[smooth])
    noise = nmad(dz[smooth])
    length = np.hypot(*np.diff(trk, axis=0).T).sum()
    print(f'bare-ground NMAD {noise:.4f} m, track length {length:.0f} m')

    rows = list(csv.DictReader(open(args.detections)))
    E = np.array([float(r['E']) for r in rows])
    N = np.array([float(r['N']) for r in rows])
    k_det = args.detection
    det = rows[k_det]
    print('highlighted detection', det)
    passes = [(p.split(':')[0], int(p.split(':')[1])) for p in args.passes]
    passed = passed_positions(args.cmp_epoch, passes, M, shift)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FuncFormatter

    x0, y0 = origin - shift[:2]
    ext = (x0, x0 + shape[0] * CELL, y0, y0 + shape[1] * CELL)
    e0, n0 = np.floor(np.array([ext[0], ext[2]]) / 100) * 100   # axis offsets
    fig = plt.figure(figsize=(13, 7.6))
    gs = GridSpec(2, 3, height_ratios=[1.0, 1.15], hspace=0.30, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(np.where(smooth, dz, np.nan).T, origin='lower', extent=ext, cmap='RdBu_r',
                   vmin=-SAT, vmax=SAT, interpolation='nearest')
    ax.plot(E, N, 'o', mfc='none', mec='0.45', ms=11, mew=1.0, label=f'candidates ({len(rows)})')
    ax.plot(E[k_det], N[k_det], 'o', mfc='none', mec='#d62728', ms=16, mew=2.2,
            label=f'{args.detection_name} (rank {k_det + 1}/{len(rows)})')
    for name, (_, p) in passed.items():
        ax.plot(p[0], p[1], '^', color='k', ms=8)
        ax.annotate(f'{name}\n(not reconstructed)', (p[0], p[1]), xytext=(0, 10), ha='center',
                    textcoords='offset points', fontsize=8.5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2] - 1, ext[3] + 1)
    ax.set_aspect('equal')
    ax.set_xlabel(f'E - {e0:.0f} [m]')
    ax.set_ylabel(f'N - {n0:.0f} [m]')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v - e0:.0f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v - n0:.0f}'))
    ax.set_title(f'{args.cmp_label.capitalize()} minus {args.ref_label}, bare-ground DSM '
                 f'difference along {length:.0f} m of trail (NMAD {noise * 100:.1f} cm)',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=9, frameon=False)

    name = args.detection_name
    panels = [(f'{name}, bare-ground cells', smooth, (E[k_det], N[k_det]), True),
              (f'{name}, all cells', valid, (E[k_det], N[k_det]), True)]
    if passed:
        pname, (frame, p) = next(iter(passed.items()))
        panels.append((f'{pname} pass ({args.cmp_label} frame {frame}), all cells', valid,
                       (p[0], p[1]), False))
    for k, (title, mask, (cx, cy), ring) in enumerate(panels):
        a = fig.add_subplot(gs[1, k])
        a.imshow(np.where(mask, dz, np.nan).T, origin='lower', extent=ext, cmap='RdBu_r',
                 vmin=-SAT, vmax=SAT, interpolation='nearest')
        if ring:
            a.plot(cx, cy, 'o', mfc='none', mec='#d62728', ms=22, mew=2.0)
        else:
            a.plot(cx, cy, '^', color='k', ms=8)
        a.set_xlim(cx - ZOOM, cx + ZOOM)
        a.set_ylim(cy - ZOOM, cy + ZOOM)
        a.set_aspect('equal')
        a.set_xticks(cx + np.arange(-2, 3))
        a.set_xticklabels([f'{v:+d}' for v in range(-2, 3)])
        a.set_yticks(cy + np.arange(-2, 3))
        a.set_yticklabels([f'{v:+d}' for v in range(-2, 3)] if k == 0 else [])
        a.set_xlabel('E [m]')
        if k == 0:
            a.set_ylabel('N [m]')
            title += (f": {float(det['dz_median']) * 100:+.0f} cm, "
                      f"{float(det['area_m2']):.2f} m$^2$")
        a.set_title(title, fontsize=9.5)
    fig.colorbar(im, ax=fig.axes, shrink=0.85, pad=0.02,
                 label=f'height change {args.cmp_label} - {args.ref_label} [m]')
    fig.savefig(args.out, dpi=110, bbox_inches='tight')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
