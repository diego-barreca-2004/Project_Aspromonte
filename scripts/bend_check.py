"""Where does an SfM model of an epoch disagree with the GoPro gravity?

For each compare_mappers.py method, the per-frame angle between the gravity mapped into the
model frame and the median model gravity is printed along the sequence (binned), together
with the GPS residual of the Sim3 fit. A model bent by a rotation-averaging error shows a
contiguous run of frames with a large angle, not scattered outliers. Reads only.

    python3 scripts/bend_check.py --epoch /data/aspromonte/seg01 --video Attempt.MP4 \
        --models /data/aspromonte/sfm_compare/seg01
"""
import argparse
import os
import sys

import numpy as np
import pycolmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_mappers import camera_arrays, fit_georef, gps_world, gravity_in_model  # noqa: E402
from gopro_gravity import frame_gravity  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--epoch', required=True, help='epoch folder with frames/ and gps.csv')
    ap.add_argument('--video', required=True, help='GoPro MP4 of the epoch (GRAV stream)')
    ap.add_argument('--models', required=True,
                    help='compare_mappers.py output folder of the epoch (<method>/best)')
    ap.add_argument('--epsg', type=int, default=32633)
    ap.add_argument('--bins', type=int, default=12)
    args = ap.parse_args()

    frames = sorted(n for n in os.listdir(os.path.join(args.epoch, 'frames'))
                    if n.lower().endswith('.jpg'))
    grav = frame_gravity(args.video, frames)
    world = gps_world(os.path.join(args.epoch, 'gps.csv'), frames, args.epsg)
    label = os.path.basename(os.path.normpath(args.models))
    for m in ('incremental', 'global', 'global_gravity'):
        best = os.path.join(args.models, m, 'best')
        if not os.path.isdir(best):
            continue
        rec = pycolmap.Reconstruction(best)
        names, R, C = camera_arrays(rec, set(world) & set(grav))
        g_model, spread = gravity_in_model(R, np.array([grav[n] for n in names]))
        W = np.array([world[n] for n in names])
        c, Rg, t, _, _ = fit_georef(C, W, g_model, 'sim3')
        res = np.linalg.norm(c * C @ Rg.T + t - W, axis=1)
        print(f'\n{label} {m}: {len(names)} frames, gravity spread median {np.median(spread):.2f} '
              f'p90 {np.percentile(spread, 90):.2f} deg, frames > 5 deg: {(spread > 5).sum()}')
        print('  bin  frames        spread med/max deg   GPS res. med m')
        for k, idx in enumerate(np.array_split(np.arange(len(names)), args.bins)):
            print(f'  {k:3d}  {names[idx[0]][6:12]}-{names[idx[-1]][6:12]}   '
                  f'{np.median(spread[idx]):6.2f} / {spread[idx].max():6.2f}        '
                  f'{np.median(res[idx]):6.2f}')


if __name__ == '__main__':
    main()
