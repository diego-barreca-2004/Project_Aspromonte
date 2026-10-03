"""Aggregate compare_mappers.py results over several epochs of the same segment.

For every method (plus the original incremental models georeferenced with geo_transform.txt)
and every georeferencing (GPS Sim3, gravity-levelled 4-DoF) it prints the per-epoch metrics
and the rotation left between epochs, measured in two independent ways:

  cloud ICP: the georeferenced sparse points of epoch B are aligned to those of epoch A by
             point-to-plane ICP (normals from 12 neighbours), coarse to fine (correspondence
             gate 20, 5, 2, 1, 0.5 m), because GPS-only georeferencings of different days
             can be metres apart;
  via DTM:   each epoch's ground points are aligned to the LiDAR DTM by ICP (rotation R_e);
             the relative rotation is R_A^T R_B. The DTM is a common absolute reference, so
             the tilt part of this is well constrained; the heading part is not.

Writes summary.md and summary.json into --out-dir (default: the compare_mappers.py output
root). Requires pycolmap>=4.2.

    python3 scripts/mapper_report.py --root /data/aspromonte \
        --epochs seg01 seg01_ep2 seg01_ep3
    python3 scripts/mapper_report.py --root /data/aspromonte --selftest seg01_ep2
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pycolmap
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_mappers import dtm_residual_rotation  # noqa: E402
from georef_cloud import parse_geo_transform  # noqa: E402
from georef_splat import icp_point_to_plane, load_dtm  # noqa: E402

METHODS = ('incremental', 'global', 'global_gravity')
GATES = (20.0, 5.0, 2.0, 1.0, 0.5)
NAN = float('nan')


def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def tilt(R):
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))


def points(rec, c, R, t):
    P = np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 3])
    return c * P @ np.asarray(R).T + np.asarray(t)


def cloud_correspondence(Q, k=12):
    """Returns gate -> correspond(P): nearest point of Q within `gate` metres and its normal."""
    tree = cKDTree(Q)
    _, nb = tree.query(Q, k)
    X = Q[nb] - Q[nb].mean(1, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    N = Vt[:, 2, :]                                   # smallest-variance direction

    def make(gate):
        def correspond(P):
            d, j = tree.query(P, distance_upper_bound=gate)
            ok = np.isfinite(d)
            j = np.where(ok, j, 0)
            return Q[j], N[j], ok
        return correspond
    return make


def inter_epoch(PA, PB):
    """Coarse-to-fine ICP of B onto A (both UTM): rotation, tilt, shift at the centroid."""
    make = cloud_correspondence(PA)
    centre = PB.mean(0)
    sel = np.random.default_rng(0).choice(len(PB), min(len(PB), 60000), replace=False)
    P = PB[sel].copy()
    R_tot, t_tot = np.eye(3), np.zeros(3)
    rms0 = rms = NAN
    for gate in GATES:
        R, t, info = icp_point_to_plane(P, make(gate), np.floor(centre), iters=30)
        if info['rms0'] is None:
            continue
        if np.isnan(rms0):
            rms0 = info['rms0']
        rms = info['rms'] if info['rms'] is not None else NAN
        P = P @ R.T + t
        R_tot, t_tot = R @ R_tot, R @ t_tot + t
    if np.isnan(rms0):
        return {'rotation_deg': NAN, 'tilt_deg': NAN, 'shift_m': NAN,
                'rms_before_m': NAN, 'rms_after_m': NAN}
    return {'rotation_deg': angle(R_tot), 'tilt_deg': tilt(R_tot),
            'shift_m': float(np.linalg.norm(R_tot @ centre + t_tot - centre)),
            'rms_before_m': rms0, 'rms_after_m': rms}


def via_dtm(RA, RB):
    Rrel = np.asarray(RA).T @ np.asarray(RB)
    return {'rotation_deg': angle(Rrel), 'tilt_deg': tilt(Rrel)}


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def selftest(compare_dir, epoch):
    """Recover a known 5 deg tilt + 6 m offset between a real cloud and a moved copy of it."""
    d = os.path.join(compare_dir, epoch)
    g = json.load(open(os.path.join(d, 'metrics.json')))['global_gravity']['georef_sim3']
    PA = points(pycolmap.Reconstruction(os.path.join(d, 'global_gravity', 'best')),
                g['scale'], g['R'], g['t'])
    c = PA.mean(0)
    Rk = rot([1.0, 0.4, 0.0], 5.0)
    rng = np.random.default_rng(1)
    PB = (PA - c) @ Rk.T + c + np.array([2.0, -3.0, 5.0]) + rng.normal(0, 0.05, PA.shape)
    r = inter_epoch(PA, PB)
    print(f"selftest: expected rot 5.00 deg, tilt {tilt(Rk.T):.2f} deg, shift "
          f"{np.linalg.norm([2.0, -3.0, 5.0]):.2f} m; got rot {r['rotation_deg']:.2f}, tilt "
          f"{r['tilt_deg']:.2f}, shift {r['shift_m']:.2f}, rms {r['rms_before_m']:.2f}->"
          f"{r['rms_after_m']:.2f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True, help='data directory holding the epoch folders')
    ap.add_argument('--epochs', nargs='+', default=['seg01', 'seg01_ep2', 'seg01_ep3'],
                    help='epoch folders, oldest first')
    ap.add_argument('--compare-dir', help='compare_mappers.py output root '
                    '(default: <root>/sfm_compare)')
    ap.add_argument('--dtm', help='LiDAR DTM in UTM (default: <root>/aspromonte_dtm_utm33n.tif)')
    ap.add_argument('--out-dir', help='where summary.md/json go (default: --compare-dir)')
    ap.add_argument('--selftest', metavar='EPOCH',
                    help='only check the inter-epoch ICP on a moved copy of this epoch')
    args = ap.parse_args()
    root = args.root
    cmp_dir = args.compare_dir or os.path.join(root, 'sfm_compare')
    out_dir = args.out_dir or cmp_dir
    if args.selftest:
        selftest(cmp_dir, args.selftest)
        return

    epochs = args.epochs
    # consecutive pairs first, then longer gaps
    pairs = sorted(itertools.combinations(epochs, 2),
                   key=lambda p: (epochs.index(p[1]) - epochs.index(p[0]), epochs.index(p[0])))
    metrics = {}
    for e in epochs:
        path = os.path.join(cmp_dir, e, 'metrics.json')
        if os.path.isfile(path):
            metrics[e] = json.load(open(path))
    dtm = load_dtm(args.dtm or os.path.join(root, 'aspromonte_dtm_utm33n.tif'))
    clouds, dtm_R = {}, {}
    summary = {'per_epoch': {}, 'inter_epoch': {}}
    # original models of track 1 (undistorted sparse, geo_transform.txt)
    for e in epochs:
        rec = pycolmap.Reconstruction(os.path.join(root, e, 'colmap', 'undistorted', 'sparse', '0'))
        c, R, t, _ = parse_geo_transform(os.path.join(root, e, 'colmap', 'geo_transform.txt'))
        clouds[('original', 'sim3', e)] = points(rec, c, R, t)
        icp = dtm_residual_rotation(rec, c, R, t, dtm)
        if icp:
            dtm_R[('original', 'sim3', e)] = icp['R']
            summary['per_epoch'][f'original/{e}'] = {'sim3': {'dtm_icp': {
                k: v for k, v in icp.items() if k != 'R'}}}
    for m in METHODS:
        for e in epochs:
            mm = metrics.get(e, {}).get(m)
            best = os.path.join(cmp_dir, e, m, 'best')
            if not mm or 'georef_sim3' not in mm or not os.path.isdir(best):
                continue
            rec = pycolmap.Reconstruction(best)
            summary['per_epoch'][f'{m}/{e}'] = {k: v for k, v in mm.items()
                                                if not isinstance(v, dict) or k == 'gravity_spread_deg'}
            for kind in ('sim3', 'gravity'):
                g = mm['georef_' + kind]
                clouds[(m, kind, e)] = points(rec, g['scale'], g['R'], g['t'])
                icp = g.get('dtm_icp') or {}
                if 'R' in icp:
                    dtm_R[(m, kind, e)] = icp['R']
                summary['per_epoch'][f'{m}/{e}'][kind] = {
                    k: v for k, v in g.items() if k not in ('R', 't', 'dtm_icp')}
                summary['per_epoch'][f'{m}/{e}'][kind]['dtm_icp'] = {
                    k: v for k, v in icp.items() if k != 'R'}
    print(time.strftime('%H:%M:%S'), 'loaded', len(clouds), 'clouds', flush=True)
    for (m, kind) in sorted({(m, k) for m, k, _ in clouds}):
        for a, b in pairs:
            if (m, kind, a) in clouds and (m, kind, b) in clouds:
                r = inter_epoch(clouds[(m, kind, a)], clouds[(m, kind, b)])
                if (m, kind, a) in dtm_R and (m, kind, b) in dtm_R:
                    v = via_dtm(dtm_R[(m, kind, a)], dtm_R[(m, kind, b)])
                    r['dtm_rotation_deg'], r['dtm_tilt_deg'] = v['rotation_deg'], v['tilt_deg']
                summary['inter_epoch'][f'{m}/{kind}/{b}->{a}'] = r
                print(f'{time.strftime("%H:%M:%S")} {m:15s} {kind:8s} {b:10s}->{a:10s} '
                      f'ICP rot {r["rotation_deg"]:5.2f} tilt {r["tilt_deg"]:5.2f} deg  '
                      f'shift {r["shift_m"]:5.2f} m  rms {r["rms_before_m"]:.2f}->'
                      f'{r["rms_after_m"]:.2f} m | via DTM rot '
                      f'{r.get("dtm_rotation_deg", NAN):5.2f} tilt {r.get("dtm_tilt_deg", NAN):5.2f}',
                      flush=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)

    lines = ['| Method | Epoch | Registered | Reproj. px | Points | Time s | Model y vs gravity | '
             'GPS res. Sim3 / grav. m | Sim3 tilt vs IMU | DTM-ICP tilt Sim3 / grav. |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for key, v in summary['per_epoch'].items():
        m, e = key.split('/')
        if m == 'original':
            continue
        s, g = v['sim3'], v['gravity']
        lines.append(f"| {m} | {e} | {v['registered']}/{v['images']} | {v['mean_reproj_px']:.2f} | "
                     f"{v['points3D']:,} | {v['seconds']:.0f} | {v['model_y_vs_gravity_deg']:.1f} | "
                     f"{s['gps_residual_median_m']:.2f} / {g['gps_residual_median_m']:.2f} | "
                     f"{s['tilt_vs_gopro_gravity_deg']:.1f} | "
                     f"{s['dtm_icp'].get('tilt_deg', NAN):.1f} / "
                     f"{g['dtm_icp'].get('tilt_deg', NAN):.1f} |")
    lines += ['', '| Method | Georef | Pair | ICP rotation deg | ICP tilt deg | Shift m | '
              'RMS m | Tilt via DTM deg |', '|---|---|---|---|---|---|---|---|']
    for key, r in summary['inter_epoch'].items():
        m, kind, pair = key.split('/')
        lines.append(f"| {m} | {kind} | {pair} | {r['rotation_deg']:.2f} | {r['tilt_deg']:.2f} | "
                     f"{r['shift_m']:.2f} | {r['rms_before_m']:.2f} -> {r['rms_after_m']:.2f} | "
                     f"{r.get('dtm_tilt_deg', NAN):.2f} |")
    open(os.path.join(out_dir, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
