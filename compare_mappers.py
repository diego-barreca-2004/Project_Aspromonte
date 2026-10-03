#!/usr/bin/env python3
"""compare_mappers.py - incremental vs global SfM (GLOMAP), with and without GoPro gravity.

For one epoch (a folder produced by run_pipeline.py: frames/, gps.csv, colmap/database.db),
this script runs three mappers on the SAME features and matches and measures what changes:

  incremental      COLMAP's incremental mapper (pycolmap.incremental_mapping)
  global           COLMAP's global mapper, i.e. GLOMAP (pycolmap.global_mapping)
  global_gravity   the global mapper with the GoPro gravity vector written as a
                   gravity-only pose prior and used in rotation averaging (use_gravity)

The source database is never modified. It is copied into --out, the shared camera is set to
the ChArUco calibration (so all epochs use identical intrinsics), and geometric
verification is re-run on the existing matches with the installed COLMAP. The gravity run
uses a second copy that also holds the priors.

Per mapper it reports: registered images, models, 3D points, mean reprojection error, wall
time; the GPS fit of two georeferencings (a 7-DoF Sim3, as geo_align.py does, and a 4-DoF
fit that takes "down" from the GoPro gravity and only estimates heading, scale and
translation from GPS); the tilt each georeferencing leaves against the gravity vector; and,
with --dtm, the rotation a point-to-plane ICP of the sparse ground points onto the LiDAR DTM
still has to apply (as georef_splat.py does for the splat).

Requires: pycolmap >= 4.2 (CPU wheel is enough), numpy, pyproj, rasterio, exiftool.

Usage:
  python3 compare_mappers.py --epoch ./seg01 --video Attempt.MP4 \
      --calibration ./calib_out/calibration_fisheye.json \
      --dtm aspromonte_dtm_utm33n.tif --out ./sfm_compare/seg01
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from geo_align import load_gps, robust_umeyama
from georef_splat import dtm_correspondence, icp_point_to_plane, load_dtm, nearest_cell
from gopro_gravity import frame_gravity, frame_time, write_database_priors
from run_colmap import camera_params_from_calibration

METHODS = ("incremental", "global", "global_gravity")
DOWN = np.array([0.0, 0.0, -1.0])                 # world "down" in a Z-up CRS


# --- database preparation ----------------------------------------------------
def prepare_database(src, dst, calibration, frames_dir, threads):
    """Copy src -> dst, set the calibrated camera, re-run geometric verification on dst."""
    import pycolmap
    if os.path.abspath(src) == os.path.abspath(dst):
        sys.exit("Refusing to work on the source database; --out must be another folder.")
    shutil.copy2(src, dst)
    model, params = camera_params_from_calibration(calibration, frames_dir)
    db = pycolmap.Database.open(dst)
    try:
        for cam in db.read_all_cameras():
            cam.model = pycolmap.CameraModelId.__members__[model]
            cam.params = np.array(params.split(","), float)
            cam.has_prior_focal_length = True
            db.update_camera(cam)
        db.clear_two_view_geometries()
    finally:
        db.close()
    opts = pycolmap.GeometricVerifierOptions()
    opts.num_threads = threads
    pycolmap.geometric_verification(dst, verifier_options=opts)


# --- mappers -----------------------------------------------------------------
def run_mapper(method, database, image_dir, out_dir, threads, seed):
    """Run one mapper; returns (largest reconstruction, number of models, seconds)."""
    import pycolmap
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if method == "incremental":
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_focal_length = False       # intrinsics fixed to the calibration,
        opts.ba_refine_principal_point = False    # like run_colmap.py --calibration
        opts.ba_refine_extra_params = False
        opts.num_threads = threads
        opts.random_seed = seed
        recs = pycolmap.incremental_mapping(database, image_dir, out_dir, opts)
    else:
        opts = pycolmap.GlobalPipelineOptions()
        opts.num_threads = threads
        opts.random_seed = seed
        m = opts.mapper
        m.random_seed = seed
        m.bundle_adjustment.refine_focal_length = False
        m.bundle_adjustment.refine_principal_point = False
        m.bundle_adjustment.refine_extra_params = False
        m.bundle_adjustment.ceres.use_gpu = False
        m.global_positioning.use_gpu = False
        m.rotation_averaging.use_gravity = method == "global_gravity"
        recs = pycolmap.global_mapping(database, image_dir, out_dir, opts)
    secs = time.time() - t0
    if not recs:
        return None, 0, secs
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    return best, len(recs), secs


# --- geometry helpers --------------------------------------------------------
def angle_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def rotation_between(a, b):
    """Smallest rotation mapping unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def rotation_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def umeyama_yaw(S, Y):
    """Similarity mapping S -> Y whose rotation is about the vertical (z) axis only."""
    ms, my = S.mean(0), Y.mean(0)
    Sc, Yc = S - ms, Y - my
    th = np.angle(np.sum(np.conj(Sc[:, 0] + 1j * Sc[:, 1]) * (Yc[:, 0] + 1j * Yc[:, 1])))
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    c = float(np.sum((Sc @ R.T) * Yc) / np.sum(Sc ** 2))
    return c, R, my - c * R @ ms


def robust_umeyama_yaw(S, Y, iters=6, k=3.0, min_keep=10):
    """umeyama_yaw with the same outlier rejection as geo_align.robust_umeyama."""
    idx = np.arange(len(S))
    c, R, t = umeyama_yaw(S, Y)
    for _ in range(iters):
        res = np.linalg.norm((c * (R @ S.T).T + t) - Y, axis=1)
        keep = np.where(res <= np.median(res[idx]) * k + 1e-6)[0]
        if len(keep) < min_keep or len(keep) == len(idx):
            break
        idx = keep
        c, R, t = umeyama_yaw(S[idx], Y[idx])
    return c, R, t, idx


# --- per-reconstruction measurements -----------------------------------------
def camera_arrays(rec, names_filter):
    imgs = sorted((im for im in rec.images.values() if im.has_pose and im.name in names_filter),
                  key=lambda im: im.name)
    names = [im.name for im in imgs]
    R = np.array([im.cam_from_world().rotation.matrix() for im in imgs])
    C = np.array([im.projection_center() for im in imgs])
    return names, R, C


def gravity_in_model(R, g_cam):
    """Per-image gravity mapped into the model frame, and its robust mean direction."""
    gw = np.einsum("nji,nj->ni", R, g_cam)          # R^T g per image
    gm = np.median(gw, axis=0)
    gm /= np.linalg.norm(gm)
    spread = np.array([angle_deg(x, gm) for x in gw])
    return gm, spread


def fit_georef(C, W, g_model, kind):
    """Georeference camera centres C to world W. kind: 'sim3' (7 DoF) or 'gravity' (4 DoF)."""
    origin = np.floor(W.min(0))
    if kind == "sim3":
        c, R, t, idx = robust_umeyama(C, W - origin)
    else:
        Ra = rotation_between(g_model, DOWN)       # level the model with gravity first
        c, Rz, t, idx = robust_umeyama_yaw(C @ Ra.T, W - origin)
        R = Rz @ Ra
    t = t + origin
    res = np.linalg.norm(c * C @ R.T + t - W, axis=1)[idx]
    return c, R, t, len(idx), float(np.median(res))


def dtm_residual_rotation(rec, c, R, t, dtm, ground_band=1.5, iters=60):
    """ICP of the georeferenced sparse ground points onto the DTM: what rotation is left."""
    Z, nrm, T, _ = dtm
    P = np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 3])
    X = c * P @ R.T + t
    ri, ci, ok = nearest_cell(T, Z, X[:, 0], X[:, 1])
    res = np.full(len(X), np.nan)
    res[ok] = X[ok, 2] - Z[ri[ok], ci[ok]]
    fin = np.isfinite(res)
    if fin.sum() < 50:
        return None
    ground = fin & (np.abs(res - np.median(res[fin])) <= ground_band)
    if ground.sum() < 50:
        return None
    Pg = X[ground]
    R2, t2, info = icp_point_to_plane(Pg, dtm_correspondence(Z, nrm, T), np.floor(Pg.mean(0)),
                                      iters=iters)
    centre = Pg.mean(0)                   # t2 is global: report the shift of the centroid
    return {"ground_points": int(ground.sum()),
            "rotation_deg": rotation_angle(R2),
            "tilt_deg": angle_deg(R2 @ np.array([0, 0, 1.0]), np.array([0, 0, 1.0])),
            "shift_m": float(np.linalg.norm(R2 @ centre + t2 - centre)),
            "rms_before_m": info["rms0"], "rms_after_m": info["rms"],
            "R": R2.tolist()}


def measure(rec, n_models, secs, n_images, world, grav, dtm):
    out = {"registered": rec.num_reg_images(), "images": n_images, "models": n_models,
           "points3D": rec.num_points3D(),
           "mean_reproj_px": float(rec.compute_mean_reprojection_error()),
           "mean_track_length": float(rec.compute_mean_track_length()),
           "seconds": round(secs, 1)}
    names, Rc, C = camera_arrays(rec, set(world) & set(grav))
    g_cam = np.array([grav[n] for n in names])
    g_model, spread = gravity_in_model(Rc, g_cam)
    out["gravity_spread_deg"] = {"median": float(np.median(spread)),
                                 "p90": float(np.percentile(spread, 90))}
    # in a gravity-aligned global model, world +y should be down
    out["model_y_vs_gravity_deg"] = angle_deg(g_model, np.array([0, 1.0, 0]))
    W = np.array([world[n] for n in names])
    for kind in ("sim3", "gravity"):
        c, R, t, n_in, med = fit_georef(C, W, g_model, kind)
        entry = {"gps_inliers": n_in, "gps_used": len(names), "gps_residual_median_m": med,
                 "tilt_vs_gopro_gravity_deg": angle_deg(R.T @ DOWN, g_model),
                 "scale": c, "R": R.tolist(), "t": t.tolist()}
        if dtm is not None:
            entry["dtm_icp"] = dtm_residual_rotation(rec, c, R, t, dtm)
        out["georef_" + kind] = entry
    return out


def gps_world(gps_csv, names, epsg):
    from pyproj import Transformer
    t_s, lat, lon, alt = load_gps(gps_csv)
    project = Transformer.from_crs(4326, epsg, always_xy=True).transform
    world = {}
    for n in names:
        ft = frame_time(n)
        if ft is None or ft < t_s[0] - 0.5 or ft > t_s[-1] + 0.5:
            continue
        e, nn = project(float(np.interp(ft, t_s, lon)), float(np.interp(ft, t_s, lat)))
        world[n] = np.array([e, nn, float(np.interp(ft, t_s, alt))])
    return world


def main():
    ap = argparse.ArgumentParser(description="Incremental vs global SfM, with/without GoPro gravity.")
    ap.add_argument("--epoch", required=True, help="epoch folder (frames/, gps.csv, colmap/database.db)")
    ap.add_argument("--video", required=True, help="the GoPro .MP4 of this epoch (gravity stream)")
    ap.add_argument("--calibration", required=True, help="calibration JSON from calibrate_camera.py")
    ap.add_argument("--out", required=True, help="new output folder (databases, models, metrics.json)")
    ap.add_argument("--dtm", default=None, help="bare-earth DTM GeoTIFF in --epsg (enables ICP check)")
    ap.add_argument("--epsg", type=int, default=32633)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure-only", action="store_true",
                    help="do not map: re-measure the models already in --out/<method>/best")
    args = ap.parse_args()

    frames = os.path.join(args.epoch, "frames")
    src_db = os.path.join(args.epoch, "colmap", "database.db")
    os.makedirs(args.out, exist_ok=True)
    methods = [m for m in args.methods.split(",") if m]
    if set(methods) - set(METHODS):
        sys.exit(f"Unknown method(s): {set(methods) - set(METHODS)}")

    names = sorted(n for n in os.listdir(frames) if n.lower().endswith(".jpg"))
    print(f"[gravity] reading the GPMF gravity stream of {args.video}")
    grav = frame_gravity(args.video, names)
    world = gps_world(os.path.join(args.epoch, "gps.csv"), names, args.epsg)
    print(f"  {len(grav)} frames with gravity, {len(world)} with GPS")

    db = os.path.join(args.out, "database.db")
    if not os.path.isfile(db):
        print(f"[db] copy + calibrated camera + geometric verification -> {db}")
        prepare_database(src_db, db, args.calibration, frames, args.threads)
    db_grav = os.path.join(args.out, "database_gravity.db")
    if "global_gravity" in methods and not os.path.isfile(db_grav):
        shutil.copy2(db, db_grav)
        print(f"[db] {write_database_priors(db_grav, grav)} gravity priors -> {db_grav}")

    dtm = load_dtm(args.dtm) if args.dtm else None
    metrics_path = os.path.join(args.out, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.isfile(metrics_path) else {}
    for method in methods:
        best_dir = os.path.join(args.out, method, "best")
        if args.measure_only:
            if not os.path.isdir(best_dir) or method not in metrics:
                print(f"[{method}] no saved model, skipped")
                continue
            import pycolmap
            print(f"[{method}] re-measuring {best_dir}")
            rec = pycolmap.Reconstruction(best_dir)
            n_models, secs = metrics[method]["models"], metrics[method]["seconds"]
        else:
            print(f"[{method}] mapping")
            rec, n_models, secs = run_mapper(method, db_grav if method == "global_gravity" else db,
                                             frames, os.path.join(args.out, method),
                                             args.threads, args.seed)
        if rec is None:
            metrics[method] = {"registered": 0, "images": len(names), "seconds": round(secs, 1)}
            print(f"  no model ({secs:.0f} s)")
            continue
        if not args.measure_only:
            os.makedirs(best_dir, exist_ok=True)
            rec.write(best_dir)
        metrics[method] = measure(rec, n_models, secs, len(names), world, grav, dtm)
        m = metrics[method]
        print(f"  {m['registered']}/{m['images']} images, {m['models']} model(s), "
              f"{m['mean_reproj_px']:.2f} px, {secs:.0f} s; GPS Sim3 "
              f"{m['georef_sim3']['gps_residual_median_m']:.2f} m, tilt vs gravity "
              f"{m['georef_sim3']['tilt_vs_gopro_gravity_deg']:.2f} deg")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=1)
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
