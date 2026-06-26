#!/usr/bin/env python3
"""Single-camera intrinsic calibration for the Aspromonte Digital Twin.

Detects a ChArUco board across a calibration video and fits the standard
(Brown-Conrady), rational and fisheye (Kannala-Brandt) distortion models.
For each model it writes the intrinsics, per-view reprojection errors and an
undistortion preview, plus a shared detection-coverage map.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import numpy as np

MODELS = ("standard", "rational", "fisheye")
ARUCO_DICT = cv2.aruco.DICT_4X4_50                     # matches a DICT_4X4 board
MIN_CORNERS = 6                                        # min ChArUco corners per usable view
HAS_NEW_API = hasattr(cv2.aruco, "CharucoDetector")    # OpenCV >= 4.7


# --- ChArUco setup -----------------------------------------------------------
def create_charuco_board(cols: int, rows: int, square_size: float, marker_size: float):
    """Build the ChArUco board and its dictionary."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    if HAS_NEW_API:
        board = cv2.aruco.CharucoBoard((cols, rows), square_size, marker_size, dictionary)
    else:
        board = cv2.aruco.CharucoBoard_create(cols, rows, square_size, marker_size, dictionary)
    return board, dictionary


def build_detector(board):
    """Reusable ChArUco detector (new API); None on the legacy path."""
    if not HAS_NEW_API:
        return None
    det_params = cv2.aruco.DetectorParameters()
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # sub-pixel marker corners
    return cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), det_params)


def detect_charuco(gray, board, dictionary, detector):
    """Return (corners, ids) for one frame, or (None, None) if the board is absent."""
    if detector is not None:                                     # OpenCV >= 4.7
        corners, ids, _, _ = detector.detectBoard(gray)
        return corners, ids
    markers, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)   # legacy API
    if marker_ids is None or len(marker_ids) == 0:
        return None, None
    _, corners, ids = cv2.aruco.interpolateCornersCharuco(markers, marker_ids, gray, board)
    return corners, ids


def board_object_points(board, ids) -> np.ndarray:
    """Object points (mm) for the detected chessboard-corner ids."""
    corners = board.getChessboardCorners() if HAS_NEW_API else board.chessboardCorners
    return np.array([corners[i[0]] for i in ids], dtype=np.float32)


# --- Data collection ---------------------------------------------------------
def collect_from_video(video_path: str, frame_step: int, board, dictionary, detector,
                       save_dir: Optional[str] = None):
    """Detect the board across the video and gather calibration observations."""
    print(f"Reading {video_path} (1 frame every {frame_step})")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Error: cannot open video file {video_path}")

    objpoints, imgpoints, names, all_corners = [], [], [], []
    image_size, first_img = None, None
    count = 0

    while True:
        if not cap.grab():                       # advance without decoding
            break
        if count % frame_step == 0:
            ok, frame = cap.retrieve()           # decode only the sampled frames
            if not ok:
                break
            name = f"frame_{count}"
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])

            corners, ids = detect_charuco(gray, board, dictionary, detector)
            if corners is not None and ids is not None and len(corners) >= MIN_CORNERS:
                objpoints.append(board_object_points(board, ids))
                imgpoints.append(corners)
                all_corners.append(corners)
                names.append(name)
                if first_img is None:
                    first_img = frame.copy()
                if save_dir is not None:
                    annotated = frame.copy()
                    cv2.aruco.drawDetectedCornersCharuco(annotated, corners, ids, (0, 255, 0))
                    cv2.imwrite(os.path.join(save_dir, f"{name}.jpg"), annotated)
                print(f"  ok:   {name} ({len(corners)} corners)")
            else:
                print(f"  skip: {name} (insufficient corners)")
        count += 1

    cap.release()
    return objpoints, imgpoints, names, all_corners, image_size, first_img


# --- Reprojection error ------------------------------------------------------
def _rms_per_view(imgp, proj) -> float:
    """RMS reprojection error (px) over one view: sqrt(mean ||obs - proj||^2).

    Note the sqrt(N) normalisation: cv2.norm returns sqrt(sum of squared
    residuals), so dividing by N (instead of sqrt(N)) understates the error by
    a factor of sqrt(N) and breaks consistency with the overall RMS.
    """
    imgp = np.asarray(imgp, np.float64).reshape(-1, 2)
    proj = np.asarray(proj, np.float64).reshape(-1, 2)
    return float(cv2.norm(imgp, proj, cv2.NORM_L2) / np.sqrt(len(proj)))


def reproj_errors_standard(objpoints, imgpoints, rvecs, tvecs, K, dist) -> List[float]:
    errs = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        errs.append(_rms_per_view(imgp, proj))
    return errs


def reproj_errors_fisheye(objpoints, imgpoints, rvecs, tvecs, K, D, keep) -> List[float]:
    errs = []
    for j, i in enumerate(keep):
        objp = objpoints[i].reshape(-1, 1, 3).astype(np.float64)
        proj, _ = cv2.fisheye.projectPoints(objp, rvecs[j], tvecs[j], K, D)
        errs.append(_rms_per_view(imgpoints[i], proj))
    return errs


# --- Calibration -------------------------------------------------------------
def calibrate_standard(objpoints, imgpoints, image_size, rational: bool = False):
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    return cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=flags)


def _illconditioned_index(msg: str, n_keep: int) -> Optional[int]:
    """View index reported by a fisheye CALIB_CHECK_COND failure, if parseable."""
    if "input array " not in msg:
        return None
    try:
        idx = int(msg.split("input array ")[1].split()[0].strip(".,)"))
    except (ValueError, IndexError):
        return None
    return idx if idx < n_keep else None


def calibrate_fisheye(objpoints, imgpoints, image_size, names):
    """Fisheye calibration; drops ill-conditioned views flagged by CALIB_CHECK_COND."""
    objp = [o.reshape(-1, 1, 3).astype(np.float64) for o in objpoints]
    imgp = [c.reshape(-1, 1, 2).astype(np.float64) for c in imgpoints]
    keep = list(range(len(objp)))
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_FIX_SKEW |
             cv2.fisheye.CALIB_CHECK_COND)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    while True:
        op = [objp[i] for i in keep]
        ip = [imgp[i] for i in keep]
        K, D = np.zeros((3, 3)), np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), np.float64) for _ in keep]
        tvecs = [np.zeros((1, 1, 3), np.float64) for _ in keep]
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                op, ip, image_size, K, D, rvecs, tvecs, flags, term)
            return rms, K, D, rvecs, tvecs, keep
        except cv2.error as exc:
            idx = _illconditioned_index(str(exc), len(keep))
            if idx is None:
                raise
            print(f"  [fisheye] dropping ill-conditioned view: {names[keep.pop(idx)]}")
            if len(keep) < 4:
                raise RuntimeError("Too few usable views left for fisheye calibration.")


def run_model(model: str, objpoints, imgpoints, image_size, names) -> dict:
    """Fit one model and return its intrinsics, distortion and per-view errors."""
    if model == "fisheye":
        rms, K, dist, rvecs, tvecs, keep = calibrate_fisheye(objpoints, imgpoints, image_size, names)
        per_img = reproj_errors_fisheye(objpoints, imgpoints, rvecs, tvecs, K, dist, keep)
        used = [names[i] for i in keep]
    else:
        rms, K, dist, rvecs, tvecs = calibrate_standard(
            objpoints, imgpoints, image_size, rational=(model == "rational"))
        per_img = reproj_errors_standard(objpoints, imgpoints, rvecs, tvecs, K, dist)
        used = list(names)
    return {"model": model, "rms": float(rms), "K": np.asarray(K),
            "dist": np.ravel(dist), "per_img": per_img, "used": used}


# --- Outputs -----------------------------------------------------------------
def undistort_one(img, K, dist, model: str, image_size):
    """Undistort one image with the given model's intrinsics."""
    if model == "fisheye":
        D = np.asarray(dist, np.float64).reshape(-1, 1)[:4]
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, image_size, np.eye(3), balance=0.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_k, image_size, cv2.CV_16SC2)
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    new_k, _ = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 1, image_size)
    return cv2.undistort(img, K, dist, None, new_k)


def _label(img, text: str):
    """Overlay a top-left caption."""
    out = img.copy()
    fs = max(1.0, out.shape[1] / 900.0)
    th = max(2, out.shape[1] // 500)
    org = (20, int(70 * fs))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th, cv2.LINE_AA)
    return out


def save_pair(img, undist, path: str, max_w: int = 1600):
    """Save an original|undistorted comparison."""
    pair = np.hstack([img, undist])
    s = max_w / pair.shape[1]
    if s < 1.0:
        pair = cv2.resize(pair, (int(pair.shape[1] * s), int(pair.shape[0] * s)))
    cv2.imwrite(path, pair)


def save_compare_grid(img, results, image_size, path: str, max_w: int = 1800):
    """Save a 2x2 grid: original plus each model's undistortion."""
    tiles = [_label(img, "original")]
    for r in results:
        und = undistort_one(img, r["K"], r["dist"], r["model"], image_size)
        tiles.append(_label(und, f"{r['model']}  RMS={r['rms']:.3f}px"))
    while len(tiles) < 4:
        tiles.append(np.zeros_like(img))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    s = max_w / grid.shape[1]
    if s < 1.0:
        grid = cv2.resize(grid, (int(grid.shape[1] * s), int(grid.shape[0] * s)))
    cv2.imwrite(path, grid)


def save_coverage(image_size, all_corners, path: str):
    """Plot every detected corner to visualise frame coverage."""
    w, h = image_size
    canvas = np.zeros((h, w, 3), np.uint8)
    r = max(2, w // 600)
    for corners in all_corners:
        for p in corners.reshape(-1, 2):
            cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))), r, (0, 255, 0), -1)
    cv2.imwrite(path, canvas)


def result_to_json(res: dict, args, image_size) -> dict:
    return {
        "model": res["model"],
        "image_size": list(image_size),
        "camera_matrix": res["K"].tolist(),
        "distortion_coefficients": res["dist"].tolist(),
        "rms_reprojection_error_px": res["rms"],
        "num_images_used": len(res["used"]),
        "board": {"cols": args.cols, "rows": args.rows, "square_size_mm": args.square_size},
        "per_image_error_px": {k: round(v, 5) for k, v in zip(res["used"], res["per_img"])},
        "opencv_version": cv2.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


# --- Entry point -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ChArUco single-camera calibration from video.")
    ap.add_argument("--video", required=True, help="Path to the calibration video.")
    ap.add_argument("--frame-step", type=int, default=1, help="Use 1 frame every N.")
    ap.add_argument("--cols", type=int, default=9, help="Board squares along X (width).")
    ap.add_argument("--rows", type=int, default=6, help="Board squares along Y (height).")
    ap.add_argument("--square-size", type=float, default=25.0, help="Square side length (mm).")
    ap.add_argument("--marker-size", type=float, default=18.75, help="Marker side length (mm).")
    ap.add_argument("--model", choices=list(MODELS) + ["all"], default="all", help="Model(s) to fit.")
    ap.add_argument("--save-frames", action="store_true",
                    help="Save annotated frames of accepted detections to <out>/extracted_frames.")
    ap.add_argument("--out", default="./calib_out", help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    save_dir = os.path.join(args.out, "extracted_frames") if args.save_frames else None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    board, dictionary = create_charuco_board(args.cols, args.rows, args.square_size, args.marker_size)
    detector = build_detector(board)

    objpoints, imgpoints, names, all_corners, image_size, first_img = collect_from_video(
        args.video, args.frame_step, board, dictionary, detector, save_dir)

    n = len(objpoints)
    print(f"\nDetected ChArUco in {n} frames.")
    if n < 5:
        sys.exit("Not enough valid frames (need >= 5). Check marker size or video quality.")
    if n < 10:
        print("WARNING: fewer than 10 valid frames -- calibration may be unstable.")

    save_coverage(image_size, all_corners, os.path.join(args.out, "coverage.png"))

    models = list(MODELS) if args.model == "all" else [args.model]
    results = []
    for m in models:
        print(f"\n--- calibrating: {m} ---")
        res = run_model(m, objpoints, imgpoints, image_size, names)
        results.append(res)
        with open(os.path.join(args.out, f"calibration_{m}.json"), "w") as fh:
            json.dump(result_to_json(res, args, image_size), fh, indent=2)
        save_pair(first_img, undistort_one(first_img, res["K"], res["dist"], m, image_size),
                  os.path.join(args.out, f"undistort_{m}.png"))

    print("\n================== COMPARISON ==================")
    print(f"{'model':<10}{'#coeffs':>9}{'RMS (px)':>11}{'imgs':>7}")
    for r in results:
        print(f"{r['model']:<10}{len(r['dist']):>9}{r['rms']:>11.4f}{len(r['used']):>7}")
    best = min(results, key=lambda r: r["rms"])
    print(f"\nLowest RMS: {best['model']} ({best['rms']:.4f} px)")

    if len(results) > 1:
        save_compare_grid(first_img, results, image_size,
                          os.path.join(args.out, "compare_undistort.png"))
        with open(os.path.join(args.out, "comparison.json"), "w") as fh:
            json.dump({"models": [
                {"model": r["model"], "rms_reprojection_error_px": r["rms"],
                 "num_coeffs": int(len(r["dist"])), "num_images_used": len(r["used"])}
                for r in results], "lowest_rms_model": best["model"]}, fh, indent=2)

    print(f"\nArtifacts written to: {args.out}")


if __name__ == "__main__":
    main()