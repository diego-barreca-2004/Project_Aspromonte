"""Undistort extracted GoPro frames to PINHOLE for VidMap's calibrated mode.

Uses only the ChArUco fisheye calibration (calib_out/calibration_fisheye.json),
scaled to the frame resolution, so VidMap does not inherit the intrinsics that
COLMAP refined. Writes only to --out, and stops if --out/images already exists.

Usage:
  python undistort_pinhole.py --frames seg01_ep2/frames \
      --out seg01_ep2/vidmap/frames_pinhole
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calib", type=Path, default=Path("calib_out/calibration_fisheye.json"))
    args = parser.parse_args()

    calib = json.loads(args.calib.read_text())
    calib_w, calib_h = calib["image_size"]
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    D = np.array(calib["distortion_coefficients"], dtype=np.float64).reshape(4, 1)

    names = sorted(p.name for p in args.frames.glob("*.jpg"))
    if not names:
        raise SystemExit(f"No .jpg frames in {args.frames}")
    h, w = cv2.imread(str(args.frames / names[0])).shape[:2]
    sx, sy = w / calib_w, h / calib_h
    if abs(sx - sy) > 1e-3:
        raise SystemExit(f"Frames {w}x{h} do not keep the calibration aspect {calib_w}x{calib_h}")
    K_frame = K.copy()
    K_frame[0] *= sx
    K_frame[1] *= sy

    # Same focal length and principal point as the fisheye camera, same image
    # size: the pinhole view keeps the centre at native resolution and crops the
    # far periphery, where the fisheye stretches most.
    K_new = K_frame.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_frame, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    for name in names:
        image = cv2.imread(str(args.frames / name))
        undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(images_dir / name), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95])

    fx, fy, cx, cy = K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]
    intrinsics = {1: {"params": [float(fx), float(fy), float(cx), float(cy)], "images": "all"}}
    (args.out / "intrinsics.yaml").write_text(yaml.safe_dump(intrinsics, sort_keys=False))
    hfov = np.degrees(2 * np.arctan(max(cx, w - cx) / fx))
    print(f"{len(names)} frames {w}x{h}, fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}, horizontal FOV about {hfov:.0f} deg")


if __name__ == "__main__":
    main()
