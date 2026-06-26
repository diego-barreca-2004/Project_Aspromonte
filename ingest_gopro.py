#!/usr/bin/env python3
"""ingest_gopro.py - Step 1 of the reconstruction pipeline: GoPro video -> frames (+ GPS).

Extracts evenly-spaced frames from a GoPro video (optional sub-clip and downscale) and the
GPS track, writing:
  out/frames/frame_000001_t00012.50.jpg ...
  out/gps.csv         columns: t_s, ts, lat, lon, alt, speed, dop, fix
  out/summary.json

GPS source (in order of preference):
  1. --gps-file: a sidecar exported by gopro-telemetry (.csv / .gpx / .json). Robust and the
     recommended path - HERO11/12/13 use the GPS9 stream, which older exiftool builds do not
     decode and which exiftool exposes as Doc<N>:GPS* sub-documents.
  2. exiftool fallback (GPS5/GPS9), used when --gps-file is omitted.

Usage:
  python3 ingest_gopro.py --video ride.mp4 --out ./seg01 --every-sec 0.5 \
                          --gps-file ride-GPS9.csv
  python3 ingest_gopro.py --video ride.mp4 --out ./seg01 --start 0 --end 120
  python3 ingest_gopro.py --video ride.mp4 --probe

Tip: for the FIRST end-to-end test, process a short sub-clip (~1-2 min) before the full 2 km.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import cv2

GPS_COLUMNS = ["t_s", "ts", "lat", "lon", "alt", "speed", "dop", "fix"]


# --- Video -------------------------------------------------------------------
def probe_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {path}\n"
                 f"(If it is HEVC/H.265 and this fails, transcode/re-mux with ffmpeg, "
                 f"or use an HEVC-capable OpenCV build.)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "n_frames": n, "width": w, "height": h,
            "duration_s": (n / fps) if fps else 0.0}


def extract_frames(path, out_dir, every_sec, start, end, longest_side, jpeg_quality, max_frames):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * every_sec)))
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    idx, saved, times = 0, 0, []
    while True:
        if not cap.grab():                       # advance without decoding
            break
        t = idx / fps
        idx += 1
        if t < start:
            continue
        if end is not None and t > end:
            break
        if (idx - 1) % step != 0:
            continue
        ok, frame = cap.retrieve()               # decode only the sampled frames
        if not ok or frame is None:
            continue
        if longest_side:
            hh, ww = frame.shape[:2]
            s = longest_side / max(hh, ww)
            if s < 1.0:
                frame = cv2.resize(frame, (int(ww * s), int(hh * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{saved:06d}_t{t:08.2f}.jpg"),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        saved += 1
        times.append(round(t, 3))
        if max_frames and saved >= max_frames:
            print(f"  reached --max-frames={max_frames}, stopping.")
            break
    cap.release()
    return saved, times


# --- GPS parsing helpers -----------------------------------------------------
def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _fix_to_int(x):
    """Normalise a GPX/numeric fix value ('3', '3d', 'none', ...) to 0/2/3."""
    if x is None:
        return None
    s = str(x).strip().lower()
    return {"none": 0, "0": 0, "2d": 2, "2": 2, "3d": 3, "3": 3}.get(s, _to_int(s))


def _parse_epoch(ts):
    """Parse an exiftool ('Y:m:d H:M:S') or ISO timestamp to epoch seconds; None if unparseable."""
    if not ts:
        return None
    s = str(ts).strip().rstrip("Z")
    for fmt in ("%Y:%m:%d %H:%M:%S.%f", "%Y:%m:%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def _row_time(r):
    """Comparable time for a row: video-relative seconds if known, else absolute epoch."""
    return r["t_s"] if r.get("t_s") is not None else _parse_epoch(r.get("ts"))


def _fill_relative_time(rows):
    """Populate t_s (video-relative s) from absolute-ts deltas where t_s is missing."""
    epochs = [_parse_epoch(r.get("ts")) for r in rows]
    t0 = next((e for e in epochs if e is not None), None)
    if t0 is not None:
        for r, e in zip(rows, epochs):
            if r.get("t_s") is None and e is not None:
                r["t_s"] = round(e - t0, 3)
    return rows


def _gps_rows_from_csv(path):
    """Parse a gopro-telemetry CSV (matches columns by name, tolerant of units/order)."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}

        def find(*tokens):
            for low, orig in cols.items():
                if all(t in low for t in tokens):
                    return orig
            return None

        c_lat = find("lat", "deg") or find("lat")
        c_lon = find("lon") or find("long")
        c_alt = find("alt", "[m]") or find("alt", "m")
        c_spd = find("2d") or find("speed")
        c_dop = find("dop")
        c_fix = find("fix")
        c_date, c_cts = cols.get("date"), cols.get("cts")
        if not (c_lat and c_lon):
            return rows

        for r in reader:
            lat, lon = _to_float(r.get(c_lat)), _to_float(r.get(c_lon))
            if lat is None or lon is None:
                continue
            cts = _to_float(r.get(c_cts)) if c_cts else None
            rows.append({"t_s": cts / 1000.0 if cts is not None else None,   # cts is in ms
                         "ts": r.get(c_date) if c_date else None,
                         "lat": lat, "lon": lon,
                         "alt": _to_float(r.get(c_alt)) if c_alt else None,
                         "speed": _to_float(r.get(c_spd)) if c_spd else None,
                         "dop": _to_float(r.get(c_dop)) if c_dop else None,
                         "fix": _to_int(r.get(c_fix)) if c_fix else None})
    return rows


def _gps_rows_from_gpx(path):
    """Parse a GPX 1.1 track (namespace-agnostic)."""
    rows = []
    root = ET.parse(path).getroot()
    name = lambda e: e.tag.rsplit("}", 1)[-1]
    for pt in (e for e in root.iter() if name(e) == "trkpt"):
        lat, lon = _to_float(pt.get("lat")), _to_float(pt.get("lon"))
        if lat is None or lon is None:
            continue
        fields = {name(ch): (ch.text or "").strip() for ch in pt}
        rows.append({"t_s": None, "ts": fields.get("time"),
                     "lat": lat, "lon": lon,
                     "alt": _to_float(fields.get("ele")), "speed": None,
                     "dop": _to_float(fields.get("hdop")),
                     "fix": _fix_to_int(fields.get("fix"))})
    return rows


def _gps_rows_from_json(path):
    """Parse gopro-telemetry native JSON; handles GPS9 (9 values) and GPS5 (5 values)."""
    with open(path) as fh:
        data = json.load(fh)
    samples = []
    for dev in (data.values() if isinstance(data, dict) else []):
        streams = dev.get("streams", {}) if isinstance(dev, dict) else {}
        for sname, stream in streams.items():
            if sname.upper().startswith("GPS"):
                samples = stream.get("samples", [])
                break
        if samples:
            break

    rows = []
    for s in samples:
        val = s.get("value")
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            continue
        lat, lon = _to_float(val[0]), _to_float(val[1])
        if lat is None or lon is None:
            continue
        cts = s.get("cts")
        rows.append({"t_s": cts / 1000.0 if isinstance(cts, (int, float)) else None,
                     "ts": s.get("date"),
                     "lat": lat, "lon": lon,
                     "alt": _to_float(val[2]) if len(val) > 2 else None,
                     "speed": _to_float(val[3]) if len(val) > 3 else None,
                     "dop": _to_float(val[7]) if len(val) > 7 else None,   # GPS9 only
                     "fix": _to_int(val[8]) if len(val) > 8 else None})    # GPS9 only
    return rows


def gps_rows_from_sidecar(path):
    """Dispatch a GPS sidecar to the right parser by extension."""
    ext = os.path.splitext(path)[1].lower()
    parser = {".csv": _gps_rows_from_csv, ".gpx": _gps_rows_from_gpx,
              ".json": _gps_rows_from_json}.get(ext)
    if parser is None:
        print(f"  unsupported GPS sidecar '{ext}' (use .csv/.gpx/.json).")
        return []
    return parser(path)


def _exiftool_version():
    try:
        return subprocess.run(["exiftool", "-ver"], capture_output=True, text=True).stdout.strip()
    except Exception:                                           # noqa: BLE001
        return "?"


def gps_rows_from_exiftool(path):
    """GPS via exiftool, GPS9-aware (groups Doc<N>: sub-documents).

    HERO11/12/13 use the GPS9 stream, decoded only by exiftool >= ~13.0 - newer than the
    version in Ubuntu's apt (12.76). Update from exiftool.org if no GPS is found.
    """
    if shutil.which("exiftool") is None:
        print("  exiftool not found - skipping auto GPS (pass --gps-file instead).")
        return []
    ver = _exiftool_version()
    try:
        raw = subprocess.run(
            ["exiftool", "-ee", "-n", "-j", "-G3", "-api", "largefilesupport=1", path],
            capture_output=True, text=True, timeout=900)
        data = json.loads(raw.stdout or "[]")
    except Exception as exc:                                     # noqa: BLE001
        print(f"  exiftool GPS extraction failed ({exc}); pass --gps-file instead.")
        return []

    rows = []
    for obj in data:
        docs = {}                                               # group keys by family-3 doc prefix
        for key, val in obj.items():
            doc, _, tag = key.partition(":") if ":" in key else ("Main", "", key)
            docs.setdefault(doc, {})[tag] = val
        for doc, tags in docs.items():
            if doc == "Main":                                   # skip the file-level summary coordinate
                continue
            lat, lon = _to_float(tags.get("GPSLatitude")), _to_float(tags.get("GPSLongitude"))
            if lat is None or lon is None:
                continue
            spd = _to_float(tags.get("GPSSpeed"))               # exiftool reports GPS9 speed in km/h
            rows.append({"t_s": _to_float(tags.get("SampleTime")),
                         "ts": tags.get("GPSDateTime"),
                         "lat": lat, "lon": lon,
                         "alt": _to_float(tags.get("GPSAltitude")),
                         "speed": spd / 3.6 if spd is not None else None,   # -> m/s, like the sidecar
                         "dop": _to_float(tags.get("GPSDOP") or tags.get("GPSHPositioningError")),
                         "fix": _fix_to_int(tags.get("GPSMeasureMode"))})   # GPS9 fix (value 8)
    if not rows:
        print(f"  exiftool {ver} found no GPS. HERO11/12/13 use GPS9, which needs "
              f"exiftool >= ~13.0 (apt ships 12.76). Update it, or pass --gps-file.")
    return rows


# --- GPS output --------------------------------------------------------------
def _haversine_m(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp, dl = math.radians(b["lat"] - a["lat"]), math.radians(b["lon"] - a["lon"])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(x)))


def _track_distance_m(rows, min_dt=1.0):
    """Approx distance, decimated to ~1 Hz so GPS jitter is not integrated at 10 Hz."""
    use, last_t = [], None
    for r in rows:
        t = _row_time(r)
        if t is None or last_t is None or (t - last_t) >= min_dt:
            use.append(r)
            if t is not None:
                last_t = t
    if len(use) < 2:
        use = rows
    return sum(_haversine_m(use[i - 1], use[i]) for i in range(1, len(use)))


def _round_row(r):
    out = dict(r)
    for k, nd in (("t_s", 3), ("lat", 7), ("lon", 7), ("alt", 2), ("speed", 3), ("dop", 2)):
        if out.get(k) is not None:
            out[k] = round(out[k], nd)
    return out


def write_gps(rows, out_dir, min_fix):
    """Filter by fix quality, write gps.csv, return (n_kept, n_dropped, distance_m)."""
    if not rows:
        return 0, 0, 0.0
    _fill_relative_time(rows)                                   # derive t_s from ts when missing
    if min_fix and min_fix > 0:
        kept = [r for r in rows if r.get("fix") is None or r["fix"] >= min_fix]
    else:
        kept = list(rows)
    dropped = len(rows) - len(kept)

    with open(os.path.join(out_dir, "gps.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GPS_COLUMNS)
        w.writeheader()
        for r in kept:
            rr = _round_row(r)
            w.writerow({k: rr.get(k) for k in GPS_COLUMNS})
    return len(kept), dropped, _track_distance_m(kept)


# --- Entry point -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GoPro video -> frames (+ GPS).")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="./ingest_out")
    ap.add_argument("--every-sec", type=float, default=0.5, help="Seconds between extracted frames.")
    ap.add_argument("--start", type=float, default=0.0, help="Start time (s).")
    ap.add_argument("--end", type=float, default=None, help="End time (s).")
    ap.add_argument("--longest-side", type=int, default=None, help="Downscale longest side to N px.")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--gps-file", default=None,
                    help="GPS sidecar from gopro-telemetry (.csv/.gpx/.json); overrides exiftool.")
    ap.add_argument("--gps-min-fix", type=int, default=2,
                    help="Drop GPS samples with fix < this (0=keep all; GPS9: 2=2D, 3=3D).")
    ap.add_argument("--probe", action="store_true", help="Only inspect the video, do not extract.")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"No such file: {args.video}")
    if args.gps_file and not os.path.isfile(args.gps_file):
        sys.exit(f"No such GPS file: {args.gps_file}")

    info = probe_video(args.video)
    print(f"Video: {info['width']}x{info['height']}  {info['fps']:.3f} fps  "
          f"{info['duration_s']:.1f} s  ({info['n_frames']} frames)")
    if args.probe:
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"Extracting frames every {args.every_sec}s"
          + (f" in [{args.start},{args.end}]s" if args.end is not None else "") + " ...")
    saved, _ = extract_frames(args.video, args.out, args.every_sec, args.start, args.end,
                              args.longest_side, args.jpeg_quality, args.max_frames)
    print(f"  saved {saved} frames -> {os.path.join(args.out, 'frames')}")

    print("Extracting GPS ...")
    if args.gps_file:
        rows = gps_rows_from_sidecar(args.gps_file)
        gps_source = os.path.basename(args.gps_file)
    else:
        rows = gps_rows_from_exiftool(args.video)
        gps_source = "exiftool"
    n_gps, dropped, dist = write_gps(rows, args.out, args.gps_min_fix)
    if n_gps:
        msg = f"  {n_gps} GPS fixes from {gps_source}, ~{dist:.0f} m"
        if dropped:
            msg += f" ({dropped} dropped: fix < {args.gps_min_fix})"
        print(msg + f" -> {os.path.join(args.out, 'gps.csv')}")
    else:
        print("  no GPS written (was GPS enabled? try --gps-file with a gopro-telemetry export).")

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"video": os.path.basename(args.video), **info, "frames_saved": saved,
                   "every_sec": args.every_sec, "clip_start_s": args.start, "clip_end_s": args.end,
                   "gps_source": gps_source if n_gps else None, "gps_fixes": n_gps,
                   "gps_dropped": dropped, "gps_distance_m": round(dist, 1) if dist else None},
                  fh, indent=2)
    print(f"Summary -> {os.path.join(args.out, 'summary.json')}")
    if saved < 30:
        print("NOTE: few frames - fine for a quick smoke test; capture more for real reconstruction.")


if __name__ == "__main__":
    main()