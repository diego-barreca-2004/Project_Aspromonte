# Pipeline: from GoPro video to a georeferenced splat

Each stage writes into a per-segment working directory (`seg01/` in the examples). The
same front-end (steps 1–3, 5–6) feeds both the navigable splat (Track 1) and the change
detection (Track 2, [CHANGE_DETECTION.md](CHANGE_DETECTION.md)).

## Scripts

| File | Stage | Description |
|------|-------|-------------|
| `calibrate_camera.py` | Calibration | ChArUco fisheye calibration; writes intrinsics and distortion as JSON. |
| `assets/GoPro_Calibration.pdf` | Calibration | Printable ChArUco board (9×6 squares, `DICT_4X4_50`), matching the script defaults. |
| `ingest_gopro.py` | Ingestion | GoPro video to frames plus a per-frame `gps.csv` (HERO13 GPS9 telemetry). |
| `gopro_gravity.py` | Ingestion | Per-frame gravity direction from the GPMF `GRAV` stream, as CSV or as COLMAP pose priors. |
| `run_colmap.py` | SfM | COLMAP wrapper: fisheye-aware, optional calibration injection, CPU by default. |
| `compare_mappers.py` | SfM | Incremental vs global (GLOMAP) mapping, with and without gravity priors; see [SFM_COMPARISON.md](SFM_COMPARISON.md). |
| `geo_align.py` | Georeferencing | GPS to world similarity transform (robust Umeyama fit, UTM). |
| `dtm_merge_reproject.py` | Georeferencing | Mosaic the open LiDAR DTM tiles and reproject them to UTM. |
| `georef_splat.py` | Georeferencing | Apply the Sim3 to the splat, refine onto the DTM by ICP, write the UTM splat and a recentred `view.ply`. |
| `georef_cloud.py` | Change detection | Apply the Sim3 to a dense MVS cloud with floater removal; float64 UTM output. |
| `run_m3c2.py` | Change detection | Scripted M3C2 between two epochs (py4dgeo), with ICP, cropping and stable-patch statistics. |
| `dsm_change.py` | Change detection | Compact-object detection by DSM differencing on bare ground inside the trail corridor. |
| `run_pipeline.py`, `pipeline.conf` | Orchestration | One command per epoch, with QC gates and a shared reconstruction contract. |
| `tests/` | Tests | Synthetic ground-truth tests for `run_m3c2.py`, `dsm_change.py` and the gravity-levelled georeferencing. |
| `scripts/` | Reports and figures | `mapper_report.py` and `bend_check.py` aggregate the SfM comparison; `change_map_figure.py` and `render_cloud_gif.py` draw the README figures. |

COLMAP and the Inria 3DGS code are not vendored; they are installed separately
([ENVIRONMENT.md](ENVIRONMENT.md)).

## Step by step

**1. Calibrate the camera** (once per lens and setting). Print
`assets/GoPro_Calibration.pdf` and film it from varied angles and distances:

```bash
python3 calibrate_camera.py --video calib.mp4 --out ./calib_out
```

The model is `OPENCV_FISHEYE` (Kannala–Brandt), the appropriate one for the GoPro Wide
field of view.

**2. Extract frames and GPS** from a survey clip:

```bash
python3 ingest_gopro.py --video seg01.mp4 --out ./seg01 --every-sec 0.2 --longest-side 1600
```

**3. Structure-from-Motion** (CPU SIFT, sequential matching; the calibration is injected as
fixed intrinsics):

```bash
python3 run_colmap.py --images ./seg01/frames --out ./seg01/colmap \
        --calibration ./calib_out/calibration_fisheye.json
```

**4. Train 3D Gaussian Splatting** with the Inria implementation:

```bash
python3 train.py -s ./seg01/colmap/undistorted -m ./seg01/gs_output --data_device cpu
# output: ./seg01/gs_output/point_cloud/iteration_30000/point_cloud.ply
```

**5. Georeference from GPS.** A robust Umeyama fit replaces COLMAP's `model_aligner`, which
is unstable on near-linear walking tracks:

```bash
python3 geo_align.py --gps ./seg01/gps.csv \
        --images ./seg01/colmap/undistorted/images \
        --model  ./seg01/colmap/undistorted/sparse/0 \
        --out    ./seg01/colmap/geo_transform.txt
```

A GPS-only fit leaves the roll about the walking direction poorly constrained on a
straight track. [SFM_COMPARISON.md](SFM_COMPARISON.md) measures this and evaluates a
gravity-levelled alternative.

**6. Prepare the DTM** (mosaic and reproject the downloaded LiDAR tiles; sources in
[ENVIRONMENT.md](ENVIRONMENT.md)):

```bash
python3 dtm_merge_reproject.py --in ./dtm_tiles --out aspromonte_dtm_utm33n.tif
```

**7. Georeference the splat** (Sim3, then ICP of the ground onto the DTM):

```bash
python3 georef_splat.py \
        --ply ./seg01/gs_output/point_cloud/iteration_30000/point_cloud.ply \
        --transform ./seg01/colmap/geo_transform.txt \
        --dtm aspromonte_dtm_utm33n.tif \
        --out ./seg01/gs_output/point_cloud_utm_icp.ply
```

The ICP runs in Python and prints the ground-to-DTM residual. Without `--dtm` (and step 6)
the splat is georeferenced by the Sim3 alone. One run writes two files:

- `--out`: the splat in absolute UTM, for GIS and CloudCompare.
- `view.ply` (beside `--out`, with a `view.ply.offset.txt` sidecar): the same splat
  recentred on a local origin, for WebGL viewers such as the
  [SuperSplat editor](https://superspl.at/editor), which cannot draw absolute UTM
  magnitudes. The data are Z-up and not re-oriented: set Rotation X = 90 on import.
  `--clip-dtm <m>` drops floaters more than `<m>` metres from the DTM from the view only.
  `--view-out` changes its path and `--no-view` skips it.

## One command per epoch: `run_pipeline.py`

Steps 2–7 (except the DTM download) run as one command, driven by `pipeline.conf`:

```bash
python3 run_pipeline.py --config pipeline.conf --video seg01.mp4 --workdir seg01
```

QC gates stop the run with a message at the points that have failed silently before: wrong
ingest resolution, thin SfM registration, a weak GPS fit, the Blackwell zero-fused-points
bug, a mislocated georeferencing. Each stage is skipped if its output exists; `--force`
re-runs it, `--from`/`--to` restrict the range, `--dry-run` prints the commands.
`splat=true` in `pipeline.conf` adds 3DGS training; otherwise the run stops at the
georeferenced dense cloud (Track 2).
