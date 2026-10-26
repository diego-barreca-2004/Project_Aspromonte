# Aspromonte: georeferenced trail reconstruction and change detection from a walking GoPro

Action-camera video of a mountain trail, turned into a georeferenced 3D Gaussian Splatting
model and into change maps between repeated passes, in UTM coordinates.

<p align="center">
  <img src="assets/splat_segment_render.jpg" width="80%" alt="Render of the 3DGS model of a 30 m stretch of the trail">
  <br><em>Rendered from the 3DGS model at one of its training camera poses
  (30 m stretch of the second pass, 121 frames at 3200 px).</em>
</p>

A GoPro HERO13 was walked three times along the same 152 m trail in the Aspromonte massif
(Calabria, Italy). This repository goes from the raw video and its telemetry (GPS and
gravity) to

- **Track 1:** a navigable 3D Gaussian Splatting (3DGS) model, georeferenced by GPS and
  refined by ICP onto an open 1 m LiDAR terrain model;
- **Track 2:** the passes reconstructed together in one co-aligned SfM model, a dense point
  cloud per pass in the same UTM frame, and change measured between passes both in the
  heights (DSM differencing with an a-contrario test) and in the images (the reference pass
  warped into the frames of the new one).

It wraps [COLMAP](https://colmap.github.io) (incremental and global SfM, dense MVS) and the
[Inria 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) code, and adds what
they do not cover: GoPro fisheye calibration, telemetry ingestion, robust GPS
georeferencing on near-linear tracks, gravity priors from the GoPro IMU, ICP onto the DTM,
co-alignment of repeated passes, and change detection whose thresholds come from an
object-free pair of passes.

<p align="center">
  <img src="assets/pipeline.svg" width="100%" alt="Pipeline overview">
</p>

## Quick start

One epoch, end to end (frames, SfM, georeferencing, dense cloud and, with `splat true`,
the 3DGS model), driven by the shared reconstruction contract in `pipeline.conf`:

```bash
python3 run_pipeline.py --config pipeline.conf --video seg01.mp4 --workdir seg01
```

Each stage resumes from existing outputs; `--from`/`--to` restrict the range, `--dry-run`
prints the plan. QC gates stop the run on the failures that are otherwise silent (thin
registration, weak GPS fit, empty dense output, mislocated georeferencing).

Compare passes (Track 2): reconstruct them in one model, then measure change against an
object-free pair of passes (here ep2 vs ep1), whose noise tails set the threshold:

```bash
python3 coalign.py --pass ep1=seg01_ep1 --pass ep2=seg01_ep2 --pass ep3=seg01_ep3 \
    --calibration calib_out/calibration_fisheye.json --out coalign/seg01
P=coalign/seg01/passes
# object-free pair: writes null_tails.json, the null of the other pairs
python3 image_change.py --ref-seg $P/ep1 --cmp-seg $P/ep2 --out img_ep2_vs_ep1
python3 image_change.py --ref-seg $P/ep2 --cmp-seg $P/ep3 \
    --null-tails img_ep2_vs_ep1/null_tails.json --out img_ep3_vs_ep2
# measure_change.py takes the same arguments and measures the change in the heights
```

The earlier per-pass workflow (`run_m3c2.py`, `dsm_change.py` after a rigid ICP) is in
[docs/CHANGE_DETECTION.md](docs/CHANGE_DETECTION.md).

Compare incremental and global SfM, with and without gravity priors
(requires `pycolmap>=4.2`):

```bash
python3 compare_mappers.py --epoch ./seg01_ep2 --video Attempt_2.MP4 \
    --calibration ./calib_out/calibration_fisheye.json \
    --dtm aspromonte_dtm_utm33n.tif --out ./sfm_compare/seg01_ep2
```

Individual stages and their options: [docs/PIPELINE.md](docs/PIPELINE.md).

## Setup

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
```

- [COLMAP](https://colmap.github.io) with CUDA for dense MVS. On Blackwell GPUs build it for
  arch 89: [docs/BUILD_COLMAP_CUDA.md](docs/BUILD_COLMAP_CUDA.md).
- The [Inria 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) code, cloned into
  `gaussian-splatting/`, for Track 1 only.
- ExifTool ≥ 13.0 for the GPS9 and GRAV telemetry streams.
- `pycolmap>=4.2` (CPU wheel) for `compare_mappers.py`, `scripts/mapper_report.py` and
  `scripts/bend_check.py`.

Tested on WSL2 (Ubuntu), RTX 5070 Ti Laptop (12 GB), CUDA 12.8, Python 3.12. Capture
settings and environment notes: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

Tests run on synthetic scenes with known ground truth:

```bash
for t in tests/test_*.py; do python3 "$t" || break; done
```

## Results

### Structure-from-Motion and georeferencing

Three passes of about 600 frames each, pycolmap 4.2.1 on CPU, one run per configuration:

| Mapper | Registered | Reprojection error | Time | Failures |
|--------|-----------|--------------------|------|----------|
| COLMAP incremental | all frames | 0.39–0.45 px | 436–655 s | none |
| COLMAP global (GLOMAP) | all frames | 0.38–0.45 px | 170–440 s | ep1 bent by up to 14° |
| global + GoPro gravity priors | all frames | 0.38–0.45 px | 188–253 s | none |

The bent ep1 model has a normal reprojection error; only the GPS residual (5.75 m) and
the disagreement with the GoPro gravity reveal it. A second random seed bends it at the
same place.

Rotation left between passes by the georeferencing (global model with gravity priors;
the incremental model gives the same values within 0.3°):

| Georeferencing | ep2 → ep1 | ep3 → ep2 | ep3 → ep1 | GPS residual (median) |
|----------------|-----------|-----------|-----------|-----------------------|
| GPS only, Sim3 (7 DoF) | 5.6° | 5.1° | 0.7° | 0.31–0.55 m |
| Gravity-levelled (4 DoF) | **0.6°** | **1.4°** | 1.1° | 0.95–1.33 m |

On a near-straight walk GPS does not constrain the roll about the direction of travel;
fixing the vertical with the GoPro gravity removes most of the tilt between passes, at
the cost of a larger GPS residual. Details: [docs/SFM_COMPARISON.md](docs/SFM_COMPARISON.md).

### 3D Gaussian Splatting

<p align="center">
  <img src="assets/splat_flythrough.gif" width="80%" alt="Fly-through of the georeferenced 3DGS model of epoch 2">
  <br><em>3DGS model of the whole second pass, rendered along the walked path.</em>
</p>

### Change detection

Reconstructed alone, each pass keeps its own non-rigid drift: after a rigid ICP the height
difference ep3 − ep2 on bare ground still bends by 10 cm along the track (NMAD 5.4 cm).
Reconstructed in one co-aligned model, the same difference is flat within 0.5 cm
(NMAD 0.95 cm), with no ICP.

Every threshold and design choice was made on the object-free pair ep2 vs ep1 and on
synthetic cylinders drawn into it, never on the placed objects. The test pair ep3 vs ep2
was then run once, with the thresholds frozen, at 1 false change per km of track:

| Object | Size | Heights (`measure_change.py`) | Images (`image_change.py`) |
|--------|------|-------------------------------|----------------------------|
| Bucket | 31 cm wide, 27 cm high | stated, 1st | not stated, ranked 1st |
| Ball | 22 cm | stated, 5th | not stated, ranked 3rd |
| Black cylinder | 7.4 cm wide, 17 cm high | missed | not stated, ranked 2nd |
| False changes | | 6 (39.5 per km) | 0 |

In the images the three objects are the three most significant changes of the whole track,
with no false change before them; the cylinder would be stated from 10.5 false changes per
km. The threshold at 1 per km is conservative because the object-free pair is noisier than
the test pair (median image score 0.134 against 0.094): on two pairs of a pass with itself,
quieter still, the same thresholds state no false change up to 100 per km. Details:
[docs/CHANGE_DETECTION.md](docs/CHANGE_DETECTION.md).

## Limitations

- **One segment, three passes.** All numbers come from a single 152 m trail on an open
  slope. Forest canopy, longer sequences and other cameras are untested.
- **Residual tilt.** Even after gravity levelling, passes differ by up to 1.5°, and the ICP
  onto the 1 m DTM still tilts each model by 1.6–3.2°. A misalignment between the GoPro
  IMU and the optical axis, a bias of the ICP on grass, or both could explain it; they
  have not been separated yet.
- **Absolute accuracy.** Consumer GPS gives metre-level georeferencing; the ICP onto the
  DTM brings the ground to 0.3–0.4 m RMS.
- **Drift between passes.** Co-alignment removes it for passes of one segment; passes
  reconstructed separately and aligned by one rigid ICP keep up to 10 cm of non-rigid drift.
- **Small objects.** A 7.4 cm dark cylinder covers one or two 5 cm cells and stays below the
  height noise of MVS at `max_image_size` 1000. In the images it ranks second but is not
  stated at 1 false change per km.
- **One object-free pair.** All thresholds come from a single pair, ep2 vs ep1, noisier
  than the test pair. A null pair recorded in the same session as the test pair
  (pass, pass, place the objects, pass) would match its noise; a new session is planned.
- **Splat views off the walked path.** A forward walk constrains the 3DGS model only near
  the camera path: renders from the walked path are sharp, while viewpoints away from it
  show floaters and a smeared background.
- **Licensing of Track 1.** The Inria 3DGS code is non-commercial; see below.

## License

The code in this repository is released under the MIT License ([LICENSE](LICENSE)).
Third-party components keep their own terms: COLMAP and GLOMAP (BSD), the Inria 3DGS code
(non-commercial research license). The Inria license applies to Track 1 only; Track 2
(COLMAP SfM and dense MVS plus the scripts here) depends only on permissive licenses.
Permissively licensed splat renderers also exist (e.g. `gsplat`, Apache-2.0).

## Acknowledgements

Built on [COLMAP](https://colmap.github.io), [GLOMAP](https://github.com/colmap/glomap),
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[py4dgeo](https://github.com/3dgeo-heidelberg/py4dgeo) and OpenCV. Elevation data: PST
LiDAR DTM (MASE, CC BY 4.0), Regione Calabria and TINITALY (INGV).
