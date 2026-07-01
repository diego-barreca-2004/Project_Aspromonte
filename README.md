# Progetto Aspromonte

A reproducible **photogrammetry → 3D Gaussian Splatting** pipeline for building a
**georeferenced digital twin** of mountain-trail terrain in the Aspromonte massif
(Reggio Calabria, southern Italy).

The pipeline turns ordinary action-camera video into a navigable 3D Gaussian Splatting
(3DGS) reconstruction registered to real-world coordinates. It wraps the standard
open-source stack — [COLMAP](https://colmap.github.io) for Structure-from-Motion and the
[Inria 3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
implementation for radiance-field reconstruction — and adds custom tooling for GoPro
fisheye calibration, GPS-telemetry ingestion, and GPS-based georeferencing.

The project has **two tracks** that share the same capture, calibration, SfM and
georeferencing front-end:

- **Track 1 — navigable splat** (the pipeline described here): the georeferenced 3DGS
  reconstruction, for visualization and inspection. **Complete.**
- **Track 2 — georeferenced change detection**: dense MVS point clouds of the same site
  at two epochs, co-registered in the same UTM frame and differenced with **M3C2** to
  detect structural change (fallen trees, erosion, washout, landslide) and separate it
  from benign change (season, foliage, lighting). See *Track 2 — georeferenced change
  detection* below.

> Status: research / thesis work in progress. Track 1 runs end-to-end; Track 2's
> single-epoch dense pipeline (SfM → CUDA dense MVS → floater-cleaned UTM cloud) is
> working, and two-epoch change detection is in progress. Scaling to a full trail network
> is discussed under *Limitations & future work*.

## Pipeline

```
 GoPro video       frames + GPS         calibration              SfM               3D Gaussians           GPS alignment     georeferenced splat
 (HERO13 Wide) ──▶ ingest_gopro.py ──▶ calibrate_camera.py ──▶ run_colmap.py ──▶ 3DGS training     ──▶ geo_align.py  ──▶ georef_splat.py
                                        (ChArUco, fisheye)       (COLMAP)          (graphdeco-inria)      (GPS → Sim3)      (Sim3 + ICP → UTM)
```

1. **Capture** — GoPro HERO13, 5.3K 8:7 Wide, fixed exposure (see *Capture configuration*).
2. **Frame & GPS extraction** (`ingest_gopro.py`) — decode frames at a chosen rate and
   parse the HERO13 **GPS9** telemetry stream into a per-frame `gps.csv`.
3. **Camera calibration** (`calibrate_camera.py`) — ChArUco board, **OPENCV_FISHEYE**
   (Kannala–Brandt) model, the correct model for the GoPro Wide field of view.
4. **Structure-from-Motion** (`run_colmap.py`) — COLMAP feature extraction, sequential
   matching, mapping and undistortion, producing a pinhole workspace ready for 3DGS.
5. **3D Gaussian Splatting** — training with the Inria implementation (cloned separately).
6. **Georeferencing** (`geo_align.py`) — associate each frame with its GPS position and
   fit the similarity transform that maps the reconstruction into a metric CRS, using a
   robust Umeyama fit (COLMAP's `model_aligner` is unstable on near-linear walking tracks).
   Apply it to the splat — and optionally refine onto an open LiDAR DTM by ICP — in one
   self-contained step with `georef_splat.py`.

## Repository contents

| File | Stage | Description |
|------|-------|-------------|
| `calibrate_camera.py` | Calibration | ChArUco fisheye calibration; outputs intrinsics + distortion as JSON. |
| `GoPro_Calibration.pdf` | Calibration | Printable ChArUco board (9×6 squares, `DICT_4X4_50`) — matches `calibrate_camera.py`'s defaults. |
| `ingest_gopro.py` | Ingestion | GoPro video → frames + per-frame `gps.csv` (HERO13 GPS9 telemetry). |
| `run_colmap.py` | SfM | COLMAP wrapper (fisheye-aware, optional calibration injection, CPU by default). |
| `geo_align.py` | Georeferencing | GPS → world similarity transform (robust Umeyama fit, UTM). |
| `dtm_merge_reproject.py` | Georeferencing | Mosaic the open LiDAR DTM tiles and reproject them to UTM. |
| `georef_splat.py` | Georeferencing | Apply the Sim3 to the 3DGS splat, optionally refine onto the DTM by ICP, and write both the absolute-UTM splat and a recentred `view.ply` for SuperSplat — one self-contained step. |
| `georef_cloud.py` | Change detection | Apply the Sim3 to a dense MVS cloud with statistical-outlier floater removal, writing a **float64** absolute-UTM cloud for M3C2 (Track 2). |
| `BUILD_COLMAP_CUDA.md` | Build guide | Build COLMAP with CUDA for Blackwell (`sm_120`, arch-89 workaround) and run the dense MVS — required for Track 2. |

Third-party components (COLMAP, the Inria 3DGS code) are **not** vendored here — they are
installed/built separately as described below.

## Requirements

**Reference environment** (what this was developed and tested on):

- WSL2 (Ubuntu) on Windows 11
- NVIDIA RTX 5070Ti (Blackwell, compute capability `sm_120`), 12 GB VRAM
- CUDA Toolkit 12.8, NVIDIA driver supporting CUDA ≥ 12.8
- Python 3.12, PyTorch built for CUDA 12.8 (`cu128`)
- [COLMAP](https://colmap.github.io) ≥ 3.7 — CUDA build **required for Track 2** dense MVS
  (Blackwell needs an arch-89 build; see `BUILD_COLMAP_CUDA.md`), optional for Track 1

**Python packages:** `numpy`, `scipy`, `opencv-python`, `pyproj`, `rasterio`, `plyfile`
(the last four for the georeferencing scripts; `scipy` is used by `georef_cloud.py`).
`ExifTool` ≥ 13.0 is required by `ingest_gopro.py` to read the GPS9 telemetry stream.

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy scipy opencv-python pyproj rasterio plyfile
```

## Quick start

Each stage writes into a per-segment working directory (`seg01/` in the examples).

**1 — Calibrate the camera** (once per lens/setting; uses a ChArUco capture clip, can also be done after the second step):

```bash
python3 calibrate_camera.py --video calib.mp4 --out ./calib_out
```

Print `GoPro_Calibration.pdf` (a 9×6 ChArUco board, `DICT_4X4_50`) and film it while moving the
camera through varied angles and distances; the script's `--cols 9 --rows 6` defaults match this
board.

**2 — Extract frames and GPS** from a survey clip:

```bash
python3 ingest_gopro.py --video seg01.mp4 --out ./seg01 --every-sec 0.2 --longest-side 1600
```

**3 — Run Structure-from-Motion** (CPU by default; injects the calibration as initial intrinsics):

```bash
python3 run_colmap.py --images ./seg01/frames --out ./seg01/colmap \
        --calibration ./calib_out/calibration_fisheye.json
```

**4 — Train 3D Gaussian Splatting** (Inria implementation, cloned and built separately):

```bash
python3 train.py -s ./seg01/colmap/undistorted -m ./seg01/gs_output --data_device cpu
# output: ./seg01/gs_output/point_cloud/iteration_30000/point_cloud.ply
```

**5 — Georeference** the reconstruction from GPS:

```bash
python3 geo_align.py --gps ./seg01/gps.csv \
        --images ./seg01/colmap/undistorted/images \
        --model  ./seg01/colmap/undistorted/sparse/0 \
        --out    ./seg01/colmap/geo_transform.txt
# writes the similarity transform (scale + rotation + translation) to georeference the splat
```

**6 — Prepare the DTM** (mosaic + reproject the downloaded LiDAR tiles to UTM; tile sources
under *Georeferencing & elevation data*):

```bash
python3 dtm_merge_reproject.py --in ./dtm_tiles --out aspromonte_dtm_utm33n.tif
```

**7 — Georeference the splat** (apply the Sim3, refine onto the DTM by ICP, and write both
outputs in one run):

```bash
python3 georef_splat.py \
        --ply ./seg01/gs_output/point_cloud/iteration_30000/point_cloud.ply \
        --transform ./seg01/colmap/geo_transform.txt \
        --dtm aspromonte_dtm_utm33n.tif \
        --out ./seg01/gs_output/point_cloud_utm_icp.ply
```

The ICP runs in Python — no CloudCompare required — and prints the ground-to-DTM residual.
Omit `--dtm` (and step 6) for a Sim3-only georeferencing without refinement.

A single run writes **two outputs**:

- **`--out`** — the splat in **absolute UTM**, the canonical deliverable (GIS / CloudCompare,
  which absorbs the large coordinates with a global shift on load).
- **`view.ply`** (written beside `--out`, with a `view.ply.offset.txt` sidecar mapping local →
  UTM) — the same splat recentred on a local origin, for WebGL viewers like the browser-based
  [SuperSplat editor](https://superspl.at/editor), which cannot draw absolute UTM magnitudes.
  Change its path with `--view-out`, or skip it entirely with `--no-view`. The data are Z-up and
  the file is **not** re-oriented, so in SuperSplat set **Rotation X = 90** on import to view it
  level. Add `--clip-dtm <m>` to drop floaters farther than `<m>` metres (vertically) from the
  DTM **from the view only** — the UTM deliverable stays intact (requires `--dtm`).

## Track 2 — georeferenced change detection

Track 2 reuses the same front-end (capture, calibration, SfM, DTM, and the Sim3 in
`geo_transform.txt`) but replaces 3DGS with **dense MVS**, and compares two epochs. The
Sim3 applies 1:1 to the dense cloud because `image_undistorter` does not move the 3D
coordinate frame.

### Dense reconstruction (CUDA COLMAP)

Dense MVS (`patch_match_stereo`) is CUDA-only. On Blackwell GPUs, COLMAP must be built
with `-DCMAKE_CUDA_ARCHITECTURES=89` (**not** 120) — arch 120 miscompiles the PatchMatch
kernels and silently yields empty depth maps (0 fused points). The full build (with the
GCC-13 source patches) and the exact `image_undistorter` → `patch_match_stereo` →
`stereo_fusion` commands are in [`BUILD_COLMAP_CUDA.md`](BUILD_COLMAP_CUDA.md).

### Reconstruction contract (identical for both epochs)

For M3C2 to measure *terrain* change rather than pipeline differences, epoch 1 and
epoch 2 must be reconstructed with identical parameters ("like-with-like"):

| Stage | Parameter | Value |
|-------|-----------|-------|
| `patch_match_stereo` | `max_image_size` | `1000` |
| `patch_match_stereo` | `geom_consistency` | `true` |
| `patch_match_stereo` | `filter` | `true` |
| `stereo_fusion` | `input_type` | `geometric` (default thresholds) |
| `georef_cloud.py` (SOR) | `iters / k / std` | `2 / 20 / 2.0` |

### Georeference the dense cloud

`georef_cloud.py` removes MVS floaters (iterative SOR), applies the Sim3, and writes the
cloud in **float64** absolute UTM — float32 would quantise the ~4.2 × 10⁶ UTM coordinates
to ~0.25 m and defeat sub-metre change detection:

```bash
python3 georef_cloud.py \
        ./seg01/colmap/dense/fused.ply \
        ./seg01/colmap/geo_transform.txt \
        ./seg01/colmap/dense/fused_utm.ply
```

Open `fused_utm.ply` in CloudCompare (accept the global shift on load — display-only, the
file stays absolute). Reference run on `seg01`: 4,533,733 → 4,460,797 points after SOR
(the floaters were ~1.6 % of points but held ~98 % of the bounding box); UTM extent
≈ 192 × 127 × 49 m; centroid E ≈ 558995, N ≈ 4214492.

### Change-detection workflow (in progress)

1. Reconstruct a second epoch (same site, a controlled change) through the identical
   pipeline to `fused_utm2.ply`.
2. Both clouds are already coarsely aligned in UTM; a fine ICP between them removes the
   residual, then **M3C2** yields the change map.
3. Distinguish structural change from benign change — the original contribution.

**Level of detection.** Before introducing a change, run an *epoch-zero-zero*
repeatability test (two clouds of the unchanged scene) and measure the M3C2 standard
deviation on stable ground — that is the detection floor. A controlled-change object must
sit ~3–5× above it: larger than the point spacing and taller than the noise. A rigid
object of known size (a 20–40 cm box) gives quantified accuracy; moved soil/rock gives a
positive + negative signature closest to the real washout/erosion use case. Thin, dark or
glossy objects (twigs, leaves) reconstruct poorly and fall under the floor — avoid them.
Acquire both epochs with matching path, camera, time of day and weather to minimise the
lighting confound at capture.

## Capture configuration

Settings chosen for dense-forest dynamic range and to minimise rolling-shutter / motion
artefacts. **Fixed exposure is essential** for a temporally consistent reconstruction.

| Setting | Value | Rationale |
|---------|-------|-----------|
| Resolution / aspect | 5.3K, 8:7, Wide | Maximum sensor area and field of view. |
| Frame rate | 30 fps | Sufficient overlap at walking pace. |
| Shutter | 1/400 s | Sharp frames; suppresses rolling-shutter "jello". |
| Anti-flicker | 50 Hz | Mains frequency (Italy / EU). |
| Stabilisation | HyperSmooth **off** | Warping breaks the rigid pinhole model. |
| Horizon lock | **off** | Same reason — no per-frame reprojection. |
| White balance | locked (5500 K) | Consistent colour across frames. |
| ISO | 100–800 | Limit noise while keeping exposure stable. |
| Colour profile | Flat | Preserves dynamic range (grade uniformly later). |

A chest mount is recommended over a handlebar mount on rough terrain (body damping reduces
vibration). Lens model for the Wide field of view is **OPENCV_FISHEYE**.

## Notes on the compute environment

A few hard-won, hardware-specific findings, documented for reproducibility:

- **CUDA toolkit (WSL).** On WSL2, install the toolkit from NVIDIA's `wsl-ubuntu` apt repo,
  which gives you the toolkit **without** a Linux GPU driver (the driver is provided by the
  Windows host — installing a Linux one breaks WSL's CUDA passthrough). The
  `cuda-keyring_*.deb` is only the one-shot package that registers that repo; it is **not**
  part of this repository (it is `.gitignore`d):

  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get -y install cuda-toolkit-12-8
  ```
  See NVIDIA's *CUDA on WSL* guide for the canonical steps.

- **3DGS on Blackwell (sm_120):** use a `cu128` PyTorch in your venv and build the CUDA
  submodules against it — do **not** use the repo's pinned conda environment. If the
  rasterizer fails to compile with errors about `uint32_t` / `uintptr_t`, add
  `#include <cstdint>` to `diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`.
- **COLMAP SIFT runs on CPU here.** COLMAP's GPU SIFT (SiftGPU) is slow/unreliable on
  recent GPUs and under WSL; CPU feature extraction and matching are faster and more
  correct. `run_colmap.py` defaults to CPU (override with `--use-gpu 1`).
- **COLMAP dense MVS on Blackwell (sm_120):** `patch_match_stereo` is CUDA-only and, when
  built for arch 120, silently produces noise depth maps / 0 fused points on RTX 50xx (a
  known codegen bug). Build COLMAP with `-DCMAKE_CUDA_ARCHITECTURES=89` so the kernels run
  via PTX-JIT to `sm_120`, and add `#include <memory>` to `src/colmap/image/line.cc` and
  `src/colmap/mvs/workspace.h` for GCC 13. Full guide: `BUILD_COLMAP_CUDA.md`.
- **VRAM:** on 12 GB, train with `--data_device cpu` so source images stay in system RAM;
  drop to `-r 2` if you still hit out-of-memory during densification.

## Georeferencing & elevation data

`geo_align.py` gives the reconstruction correct **scale** and metre-level georeferencing
from the GoPro GPS track. Consumer GPS limits absolute accuracy to a few metres; for
sub-metre registration, reproject an open LiDAR Digital Terrain Model (DTM) to the same CRS
and refine with ICP on the ground portions of the model — `georef_splat.py` does this in Python
(step 7), so CloudCompare is not required.

Open elevation data for the area:

- **PST LiDAR DTM** (Ministero dell'Ambiente / MASE), up to 1 m, CC BY 4.0 —
  [gn.mase.gov.it](https://gn.mase.gov.it)
- **Regione Calabria** DTM 5 m — [geoportale.regione.calabria.it/opendata](http://geoportale.regione.calabria.it/opendata)
- **TINITALY** DTM 10 m, nationwide (INGV) — [tinitaly.pi.ingv.it](http://tinitaly.pi.ingv.it)

The study area sits in **UTM zone 33N** (EPSG:32633), the default target CRS in `geo_align.py`.
Downloaded PST tiles (WGS84 GeoTIFFs) can be mosaicked and reprojected to that CRS in one step
with `dtm_merge_reproject.py`.

## Limitations & future work

- **Capture geometry.** A straight, forward-facing walk gives little parallax off-axis,
  producing floating artefacts beside the trail. A weaving path and occasional lateral /
  look-around motion provide the multi-view coverage 3DGS needs.
- **Georeferencing accuracy.** GPS-only alignment is metre-level. Ground control points or
  RTK GPS would be required for survey-grade accuracy.
- **Scale.** A single segment is demonstrated. Scaling to a trail network calls for a
  different strategy — anchoring to existing georeferenced LiDAR/DTM rather than
  reconstructing terrain from scratch, selective high-fidelity capture, and faster global
  SfM ([GLOMAP](https://github.com/colmap/glomap)) — evaluated empirically against
  incremental COLMAP, which is more robust on repetitive forest texture.

## License

The code in this repository is released under the **MIT License** (see [`LICENSE`](LICENSE)).

This pipeline depends on third-party software distributed under its own terms — notably the
Inria 3D Gaussian Splatting code (non-commercial research license), COLMAP (BSD), and GLOMAP
(BSD-3). Those licenses govern their respective components.

## Acknowledgements

Built on the work of the [COLMAP](https://colmap.github.io),
[GLOMAP](https://github.com/colmap/glomap), and
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) projects, and
on OpenCV's ChArUco calibration tools. Open elevation data courtesy of MASE, Regione
Calabria, and INGV (TINITALY).