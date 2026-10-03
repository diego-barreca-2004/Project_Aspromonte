# Capture, compute environment and elevation data

## Capture configuration

The settings favour a temporally consistent reconstruction and limit rolling-shutter and
motion artefacts. Fixed exposure is essential.

| Setting | Value | Rationale |
|---------|-------|-----------|
| Resolution / aspect | 5.3K, 8:7, Wide | Largest sensor area and field of view. |
| Frame rate | 30 fps | Enough overlap at walking pace. |
| Shutter | 1/400 s | Sharp frames; suppresses rolling-shutter wobble. |
| Anti-flicker | 50 Hz | Mains frequency in Italy and the EU. |
| Stabilisation | HyperSmooth off | Warping breaks the rigid camera model. |
| Horizon lock | off | Same reason: no per-frame reprojection. |
| White balance | locked (5500 K) | Consistent colour across frames. |
| ISO | 100–800 | Limits noise while keeping exposure stable. |
| Colour profile | Flat | Preserves dynamic range. |

A chest mount damps vibration better than a handlebar mount on rough terrain. The lens
model for the Wide field of view is `OPENCV_FISHEYE`. HERO11 and later also record a
per-frame gravity vector (GPMF `GRAV`), read by `gopro_gravity.py`.

## Reference environment

- WSL2 (Ubuntu) on Windows 11
- NVIDIA RTX 5070 Ti Laptop (Blackwell, `sm_120`), 12 GB VRAM
- CUDA Toolkit 12.8; Python 3.12; PyTorch for CUDA 12.8 (`cu128`)
- COLMAP 3.9 built for CUDA arch 89 (dense MVS, [BUILD_COLMAP_CUDA.md](BUILD_COLMAP_CUDA.md))
- pycolmap 4.2 (CPU wheel) for `compare_mappers.py` and `gopro_gravity.py --database`
- ExifTool ≥ 13.0 for the GPS9 and GRAV telemetry streams

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
```

## Notes

- **CUDA toolkit on WSL.** Install it from NVIDIA's `wsl-ubuntu` apt repository, which
  provides the toolkit without a Linux GPU driver (the Windows host provides the driver;
  a Linux driver breaks CUDA passthrough):

  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-8
  ```

- **3DGS on Blackwell.** Use a `cu128` PyTorch and build the CUDA submodules against it,
  not the repository's pinned conda environment. If the rasterizer fails on `uint32_t` /
  `uintptr_t`, add `#include <cstdint>` to
  `diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`.
- **COLMAP SIFT on CPU.** GPU SIFT is slow or unreliable on recent GPUs under WSL;
  `run_colmap.py` defaults to CPU (`--use-gpu 1` to override).
- **Dense MVS on Blackwell.** Built for arch 120, `patch_match_stereo` silently produces
  empty depth maps on RTX 50xx. Build with `-DCMAKE_CUDA_ARCHITECTURES=89` (PTX-JIT to
  `sm_120`) and add `#include <memory>` to `src/colmap/image/line.cc` and
  `src/colmap/mvs/workspace.h` for GCC 13.
- **VRAM.** On 12 GB, train 3DGS with `--data_device cpu`; add `-r 2` if densification
  still runs out of memory.

## Elevation data

Consumer GPS limits absolute accuracy to metres. For sub-metre registration the ground of
the model is aligned by ICP to an open LiDAR DTM in the same CRS (`georef_splat.py`).

- PST LiDAR DTM (MASE), up to 1 m, CC BY 4.0: [gn.mase.gov.it](https://gn.mase.gov.it)
- Regione Calabria DTM, 5 m: [geoportale.regione.calabria.it/opendata](http://geoportale.regione.calabria.it/opendata)
- TINITALY DTM, 10 m, nationwide (INGV): [tinitaly.pi.ingv.it](http://tinitaly.pi.ingv.it)

The study area is in UTM zone 33N (EPSG:32633), the default CRS of the scripts.
`dtm_merge_reproject.py` mosaics the downloaded PST tiles (WGS84 GeoTIFF) and reprojects
them in one step.
