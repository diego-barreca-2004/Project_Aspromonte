# Building COLMAP with CUDA for Blackwell (sm_120) and running dense MVS

This guide compiles a CUDA-enabled COLMAP that **actually works on RTX 50xx
(Blackwell)** and runs the dense Multi-View Stereo for Track 2. It encodes several
traps that cost real debugging time — read the Blackwell section before building.

## Why this is needed

The dense MVS step `patch_match_stereo` is **CUDA-only** (no CPU implementation).
The project's default COLMAP 3.9.1 is CPU-only, so it can run SfM but cannot produce
the dense point clouds that the M3C2 change-detection pipeline differentiates.
`image_undistorter` and `stereo_fusion` run on CPU; only `patch_match_stereo` needs
the GPU.

---

## ⚠️ The Blackwell trap that matters most — READ FIRST

On RTX 50xx (Blackwell, compute capability **sm_120**, i.e. CUDA compute ≥ 100),
COLMAP's PatchMatch CUDA kernels are **miscompiled if you build for arch 120**.
The failure is silent and misleading:

- SfM works, `patch_match_stereo` runs, the GPU spins (you see `Sweep 1/2/3/4`),
- but the kernels produce **random-noise depth maps and empty normal maps**,
- so `stereo_fusion` yields **0 points** with:
  `Could not fuse any points ... filtering must be enabled for the last call to patch match stereo`.

This is a **known COLMAP bug on compute ≥ 100** (GitHub issues #3514, #4090, #3186),
**not** a settings problem. Things that look like the cause but are NOT:
`depth_min`/scale (COLMAP units are unscaled, but fusion's `max_depth_error` is
*relative*, so scale is irrelevant), low-texture terrain, and fusion thresholds.
Do not waste time tuning those — none of them fix a noise depth map.

**THE FIX:** build with `-DCMAKE_CUDA_ARCHITECTURES=89`, **not 120**. The compiled
sm_89 PTX is JIT-compiled to sm_120 by the driver at runtime, which sidesteps the
broken sm_120 native codegen path. Confirmed working on RTX 5090; on some *mobile*
50xx parts it may still fail — see Troubleshooting for fallbacks.

---

## Target environment

- WSL2, Ubuntu 24.04 (GCC 13)
- RTX 50xx (Blackwell, sm_120), 12 GB VRAM
- CUDA Toolkit **12.8** at `/usr/local/cuda-12.8` (Blackwell needs ≥ 12.8)
- CMake ≥ 3.28, Ninja

## Step 1 — Verify the toolkit and GPU

Point the shell at CUDA 12.8 (add to `~/.bashrc` so every build/run sees it):

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
```

```bash
nvcc --version     # must report "release 12.8" (NOT 12.0 — that's the apt toolkit)
nvidia-smi         # must list the RTX 50xx; "CUDA Version" top-right must be ≥ 12.8
```

The apt package `nvidia-cuda-toolkit` is CUDA 12.0 and rejects sm_120. If
`nvcc --version` shows 12.0, the apt toolkit is shadowing 12.8 — fix `PATH` first.

## Step 2 — Install build dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake ninja-build build-essential \
  libboost-program-options-dev libboost-graph-dev libboost-system-dev \
  libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
  libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
  libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev
```

(Ubuntu 24.04 ships Ceres 2.x and Boost 1.83, both fine for COLMAP 3.9.1.)

## Step 3 — Clone COLMAP

Pin the same version used for the sparse reconstruction, for consistency with Track 1:

```bash
cd ~
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout 3.9.1
```

## Step 4 — Patch the GCC 13 / missing-`<memory>` errors

COLMAP 3.9.1 does not compile cleanly on Ubuntu 24.04 (GCC 13): two files use
`std::unique_ptr` without including `<memory>` (older GCC pulled it in transitively,
GCC 13 does not). Apply both patches before configuring:

```bash
sed -i '0,/#include/s//#include <memory>\n&/' src/colmap/image/line.cc
sed -i '0,/#include/s//#include <memory>\n&/' src/colmap/mvs/workspace.h
```

**General recipe** if the build later stops with another `'X' is not a member of 'std'`:
the `FAILED:` line names the offending `.cc`/`.h`, and the `note: 'X' is defined in
header '<Y>'` line names the header to add. Fix the **header** if the error originates
in an included `.h` (cures all `.cc` that include it), then re-run `ninja`:

```bash
sed -i '0,/#include/s//#include <Y>\n&/' src/colmap/<path>/<file>
```

Common culprits: `<memory>` (`unique_ptr`/`shared_ptr`/`make_unique`),
`<cstdint>` (`uint8_t`/`int64_t`), `<algorithm>`, `<limits>`.

## Step 5 — Configure with CUDA + arch 89 (NOT 120)

```bash
mkdir build && cd build
cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ENABLED=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

Confirm the output says **`Enabling CUDA support (version: 12.8.93, archs: 89)`**.
If it says `archs: 120`, stop — you will get the silent noise-depth-map bug.

Ignorable noise in the configure log: `CMP0167`/FindBoost warnings and
`SQLite::SQLite3 is deprecated` warnings (both "for project developers").

## Step 6 — Build and install

```bash
ninja
sudo ninja install
```

Build takes ~15–40 min (CUDA kernels compile from scratch). If a compile job is
killed for RAM (`Killed` / signal 9), lower parallelism — it resumes, no full rebuild:

```bash
ninja -j4    # or -j2
```

The binary installs to `/usr/local/bin/colmap`, replacing any CPU-only build (fine —
the CUDA build is a superset). Confirm CUDA libs were installed:
`libcolmap_mvs_cuda.a` and `libcolmap_sift_gpu.a` should appear in the install log.

Verify:

```bash
which colmap                    # /usr/local/bin/colmap
colmap -h 2>&1 | head -n 3      # COLMAP 3.9.1
```

> **JIT note (arch-89 build):** the *first* CUDA kernel launch JIT-compiles the PTX to
> sm_120 — a few extra seconds on the first `Initialization`, then cached in
> `~/.nv/ComputeCache`. This is local (no network) and not a hang.

---

## Step 7 — Dense MVS

### 7a. Pick the RIGHT model

A COLMAP run can leave several sub-models. Use the one with **full image coverage**,
not a small spurious fragment. Check each:

```bash
SEG=~/Progetto_Aspromonte/seg01
colmap model_analyzer --path $SEG/colmap/sparse/0
colmap model_analyzer --path $SEG/colmap/sparse/1
colmap model_analyzer --path $SEG/colmap/undistorted/sparse/0
```

Read the `Images:` line. For seg01 the good model has **603 images / 191218 points**
(present in both `sparse/1` raw and `undistorted/sparse/0`); `sparse/0` is a 2-image
fragment — ignore it. Use the model the splat and the Sim3 (`geo_transform.txt`) were
built from, so the transform applies 1:1 to the dense cloud.

### 7b. Undistort into a dense workspace

Two valid inputs — **never double-undistort**:

- **Already-undistorted model** (recommended; matches the splat/Sim3 model): pass the
  undistorted images *and* model together. `image_undistorter` is then effectively an
  identity that only builds the `stereo/` workspace.

  ```bash
  SEG=~/Progetto_Aspromonte/seg01
  DENSE=$SEG/colmap/dense
  rm -rf $DENSE
  colmap image_undistorter \
    --image_path  $SEG/colmap/undistorted/images \
    --input_path  $SEG/colmap/undistorted/sparse/0 \
    --output_path $DENSE \
    --output_type COLMAP
  ```

- **Raw (still-fisheye) model**: pass the *original* frames (`$SEG/frames`) + the raw
  model (`sparse/1`); COLMAP undistorts once using the camera params it reads.

Sanity checks before patch-match:

```bash
ls $DENSE/stereo/fusion.cfg $DENSE/stereo/patch-match.cfg && echo OK
grep -c "" $DENSE/stereo/patch-match.cfg    # == 2 × number_of_images (e.g. 1206 for 603)
ls $DENSE/images | wc -l                    # == number_of_images
```

`image_undistorter` is also what writes `fusion.cfg`/`patch-match.cfg`; if fusion later
complains it can't find them, re-run this step.

### 7c. PatchMatch stereo — ONE clean pass

Run a **single** geometric pass with filtering on. Do **not** run `geom_consistency
false` then `true` on the same workspace — that leaves the depth maps in an unfiltered
intermediate state and fusion returns 0 points. If you ever need to re-run, wipe the
map **contents** first — delete what is *inside* the folders, not the folders
themselves. `patch_match_stereo` does not create these directories (`image_undistorter`
does), so removing them makes the run crash on the first write
(`Check failed: file.is_open() ... .photometric.bin`):

```bash
rm -rf $DENSE/stereo/depth_maps/* $DENSE/stereo/normal_maps/* $DENSE/stereo/consistency_graphs/*
```

(If you already deleted the folders, recreate them with
`mkdir -p $DENSE/stereo/{depth_maps,normal_maps,consistency_graphs}` or just re-run
`image_undistorter`.)

```bash
colmap patch_match_stereo \
  --workspace_path "$DENSE" \
  --workspace_format COLMAP \
  --PatchMatchStereo.gpu_index 0 \
  --PatchMatchStereo.max_image_size 1000 \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.filter true
```

Healthy signs: `Configuration has N problems` with N = number of images (hundreds, not
2); each view's `src_image_idxs` lists ~10–20 source images (not 1); the GPU is busy in
`nvidia-smi`.

### 7d. Fuse

```bash
colmap stereo_fusion \
  --workspace_path "$DENSE" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$DENSE/fused.ply"

ls -lh "$DENSE/fused.ply"
```

A healthy `fused.ply` is tens/hundreds of MB. A few KB = empty (see Troubleshooting).
Per-image `0 points` mid-list can be normal (pixels already consumed by neighbouring
views); per-image `0 points` for the *first* images is not.

### Parameters that are now LAW for epoch 2

The dense baseline of epoch 1 and epoch 2 **must be reconstructed identically**, or
M3C2 measures pipeline differences instead of terrain change ("like-with-like"):

- `--PatchMatchStereo.max_image_size 1000`
- `--PatchMatchStereo.geom_consistency true`
- `--PatchMatchStereo.filter true`
- `stereo_fusion --input_type geometric` (default fusion thresholds)

If you later want higher detail for thesis figures, re-run **both** epochs at the
higher resolution. Do not mix.

### Timing / resolution

`geom_consistency true` runs **two full passes** over all frames (photometric, then
geometric), so ~2× the single-pass cost. On 603 frames: ≈ 4 h at 1500 px, ≈ 1.8 h at
1000 px. 1000 px is plenty for decimetre-scale structural change (fallen trees, washout,
erosion) and 603 frames over ~142 m is heavy overlap. A bigger future lever is
subsampling frames (603 is redundant for dense), but it is more invasive — not needed now.

---

## Troubleshooting

- **0 points / "filtering must be enabled" on Blackwell** — the arch bug. Rebuild with
  `-DCMAKE_CUDA_ARCHITECTURES=89` (Step 5). This is the whole reason this guide exists.
- **`no kernel image available for execution on the device`** after an arch-89 build —
  the binary lacks runnable code for sm_120. Reconfigure with PTX-only so the driver is
  forced to JIT: `-DCMAKE_CUDA_ARCHITECTURES=89-virtual`, then rebuild/reinstall.
- **Still 0 points after arch 89** (seen on some mobile 50xx) — build the upstream fix
  branch instead of 3.9.1: the root cause is a non-trivially-copyable argument passed to
  a CUDA kernel, fixed on `user/jsch/fix-non-trivially-copyable-cuda-arg`
  (`git fetch origin user/jsch/fix-non-trivially-copyable-cuda-arg && git checkout FETCH_HEAD`),
  or use a recent COLMAP release where it has merged. Re-apply any needed `<memory>`
  patches for your compiler.
- **patch_match OOM** — lower `--PatchMatchStereo.max_image_size` (1000 → 800 → 640).
- **`nvcc --version` is 12.0** — apt toolkit shadowing 12.8 in `PATH` (Step 1).
- **`nvcc fatal: unsupported host compiler`** — host GCC newer than CUDA 12.8 allows;
  add `-DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"` or point
  `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12`.
- **patch_match runs but no GPU activity** — binary built CPU-only; recheck Step 5
  reported CUDA `ON`.

---

## Next step — georeference into UTM (Track 2, Phase 1)

`fused.ply` is in COLMAP's arbitrary frame. `image_undistorter` does not move the 3D
coordinate system, so the same Sim3 from `geo_transform.txt` (scale, R, t) applies 1:1.
This is done by `georef_cloud.py`:

```bash
python georef_cloud.py \
  seg01/colmap/dense/fused.ply \
  seg01/colmap/geo_transform.txt \
  seg01/colmap/dense/fused_utm.ply
```

The script (1) removes MVS floaters with iterative SOR, (2) applies
`X_utm = scale * R @ X + t`, and (3) writes **float64** x/y/z — mandatory, since
float32 would quantise the ~4.2e6 UTM coordinates to ~0.25 m. SOR parameters
(`--sor-iters 2 --sor-k 20 --sor-std 2.0`) are part of the like-with-like contract:
identical for both epochs.

Validate in CloudCompare (accept the Global Shift on load; it is display-only, the file
stays absolute): the cloud should sit at the correct UTM location and the vertical
extent should collapse to the real relief (tens of metres), not the floater-inflated raw
range. Reference run on seg01: 4,533,733 → 4,460,797 points after SOR, UTM extent
~192 × 127 × 49 m, centroid E≈558995 / N≈4214492. An optional ICP-to-DTM refinement
(as in `georef_splat.py`) can tighten vertical alignment, but for change detection the
critical thing is that both epochs pass through the *identical* transform + SOR.