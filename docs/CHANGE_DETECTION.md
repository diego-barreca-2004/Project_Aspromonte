# Track 2: georeferenced change detection

Track 2 reuses the front-end of [PIPELINE.md](PIPELINE.md) (capture, calibration, SfM, DTM
and the Sim3 in `geo_transform.txt`), replaces 3DGS with dense MVS, and compares epochs.
The Sim3 applies unchanged to the dense cloud, because `image_undistorter` does not move
the 3D frame.

## Dense reconstruction

`patch_match_stereo` is CUDA-only. On Blackwell GPUs COLMAP must be built with
`-DCMAKE_CUDA_ARCHITECTURES=89`, not 120: arch 120 miscompiles the PatchMatch kernels and
silently yields empty depth maps. Build and commands: [BUILD_COLMAP_CUDA.md](BUILD_COLMAP_CUDA.md).

All epochs are reconstructed with the same parameters, so that differences reflect the
terrain and not the pipeline:

| Stage | Parameter | Value |
|-------|-----------|-------|
| `patch_match_stereo` | `max_image_size` | `1000` |
| `patch_match_stereo` | `geom_consistency` | `true` |
| `patch_match_stereo` | `filter` | `true` |
| `stereo_fusion` | `input_type` | `geometric` (default thresholds) |
| `georef_cloud.py` (SOR) | `iters / k / std` | `2 / 20 / 2.0` |

## Georeferencing the dense cloud

`georef_cloud.py` removes MVS floaters (iterative statistical outlier removal), applies the
Sim3 and writes float64 absolute UTM; float32 would quantise the ~4.2 × 10⁶ m northings to
~0.25 m.

```bash
python3 georef_cloud.py ./seg01/colmap/dense/fused.ply \
        ./seg01/colmap/geo_transform.txt ./seg01/colmap/dense/fused_utm.ply
```

Reference run on `seg01`: 4,533,733 points, 4,460,797 after SOR (the removed 1.6 % held
~98 % of the bounding box).

## Two methods

**M3C2** (`run_m3c2.py`, py4dgeo) measures distributed surface change along local
normals. It crops both clouds to the mutual footprint, registers them (a CloudCompare ICP
matrix via `--icp`, or the built-in `--auto-icp`), picks core points, runs M3C2 and writes
the distance and significance map with stable-patch statistics:

```bash
python3 run_m3c2.py --ref seg01/colmap/dense/fused_utm.ply \
                    --cmp seg01_ep2/colmap/dense/fused_utm.ply \
                    --icp icp_ep2_to_ep1.txt --cc-shift -559000 -4214000 0 \
                    --patch 559010,4214480,3 --out m3c2_out
```

Run once with `--reg-error 0`, then again with the value suggested by the stable-patch
statistics, so that significance accounts for the residual registration error.

**DSM differencing** (`dsm_change.py`) targets compact objects on a near-horizontal
trail. Inside a corridor around the GPS track it grids both epochs (90th height percentile
per 5 cm cell), keeps cells that are bare ground in the epoch the change is measured
against, removes the median offset, and reports connected blobs above
max(10 cm, 4 × NMAD):

```bash
python3 dsm_change.py --ref seg01_ep2/colmap/dense/fused_utm.ply \
    --cmp seg01_ep3/colmap/dense/fused_utm.ply \
    --icp m3c2_ep3_vs_ep2_v2/icp_auto.txt --cc-shift -558997 -4214492 -123 \
    --track seg01_ep2/gps.csv --out dsm_ep3_vs_ep2
```

## Results on seg01

Three epochs of the same 152 m dirt road on an open dry-grass slope, walked with a
chest-mounted GoPro. Objects placed on the trail: a flat plasterboard sheet in epoch 2; a
black cylinder, a ball and a white bucket in epoch 3. Epoch 3 was filmed in low light,
epoch 2 in sun.

Noise on bare ground (NMAD):

| Pair | M3C2 | DSM difference |
|------|------|----------------|
| ep2 vs ep1 | 3.9 cm | 1.3 cm |
| ep3 vs ep2 | 6.3 cm | 4.0 cm |

Per object:

| Object | Outcome |
|--------|---------|
| Bucket | Reconstructed; found 8 cm from its expected position, 0.11 m², +27 cm. |
| Cylinder, ball | Not reconstructed by MVS at `max_image_size` 1000. |
| Plasterboard (~1.3 cm thick) | Below the detection floor. |

Automatic detection with default parameters is **not yet discriminative**: ep3 vs ep2
gives 44 candidates with the bucket ranked 25th; the object-free control (ep2 vs ep1)
gives 48. False positives come from the sequence ends, an eastern stretch with non-rigid
drift, and grassy edges. Next steps: full-resolution MVS inside the corridor, piecewise
(non-rigid) registration, larger objects.

![DSM difference ep3 - ep2](../assets/change_map.png)

The figure is drawn by `scripts/change_map_figure.py` (command in its docstring).

## Designing a controlled change

Measure the noise floor first, on two captures of the unchanged scene. A test object should
sit 3–5 times above it: larger than the point spacing, taller than the noise. A rigid
object of known size gives quantified accuracy; moved soil or rock gives the paired
positive and negative signature closest to real erosion. Thin, dark or glossy objects
reconstruct poorly. Keep path, camera, time of day and weather the same across epochs.
