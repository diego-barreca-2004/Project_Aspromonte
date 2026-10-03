# Incremental vs global SfM, and gravity-levelled georeferencing

`compare_mappers.py` asks two questions on the three seg01 epochs:

1. Does COLMAP's global mapper (GLOMAP, now part of COLMAP 4) match the incremental
   mapper on a walked trail sequence, and does the GoPro gravity vector help it?
2. Does gravity fix the tilt that GPS-only georeferencing leaves between epochs?

## Setup

All mappers run on the same features and matches. The script copies each epoch's
`database.db` (the original is never modified), sets the shared camera to the ChArUco
calibration with fixed intrinsics, and re-runs geometric verification with pycolmap 4.2.1
(CPU). Three mappers:

| Name | Mapper | Gravity |
|------|--------|---------|
| `incremental` | `pycolmap.incremental_mapping` | none |
| `global` | `pycolmap.global_mapping` (GLOMAP) | none |
| `global_gravity` | `pycolmap.global_mapping` | gravity-only pose prior per image, `rotation_averaging.use_gravity` |

**Gravity.** HERO11 and later cameras record a `GRAV` stream in the GPMF telemetry, one
unit vector per video frame. `gopro_gravity.py` reads it with ExifTool and interpolates it
at each extracted frame. On the HERO13 the vector is already in COLMAP camera axes
(x right, y down, z forward): against the camera rotations of an existing model the identity
mapping agrees to 1.5° (median), and every other signed axis permutation is worse by more
than 5°. No time offset is needed.

**Two georeferencings** of each model, from the per-frame GPS:

- *Sim3* (7 DoF), the robust Umeyama fit of `geo_align.py`;
- *gravity-levelled* (4 DoF): the model is first rotated so that the median gravity
  direction, mapped through the camera rotations, points down; GPS then fixes only
  heading, scale and translation.

**Measurements.** Registered images, 3D points, mean reprojection error, wall time;
the median GPS residual of each fit; the tilt of each fit against the GoPro gravity; the
rotation still needed by a point-to-plane ICP of the georeferenced sparse ground points
onto the 1 m LiDAR DTM; and the rotation between epochs. The last one is measured by a
point-to-plane ICP of one epoch's georeferenced sparse points onto another's, coarse to
fine (correspondence gate 20, 5, 2, 1, 0.5 m), since GPS-only georeferencings of different
days can be several metres apart. On a moved copy of a real cloud (5° rotation, 6.2 m
offset) it recovers both exactly. As a cross-check, the relative rotation is also computed
from the two per-epoch DTM-ICP rotations. The aggregation script is outside the
repository.

```bash
python3 compare_mappers.py --epoch ./seg01_ep2 --video Attempt_2.MP4 \
    --calibration ./calib_out/calibration_fisheye.json \
    --dtm aspromonte_dtm_utm33n.tif --out ./sfm_compare/seg01_ep2
```

## Results

Three passes of seg01 (603, 601 and 605 frames), pycolmap 4.2.1 on CPU (24 threads),
one run per configuration with seed 0; the gravity-free global run on ep1 was repeated
with seed 1.

**Mapping.** Every mapper registers every frame in a single model:

| Mapper | Reprojection error px | 3D points | Time s | Model *y* vs gravity |
|--------|----------------------|-----------|--------|----------------------|
| `incremental` | 0.39 / 0.45 / 0.44 | 194k / 182k / 178k | 436 / 655 / 540 | arbitrary |
| `global` | 0.38 / 0.45 / 0.42 | 204k / 185k / 181k | 440 / 239 / 170 | arbitrary |
| `global_gravity` | 0.38 / 0.45 / 0.42 | 203k / 186k / 181k | 216 / 253 / 188 | 1.5° / 1.8° / 2.2° |

Values are for ep1 / ep2 / ep3. The global mapper is 2–3× faster than the incremental one.
With gravity priors it produces gravity-aligned models.

**The global mapper without gravity bent one sequence.** On ep1 the gravity-free global
model has the same reprojection error as the others (0.38 px), but its GPS fit is poor
(median residual 5.75 m against 0.31–0.39 m). The camera rotations disagree with the
GoPro gravity by 8° from frame ~350 and by 14° from frame ~400, in two abrupt steps.
Reprojection error does not reveal this; GPS and gravity do. With seed 1 the model bends
at the same place (14° from frame ~400, median GPS residual 7.46 m, 0.38 px), so the
failure is systematic, not an unlucky random draw. With gravity priors the same sequence
gives the best GPS fit of all (0.31 m) and no step (90th-percentile disagreement 1.3°).

**Georeferencing.** Rotation left between passes after georeferencing, from the
cloud-to-cloud ICP (cross-check via DTM in parentheses):

| Model | Georeferencing | ep2 → ep1 | ep3 → ep2 | ep3 → ep1 |
|-------|----------------|-----------|-----------|-----------|
| original (COLMAP 3.9) | GPS Sim3 | 5.3° (3.5°) | 4.8° (3.6°) | 0.7° (0.4°) |
| `incremental` | GPS Sim3 | 5.5° (3.3°) | 4.8° (3.3°) | 0.7° (0.4°) |
| `global_gravity` | GPS Sim3 | 5.6° (3.5°) | 5.1° (3.8°) | 0.7° (0.5°) |
| `incremental` | gravity-levelled | **0.5°** (0.4°) | **1.5°** (1.7°) | 1.2° (1.5°) |
| `global_gravity` | gravity-levelled | **0.6°** (0.3°) | **1.4°** (1.4°) | 1.1° (1.3°) |

The rotation is almost entirely tilt. After ICP the clouds agree to 0.06–0.09 m RMS.

- With GPS alone, two of the three pairs are tilted by about 5°. The Sim3 fits are
  themselves tilted by 1.8–7.3° against the GoPro gravity: on a near-straight walk, GPS does
  not constrain the roll about the direction of travel, and the fit tilts to absorb the GPS
  altitude error.
- Levelling with gravity reduces the worst pair from 5.6° to 1.5°. The pair that GPS left
  at 0.7° goes to 1.1–1.2°: gravity bounds the error, it does not remove it.
- The mapper does not change these numbers; the georeferencing does.
- The price is the GPS residual: 0.95–1.42 m with the levelled fit against 0.31–0.68 m with
  the Sim3, which uses its extra degrees of freedom to absorb GPS noise.

**Against the DTM.** The DTM-ICP still tilts the levelled models by 1.6–1.8° (ep1, ep2)
and 2.8–3.2° (ep3), against 2.3–6.3° for the Sim3. A misalignment between the IMU and
the optical axis, a bias of the DTM-ICP on the grassy slope, or both could explain this
residual; it has not been separated yet. The remaining 1–1.5° between passes is of the
same order.

Gravity-free global models are excluded from the inter-epoch table: the ep1 model is bent,
so any pair involving it measures the bend.
