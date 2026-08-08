#!/usr/bin/env python3
"""test_run_m3c2.py - end-to-end validation of run_m3c2.py on synthetic ground truth.

Scenario (all analytically known):
  * rough terrain 30 x 8 m, ~400 pts/m^2, 2 cm gaussian noise, in fake UTM coords;
  * epoch 2 = independent resample + BOX top +0.30 m (0.4 x 0.4) + HOLE -0.20 m (r 0.5);
  * epoch 2 is delivered MISALIGNED by a known rigid transform (3 deg tilt + [0.3,-0.2,6.0] m)
    -> the script must recover alignment via the CC matrix + Global-Shift path;
  * a far junk cluster is added to epoch 2 -> the mutual-footprint crop must remove it;
  * the matrix file mimics a pasted CC console: timestamps, prose, TWO matrices
    (shifted-frame first) -> parser must find both, warn, use index 0.

Asserts: background patches unbiased (|mean| < 1 cm) and tight; box mean ~ +0.30
significant; hole mean ~ -0.20; sign convention; junk cropped; --reg-error raises
the LoD and shrinks the significant set; all output files written.
"""
import csv
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from plyfile import PlyData, PlyElement

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "testdata")
G = np.array([559000.0, 4214000.0, 120.0])       # fake UTM placement
SHIFT = np.array([-559000.0, -4214000.0, 0.0])   # what CC would print at load


def terrain_z(x, y):
    return 0.15 * np.sin(0.8 * x) + 0.10 * np.cos(1.3 * y) + 0.05 * x


def sample_epoch(rng, n):
    x = rng.uniform(0, 30, n)
    y = rng.uniform(0, 8, n)
    z = terrain_z(x, y) + rng.normal(0, 0.02, n)
    return np.c_[x, y, z]


def write_ply(path, pts):
    arr = np.empty(len(pts), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"),
                                    ("nx", "f4"), ("ny", "f4"), ("nz", "f4")])
    arr["x"], arr["y"], arr["z"] = pts.T
    arr["nx"] = arr["ny"] = 0.0
    arr["nz"] = 1.0                                  # dummy normals like fused_utm
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


def build_fixture():
    os.makedirs(T, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 96000

    ep1 = sample_epoch(rng, n)                       # epoch 1 (reference)
    ep2 = sample_epoch(rng, n)                       # epoch 2, aligned frame
    box = (ep2[:, 0] >= 10) & (ep2[:, 0] <= 10.4) \
        & (ep2[:, 1] >= 4) & (ep2[:, 1] <= 4.4)
    ep2[box, 2] += 0.30
    hole = np.hypot(ep2[:, 0] - 20, ep2[:, 1] - 3) <= 0.5
    ep2[hole, 2] -= 0.20

    ep1_local = ep1 + (G + SHIFT)                    # local frame = CC shifted frame
    ep2_local = ep2 + (G + SHIFT)

    # known rigid misalignment (in the local frame): tilt 3 deg about x + offset
    a = np.radians(3.0)
    R = np.array([[1, 0, 0],
                  [0, np.cos(a), -np.sin(a)],
                  [0, np.sin(a), np.cos(a)]])
    c = ep2_local.mean(0)
    delta = np.array([0.3, -0.2, 6.0])
    t = c - R @ c + delta                            # p' = R p + t  (ICP-style, local)
    Rinv, tinv = R.T, -R.T @ t
    ep2_raw_local = (Rinv @ ep2_local.T).T + tinv    # what "came from the pipeline"

    junk_local = ep2_raw_local[:5000] + np.array([80.0, 40.0, 15.0])
    ep2_raw_local = np.vstack([ep2_raw_local, junk_local])

    write_ply(os.path.join(T, "ep1_utm.ply"), ep1_local - SHIFT)      # global coords
    write_ply(os.path.join(T, "ep2_utm.ply"), ep2_raw_local - SHIFT)

    from pyproj import Transformer
    inv = Transformer.from_crs(32633, 4326, always_xy=True)
    with open(os.path.join(T, "track.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "ts", "lat", "lon", "alt", "speed", "dop", "fix"])
        for i, x in enumerate(np.arange(0, 30.01, 0.25)):
            lon, lat = inv.transform(x + G[0], 4.0 + G[1])
            w.writerow([f"{i * 0.2:.1f}", "", f"{lat:.8f}", f"{lon:.8f}", "120", "1", "1", "3"])

    Mglobal = np.eye(4)                              # decoy "global" matrix, CC-style
    Mglobal[:3, :3], Mglobal[:3, 3] = R, t - SHIFT
    with open(os.path.join(T, "icp_console_paste.txt"), "w") as f:
        f.write("[18:58:27] [Register] Final RMS*: 0.021 (computed on 99999 points)\n")
        f.write("[18:58:27] [Register] Applied transformation matrix:\n")
        for row in np.c_[np.r_[R, [[0, 0, 0]]], np.r_[t, 1.0]]:
            f.write("[18:58:27] " + " ".join(f"{v:.12f}" for v in row) + "\n")
        f.write("[18:58:27] Hint: copy it (CTRL+C) and apply it\n")
        f.write("[18:58:28] [ICP] Transformation to global coordinates:\n")
        for row in Mglobal:
            f.write("[18:58:28] " + " ".join(f"{v:.12f}" for v in row) + "\n")
    return dict(n_ep2=len(ep2_raw_local))


def run_cli(outdir, extra):
    cmd = [sys.executable, os.path.join(HERE, "run_m3c2.py"),
           "--ref", os.path.join(T, "ep1_utm.ply"),
           "--cmp", os.path.join(T, "ep2_utm.ply"),
           "--icp", os.path.join(T, "icp_console_paste.txt"),
           "--cc-shift", "-559000", "-4214000", "0",
           "--out", outdir, "--core-spacing", "0.05",
           "--patch", "559005,4214002,1.0",          # 0 background
           "--patch", "559025,4214006,1.0",          # 1 background
           "--patch", "559010.2,4214004.2,0.15",     # 2 box centre (+0.30)
           "--patch", "559020,4214003,0.3",          # 3 hole centre (-0.20)
           ] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("run_m3c2.py failed")
    return r.stdout


def run_cli_auto(outdir):
    cmd = [sys.executable, os.path.join(HERE, "run_m3c2.py"),
           "--ref", os.path.join(T, "ep1_utm.ply"),
           "--cmp", os.path.join(T, "ep2_utm.ply"),
           "--auto-icp", "--out", outdir, "--core-spacing", "0.05",
           "--track", os.path.join(T, "track.csv"), "--max-track-dist", "3",
           "--patch", "559005,4214002,1.0",
           "--patch", "559025,4214006,1.0",
           "--patch", "559010.2,4214004.2,0.15",
           "--patch", "559020,4214003,0.3",
           "--patch", "559015,4214007.9,0.5",
           ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("run_m3c2.py --auto-icp failed")
    return r.stdout


def read_stats(outdir):
    with open(os.path.join(outdir, "m3c2_patch_stats.csv")) as f:
        return {int(r["patch"]): {k: float(v) for k, v in r.items()}
                for r in csv.DictReader(f)}


def main():
    if os.path.isdir(T):
        shutil.rmtree(T)
    fx = build_fixture()
    checks = []

    def ok(name, cond, detail=""):
        checks.append((name, bool(cond)))
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    out1 = os.path.join(T, "out_pass1")
    stdout1 = run_cli(out1, ["--reg-error", "0"])
    s = read_stats(out1)

    print("== assertions, pass 1 (reg-error 0) ==")
    ok("parser found 2 matrices and warned",
       re.search(r"WARNING: 2 matrices", stdout1))
    ok("matrix is rigid (scale gate passed)", "must be ~1" in stdout1)
    m = re.search(r"cmp: kept ([\d,]+) / ([\d,]+)", stdout1)
    kept, tot = (int(x.replace(",", "")) for x in m.groups())
    ok("crop removed the junk cluster", tot - kept >= 5000, f"(removed {tot - kept})")
    for i in (0, 1):
        ok(f"background patch {i} unbiased (|mean| < 1 cm)",
           abs(s[i]["mean"]) < 0.01, f"(mean {s[i]['mean']:+.4f})")
        ok(f"background patch {i} tight (std in 2..60 mm)",
           0.002 < s[i]["std"] < 0.06, f"(std {s[i]['std']:.4f})")
        ok(f"background patch {i} mostly non-significant",
           s[i]["sig_frac"] < 0.2, f"(sig {100 * s[i]['sig_frac']:.0f}%)")
    ok("box detected ~ +0.30 (sign: added = positive)",
       0.24 < s[2]["mean"] < 0.34, f"(mean {s[2]['mean']:+.4f})")
    ok("box significant", s[2]["sig_frac"] > 0.7, f"(sig {100 * s[2]['sig_frac']:.0f}%)")
    ok("hole detected ~ -0.20 (sign: removed = negative)",
       -0.24 < s[3]["mean"] < -0.16, f"(mean {s[3]['mean']:+.4f})")
    for fn in ("m3c2_core.ply", "m3c2_map.png", "m3c2_patches_hist.png",
               "m3c2_patch_stats.csv", "m3c2_report.txt"):
        ok(f"output exists: {fn}", os.path.isfile(os.path.join(out1, fn)))
    v = PlyData.read(os.path.join(out1, "m3c2_core.ply"))["vertex"]
    ok("PLY carries CC scalar fields",
       "scalar_M3C2_distance" in v.data.dtype.names
       and "scalar_significant" in v.data.dtype.names)
    ok("PLY coords are back in global UTM",
       abs(float(np.median(v["x"])) - 559015) < 20,
       f"(median E {float(np.median(v['x'])):.1f})")

    sig1 = int(re.search(r"significant: ([\d,]+)", stdout1).group(1).replace(",", ""))
    out2 = os.path.join(T, "out_pass2")
    stdout2 = run_cli(out2, ["--reg-error", "0.05"])
    sig2 = int(re.search(r"significant: ([\d,]+)", stdout2).group(1).replace(",", ""))
    print("== assertions, pass 2 (reg-error 0.05) ==")
    ok("higher reg-error shrinks the significant set", sig2 < sig1,
       f"({sig1:,} -> {sig2:,})")
    s2 = read_stats(out2)
    ok("box still significant with reg-error 0.05", s2[2]["sig_frac"] > 0.7)
    ok("background significance collapses",
       s2[0]["sig_frac"] < 0.05 and s2[1]["sig_frac"] < 0.05)

    out3 = os.path.join(T, "out_auto")
    stdout3 = run_cli_auto(out3)
    s3 = read_stats(out3)
    print("== assertions, pass 3 (auto-ICP + corridor mask) ==")
    mrot = re.search(r"auto-ICP result: rotation ([\d.]+) deg", stdout3)
    ok("auto-ICP recovered the known 3-deg misalignment",
       mrot and 2.5 < float(mrot.group(1)) < 3.5,
       f"({mrot.group(1) if mrot else 'none'} deg)")
    import run_m3c2 as rm
    ok("icp_auto.txt written and self-parseable",
       os.path.isfile(os.path.join(out3, "icp_auto.txt"))
       and len(rm.parse_cc_matrices(os.path.join(out3, "icp_auto.txt"))) == 1)
    mk = re.search(r"corridor mask <= .*: kept ([\d,]+) / ([\d,]+)", stdout3)
    k3, t3 = (int(x.replace(",", "")) for x in mk.groups())
    ok("corridor mask reduced the core set", k3 < t3, f"({t3:,} -> {k3:,})")
    for i in (0, 1):
        ok(f"auto-ICP background patch {i} unbiased (|mean| < 1 cm)",
           abs(s3[i]["mean"]) < 0.01, f"(mean {s3[i]['mean']:+.4f})")
    ok("auto-ICP box ~ +0.30", 0.24 < s3[2]["mean"] < 0.34, f"(mean {s3[2]['mean']:+.4f})")
    ok("auto-ICP hole ~ -0.20", -0.24 < s3[3]["mean"] < -0.16, f"(mean {s3[3]['mean']:+.4f})")
    import math
    ok("off-corridor probe patch masked out", math.isnan(s3[4]["mean"]))
    v3 = PlyData.read(os.path.join(out3, "m3c2_core.ply"))["vertex"]
    ok("PLY carries dist_to_track", "scalar_dist_to_track" in v3.data.dtype.names)

    # pass 4: unit-level regression - trimmed stage 2 must resist growing vegetation
    print("== assertions, pass 4 (trimmed ICP vs vegetation bias) ==")
    import py4dgeo
    import run_m3c2 as rm
    r4 = np.random.default_rng(3)
    stones = np.c_[r4.uniform(2, 148, 25), r4.uniform(2.6, 5.4, 25)]

    def trail(n, veg_boost, r):
        x, y = r.uniform(0, 150, n), r.uniform(0, 8, n)
        z = np.where(x < 100, 0.25 * x, 25.0 - 0.20 * (x - 100))
        z = z + 0.30 * ((np.abs(y - 4) / 4) ** 2) + r.normal(0, 0.02, n)
        for cx, cy in stones:
            m = (x - cx) ** 2 + (y - cy) ** 2 < 0.15 ** 2
            z[m] += 0.04
        veg = np.abs(y - 4) > 2.5
        z[veg] += r.uniform(0, 0.7, veg.sum()) + veg_boost
        return np.c_[x, y, z]

    p_ref = trail(120000, 0.0, np.random.default_rng(10))
    p_true = trail(120000, 0.25, np.random.default_rng(20))
    a = np.radians(5.6)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    c = p_true.mean(0)
    t = c - R @ c + [1.5, -0.3, 5.0]
    p_raw = (R.T @ (p_true - t).T).T

    def vox(p, s):
        ijk = np.floor((p - p.min(0)) / s).astype(np.int64)
        k = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
        _, i = np.unique(k, return_index=True)
        return p[np.sort(i)]

    sub_r, sub_c = vox(p_ref, 0.15), vox(p_raw, 0.15)
    e_r = py4dgeo.Epoch(np.ascontiguousarray(sub_r))
    tr1 = py4dgeo.iterative_closest_point(
        e_r, py4dgeo.Epoch(np.ascontiguousarray(sub_c)), max_iterations=100)
    A1, rp1 = tr1.affine_transformation, np.asarray(tr1.reduction_point, float)
    R1 = A1[:3, :3]
    b1 = A1[:3, 3] + rp1 - R1 @ rp1
    s1 = (R1 @ sub_c.T).T + b1
    e_r.calculate_normals(radius=0.5)
    R2, b2 = rm.trimmed_p2plane_icp(sub_r, e_r.normals, s1)
    ali = ((R2 @ R1) @ p_raw.T).T + (R2 @ b1 + b2)
    d = ali - p_true
    bed = np.abs(p_true[:, 1] - 4) < 1.2
    bed_dz = abs(float(d[bed, 2].mean()))
    ok("trimmed cascade: bed bias < 1.2 cm despite +25 cm vegetation growth",
       bed_dz < 0.012, f"(bed dz {bed_dz:.4f})")
    ok("trimmed cascade: along-track residual < 2 cm",
       abs(float(d[:, 0].mean())) < 0.02, f"(x {float(d[:, 0].mean()):+.4f})")

    failed = [n for n, c in checks if not c]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    if failed:
        sys.exit("FAILED: " + "; ".join(failed))
    print("ALL GREEN")


if __name__ == "__main__":
    main()