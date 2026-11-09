#!/usr/bin/env python3
"""test_run_pipeline.py - a redone SfM must not leave an old cloud or an old Sim3 behind.

Fixture: a stand-in for the COLMAP binary that keeps the two behaviours that matter here: stereo
workspace built from the model it is given, PatchMatch skipping the depth maps already on disk.
The model is a tagged file, and the tag travels through the workspace and the depth map into
fused.ply, so the fused cloud says which model it was built from. Checks:
  - a resumed dense keeps the workspace and the depth maps;
  - --force after a redone SfM rebuilds the cloud from the new model;
  - a model newer than the cloud (no --force) does the same, and drops the old fused_utm.ply;
  - dense, geo and georef skip when nothing upstream changed, and redo when it did.

Run:  python3 tests/test_run_pipeline.py
"""
import os
import shutil
import stat
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import run_pipeline  # noqa: E402

checks = []

FAKE_COLMAP = r'''#!/usr/bin/env python3
import os, shutil, sys
cmd, argv = sys.argv[1], sys.argv[2:]
opt = lambda key: argv[argv.index(key) + 1]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(cmd + "\n")
if cmd == "image_undistorter":
    out = opt("--output_path")
    os.makedirs(os.path.join(out, "stereo"), exist_ok=True)
    os.makedirs(os.path.join(out, "sparse"), exist_ok=True)
    shutil.copyfile(os.path.join(opt("--input_path"), "images.bin"),
                    os.path.join(out, "sparse", "images.bin"))
    open(os.path.join(out, "stereo", "patch-match.cfg"), "w").write("all views\n")
elif cmd == "patch_match_stereo":
    ws = opt("--workspace_path")
    depth = os.path.join(ws, "stereo", "depth_map.bin")
    if not os.path.exists(depth):                      # COLMAP skips the views already done
        shutil.copyfile(os.path.join(ws, "sparse", "images.bin"), depth)
elif cmd == "stereo_fusion":
    tag = open(os.path.join(opt("--workspace_path"), "stereo", "depth_map.bin"), "rb").read()
    with open(opt("--output_path"), "wb") as f:
        f.write(b"ply\nelement vertex 5\nend_header\n" + tag)
'''

FAKE_CLOUD = r'''#!/usr/bin/env python3
import sys
open(sys.argv[3], "w").write("utm from " + open(sys.argv[1], "rb").read().decode().split()[-1] + "\n")
'''

FAKE_GEO = r'''#!/usr/bin/env python3
import sys
out = sys.argv[sys.argv.index("--out") + 1]
open(out, "w").write("# 90 / 100 inliers, residual median 0.30 m\n")
'''


def ok(name, cond, extra=""):
    checks.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")


def write_script(path, text):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def age(root, seconds=1000):
    """Move a file, or every file under a folder, back in time, as if from an earlier run."""
    then = time.time() - seconds
    if os.path.isfile(root):
        os.utime(root, (then, then))
    for d, _, files in os.walk(root):
        for name in files:
            os.utime(os.path.join(d, name), (then, then))


def calls(log):
    if not os.path.exists(log):
        return []
    with open(log) as f:
        out = f.read().split()
    os.remove(log)
    return out


def main():
    T = tempfile.mkdtemp(prefix="test_run_pipeline_")
    log = os.path.join(T, "colmap.log")
    os.environ["FAKE_LOG"] = log
    cfg = dict(run_pipeline.DEFAULTS, min_dense_points="1", python=sys.executable,
               colmap=write_script(os.path.join(T, "fake_colmap"), FAKE_COLMAP),
               cloud_script=write_script(os.path.join(T, "fake_cloud.py"), FAKE_CLOUD),
               geo_script=write_script(os.path.join(T, "fake_geo.py"), FAKE_GEO))
    p = run_pipeline.P(os.path.join(T, "seg"))
    os.makedirs(p.model0)
    os.makedirs(p.undist_images)
    model = os.path.join(p.model0, "images.bin")
    fused_tag = lambda: open(p.fused, "rb").read().split()[-1].decode()

    def set_model(tag):
        with open(model, "w") as f:
            f.write(tag)

    print("first run, then skip:")
    set_model("gen1")
    run_pipeline.stage_dense(cfg, p, SimpleNamespace(force=False))
    ok("cloud built from the model", fused_tag() == "gen1", f"({fused_tag()})")
    ok("workspace built once", calls(log).count("image_undistorter") == 1)
    run_pipeline.stage_dense(cfg, p, SimpleNamespace(force=False))
    ok("nothing changed: dense skipped", calls(log) == [])

    print("resume after a crash before fusion:")
    os.remove(p.fused)
    run_pipeline.stage_dense(cfg, p, SimpleNamespace(force=False))
    c = calls(log)
    ok("workspace kept, PatchMatch and fusion rerun", "image_undistorter" not in c
       and c == ["patch_match_stereo", "stereo_fusion"], f"({c})")

    print("--force after a redone SfM:")
    age(p.colmap)
    set_model("gen2")
    with open(p.fused_utm, "w") as f:
        f.write("old utm cloud\n")
    run_pipeline.stage_dense(cfg, p, SimpleNamespace(force=True))
    ok("cloud in the frame of the new model", fused_tag() == "gen2", f"({fused_tag()})")
    ok("old fused_utm.ply removed", not os.path.exists(p.fused_utm))
    ok("workspace rebuilt", "image_undistorter" in calls(log))

    print("model newer than the cloud, no --force:")
    age(p.colmap)
    set_model("gen3")
    with open(p.fused_utm, "w") as f:
        f.write("old utm cloud\n")
    age(p.fused_utm)
    run_pipeline.stage_dense(cfg, p, SimpleNamespace(force=False))
    ok("cloud rebuilt from the new model", fused_tag() == "gen3", f"({fused_tag()})")
    ok("old fused_utm.ply removed", not os.path.exists(p.fused_utm))
    calls(log)

    print("crash before fusion, then the model changes:")
    age(p.colmap)
    os.remove(p.fused)
    set_model("gen4")
    run_pipeline.stage_dense(cfg, p, SimpleNamespace(force=False))
    ok("stale workspace not resumed", fused_tag() == "gen4", f"({fused_tag()})")
    calls(log)

    print("geo:")
    run_pipeline.stage_geo(cfg, p, SimpleNamespace(force=False))
    age(p.geo)
    aged = os.path.getmtime(p.geo)
    age(model, 2000)
    run_pipeline.stage_geo(cfg, p, SimpleNamespace(force=False))
    ok("Sim3 newer than the model: skipped", os.path.getmtime(p.geo) == aged)
    set_model("gen5")
    run_pipeline.stage_geo(cfg, p, SimpleNamespace(force=False))
    ok("model newer than the Sim3: refitted", os.path.getmtime(p.geo) > aged)

    print("georef:")
    age(model, 3000)
    run_pipeline.stage_georef(cfg, p, SimpleNamespace(force=False))
    ok("UTM cloud written", os.path.exists(p.fused_utm))
    age(p.fused, 2000)
    age(p.geo, 2000)
    age(p.fused_utm)
    aged = os.path.getmtime(p.fused_utm)
    run_pipeline.stage_georef(cfg, p, SimpleNamespace(force=False))
    ok("UTM cloud newer than its inputs: skipped", os.path.getmtime(p.fused_utm) == aged)
    os.utime(p.geo, None)
    run_pipeline.stage_georef(cfg, p, SimpleNamespace(force=False))
    ok("Sim3 newer than the UTM cloud: redone", os.path.getmtime(p.fused_utm) > aged)
    age(p.fused_utm)
    aged = os.path.getmtime(p.fused_utm)
    age(p.geo, 2000)
    os.utime(p.fused, None)
    run_pipeline.stage_georef(cfg, p, SimpleNamespace(force=False))
    ok("fused cloud newer than the UTM cloud: redone", os.path.getmtime(p.fused_utm) > aged)

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    shutil.rmtree(T)
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
