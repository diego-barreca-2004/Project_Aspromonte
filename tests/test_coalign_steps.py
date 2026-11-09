#!/usr/bin/env python3
"""test_coalign_steps.py - redoing a step of coalign.py must not keep what came after it.

Fixture: coalign.main() on two passes of 20 frames, with COLMAP and the georeferencing replaced by
stand-ins that keep COLMAP's resume behaviour: feature_extractor keeps a database's camera,
matches_importer skips pairs already matched, the mapper adds to the sparse folder, PatchMatch
skips the depth maps on disk. A generation number travels from the database to the clouds, so each
output says which run produced it. Checks, for each marker deleted:
  - the step and everything after it are redone, in every --passes-dir;
  - what comes before it is kept;
  - a stale sparse/1 larger than the new sparse/0 is not picked;
  - a plain re-run does nothing, and a dense cut short resumes in its own workspace.

Run:  python3 tests/test_coalign_steps.py
"""
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import coalign  # noqa: E402

checks = []
state = {"gen": 0, "log": []}
NAMES = [f"frame_{k:06d}_t{k:05d}.00.jpg" for k in range(20)]
LABELS = ("ep2", "ep3")


def ok(name, cond, extra=""):
    checks.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")


def write_images_bin(path, names):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, name in enumerate(names, 1):
            f.write(struct.pack("<i4d3di", i, 1, 0, 0, 0, 0, 0, 0, 1))
            f.write(name.encode() + b"\0" + struct.pack("<Q", 0))


def write_model(path, names, gen):
    os.makedirs(path, exist_ok=True)
    write_images_bin(os.path.join(path, "images.bin"), names)
    with open(os.path.join(path, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", 0))
    with open(os.path.join(path, "GEN"), "w") as f:
        f.write(str(gen))


def fake_run(cmd):
    """COLMAP and georef_cloud.py, doing just what the test needs to see."""
    tool = os.path.basename(cmd[1])
    state["log"].append(tool)
    opt = lambda key: cmd[cmd.index(key) + 1]
    if tool == "feature_extractor":
        con = sqlite3.connect(opt("--database_path"))
        con.execute("CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY, name TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS matches (pair_id INTEGER PRIMARY KEY, gen INTEGER)")
        con.execute("CREATE TABLE IF NOT EXISTS two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
        con.execute("CREATE TABLE IF NOT EXISTS camera (params TEXT)")
        if not con.execute("SELECT * FROM camera").fetchall():      # keeps the camera it has
            con.execute("INSERT INTO camera VALUES (?)", (opt("--ImageReader.camera_params"),))
            names = [f"{lab}/{n}" for lab in LABELS for n in NAMES]
            con.executemany("INSERT INTO images (name) VALUES (?)", [(n,) for n in names])
        con.commit()
        con.close()
    elif tool == "matches_importer":
        con = sqlite3.connect(opt("--database_path"))
        n = sum(1 for _ in open(opt("--match_list_path")))
        con.executemany("INSERT OR IGNORE INTO matches VALUES (?, ?)",   # matched pairs are skipped
                        [(i, state["gen"]) for i in range(n)])
        con.commit()
        con.close()
    elif tool == "mapper":
        all_names = [f"{lab}/{n}" for lab in LABELS for n in NAMES]
        out = opt("--output_path")
        write_model(os.path.join(out, "0"), all_names[:30], state["gen"])
        if state["gen"] == 1:                                        # first run: a larger model 1
            write_model(os.path.join(out, "1"), all_names, state["gen"])
    elif tool == "image_deleter":
        src, dst = opt("--input_path"), opt("--output_path")
        keep = set(open(opt("--image_names_path")).read().split())
        write_images_bin(os.path.join(dst, "images.bin"),
                         [n for n in coalign.read_image_names(src).values() if n not in keep])
        shutil.copyfile(os.path.join(src, "GEN"), os.path.join(dst, "GEN"))
    elif tool == "image_undistorter":
        out = opt("--output_path")
        os.makedirs(os.path.join(out, "stereo"), exist_ok=True)
        open(os.path.join(out, "stereo", "patch-match.cfg"), "w").write("all views\n")
        shutil.copyfile(os.path.join(opt("--input_path"), "GEN"), os.path.join(out, "stereo", "GEN"))
    elif tool == "patch_match_stereo":
        depth = os.path.join(opt("--workspace_path"), "stereo", "depth.bin")
        if not os.path.exists(depth):                                # views done are skipped
            shutil.copyfile(os.path.join(opt("--workspace_path"), "stereo", "GEN"), depth)
    elif tool == "stereo_fusion":
        gen = open(os.path.join(opt("--workspace_path"), "stereo", "depth.bin")).read()
        with open(opt("--output_path"), "w") as f:
            f.write(f"ply\nelement vertex 5\nend_header\n{gen}")
    elif tool == "georef_cloud.py":
        fused, geo, out = cmd[2:5]
        with open(out, "w") as f:
            f.write(f"{open(fused).read().split()[-1]} {open(geo).read().split()[-1]}")
    else:
        raise AssertionError(f"unexpected command {cmd}")


def fake_georeference(model, segs, epsg, path):
    state["log"].append("georeference")
    with open(path, "w") as f:
        f.write(f"# Sim3\n# fit line\nscale {open(os.path.join(model, 'GEN')).read()}\n")
    return "georeference: fake"


def take_log():
    out, state["log"] = state["log"], []
    return out


def main():
    T = tempfile.mkdtemp(prefix="test_coalign_steps_")
    segs = []
    for lab in LABELS:
        seg = os.path.join(T, lab)
        os.makedirs(os.path.join(seg, "frames"))
        for n in NAMES:
            open(os.path.join(seg, "frames", n), "w").close()
        with open(os.path.join(seg, "gps.csv"), "w") as f:
            f.write("t_s,lat,lon,alt\n")
            f.writelines(f"{k},{38.0 + 1e-5 * k:.7f},{15.7:.7f},100\n" for k in range(20))
        segs.append(seg)
    out = os.path.join(T, "out")
    calib = os.path.join(T, "calib.json")

    def set_calibration(fx):
        with open(calib, "w") as f:
            json.dump(dict(model="fisheye", image_size=[1600, 1400], distortion_coefficients=[0, 0, 0, 0],
                           camera_matrix=[[fx, 0, 800], [0, fx, 700], [0, 0, 1]]), f)

    def run_main(passes_dir="passes"):
        argv = ["coalign.py", "--pass", f"ep2={segs[0]}", "--pass", f"ep3={segs[1]}", "--out", out,
                "--calibration", calib, "--passes-dir", passes_dir]
        sys.argv = argv
        coalign.main()

    coalign.run, coalign.georeference = fake_run, fake_georeference
    cm = lambda lab="ep2", pd="passes": os.path.join(out, pd, lab, "colmap")
    fused = lambda lab="ep2", pd="passes": os.path.join(cm(lab, pd), "dense", "fused.ply")
    utm = lambda lab="ep2", pd="passes": os.path.join(cm(lab, pd), "dense", "fused_utm.ply")
    gen_of = lambda path: open(path).read().split()[-1]
    text = lambda path: open(path).read().strip()
    marker = lambda step: os.path.join(out, f".done_{step}")
    db_rows = lambda sql: (lambda c: (c.execute(sql).fetchall(), c.close())[0])(sqlite3.connect(os.path.join(out, "database.db")))

    print("first run, in two --passes-dir:")
    state["gen"] = 1
    set_calibration(700.0)
    run_main()
    run_main("passes_1600px")
    take_log()
    ok("all outputs from generation 1", gen_of(fused()) == "1" and text(utm()) == "1 1")

    print("plain re-run:")
    state["gen"] = 2
    run_main()
    ok("nothing redone", take_log() == [])

    print("dense cut short (fused.ply missing, workspace there):")
    os.remove(fused())
    with open(utm(), "w") as f:
        f.write("old old")
    run_main()
    c = take_log()
    ok("resumes in the workspace, without image_undistorter",
       c == ["patch_match_stereo", "stereo_fusion", "georef_cloud.py"], f"({c})")
    ok("fused_utm.ply rebuilt from the new fused.ply", text(utm()) == "1 1", f"({open(utm()).read()})")

    print("redo georef (marker deleted):")
    os.remove(marker("georef"))
    fused_before = os.path.getmtime(fused())
    for m in os.listdir(os.path.join(out, "sparse")):         # stands for a changed GPS log
        with open(os.path.join(out, "sparse", m, "GEN"), "w") as f:
            f.write("7")
    run_main()
    c = take_log()
    ok("Sim3 refitted", "georeference" in c and gen_of(os.path.join(out, "geo_transform.txt")) == "7")
    ok("UTM clouds of this --passes-dir rebuilt", text(utm()) == "1 7" and text(utm("ep3")) == "1 7",
       f"({open(utm()).read()})")
    ok("dense not redone", "patch_match_stereo" not in c and os.path.getmtime(fused()) == fused_before)
    ok("old UTM cloud of the other --passes-dir removed", not os.path.exists(utm(pd="passes_1600px")))
    ok("marker back", os.path.exists(marker("georef")))
    run_main("passes_1600px")
    ok("other --passes-dir rebuilt with the new Sim3", text(utm(pd="passes_1600px")) == "1 7")
    take_log()

    print("redo map (marker deleted):")
    state["gen"] = 3
    os.remove(marker("map"))
    run_main()
    c = take_log()
    ok("mapper, georeference, split and dense all redone",
       all(t in c for t in ("mapper", "georeference", "image_deleter", "image_undistorter",
                            "patch_match_stereo", "stereo_fusion", "georef_cloud.py")), f"({c})")
    ok("stale larger sparse/1 gone", not os.path.exists(os.path.join(out, "sparse", "1")))
    ok("pass split from the new model", gen_of(os.path.join(cm(), "sparse", "0", "GEN")) == "3")
    ok("Sim3 from the new model", gen_of(os.path.join(out, "geo_transform.txt")) == "3")
    ok("cloud and UTM cloud from the new model", gen_of(fused()) == "3" and text(utm()) == "3 3",
       f"({open(utm()).read()})")
    ok("the other --passes-dir does not keep the old model", not os.path.exists(fused(pd="passes_1600px")))
    run_main("passes_1600px")
    ok("other --passes-dir rebuilt", gen_of(fused(pd="passes_1600px")) == "3"
       and text(utm(pd="passes_1600px")) == "3 3")
    take_log()

    print("redo match (marker deleted):")
    state["gen"] = 4
    os.remove(marker("match"))
    run_main()
    c = take_log()
    ok("matches replaced, not kept", {r[0] for r in db_rows("SELECT gen FROM matches")} == {4})
    ok("features kept", "feature_extractor" not in c)
    ok("everything after it redone", text(utm()) == "4 4" and "mapper" in c, f"({c})")

    print("redo extract with a new calibration (marker deleted):")
    state["gen"] = 5
    set_calibration(900.0)
    os.remove(marker("extract"))
    run_main()
    c = take_log()
    params = db_rows("SELECT params FROM camera")[0][0]
    ok("new calibration in the database", params.startswith("900"), f"({params})")
    ok("everything after it redone", text(utm()) == "5 5" and "matches_importer" in c, f"({c})")

    print("after all the redos:")
    run_main()
    ok("next plain re-run does nothing", take_log() == [])

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    shutil.rmtree(T)
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
