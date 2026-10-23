#!/usr/bin/env python3
"""cd_eval.py - score the change candidates of one run against objects and null zones.

Reads the candidates of a run (CSV with at least kind, E, N; best first, or ranked by a
`score` column, higher is better), the GPS track of the reference epoch (to measure the
evaluated length) and any of:

  --gt         real objects (JSON as written by scripts/object_ground_truth.py). Their
               results are printed only with --reveal-objects: the method is developed on
               the null pairs and on synthetic objects, never on the real ones.
  --synthetic  inserted objects (same JSON layout: label -> {E, N, z, height_m, ...});
               always reported.
  --exclude    E N R zones left out of the false-alarm count (a real change below the
               limit, e.g. the plasterboard in ep2 vs ep1).

A candidate whose centre lies within --match-radius of an object belongs to that object;
every other candidate outside the excluded zones is a false alarm. With --null-candidates
(the same detector on the same pair without inserted objects), a candidate that was already
there (same kind, centre within --match-radius) stays a false alarm even next to an object:
otherwise noise blobs that happen to lie near an inserted object inflate the recall.
Reported:

  - false alarms per km and per 100 m, and the Poisson probability of at least that many
    when the expected rate is --epsilon per km (criterion C1);
  - per object: rank of its best candidate of the expected kind, distance, measured height
    |dz_peak| against the ground-truth interval widened by --height-tol (C2, C4);
  - recall against false alarms per km as the score threshold drops (FROC).

Object positions are in the frame of the compared epoch (--cmp of dsm_change.py): pass the
same --icp / --cc-shift as the run so they land in the frame of the candidates.

  python3 cd_eval.py --candidates dsm_ep3_vs_ep2/dsm_detections.csv \\
      --track seg01_ep2/gps.csv --gt change_detection/v2/gt/objects.json \\
      --icp m3c2_ep3_vs_ep2_v2/icp_auto.txt --cc-shift -558997 -4214492 -123 \\
      --label base --out eval_base.json
"""
import argparse
import csv
import json
import sys

import numpy as np

from dsm_change import read_track
from run_m3c2 import parse_cc_matrices


def load_candidates(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("E", "N", "dz_peak", "score"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    if rows and isinstance(rows[0].get("score"), float):
        rows.sort(key=lambda r: -r["score"])
    return rows


def load_objects(path, M=None, shift=None):
    """label -> dict(xy, height interval, kind) in the frame of the candidates."""
    d = json.load(open(path))
    intervals = d.get("_gt_interval", {})
    objs = {}
    for label, o in d.items():
        if label.startswith("_"):
            continue
        p = np.array([o["E"], o["N"], o.get("z", 0.0)], float)
        if M is not None:
            p = M[:3, :3] @ (p + shift) + M[:3, 3] - shift
        iv = intervals.get(label, {})
        h = iv.get("height_m") or iv.get("diameter_m") or [o["height_m"]] * 2
        objs[label] = dict(xy=p[:2], height=tuple(h), kind=o.get("kind", "appeared"))
    return objs


def track_length(path, epsg, trim_ends):
    tp = read_track([path], epsg, trim_ends)
    return float(np.hypot(*np.diff(tp, axis=0).T).sum())


def score(cands, objs, zones, radius, null=()):
    """Per candidate: the object it belongs to (or None) and whether it is a false alarm."""
    labels, false = [], []
    for c in cands:
        xy = np.array([c["E"], c["N"]])
        old = any(n["kind"] == c["kind"] and np.hypot(n["E"] - c["E"], n["N"] - c["N"]) <= radius
                  for n in null)
        near = [] if old else [lab for lab, o in objs.items()
                               if np.hypot(*(xy - o["xy"])) <= radius]
        labels.append(near[0] if near else None)
        excluded = any(np.hypot(c["E"] - e, c["N"] - n) <= r for e, n, r in zones)
        false.append(not near and not excluded)
    return labels, np.array(false, bool)


def object_results(cands, labels, objs, height_tol):
    res = {}
    for lab, o in objs.items():
        hits = [i for i, l in enumerate(labels) if l == lab and cands[i]["kind"] == o["kind"]]
        if not hits:
            res[lab] = dict(rank=None)
            continue
        c = cands[hits[0]]
        h = abs(c["dz_peak"]) if isinstance(c.get("dz_peak"), float) else None
        lo, hi = o["height"]
        res[lab] = dict(rank=hits[0] + 1, n_candidates=len(hits),
                        dist_m=float(np.hypot(c["E"] - o["xy"][0], c["N"] - o["xy"][1])),
                        height_m=h, height_gt_m=[lo, hi],
                        height_ok=None if h is None else bool(lo - height_tol <= h <= hi + height_tol))
    return res


def froc(labels, false, objs, km):
    """(false alarms per km, recall) after each candidate, best first."""
    found, pts = set(), []
    for i, lab in enumerate(labels):
        if lab is not None:
            found.add(lab)
        pts.append((false[:i + 1].sum() / km, len(found) / len(objs)))
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score change candidates against objects and null zones.")
    ap.add_argument("--candidates", required=True, help="candidates CSV (kind, E, N, ...)")
    ap.add_argument("--track", required=True, help="gps.csv of the reference epoch")
    ap.add_argument("--track-epsg", type=int, default=32633)
    ap.add_argument("--trim-ends", type=float, default=0.0,
                    help="metres dropped at each track end, as in the run [m]")
    ap.add_argument("--gt", help="real objects JSON (results hidden without --reveal-objects)")
    ap.add_argument("--synthetic", help="inserted objects JSON (always reported)")
    ap.add_argument("--icp", help="CloudCompare matrix of the run, applied to object positions")
    ap.add_argument("--matrix-index", type=int, default=0)
    ap.add_argument("--cc-shift", nargs=3, type=float, metavar=("SX", "SY", "SZ"))
    ap.add_argument("--null-candidates",
                    help="candidates CSV of the same detector without inserted objects")
    ap.add_argument("--exclude", nargs=3, type=float, action="append", default=[],
                    metavar=("E", "N", "R"), help="zone left out of the false-alarm count")
    ap.add_argument("--match-radius", type=float, default=0.5,
                    help="max distance candidate centre - object centre [m]")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="accepted false alarms per km (C1)")
    ap.add_argument("--height-tol", type=float, default=0.03,
                    help="tolerance around the ground-truth height interval (C4) [m]")
    ap.add_argument("--reveal-objects", action="store_true",
                    help="print the results on the real objects (after the freeze tag)")
    ap.add_argument("--label", default="", help="row name in the ablation table")
    ap.add_argument("--out", help="write the metrics as JSON here")
    args = ap.parse_args(argv)
    if args.icp and args.cc_shift is None:
        sys.exit("--cc-shift is REQUIRED with --icp: the matrix lives in the shifted frame.")

    M = shift = None
    if args.icp:
        M = np.asarray(parse_cc_matrices(args.icp)[args.matrix_index])
        shift = np.array(args.cc_shift, float)
    cands = load_candidates(args.candidates)
    real = load_objects(args.gt, M, shift) if args.gt else {}
    synth = load_objects(args.synthetic, M, shift) if args.synthetic else {}
    if set(real) & set(synth):
        sys.exit("--gt and --synthetic share labels: rename the synthetic objects.")
    objs = {**real, **synth}
    km = track_length(args.track, args.track_epsg, args.trim_ends) / 1000.0
    null = load_candidates(args.null_candidates) if args.null_candidates else ()
    labels, false = score(cands, objs, args.exclude, args.match_radius, null)

    from scipy.stats import poisson
    n_false, mu = int(false.sum()), args.epsilon * km
    out = dict(label=args.label, candidates=args.candidates, n_candidates=len(cands),
               track_km=km, false_alarms=n_false, false_per_km=n_false / km,
               false_per_100m=n_false / km / 10, epsilon_per_km=args.epsilon,
               expected_false=mu, p_at_least=float(poisson.sf(n_false - 1, mu)))
    print(f"# cd_eval.py {args.label}\n  candidates: {args.candidates} ({len(cands)})")
    print(f"  track: {km * 1000:.1f} m | excluded zones: {len(args.exclude)} "
          f"+ {len(real)} real + {len(synth)} synthetic objects (r = {args.match_radius:g} m)")
    print(f"  false alarms: {n_false} = {out['false_per_km']:.1f} per km "
          f"({out['false_per_100m']:.2f} per 100 m)")
    print(f"  C1: expected {mu:.3f} at {args.epsilon:g} per km, "
          f"P(N >= {n_false}) = {out['p_at_least']:.2e}")

    shown = dict(synth)
    if args.reveal_objects:
        shown.update(real)
    elif real:
        print(f"  real objects: hidden ({len(real)}); --reveal-objects after the freeze tag")
    if shown:
        res = object_results(cands, labels, shown, args.height_tol)
        out["objects"] = res
        print(f"\n  {'object':>10} {'rank':>5} {'dist m':>7} {'|dz| m':>7} {'GT m':>12} {'C4':>4}")
        for lab, r in res.items():
            if r["rank"] is None:
                print(f"  {lab:>10} {'-':>5}")
                continue
            h = "-" if r["height_m"] is None else f"{r['height_m']:.3f}"
            gt = f"{r['height_gt_m'][0]:.3f}-{r['height_gt_m'][1]:.3f}"
            ok = {None: "-", True: "ok", False: "no"}[r["height_ok"]]
            print(f"  {lab:>10} {r['rank']:>5} {r['dist_m']:7.2f} {h:>7} {gt:>12} {ok:>4}")
        sub = [lab if lab in shown and c["kind"] == shown[lab]["kind"] else None
               for lab, c in zip(labels, cands)]
        pts = froc(sub, false, shown, km)
        out["froc"] = pts
        print("\n  FROC (false per km -> recall), at each recall step:")
        last = 0.0
        for fpk, rec in pts:
            if rec > last:
                print(f"    {fpk:7.1f} -> {rec:.2f}")
                last = rec
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
