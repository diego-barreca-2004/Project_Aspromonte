#!/usr/bin/env python3
"""test_measure_change.py - validation of the null tails of measure_change.py (step 7).

Fixture: synthetic window contrasts D at one scale, with a continuous window roughness f (jittered,
so the class edges never fall on ties). Checks:
  - fit_null_tails + tail_log10p recover a heavy tail: D Student-t with 4 degrees of freedom,
    log10 P at the contrast of true P = 1e-6 (four orders beyond the 99th percentile the fit
    starts from) within 0.7;
  - a bounded null (D uniform on [-1, 1], maximum likelihood shape < 0) never gives P = 0 past
    its maximum: P stays finite and keeps falling, so a statement needs a contrast beyond what
    the null pair showed, not any contrast past its end;
  - classes of roughness: where rough windows spread 4 times more than smooth ones, the same
    contrast is more significant in a smooth window, and both signs are fitted apart.

Run:  python3 tests/test_measure_change.py
"""
import os
import sys

import numpy as np
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # scripts live in the repository root

from measure_change import fit_null_tails, tail_log10p  # noqa: E402

checks = []


def ok(name, cond, extra=""):
    checks.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")


def log10p(tails, d, f, kind="appeared"):
    d, f = np.atleast_1d(np.asarray(d, float)), np.broadcast_to(np.asarray(f, float), np.shape(d))
    return tail_log10p(d, np.ones(d.shape, bool), f, tails["scales"]["5"], kind, tails["q"])


def main():
    rng = np.random.default_rng(1)
    n = 1_000_000
    f = rng.uniform(0, 1, n)
    ok_all = np.ones(n, bool)

    D = student_t.rvs(4, size=n, random_state=rng)
    tails = fit_null_tails([(5, D, ok_all, f)])
    d6 = student_t.isf(1e-6, 4)
    got = log10p(tails, [d6] * 5, [0.1, 0.3, 0.5, 0.7, 0.9])
    ok("Student-t tail: log10 P at true P = 1e-6 within 0.7, every class", np.all(np.abs(got + 6) < 0.7),
       f"(d = {d6:.1f}: {' '.join(f'{x:.2f}' for x in got)})")
    got = log10p(tails, [-d6], [0.5], "removed")
    ok("Student-t tail, removed side fitted on -D", abs(got[0] + 6) < 0.7, f"({got[0]:.2f})")

    D = rng.uniform(-1, 1, n)
    tails = fit_null_tails([(5, D, ok_all, f)])
    lp = log10p(tails, [0.995, 1.5, 3.0], [0.5])
    ok("bounded null: P finite past the maximum and falling", np.all(np.isfinite(lp)) and lp[0] > lp[1] > lp[2]
       and lp[1] > -30, f"({' '.join(f'{x:.1f}' for x in lp)})")
    ok("bounded null: shape kept >= 0", all(xi >= 0 for _, xi, _ in tails["scales"]["5"]["appeared"]))

    D = rng.normal(0, 1, n) * (1 + 3 * f) + np.where(f > 0.5, 0.5, 0.0)
    tails = fit_null_tails([(5, D, ok_all, f)])
    smooth, rough = log10p(tails, [6.0, 6.0], [0.05, 0.95])
    ok("same contrast more significant in a smooth window", smooth < rough - 3, f"({smooth:.1f} vs {rough:.1f})")
    ok("below the 99th percentile P = 1 (log10 P = 0)", log10p(tails, [0.5], [0.05])[0] == 0)
    edges = tails["scales"]["5"]["edges"]
    ok("class edges at the quintiles of f", np.allclose(edges, [0.2, 0.4, 0.6, 0.8], atol=0.01),
       f"({' '.join(f'{e:.3f}' for e in edges)})")

    n_ok = sum(c for _, c in checks)
    print(f"\n{n_ok}/{len(checks)} checks passed.")
    print("ALL GREEN" if n_ok == len(checks) else "FAILURES")
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
