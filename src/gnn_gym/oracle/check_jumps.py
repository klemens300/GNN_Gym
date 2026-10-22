"""Quality check of reference jump datasets.

Per file:
    completeness   row count, duplicate and missing indices, convergence flags
    sampling       subsystem histogram (all 2^m - 1 subsets hit?), cardinality
    labels         dE, B_sym, E_fwd, E_bwd percentiles; rows with
                   B_sym < |dE|/2, for which B_sym + dE/2 would be negative
    noise floor    scatter of dE and B_sym over the unary cells, where every
                   jump is the same jump: a direct measure of the label noise
                   of the NEB
    numerics       |E_fwd - E_bwd - dE|, zero in exact arithmetic, against the
                   single-precision step of the total energy at that cell size
    streams        seed ranges (two files must use disjoint seeds)
    cost           wall time per jump (relaxations and NEB)

    python -m gnn_gym.oracle.check_jumps data/jumps/MoNbTaW_7cube_train.csv \
        data/jumps/MoNbTaW_7cube_referee.csv
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..paths import resolve

# Typical |E| per atom of the refractory cells (eV), for the float32 step.
E_PER_ATOM_EV = 11.5


def load(path):
    with open(resolve(path), newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(rows, col):
    return np.array([float(r[col]) for r in rows])


def check(path):
    rows = load(path)
    n = len(rows)
    elements = [c[len("frac_"):] for c in rows[0] if c.startswith("frac_")]
    print(f"\n{'=' * 74}\n{Path(path).name}: {n} rows, elements {elements}")

    idxs = [int(r["idx"]) for r in rows]
    dup = [i for i, c in Counter(idxs).items() if c > 1]
    missing = sorted(set(range(min(idxs), max(idxs) + 1)) - set(idxs))
    bad_conv = sum(1 for r in rows
                   if str(r["relax_conv"]).lower() not in ("true", "1")
                   or str(r["neb_conv"]).lower() not in ("true", "1"))
    print(f"  idx {min(idxs)}..{max(idxs)} | duplicates: {len(dup)} | "
          f"missing: {len(missing)} | not converged: {bad_conv}")

    subs = Counter(r["subsystem"] for r in rows)
    card = Counter(sum(1 for e in elements if float(r[f"frac_{e}"]) > 0)
                   for r in rows)
    print(f"  subsystems hit: {len(subs)}/{2 ** len(elements) - 1} | by "
          "cardinality: " + " ".join(f"{k}-ary {card[k]}" for k in sorted(card)))
    print(f"  subsystem counts min/max: {min(subs.values())}/{max(subs.values())}")

    dE, bsym = fnum(rows, "dE_eV"), fnum(rows, "B_sym_eV")
    fwd, bwd = fnum(rows, "E_fwd_eV"), fnum(rows, "E_bwd_eV")
    for name, x in ((" dE", dE), ("Bsym", bsym), ("Efwd", fwd), ("Ebwd", bwd)):
        q = np.percentile(x, [0, 1, 50, 99, 100])
        print(f"  {name} [eV]: min {q[0]:+.3f} | p1 {q[1]:+.3f} | med "
              f"{q[2]:+.3f} | p99 {q[3]:+.3f} | max {q[4]:+.3f}")
    neg = int(np.sum(bsym < np.abs(dE) / 2))
    print(f"  rows with B_sym < |dE|/2: {neg} ({neg / n * 100:.2f} %)")
    print(f"  smallest barriers: E_fwd {fwd.min():.4f}, E_bwd {bwd.min():.4f} eV")

    unary = defaultdict(lambda: ([], []))
    for r in rows:
        if r["subsystem"] in elements:
            unary[r["subsystem"]][0].append(float(r["dE_eV"]))
            unary[r["subsystem"]][1].append(float(r["B_sym_eV"]))
    if unary:
        print("  unary noise floor (identical jumps; std in meV):")
        for s in sorted(unary):
            d, b = unary[s]
            print(f"    {s:>3} n={len(d):>4}  dE std {np.std(d) * 1e3:5.2f}"
                  f"  B_sym std {np.std(b) * 1e3:5.2f}"
                  f"  B_sym mean {np.mean(b):.4f} eV")

    db = np.abs(fnum(rows, "db_residual_eV")) * 1e3
    n_atoms = int(rows[0]["n_sites"]) - 1
    floor = E_PER_ATOM_EV * n_atoms * 2 ** -23 * 1e3
    print(f"  |db_residual| meV: mean {db.mean():.3f} | p99 "
          f"{np.percentile(db, 99):.3f} | max {db.max():.3f} "
          f"(float32 step at N={n_atoms}: {floor:.2f})")

    seeds = fnum(rows, "seed")
    a = fnum(rows, "a_relaxed_A")
    t = fnum(rows, "t_s")
    print(f"  seeds {int(seeds.min())}..{int(seeds.max())} | a_relaxed "
          f"{a.min():.4f}..{a.max():.4f} A | models: "
          f"{sorted(set(r['model'] for r in rows))}")
    print(f"  time per jump: mean {t.mean():.1f} s, median {np.median(t):.1f} s")
    return int(seeds.min()), int(seeds.max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    args = ap.parse_args(argv)
    ranges = [check(p) for p in args.csv]
    if len(ranges) > 1:
        ok = all(a[1] < b[0] or b[1] < a[0]
                 for i, a in enumerate(ranges) for b in ranges[i + 1:])
        print(f"\nseed streams disjoint: {'yes' if ok else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
