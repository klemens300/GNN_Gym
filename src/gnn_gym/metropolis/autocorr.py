"""Equilibration of Metropolis chains, measured from their per-sweep traces.

A block error over the sampled sweeps says how noisy an average is, not
whether the chain had reached equilibrium before the averaging started: a
frozen chain has a small block error and a wrong mean. Two tests address
that (van de Walle and Asta, Modelling Simul. Mater. Sci. Eng. 10, 521
(2002); Kostiuchenko et al., npj Comput. Mater. 5, 55 (2019)):

    tau_int   integrated autocorrelation time in sweeps, with Sokal's
              self-consistent window (tau = 1/2 + sum rho(t), summed up to
              the first M >= 5 tau). An independent sample costs 2 tau sweeps.
    t_eq      first sweep from which the two halves of the remaining trace
              agree within 2 sigma of their correlation-corrected errors (the
              block-equality test); averaging may start there.
    N_eff     independent samples in the averaging window.
    err       standard error of the mean with the correlation factor,
              sqrt(2 tau Var / N).
    need      sweeps for a target precision, 2 tau Var / p^2 + t_eq.

    Two-start test: when one temperature was run from more than one start
    state (or chain), the difference of the tail means with its combined
    error. Beyond 3 sigma the chains have not met and neither value is the
    equilibrium value.

Each observable is analysed per (trace file, temperature, start state): the
energy and the Warren-Cowley parameter of one pair, by default the pair with
the largest |mean alpha_1|.

    python -m gnn_gym.metropolis.autocorr runs/metropolis/tord/TaW_*.trace.csv \
        --csv results/metropolis/equilibration_TaW.csv
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..paths import resolve

ENERGY = ("E [meV/at]", "E_rel_meV_per_atom")


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def tau_int(x, c=5.0):
    """Integrated autocorrelation time in sampling intervals, Sokal window.
    Returns (tau, window); tau >= 0.5, 0.5 meaning uncorrelated."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 16:
        return float("nan"), 0
    x = x - x.mean()
    var = float(x @ x) / n
    if var <= 0:
        return 0.5, 0
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n] / n
    rho = acov / acov[0]
    tau, window = 0.5, 0
    for m in range(1, n):
        tau += rho[m]
        if m >= c * tau:
            window = m
            break
        window = m
    return float(max(tau, 0.5)), window


def err_of_mean(x, tau):
    """Standard error of the mean with the correlation factor 2 tau."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2 or not np.isfinite(tau):
        return float("nan")
    return math.sqrt(2.0 * tau * float(x.var(ddof=1)) / n)


def equilibration_onset(x, n_grid=40, k_sigma=2.0):
    """Earliest t0 from which the two halves of x[t0:] agree within k_sigma
    of their combined correlation-corrected errors; None if never."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    for t0 in np.linspace(0, int(0.8 * n), n_grid, dtype=int):
        seg = x[t0:]
        if seg.size < 32:
            break
        h = seg.size // 2
        a, b = seg[:h], seg[h:]
        t_a, _ = tau_int(a)
        t_b, _ = tau_int(b)
        e = math.hypot(err_of_mean(a, t_a), err_of_mean(b, t_b))
        if not np.isfinite(e):
            continue
        if abs(a.mean() - b.mean()) <= k_sigma * e:
            return int(t0)
    return None


def tail_statistics(x):
    """(t0, tail, tau, err) of one series: the tail starts at the onset, or
    at the middle when the block test never passes."""
    t0 = equilibration_onset(x)
    tail = x[t0:] if t0 is not None else x[len(x) // 2:]
    tau, _ = tau_int(tail)
    return t0, tail, tau, err_of_mean(tail, tau)


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------

def read_traces(paths):
    """{(file name, T, start): {column: array}} from Metropolis trace files."""
    out = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(resolve(p), newline="") as fh:
            for row in csv.DictReader(fh):
                key = (Path(p).name, float(row["T_K"]), row.get("start", "?"))
                for col, val in row.items():
                    if col in ("T_K", "mode", "seed", "start", "phase"):
                        continue
                    try:
                        out[key][col].append(float(val))
                    except (TypeError, ValueError):
                        pass
    return {k: {c: np.asarray(v) for c, v in d.items()} for k, d in out.items()}


def pick_pair(cols, wanted=None):
    """Column of the Warren-Cowley pair to analyse: `wanted` ('A-B', either
    order), or the pair with the largest |mean| (the first of equal ones)."""
    alphas = [c for c in cols if c.startswith("alpha_")]
    if wanted:
        a, b = wanted.split("-")
        hit = [c for c in alphas if c[len("alpha_"):] in (f"{a}-{b}", f"{b}-{a}")]
        if not hit:
            raise ValueError(f"pair {wanted} not in the trace "
                             f"({[c[6:] for c in alphas]})")
        return hit[0]
    return max(alphas, key=lambda c: abs(float(np.mean(cols[c])))) if alphas else None


def equilibration_table(data, pair=None, target=0.01, target_E=0.10):
    """One row per (trace, T, start, observable), sorted by file, falling T
    and start. target / target_E: precision for the 'need' column (alpha_1,
    and E in meV/atom)."""
    rows = []
    for key in sorted(data, key=lambda k: (k[0], -k[1], k[2])):
        fname, T, start = key
        cols = data[key]
        col_pair = pick_pair(cols, pair)
        observables = [(ENERGY[0], ENERGY[1], target_E)]
        if col_pair:
            observables.append((col_pair[len("alpha_"):], col_pair, target))
        for label, col, tgt in observables:
            x = cols[col]
            t0, tail, tau, e = tail_statistics(x)
            n_eff = tail.size / (2 * tau) if np.isfinite(tau) else float("nan")
            need = (2 * tau * float(tail.var(ddof=1)) / tgt ** 2
                    + (t0 or 0)) if np.isfinite(tau) else float("nan")
            rows.append(dict(file=fname, T_K=T, start=start, observable=label,
                             sweeps=x.size, t_eq="-" if t0 is None else str(t0),
                             tau_int=f"{tau:.2f}", n_eff=f"{n_eff:.1f}",
                             mean=f"{tail.mean():.5f}", err=f"{e:.5f}",
                             need_sweeps=f"{need:.0f}"))
    return rows


def two_start_table(data, pair=None):
    """Tail means of every pair of traces at the same temperature, their
    difference, its combined error and the verdict (3 sigma)."""
    by_T = defaultdict(list)
    for (fname, T, start), cols in data.items():
        by_T[T].append((start, fname, cols))
    rows = []
    for T in sorted(by_T, reverse=True):
        entries = by_T[T]
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                (s1, f1, c1), (s2, f2, c2) = entries[i], entries[j]
                col_pair = pick_pair(c1, pair)
                todo = [ENERGY] + ([(col_pair[6:], col_pair)] if col_pair else [])
                for label, col in todo:
                    if col not in c1 or col not in c2:
                        continue
                    (_, t1, _, e1), (_, t2, _, e2) = (tail_statistics(c1[col]),
                                                      tail_statistics(c2[col]))
                    d = t1.mean() - t2.mean()
                    sd = math.hypot(e1, e2)
                    rows.append(dict(T_K=T, observable=label, file_1=f1,
                                     start_1=s1, mean_1=float(t1.mean()),
                                     file_2=f2, start_2=s2,
                                     mean_2=float(t2.mean()), diff=float(d),
                                     sigma=float(sd),
                                     met=bool(abs(d) <= 3 * sd)))
    return rows


def write_csv(path, rows):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_tables(rows, pairs):
    print(f"{'T [K]':>7} {'start':<28} {'obs':<10} {'sweeps':>7} {'t_eq':>6} "
          f"{'tau':>7} {'N_eff':>7} {'mean':>11} {'err':>9} {'need':>8}")
    for r in rows:
        print(f"{r['T_K']:7.0f} {r['start'][:28]:<28} {r['observable']:<10} "
              f"{r['sweeps']:7d} {r['t_eq']:>6} {float(r['tau_int']):7.1f} "
              f"{float(r['n_eff']):7.1f} {float(r['mean']):11.4f} "
              f"{float(r['err']):9.4f} {float(r['need_sweeps']):8.0f}")
    if pairs:
        print("\ntwo-start test")
    for r in pairs:
        ratio = abs(r["diff"]) / r["sigma"] if r["sigma"] else float("nan")
        print(f"{r['T_K']:7.0f} {r['observable']:<10} {r['start_1'][:22]:<22} vs "
              f"{r['start_2'][:22]:<22} d = {r['diff']:+9.4f} +/- "
              f"{r['sigma']:7.4f} ({ratio:5.1f} sigma)  "
              f"{'met' if r['met'] else 'NOT EQUILIBRATED'}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="<chain>.trace.csv files")
    ap.add_argument("--pair", default=None, help="pair to analyse, e.g. Mo-Ta")
    ap.add_argument("--target", type=float, default=0.01,
                    help="target precision on alpha_1 for 'need'")
    ap.add_argument("--target-E", type=float, default=0.10,
                    help="target precision on E (meV/atom) for 'need'")
    ap.add_argument("--csv", default=None, help="write the table here")
    args = ap.parse_args(argv)
    data = read_traces(args.traces)
    if not data:
        raise SystemExit("no usable rows in the given traces")
    rows = equilibration_table(data, args.pair, args.target, args.target_E)
    print_tables(rows, two_start_table(data, args.pair))
    if args.csv:
        write_csv(args.csv, rows)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
