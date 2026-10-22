"""The escape layer against plain BKL KMC over the same steps.

    python -m gnn_gym.analysis.escape_vs_plain configs/analysis/escape_vs_plain.yaml

Each pair is one run with the escape layer and one without, from the same
seed: the same start configuration and the same random-number stream, so the
two trajectories coincide step for step until the first escape. Both are cut
to the same number of steps (gnn_gym.kmc.result.KMCRun.head; the escape runs
may be longer) and compared on what the escape layer is meant to change, the
simulated time per step, and on what it must not change, the vacancy
diffusion coefficient. D_vac is the time-averaged MSD slope of the whole cut
run and the exponent its log-log slope, as in analyze_campaign; the flags are
those of check_trapping. The same temperature gives both runs the same
conversion to physical time, so the ratios hold in physical time as well.

One row per pair is written to `out`.
"""
import argparse
import csv
import sys

import numpy as np

from ..kmc.checkpoint import find_checkpoint, load_run
from ..paths import load_yaml, resolve
from .check_trapping import decile_clock, flags, halves, window_sites


def run_summary(run):
    """t_sim, D_vac, MSD exponent, flags and visited sites of a (cut) run."""
    n = run.n_steps
    dt = np.asarray(run.delta_t_s[:n], np.float64)
    vac = np.asarray(run.vacancy_indices[:n + 1], np.int64)
    t_pos = run.times_with_zero_s[:n + 1]
    h = halves(np.asarray(run.vacancy_positions_unfolded[:n + 1]), t_pos, n)
    _, shares, mdts = decile_clock(dt, n)
    sites = window_sites(vac, n)
    return dict(t_sim_s=float(dt.sum()), D_vac_A2s=h["dall"],
                msd_exp=h["aall"], D_ratio_halves=h["ratio"],
                escape_frac=float(np.mean(run.is_escape[:n])),
                sites_total=int(len(np.unique(vac))),
                last_decile_share=float(shares[-1]),
                flags=";".join(flags(shares, mdts, sites, h["aall"], h["ratio"])))


def compare(escape_dir, plain_dir, n_steps):
    esc = load_run(find_checkpoint(resolve(escape_dir)), snapshots=False)
    pln = load_run(find_checkpoint(resolve(plain_dir)), snapshots=False)
    if (esc.system, esc.T_K) != (pln.system, pln.T_K):
        raise ValueError(f"{escape_dir} and {plain_dir} are not the same point")
    n = min(int(n_steps), esc.n_steps, pln.n_steps)
    esc, pln = esc.head(n), pln.head(n)
    same_start = bool(np.array_equal(esc.initial_species, pln.initial_species)
                      and esc.initial_vacancy_index == pln.initial_vacancy_index)
    differ = np.flatnonzero(esc.vacancy_indices != pln.vacancy_indices)
    first_escape = np.flatnonzero(esc.is_escape)
    e, p = run_summary(esc), run_summary(pln)
    row = dict(system=esc.system, T_K=esc.T_K, n_steps=n, same_start=same_start,
               first_escape_step=int(first_escape[0]) if first_escape.size else -1,
               first_difference_step=int(differ[0]) if differ.size else -1,
               time_ratio=e["t_sim_s"] / p["t_sim_s"],
               D_vac_ratio=e["D_vac_A2s"] / p["D_vac_A2s"])
    for tag, s in (("escape", e), ("plain", p)):
        row.update({f"{k}_{tag}": v for k, v in s.items()})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/analysis/escape_vs_plain.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    rows = []
    for pair in cfg["pairs"]:
        r = compare(pair["escape"], pair["plain"], cfg["n_steps"])
        rows.append(r)
        print(f"{r['system']} {r['T_K']:.0f} K, {r['n_steps']} steps: "
              f"t_sim escape/plain = {r['time_ratio']:.3g}, D_vac escape/plain "
              f"= {r['D_vac_ratio']:.3g} (MSD exponents {r['msd_exp_escape']:.2f}"
              f" / {r['msd_exp_plain']:.2f}; plain flags "
              f"{r['flags_plain'] or '-'}); same start {r['same_start']}, "
              f"first escape at step {r['first_escape_step']}, trajectories "
              f"differ from step {r['first_difference_step']}")
    out = resolve(args.out or cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
