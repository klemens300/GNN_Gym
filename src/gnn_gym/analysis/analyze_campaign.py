"""Tables of a KMC campaign: diffusion, order kinetics, trapping diagnostics.

    python -m gnn_gym.analysis.analyze_campaign configs/analysis/campaign_alloys.yaml
        [--out-dir DIR] [--runs PATTERN ...] [--alpha-eq CSV]

Every run checkpoint is read once and the same criteria are applied to all
runs. Three CSV files are written:

campaign_runs.csv    one row per run: time conversion factor R(T) =
                     (nu0_sim / nu0_phys) / (N c_v(T)) (gnn_gym.realtime),
                     clock and basin diagnostics with their flags
                     (gnn_gym.analysis.check_trapping), vacancy D from the
                     whole run and from each half with the MSD exponents,
                     the correlation factor f = D / (Gamma d^2 / 6), and the
                     tracer D of every element.
campaign_tau.csv     one row per run and element pair: the first-shell
                     Warren-Cowley alpha_1(t) fitted by a relaxation
                     (gnn_gym.kmc.tau_order), with alpha_inf free and, when an
                     equilibrium table is given for the system, pinned to the
                     equilibrium value at T; times simulated and physical.
campaign_alpha.csv   alpha_1(t) of every run on `nsub` snapshots evenly spaced
                     in steps, for the figures.

Conventions. D_vac is the time-averaged vacancy MSD slope over the whole run
(gnn_gym.kmc.observables.vacancy_diffusion). The tracer D of element X is the
slope of the single-origin MSD of its atoms (replayed hop history, escapes
included) over the second half of the run, where it is least noisy; in the
cell with one vacancy in N sites. Snapshots are stored every k steps and
after every escape, so they are mapped to their step through the time axis
and subsampled evenly in steps. The equilibrium values are those of the
Metropolis campaign at the KMC temperatures, alpha_eq_kmc.csv of
gnn_gym.metropolis.analysis (columns system, host, other, T_K, alpha_eq);
between its temperatures the value is interpolated linearly in T, and a run
outside its range gets no pinned fit. Runs are labelled
<run directory>@<campaign directory>.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .. import realtime
from ..kmc.checkpoint import find_checkpoint, load_run
from ..kmc.observables import linear_diffusion, tracer_msd, warren_cowley
from ..kmc.tau_order import fit_tau_order
from ..paths import load_yaml, resolve
from .check_trapping import decile_clock, expand, flags, halves


# ---------------------------------------------------------------------------
# equilibrium tables
# ---------------------------------------------------------------------------

def load_alpha_eq(path):
    """{system: {(host, other): (T, alpha_eq)}} from alpha_eq_kmc.csv."""
    acc = {}
    with open(resolve(path), newline="") as fh:
        for row in csv.DictReader(fh):
            acc.setdefault(row["system"], {}) \
               .setdefault((row["host"], row["other"]), []) \
               .append((float(row["T_K"]), float(row["alpha_eq"])))
    return {system: {key: (np.array([t for t, _ in sorted(v)]),
                           np.array([a for _, a in sorted(v)]))
                     for key, v in pairs.items()}
            for system, pairs in acc.items()}


def alpha_eq_at(table, ei, ej, T):
    for key in ((ei, ej), (ej, ei)):
        if key in table:
            Ts, av = table[key]
            if Ts[0] <= T <= Ts[-1]:
                return float(np.interp(T, Ts, av))
    return np.nan


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def analyse_run(run_dir, nsub, alpha_eq):
    name = Path(run_dir).name
    path = find_checkpoint(run_dir)
    if path is None:
        print(f"  {name}: no checkpoint, skipped")
        return None
    r = load_run(path)
    variant = Path(run_dir).resolve().parent.name
    label = f"{name}@{variant}"
    system, T, n = r.system, float(r.T_K), int(r.n_steps)
    els = list(r.elements)
    present = [e for e in els if (r.initial_species == els.index(e)).any()]

    tf = realtime.time_factors(system, T, r.n_sites, r.attempt_frequency_Hz)
    R = tf["R"]

    dt = r.delta_t_s[:n]
    t_total = float(dt.sum())
    _, shares, mdts = decile_clock(dt, n)
    vac = r.vacancy_indices
    probes = np.linspace(0, n - 200, 12).astype(int)
    sites = [len(np.unique(vac[p:p + 200])) for p in probes]

    t_pos = r.times_with_zero_s[:n + 1]
    pos = r.vacancy_positions_unfolded[:n + 1]
    h = halves(pos, t_pos, n)
    jd = float(r.jump_distance_A)
    f_app = h["dall"] / ((n / t_total) * jd ** 2 / 6.0)

    D_el = {}
    try:
        msd = tracer_msd(r)
        for e in present:
            D_el[e] = linear_diffusion(t_pos, msd[e])
    except Exception as exc:
        print(f"  {name}: tracer MSD failed ({exc})")
        D_el = {e: np.nan for e in present}

    curves, tau_rows = None, []
    if r.snapshot_species is not None and r.n_snapshots >= 5:
        snaps = r.snapshot_species
        stimes = np.asarray(r.snapshot_times_s)
        snap_steps = np.searchsorted(r.times_s[:n], stimes, side="right")
        snap_steps[0] = 0
        idx = np.unique(np.searchsorted(snap_steps, np.linspace(0, n, nsub),
                                        side="left").clip(0, len(snaps) - 1))
        nn = r.final_state.neighbours
        A = np.stack([warren_cowley(snaps[k], nn, len(els)) for k in idx])
        t = stimes[idx]
        curves = dict(step=snap_steps[idx], t_sim=t, t_real=t * R, A=A)
        n_tail = max(3, len(idx) // 10)
        for i, ei in enumerate(present):
            for ej in present[i:]:
                y = A[:, els.index(ei), els.index(ej)]
                row = dict(run=label, variant=variant, alloy=system, T_K=T,
                           pair=f"{ei}-{ej}", R=R, alpha_start=float(y[0]),
                           alpha_end=float(np.nanmean(y[-n_tail:])),
                           alpha_end_scatter=float(np.nanstd(y[-n_tail:])))
                fit = fit_tau_order(t, y)
                if fit is None:
                    row.update(status="failed")
                else:
                    row.update(status=fit.status, tau_sim_s=fit.tau_order_s,
                               tau_real_s=fit.tau_order_s * R,
                               alpha_0_fit=fit.alpha_0,
                               alpha_inf_fit=fit.alpha_inf,
                               fraction_completed=fit.fraction_completed,
                               tau_upper_bound_sim_s=fit.tau_upper_bound_s,
                               tau_upper_bound_real_s=fit.tau_upper_bound_s * R,
                               rmse=fit.rmse)
                aeq = (alpha_eq_at(alpha_eq[system], ei, ej, T)
                       if system in alpha_eq else np.nan)
                row["alpha_eq"] = aeq
                if np.isfinite(aeq):
                    fx = fit_tau_order(t, y, alpha_inf_fixed=aeq)
                    if fx is not None:
                        row.update(status_fixed=fx.status,
                                   tau_sim_fixed_s=fx.tau_order_s,
                                   tau_real_fixed_s=fx.tau_order_s * R,
                                   fraction_completed_fixed=fx.fraction_completed)
                tau_rows.append(row)

    run_row = dict(
        run=label, variant=variant, alloy=system, T_K=T,
        n_sites=r.n_sites, n_steps=n,
        t_sim_s=t_total, t_real_s=t_total * R, R=R, c_v=tf["c_v"],
        nu0_sim=float(r.attempt_frequency_Hz), nu0_phys=tf["nu0_phys_Hz"],
        escape_frac=float(r.is_escape[:n].mean()),
        last_decile_share=float(shares[-1]),
        dt_ratio_last_median=float(mdts[-1] / np.median(mdts)),
        sites_per_window_median=float(np.median(sites)),
        sites_total=int(len(np.unique(vac[:n + 1]))),
        D_vac_A2s=h["dall"], D_vac_first=h["d1"], D_vac_second=h["d2"],
        D_ratio=h["ratio"], msd_exp=h["aall"], msd_exp_first=h["a1"],
        msd_exp_second=h["a2"], f_apparent=f_app,
        net_disp_A=float(np.sqrt(h["r2"])), jump_distance_A=jd,
        flags=";".join(flags(shares, mdts, sites, h["aall"], h["ratio"])),
    )
    for e in els:
        run_row[f"D_{e}_A2s"] = D_el.get(e, np.nan)
    return run_row, tau_rows, curves, present, els


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"  -> {path}  ({len(rows)} rows)")


def print_run(rr, tr):
    print(f"  T={rr['T_K']:.0f} K  t_sim={rr['t_sim_s']:.3g} s  "
          f"R={rr['R']:.3g}  D_vac={rr['D_vac_A2s']:.3g} A^2/s  "
          f"exp={rr['msd_exp']:.2f}  halves={rr['D_ratio']:.2f}  "
          f"f={rr['f_apparent']:.3g}  flags={rr['flags'] or '-'}")
    for t in tr:
        msg = (f"    {t['pair']:6} alpha {t['alpha_start']:+.3f} -> "
               f"{t['alpha_end']:+.3f}  {t['status']:12}")
        if "tau_sim_s" in t:
            msg += (f" tau_sim={t['tau_sim_s']:.3g} s "
                    f"(seen {t['fraction_completed']:.0%}, "
                    f"ub {t['tau_upper_bound_sim_s']:.3g})")
        if "tau_sim_fixed_s" in t:
            msg += (f"  | fixed a_eq={t['alpha_eq']:+.3f}: "
                    f"tau_sim={t['tau_sim_fixed_s']:.3g} s "
                    f"{t['status_fixed']}")
        print(msg)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/analysis/*.yaml")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run directories or glob patterns (replace the "
                         "config's list)")
    ap.add_argument("--alpha-eq", default=None, metavar="CSV",
                    help="equilibrium alpha_1 table (replaces the config's)")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)

    eq_path = args.alpha_eq or cfg.get("alpha_eq")
    alpha_eq = load_alpha_eq(eq_path) if eq_path else {}
    if eq_path:
        print(f"alpha_eq for {', '.join(alpha_eq)}: {eq_path}")

    out = resolve(args.out_dir or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    nsub = int(cfg.get("nsub", 400))

    run_rows, tau_rows, alpha_rows = [], [], []
    for rd in expand(args.runs or cfg["runs"]):
        print(f"\n{Path(rd).name}")
        res = analyse_run(rd, nsub, alpha_eq)
        if res is None:
            continue
        rr, tr, curves, present, els = res
        run_rows.append(rr)
        tau_rows.extend(tr)
        print_run(rr, tr)
        if curves is not None:
            for k in range(len(curves["step"])):
                row = dict(run=rr["run"], alloy=rr["alloy"], T_K=rr["T_K"],
                           step=int(curves["step"][k]),
                           t_sim_s=float(curves["t_sim"][k]),
                           t_real_s=float(curves["t_real"][k]))
                for i, a in enumerate(present):
                    for b in present[i:]:
                        row[f"{a}-{b}"] = float(
                            curves["A"][k, els.index(a), els.index(b)])
                alpha_rows.append(row)

    print()
    write_csv(out / "campaign_runs.csv", run_rows)
    write_csv(out / "campaign_tau.csv", tau_rows)
    write_csv(out / "campaign_alpha.csv", alpha_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
