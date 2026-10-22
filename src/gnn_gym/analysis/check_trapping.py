"""Is a KMC run diffusing, or confined to a basin?

A run that has reached its step target can still be uninformative. At low
temperature the BKL loop can cycle through a few configurations joined by
low barriers: every step is accepted with a tiny residence time, the vacancy
does not get anywhere, and the clock stalls. A diffusion coefficient
computed from such a run looks like any other. This module measures it.

    clock per decile   share of the simulated time earned in each tenth of
                       the steps; a stalled run earns almost all of it early
    back-jumps         fraction of steps that return the vacancy to the site
                       it held before the previous step (1/8 for equal
                       barriers; with spread barriers much higher, because
                       detailed balance makes a jump downhill cheap to undo)
    basin size         distinct vacancy sites in a 200-step window and over
                       the run
    D, MSD exponent    from each half and from the whole run
                       (gnn_gym.kmc.observables.vacancy_diffusion); the
                       halves must agree and alpha >= 0.8 for a converged D
    correlation f      D / (Gamma d^2 / 6), Gamma = steps per simulated time;
                       an escape counts as one step, so f is an upper bound
                       in runs with escapes
    min barrier        lowest candidate barrier per decile (BKL steps only)

Flags (from the clock, not from the back-jump fraction):
    clock_stopped      the last tenth of the steps earned < 1 % of the time
    clock_slowing      mean dt of the last decile < 1/10 of the median decile
    limit_cycle        deciles 2-10 carry equal time shares (CV < 5 %) and the
                       vacancy visits fewer than 10 sites per 200-step window
    subdiffusive       MSD exponent of the whole run < 0.8
    halves_disagree    D of the two halves differ by more than a factor 2

    python -m gnn_gym.analysis.check_trapping runs/escape_local/kmc_* \\
        [--csv results/kmc/trapping.csv]
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

from ..kmc.checkpoint import find_checkpoint, load_run
from ..kmc.observables import vacancy_diffusion
from ..paths import resolve

F_TRACER_BCC = 0.727       # reference printed with f
WINDOW = 200               # steps per basin-size window


def expand(patterns):
    """Run directories from paths or glob patterns (repository-relative)."""
    out = []
    for p in patterns:
        p = str(resolve(p))
        out.extend(sorted(glob.glob(p)) or [p])
    return [d for d in out if Path(d).is_dir()]


def decile_clock(dt, n):
    """(edges, time share, mean dt) of the ten deciles of the steps."""
    edges = np.linspace(0, n, 11).astype(int)
    t_total = dt[:n].sum()
    shares = np.array([dt[a:b].sum() / t_total if t_total > 0 else np.nan
                       for a, b in zip(edges[:-1], edges[1:])])
    mean_dt = np.array([dt[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    return edges, shares, mean_dt


def window_sites(vac, n, win=WINDOW):
    """Distinct vacancy sites in 12 windows of `win` steps across the run."""
    if n <= 4 * win:
        return []
    probes = np.linspace(0, n - win, 12).astype(int)
    return [len(np.unique(vac[p:p + win])) for p in probes]


def flags(shares, mean_dt, sites, alpha, ratio):
    out = []
    med = np.median(mean_dt)
    dt_ratio = mean_dt[-1] / med if med > 0 else np.nan
    if shares[-1] < 0.01:
        out.append("clock_stopped")
    elif dt_ratio < 0.1:
        out.append("clock_slowing")
    cv = (shares[1:].std() / shares[1:].mean()
          if shares[1:].mean() > 0 else np.nan)
    if cv < 0.05 and bool(sites) and np.median(sites) < 10:
        out.append("limit_cycle")
    if np.isfinite(alpha) and alpha < 0.8:
        out.append("subdiffusive")
    if np.isfinite(ratio) and not (0.5 <= ratio <= 2.0):
        out.append("halves_disagree")
    return out


def halves(pos, t_pos, n):
    """D and MSD exponent of the first half, second half, whole run."""
    t_half = t_pos[n // 2]
    d1, _, a1 = vacancy_diffusion(pos, t_pos, t_pos[0], t_half)
    d2, _, a2 = vacancy_diffusion(pos, t_pos, t_half, t_pos[n])
    dall, r2, aall = vacancy_diffusion(pos, t_pos, t_pos[0], t_pos[n])
    ratio = d2 / d1 if (np.isfinite(d1) and np.isfinite(d2) and d1 > 0) \
        else np.nan
    return dict(t_half=t_half, d1=d1, a1=a1, d2=d2, a2=a2, dall=dall,
                r2=r2, aall=aall, ratio=ratio)


def report(run_dir):
    path = find_checkpoint(run_dir)
    name = Path(run_dir).name
    if path is None:
        print(f"  {name}: no checkpoint")
        return None
    r = load_run(path, snapshots=False)
    dt = np.asarray(r.delta_t_s, np.float64)
    vac = np.asarray(r.vacancy_indices, np.int64)
    pos = np.asarray(r.vacancy_positions_unfolded, np.float64)
    bar = np.asarray(r.barriers_eV, np.float64)
    esc = r.is_escape
    n = min(r.n_steps, len(dt))
    T = r.T_K

    print(f"\n{'=' * 74}")
    print(f"{name}   T = {T:.0f} K   {n:,} steps   "
          f"t_sim = {r.total_time_s:.4g} s")
    print("=" * 74)

    edges, shares, mean_dts = decile_clock(dt, n)
    back = vac[2:n + 1] == vac[0:n - 1]
    has_esc = r.escape_layer
    head = (f"{'decile':>7} {'steps':>9} {'t share':>9} {'mean dt':>11} "
            f"{'back-jump':>10} {'min E_a':>9} {'sites':>6}")
    if has_esc:
        head += f" {'escapes':>9}"
    print(head)
    emins = []
    for k in range(10):
        a, b = edges[k], edges[k + 1]
        bj = back[max(a - 1, 0):max(b - 1, 0)]
        blk = bar[a:b]
        row_ok = np.isfinite(blk).all(axis=1) if blk.size else np.zeros(0, bool)
        emin = blk[row_ok].min(axis=1).mean() if row_ok.any() else np.nan
        emins.append(emin)
        row = (f"{k + 1:>7} {b - a:>9,} {shares[k]:>8.2%} "
               f"{dt[a:b].mean():>11.3e} "
               f"{(bj.mean() if bj.size else np.nan):>9.1%} {emin:>8.3f} "
               f"{len(np.unique(vac[a:b + 1])):>6}")
        if has_esc:
            row += f" {esc[a:b].mean():>8.2%}"
        print(row)
    if has_esc:
        print(f"\n  escape layer fired on {esc[:n].mean():.2%} of all steps "
              f"({int(esc[:n].sum()):,} of {n:,})")
    else:
        print("\n  plain BKL: no escape layer in this run")
    print("  back-jump reference: 12.5% for EQUAL barriers; with spread "
          "barriers 60-75% is expected (E_back = E_fwd - dE)")

    sites = window_sites(vac, n)
    if sites:
        print(f"  distinct sites per {WINDOW}-step window, across the run:")
        print("   ", " ".join(f"{u:>4}" for u in sites))
    n_sites_run = len(np.unique(vac[:n + 1]))
    print(f"  distinct sites over the whole run: {n_sites_run:,}")

    t_pos = np.concatenate([[0.0], np.asarray(r.times_s, np.float64)])[:n + 1]
    h = halves(pos[:n + 1], t_pos, n)
    print(f"\n  {'':16} {'D [A^2/s]':>12} {'MSD exp.':>9}")
    print(f"  {'first half':16} {h['d1']:>12.4g} {h['a1']:>9.2f}   "
          f"(t = 0 ... {h['t_half']:.4g} s)")
    print(f"  {'second half':16} {h['d2']:>12.4g} {h['a2']:>9.2f}   "
          f"(t = {h['t_half']:.4g} ... {t_pos[n]:.4g} s)")
    print(f"  {'whole run':16} {h['dall']:>12.4g} {h['aall']:>9.2f}")
    if np.isfinite(h["ratio"]):
        print(f"  ratio second/first  {h['ratio']:>8.3f}"
              "   (agreement within a factor ~2 is the bar for one "
              "realisation)")
    print(f"  net displacement^2 {h['r2']:>11.4g} A^2  "
          f"({np.sqrt(h['r2']):.1f} A over the run)")

    t_total = dt[:n].sum()
    f_app = np.nan
    jd = r.jump_distance_A
    if np.isfinite(jd) and t_total > 0 and np.isfinite(h["dall"]):
        gamma = n / t_total
        f_app = h["dall"] / (gamma * jd ** 2 / 6.0)
        note = (" (upper bound: escape macro-steps counted as one jump)"
                if has_esc and esc[:n].any() else "")
        print(f"  correlation factor f = D/(Gamma d^2/6) = {f_app:.3g}   "
              f"against {F_TRACER_BCC} for an ideal bcc tracer{note}")

    med = np.median(mean_dts)
    dt_ratio = mean_dts[-1] / med if med > 0 else np.nan
    share_cv = (shares[1:].std() / shares[1:].mean()
                if shares[1:].mean() > 0 else np.nan)
    fl = flags(shares, mean_dts, sites, h["aall"], h["ratio"])
    print()
    if "clock_stopped" in fl:
        print(f"  ! clock stopped: the final tenth of the steps earned "
              f"{shares[-1]:.2%} of the simulated time.")
    if "clock_slowing" in fl:
        print(f"  ! clock slowing: final-decile mean dt is {dt_ratio:.2f} of "
              f"the median decile. The run is entering a trap.")
    if "limit_cycle" in fl:
        print(f"  ! limit cycle: deciles 2-10 carry equal time shares "
              f"(CV {share_cv:.1%}) on a basin of ~{int(np.median(sites))} "
              f"sites. The trajectory is a periodic orbit; steps after the "
              f"first decile carry no information.")
    if "subdiffusive" in fl:
        print(f"  ! MSD exponent {h['aall']:.2f} < 0.8: not in the diffusive "
              f"regime on the lags probed; D is not a converged number.")
    if "halves_disagree" in fl:
        print(f"  ! D of the two halves differ by a factor {h['ratio']:.2f}.")
    if not fl:
        print(f"  no trapping signature: final tenth carries "
              f"{shares[-1]:.1%} of the time, MSD exponent {h['aall']:.2f}, "
              f"halves agree ({h['ratio']:.2f}).")

    return dict(
        run=name, T_K=T, n_steps=n, t_sim_s=float(t_total),
        last_decile_share=float(shares[-1]), dt_ratio_last_median=float(dt_ratio),
        share_cv=float(share_cv),
        sites_per_window_median=float(np.median(sites)) if sites else np.nan,
        sites_total=n_sites_run,
        escape_frac=float(esc[:n].mean()) if has_esc else 0.0,
        D_first=h["d1"], D_second=h["d2"], D_all=h["dall"], D_ratio=h["ratio"],
        msd_exp_first=h["a1"], msd_exp_second=h["a2"], msd_exp_all=h["aall"],
        f_apparent=f_app, min_Ea_mean=float(np.nanmean(emins)),
        flags=";".join(fl))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories or glob patterns")
    ap.add_argument("--csv", default=None,
                    help="write one summary row per run to this CSV")
    args = ap.parse_args(argv)
    rows = [r for r in map(report, expand(args.runs)) if r is not None]
    if args.csv and rows:
        out = resolve(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
