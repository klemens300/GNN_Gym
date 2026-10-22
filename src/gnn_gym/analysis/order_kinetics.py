"""How much of the equilibrium order the vacancy builds, and how fast.

    python -m gnn_gym.analysis.order_kinetics configs/analysis/order_kinetics.yaml

Reads the tables of analyze_campaign (campaign_runs, campaign_alpha,
campaign_tau of the alloy campaign) and the equilibrium values of the
Metropolis campaign. The order of a run is the first-shell Warren-Cowley
parameter of the leading pair of its alloy; alpha_eq is the cooling branch of
the Metropolis campaign, mean over the seeds, over the equilibrated
temperatures, interpolated linearly in T to the KMC temperatures
(gnn_gym.metropolis.analysis, alpha_eq_kmc.csv).

order_reached.csv          per run: alpha_1 at the end of the run (mean over
                           the last `end_fraction` of its curve), alpha_eq,
                           their ratio (the share of the equilibrium order),
                           the physical time the run covers
order_reached_summary.csv  per alloy and temperature: mean, smallest and
                           largest share and alpha_1 over the realisations,
                           the physical time of the run (mean) and of its
                           first `step_max` steps (mean), the share reached
                           at `step_max` (curve of order_kinetics_curves)
order_kinetics_curves.csv  alpha_1 / alpha_eq against the KMC step, first
                           `step_max` steps, mean over the realisations on
                           the step grid of the shortest run (figure data)
ordering_times.csv         per run: the ordering time, the relaxation
                           alpha(t) = alpha_eq + (alpha_0 - alpha_eq)
                           exp(-t / tau) with alpha_eq pinned, fitted to the
                           physical-time curve (campaign_tau, tau_real_fixed_s),
                           with the number of tau the run covers
start_order.csv            per alloy: range of alpha_1 over all pairs and
                           runs in the random start configurations
"""
import argparse
import sys

import numpy as np
import pandas as pd

from ..paths import load_yaml, resolve
from .analyze_campaign import alpha_eq_at, load_alpha_eq


def end_value(curve, pair, fraction):
    """alpha_1 of `pair` at the end of a run: mean over the last points."""
    n = max(3, int(len(curve) * fraction))
    return float(curve[pair].iloc[-n:].mean())


def order_reached(runs, alpha, eq, lead, temperatures, fraction):
    rows = []
    for _, r in runs.iterrows():
        system, T = r.alloy, float(r.T_K)
        if system not in lead or T not in temperatures:
            continue
        pair = lead[system]
        curve = alpha[alpha.run == r.run].sort_values("step")
        a_end = end_value(curve, pair, fraction)
        a_eq = alpha_eq_at(eq[system], *pair.split("-"), T)
        rows.append(dict(run=r.run, system=system, T_K=T, pair=pair,
                         alpha_end=a_end, alpha_eq=a_eq, share=a_end / a_eq,
                         t_real_s=float(curve.t_real_s.iloc[-1]),
                         n_steps=int(r.n_steps),
                         flags=r["flags"] if isinstance(r["flags"], str) else ""))
    return pd.DataFrame(rows)


def kinetics_curves(tab, alpha, step_max):
    """Mean alpha_1/alpha_eq over the realisations against the step, and per
    (system, T) the mean physical time of the first step_max steps."""
    rows, t_head = [], {}
    for (system, T), sub in tab.groupby(["system", "T_K"]):
        curves, t_end = [], []
        for _, r in sub.iterrows():
            d = alpha[alpha.run == r.run].sort_values("step")
            curves.append(pd.Series(d[r.pair].values / r.alpha_eq,
                                    index=d.step.values))
            t_end.append(float(d[d.step <= step_max].t_real_s.iloc[-1]))
        grid = min(curves, key=len).index.values
        y = np.mean([np.interp(grid, c.index.values, c.values) for c in curves],
                    axis=0)
        keep = grid <= step_max
        rows.extend(dict(system=system, T_K=T, step=int(s), share=float(v))
                    for s, v in zip(grid[keep], y[keep]))
        t_head[(system, T)] = float(np.mean(t_end))
    return pd.DataFrame(rows), t_head


def summary(tab, curves, t_head):
    agg = (tab.groupby(["system", "T_K"])
           .agg(n_runs=("share", "size"), alpha_end=("alpha_end", "mean"),
                alpha_end_min=("alpha_end", "min"),
                alpha_end_max=("alpha_end", "max"),
                alpha_eq=("alpha_eq", "first"), share=("share", "mean"),
                share_min=("share", "min"), share_max=("share", "max"),
                t_real_s=("t_real_s", "mean"))
           .reset_index())
    last = curves.sort_values("step").groupby(["system", "T_K"]).last()
    agg["step_max"] = [int(last.loc[(s, T), "step"])
                       for s, T in zip(agg.system, agg.T_K)]
    agg["share_at_step_max"] = [float(last.loc[(s, T), "share"])
                                for s, T in zip(agg.system, agg.T_K)]
    agg["t_real_at_step_max_s"] = [t_head[(s, T)]
                                   for s, T in zip(agg.system, agg.T_K)]
    return agg


def ordering_times(tab, tau):
    rows = []
    for _, r in tab.iterrows():
        f = tau[(tau.run == r.run) & (tau.pair == r.pair)]
        if f.empty:
            continue
        f = f.iloc[0]
        t = float(f.get("tau_real_fixed_s", np.nan))
        rows.append(dict(run=r.run, system=r.system, T_K=r.T_K, pair=r.pair,
                         alpha_eq=float(f.alpha_eq), tau_real_s=t,
                         status=f.get("status_fixed", ""),
                         fraction_completed=float(
                             f.get("fraction_completed_fixed", np.nan)),
                         t_real_run_s=r.t_real_s, spans=r.t_real_s / t))
    return pd.DataFrame(rows)


def start_order(alpha, systems):
    rows = []
    for system in systems:
        d = alpha[(alpha.alloy == system) & (alpha.step == 0)]
        pairs = [c for c in d.columns if "-" in c and d[c].notna().any()]
        unlike = [p for p in pairs if p.split("-")[0] != p.split("-")[1]]
        v_all = d[pairs].values.ravel()
        v_unlike = d[unlike].values.ravel()
        rows.append(dict(system=system, n_runs=len(d),
                         alpha_min=float(np.nanmin(v_all)),
                         alpha_max=float(np.nanmax(v_all)),
                         alpha_unlike_min=float(np.nanmin(v_unlike)),
                         alpha_unlike_max=float(np.nanmax(v_unlike))))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/analysis/order_kinetics.yaml")
    ap.add_argument("--campaign", default=None,
                    help="directory of the analyze_campaign tables")
    ap.add_argument("--alpha-eq", default=None, metavar="CSV",
                    help="replaces the config's alpha_eq table")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)

    camp = resolve(args.campaign or cfg["campaign"])
    runs = pd.read_csv(camp / "campaign_runs.csv")
    alpha = pd.read_csv(camp / "campaign_alpha.csv")
    tau = pd.read_csv(camp / "campaign_tau.csv")
    eq = load_alpha_eq(args.alpha_eq or cfg["alpha_eq"])
    lead = dict(cfg["lead_pair"])
    temps = [float(t) for t in cfg["temperatures"]]
    out = resolve(args.out_dir or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    tab = order_reached(runs, alpha, eq, lead, temps, float(cfg["end_fraction"]))
    curves, t_head = kinetics_curves(tab, alpha, int(cfg["step_max"]))
    agg = summary(tab, curves, t_head)
    times = ordering_times(tab, tau)
    start = start_order(alpha, list(lead))
    for name, frame in (("order_reached.csv", tab),
                        ("order_reached_summary.csv", agg),
                        ("order_kinetics_curves.csv", curves),
                        ("ordering_times.csv", times),
                        ("start_order.csv", start)):
        frame.to_csv(out / name, index=False)
        print(f"-> {out / name}")

    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3g}"                                  # noqa: E731
    print("\norder reached (share of alpha_eq), per alloy and T:")
    print(agg.to_string(index=False, float_format=fmt))
    print("\nordering times (physical s), per alloy and T:")
    print(times.groupby(["system", "T_K"])
          .agg(tau_min_s=("tau_real_s", "min"), tau_max_s=("tau_real_s", "max"),
               spans_min=("spans", "min"), spans_max=("spans", "max"),
               n=("spans", "size"))
          .reset_index().to_string(index=False, float_format=fmt))
    print("\nrandom start configurations:")
    print(start.to_string(index=False, float_format=fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
