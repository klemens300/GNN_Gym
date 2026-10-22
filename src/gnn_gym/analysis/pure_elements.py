"""Self-diffusion of the pure metals through the alloy pipeline.

    python -m gnn_gym.analysis.pure_elements configs/analysis/pure_elements.yaml

The pure metals are simulated and analysed exactly like the alloys (lattice
model, KMC with the same cell, steps and escape layer, analyze_campaign);
this module applies the alloy relations to the campaign table of the pure
runs and fits Arrhenius laws, which is the one place where the absolute
scale of the chain can be compared with measured self-diffusion.

Per run (physical time via the Debye nu0 of the element, gnn_gym.realtime):

    D_vac        vacancy MSD slope, D_vac_phys = D_vac_sim nu0 / nu0_sim
    D*_X         = c_v(T) N D_X: tracer coefficient of the atoms (D_X their
                 MSD slope in the cell with one vacancy in N sites), with c_v
                 from the formation energy of the unary reference cells
    f, f_X       correlation factors against the uncorrelated walk with the
                 jump rate Gamma = n_steps / t_sim: f = D_vac / (Gamma d^2/6),
                 f_X = N D_X / (Gamma d^2/6); in a pure metal f = 1 and f_X is
                 the tracer correlation factor of the lattice
    D_vac,rate   Gamma d^2 / 6, the noise-free uncorrelated reference

Per element: geometric means over the realisations at each temperature, and
Arrhenius fits ln D = ln D0 - Q / k_B T of D*_X (Q, D0*) and of D_vac (the
migration energy E_m), against the barrier of the unary reference cells from
the uMLIP (reference dataset) and from the model (referee predictions).

Output: pure_elements_diffusion.csv (one row per element) and
pure_elements_points.csv (one row per element and temperature).
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

from .. import realtime
from ..paths import load_yaml, resolve
from ..realtime.nu0 import load_table
from ..realtime.vacancy import pure_formation_energy

K_B = 8.617333262e-5          # eV/K
EV_KJ = 96.485                # kJ/mol per eV


def read_csv(path):
    with open(resolve(path), newline="") as fh:
        return list(csv.DictReader(fh))


def barrier_checks(elements, jump_csvs, predictions_csv):
    """(uMLIP, model) barrier of the unary reference cells: (mean, sd, n)."""
    uma = {e: [] for e in elements}
    for path in jump_csvs:
        for r in read_csv(path):
            if r["subsystem"] in uma and r.get("neb_conv", "True") == "True":
                uma[r["subsystem"]].append(float(r["E_fwd_eV"]))
    mod = {e: [] for e in elements}
    for r in read_csv(predictions_csv):
        if r["subsystem"] in mod:
            mod[r["subsystem"]].append(float(r["Ea_pred_eV"]))
    stat = lambda v: (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)  # noqa: E731
    return ({e: stat(v) for e, v in uma.items()},
            {e: stat(v) for e, v in mod.items()})


def arrhenius(T, D):
    """(Q [eV], D0, Q_err, D0_err) of ln D = ln D0 - Q / k_B T."""
    x = 1.0 / (K_B * np.asarray(T, float))
    y = np.log(np.asarray(D, float))
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan
    if len(x) == 2:
        slope, icpt = np.polyfit(x, y, 1)
        return -slope, float(np.exp(icpt)), np.nan, np.nan
    (slope, icpt), cov = np.polyfit(x, y, 1, cov=True)
    return (-slope, float(np.exp(icpt)),
            float(np.sqrt(cov[0, 0])), float(np.exp(icpt) * np.sqrt(cov[1, 1])))


def gmean(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, float)))))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/analysis/pure_elements.yaml")
    ap.add_argument("--runs-csv", default=None,
                    help="campaign_runs.csv of the pure runs (replaces the "
                         "config's)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    elements = list(cfg["systems"])
    f0 = float(cfg["tracer_correlation_factor"])
    rt = load_yaml(realtime.CONFIG)
    nu_cfg = rt["nu0"]
    table = load_table(nu_cfg["grid"], nu_cfg["fit"], nu_cfg["lattice"])

    runs = read_csv(args.runs_csv or cfg["runs_csv"])
    uma_b, mod_b = barrier_checks(elements, cfg["jumps"], cfg["predictions"])

    points, rows = [], []
    print(f"[pure] {len(runs)} runs")
    for e in elements:
        R = [r for r in runs if r["alloy"] == e]
        if not R:
            print(f"[pure] {e}: no runs")
            continue
        el = table.elastic({e: 1.0})
        nu0 = realtime.nu0_phys(e)
        S_f = float(rt["S_f_kB"])
        E_f = pure_formation_energy(rt["c_v"][e]["pure"], e)
        print(f"\n{e}: E_f {E_f:.3f} eV, nu0 {nu0/1e12:.2f} THz, a {el['a']:.4f} A, "
              f"{len(R)} runs")
        print(f"  {'T':>6} {'n':>2} {'D_vac phys':>11} {'(min/max)':>9} {'f':>6} "
              f"{'f_X':>6} {'c_v':>10} {'D*_X':>10} {'(min/max)':>9}  flags")
        per_T = defaultdict(list)
        for r in R:
            Tk = float(r["T_K"])
            nu_sim = float(r["nu0_sim"])
            n_at = int(r["n_sites"]) - 1
            gam = int(r["n_steps"]) / float(r["t_sim_s"])
            jd = float(r["jump_distance_A"])
            d_vac = float(r["D_vac_A2s"]) * 1e-20 * nu0 / nu_sim     # m^2/s
            dX = float(r.get(f"D_{e}_A2s", "nan") or "nan")
            f_vac = float(r["D_vac_A2s"]) / (gam * jd ** 2 / 6.0)
            f_X = dX * n_at / (gam * jd ** 2 / 6.0)
            cv = realtime.c_v(e, Tk)
            d_star = cv * n_at * dX * 1e-20 * nu0 / nu_sim
            d_rate = gam * jd ** 2 / 6.0 * 1e-20 * nu0 / nu_sim
            per_T[Tk].append(dict(run=r["run"], d_vac=d_vac, d_star=d_star,
                                  f=f_vac, f_X=f_X, cv=cv, d_rate=d_rate,
                                  exp=float(r["msd_exp"]),
                                  esc=float(r["escape_frac"]),
                                  flags=r.get("flags", "") or ""))
        T, Dv, Ds, Dr, fs, fX = [], [], [], [], [], []
        for Tk in sorted(per_T):
            P = per_T[Tk]
            dv, ds = [p["d_vac"] for p in P], [p["d_star"] for p in P]
            fl = ";".join(sorted({p["flags"] for p in P if p["flags"]})) or "-"
            T.append(Tk)
            Dv.append(gmean(dv))
            Ds.append(gmean(ds))
            Dr.append(gmean([p["d_rate"] for p in P]))
            fs.append(np.mean([p["f"] for p in P]))
            fX.append(np.mean([p["f_X"] for p in P]))
            points.append(dict(
                element=e, T_K=Tk, n_realisations=len(P),
                D_vac_phys_m2s=Dv[-1], D_vac_min_m2s=min(dv),
                D_vac_max_m2s=max(dv), f_mean=fs[-1], f_X_mean=fX[-1],
                c_v=P[0]["cv"], D_star_m2s=Ds[-1], D_star_min_m2s=min(ds),
                D_star_max_m2s=max(ds), D_vac_rate_phys_m2s=Dr[-1],
                msd_exp_mean=np.mean([p["exp"] for p in P]),
                escape_frac_max=max(p["esc"] for p in P), nu0_phys_Hz=nu0,
                flags=fl))
            print(f"  {Tk:6.0f} {len(P):2d} {Dv[-1]:11.3e} {max(dv)/min(dv):9.2f} "
                  f"{fs[-1]:6.3f} {fX[-1]:6.3f} {P[0]['cv']:10.3e} {Ds[-1]:10.3e} "
                  f"{max(ds)/min(ds):9.2f}  {fl}")
        Em, D0v, Em_err, _ = arrhenius(T, Dv)
        Em_rate, _, _, _ = arrhenius(T, Dr)
        Q, D0s, Q_err, D0s_err = arrhenius(T, Ds)
        um, mo = uma_b[e], mod_b[e]
        print(f"  D*_X Arrhenius: Q = {Q:.3f} +- {Q_err:.3f} eV = "
              f"{Q*EV_KJ:.0f} kJ/mol, D0* = {D0s*1e4:.3e} cm2/s;  "
              f"Q - E_f = {Q - E_f:.3f} eV")
        print(f"  D_vac Arrhenius: E_m = {Em:.3f} +- {Em_err:.3f} eV "
              f"(rate-based {Em_rate:.3f}); model barrier (referee) "
              f"{mo[0]:.3f} +- {mo[1]:.3f} (n={mo[2]}), uMLIP {um[0]:.3f} "
              f"(n={um[2]})")
        print(f"  checks: <f> = {np.mean(fs):.3f} (exact 1), <f_X> = "
              f"{np.mean(fX):.3f} (lattice f0 {f0})")
        rows.append(dict(
            element=e, n_T=len(T), n_runs=len(R),
            T_K=";".join(f"{t:.0f}" for t in T),
            Q_eV=Q, Q_err_eV=Q_err, Q_kJ_mol=Q * EV_KJ, D0_star_m2s=D0s,
            D0_star_cm2s=D0s * 1e4, D0_star_err_cm2s=D0s_err * 1e4,
            E_m_kmc_eV=Em, E_m_kmc_err_eV=Em_err, E_m_rate_eV=Em_rate,
            D0_vac_m2s=D0v, E_m_model_referee_eV=mo[0],
            E_m_model_referee_sd_eV=mo[1], n_referee=mo[2],
            E_m_uma_eV=um[0], E_m_uma_sd_eV=um[1], n_uma=um[2],
            E_f_eV=E_f, S_f_kB=S_f, f_mean=float(np.mean(fs)),
            f_X_mean=float(np.mean(fX)), f0_bcc=f0,
            a_A=el["a"], C11_GPa=el["C11"], C12_GPa=el["C12"],
            C44_GPa=el["C44"], nu0_Hz=nu0,
            flags=";".join(sorted({p["flags"] for P in per_T.values()
                                   for p in P if p["flags"]}))))

    out = resolve(args.out_dir or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("pure_elements_diffusion.csv", rows),
                       ("pure_elements_points.csv", points)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"[pure] -> {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
