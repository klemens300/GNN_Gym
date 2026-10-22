"""Diffusion in the alloys and in the pure metals, as reported.

    python -m gnn_gym.analysis.diffusion_summary configs/analysis/diffusion_summary.yaml

Reads the tables of analyze_campaign (alloy campaign), order_kinetics,
escape_vs_plain and pure_elements, and the measured self-diffusion of the
pure metals (data/literature). The vacancy's own coefficient in physical
time takes the physical attempt frequency, D_vac = D_vac(sim) nu0_phys /
nu0_sim (gnn_gym.realtime); the alloy's coefficient is D = c_v D_vac, and
the tracer coefficient of element X is D*_X = D_X(sim) N c_v nu0_phys /
nu0_sim (N = sites - 1 atoms), D times the share f_X / f = N D_X / D_vac
that X carries. Values over realisations are geometric means
for diffusion coefficients, arithmetic means otherwise. Arrhenius fits are
least squares of ln D against 1/T over the per-temperature means in
`arrhenius_K`; tracer fits use only runs that are not sub-diffusive.

kmc_table.csv              per alloy and T: D_vac (phys), its spread over the
                           realisations, MSD exponents and the number of
                           sub-diffusive runs, f, c_v, D = c_v D_vac, the
                           physical time covered, the share of the
                           equilibrium order (order_kinetics)
arrhenius_alloys.csv       per alloy: E and D0 of D_vac and of D
arrhenius_alloy_points.csv every point against the extrapolated line, also
                           the first `head_steps` of the longer runs
                           (escape_vs_plain)
alloy_ratios.csv           per T: Mo-Nb-Ta-W over Ta-W for D_vac, c_v, D, f
alloy_vs_pure_diffusion.csv  per element: D*_X and its Arrhenius parameters
                           as pure metal (pure_elements) and in each alloy
                           (figure data)
element_shares.csv         per alloy, element and T: f_X / f = D*_X / D of the
                           geometric means, and its range over the runs
element_ratios.csv         per element and T: pure metal against alloy, and
                           the spread of D*_X over the elements
pure_vs_measured.csv       per element: D* of the chain against each
                           measured Arrhenius term and their sum, at 1100 K
                           and at the ends of the measured range
pure_metals_summary.csv    per element: E_m = Q - E_f, the model and uMLIP
                           barriers, E_f, Q and D0* against the measurement,
                           correlation factors; last row over all runs
"""
import argparse
import csv
import sys

import numpy as np
import pandas as pd

from ..paths import load_yaml, resolve

K_B = 8.617333262e-5           # eV/K
EV_KJ = 96.485                 # kJ/mol per eV (as pure_elements)
R_KJ = 8.314462618e-3          # kJ/(mol K)


def gmean(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, float)))))


def fit_ev(T, D):
    """(E [eV], D0) of ln D = ln D0 - E / k_B T."""
    slope, icpt = np.polyfit(1.0 / (K_B * np.asarray(T, float)),
                             np.log(np.asarray(D, float)), 1)
    return float(-slope), float(np.exp(icpt))


def fit_kj(T, D):
    """(D0, Q [kJ/mol]) of ln D = ln D0 - Q / R T."""
    slope, icpt = np.polyfit(1.0 / (R_KJ * np.asarray(T, float)),
                             np.log(np.asarray(D, float)), 1)
    return float(np.exp(icpt)), float(-slope)


# ---------------------------------------------------------------------------
# vacancy in the alloys
# ---------------------------------------------------------------------------

def alloy_runs(runs, systems, temperatures):
    d = runs[runs.alloy.isin(systems)].copy()
    d["T_K"] = d.T_K.astype(float)
    d = d[d.T_K.isin(temperatures)]
    d["D_vac_m2s"] = d.D_vac_A2s * 1e-20 * d.nu0_phys / d.nu0_sim
    d["subdiffusive"] = d["flags"].fillna("").str.contains("subdiffusive")
    return d


def kmc_table(d, order):
    rows = []
    for (system, T), g in d.groupby(["alloy", "T_K"]):
        dv = gmean(g.D_vac_m2s)
        cv = float(g.c_v.iloc[0])
        o = order[(order.system == system) & (order.T_K.astype(float) == T)]
        rows.append(dict(
            system=system, T_K=T, n_runs=len(g), n_steps_max=int(g.n_steps.max()),
            D_vac_m2s=dv, D_vac_min_m2s=float(g.D_vac_m2s.min()),
            D_vac_max_m2s=float(g.D_vac_m2s.max()),
            D_vac_spread=float(g.D_vac_m2s.max() / g.D_vac_m2s.min()),
            msd_exp_min=float(g.msd_exp.min()), msd_exp_max=float(g.msd_exp.max()),
            n_subdiffusive=int(g.subdiffusive.sum()),
            f=float(g.f_apparent.mean()), c_v=cv, D_m2s=cv * dv,
            t_real_s=float(g.t_real_s.mean()),
            t_real_a=float(g.t_real_s.mean()) / (365.25 * 86400.0),
            order_share=float(o.share.iloc[0]) if len(o) else np.nan,
            sites_total_max=int(g.sites_total.max())))
    return pd.DataFrame(rows)


def arrhenius_alloys(table, d, head, T_fit):
    fits, points = [], []
    for system, g in table.groupby("system"):
        sel = g[(g.T_K >= T_fit[0]) & (g.T_K <= T_fit[1])]
        g_nu = d[d.alloy == system]
        nu_of = (g_nu.nu0_phys / g_nu.nu0_sim).groupby(g_nu.T_K).first().to_dict()
        for q in ("D_vac", "D"):
            E, D0 = fit_ev(sel.T_K, sel[f"{q}_m2s"])
            fits.append(dict(system=system, quantity=q, E_eV=E,
                             Q_kJ_mol=E * EV_KJ, D0_m2s=D0,
                             T_min_K=T_fit[0], T_max_K=T_fit[1], n_T=len(sel)))
            for _, r in g.iterrows():
                line = D0 * np.exp(-E / (K_B * r.T_K))
                points.append(dict(system=system, quantity=q, T_K=r.T_K,
                                   steps=int(r.n_steps_max), value_m2s=r[f"{q}_m2s"],
                                   line_m2s=line, ratio=r[f"{q}_m2s"] / line,
                                   n_subdiffusive=int(r.n_subdiffusive)))
            if q == "D_vac" and head is not None:
                for _, h in head[head.system == system].iterrows():
                    if float(h.T_K) not in nu_of:
                        continue
                    v = h.D_vac_A2s_escape * 1e-20 * nu_of[float(h.T_K)]
                    line = D0 * np.exp(-E / (K_B * h.T_K))
                    points.append(dict(system=system, quantity=q, T_K=h.T_K,
                                       steps=int(h.n_steps), value_m2s=v,
                                       line_m2s=line, ratio=v / line,
                                       n_subdiffusive=int("subdiffusive" in str(
                                           h.flags_escape))))
    return pd.DataFrame(fits), pd.DataFrame(points)


def alloy_ratios(table, a, b):
    A = table[table.system == a].set_index("T_K")
    B = table[table.system == b].set_index("T_K")
    rows = []
    for T in sorted(set(A.index) & set(B.index)):
        rows.append(dict(T_K=T, D_vac_ratio=A.D_vac_m2s[T] / B.D_vac_m2s[T],
                         c_v_ratio=A.c_v[T] / B.c_v[T],
                         D_ratio=A.D_m2s[T] / B.D_m2s[T],
                         f_ratio=A.f[T] / B.f[T]))
    return pd.DataFrame(rows, columns=["T_K", "D_vac_ratio", "c_v_ratio",
                                       "D_ratio", "f_ratio"])


# ---------------------------------------------------------------------------
# elements: alloy against pure metal
# ---------------------------------------------------------------------------

def tracer_points(rows, system, e, T_min):
    """{T: D*_X [cm^2/s]}, geometric mean over the diffusive realisations."""
    by_T = {}
    for r in rows:
        if r["alloy"] != system or r.get(f"D_{e}_A2s", "") in ("", "nan"):
            continue
        T = float(r["T_K"])
        if T < T_min or "subdiffusive" in (r.get("flags") or ""):
            continue
        n_at = int(r["n_sites"]) - 1
        d = (float(r[f"D_{e}_A2s"]) * 1e-20 * n_at * float(r["c_v"])
             * float(r["nu0_phys"]) / float(r["nu0_sim"])) * 1e4
        if d > 0:
            by_T.setdefault(T, []).append(d)
    return {T: float(np.exp(np.mean(np.log(v)))) for T, v in sorted(by_T.items())}


def alloy_vs_pure(rows, pure_fit, pure_pts, elements, systems, T_min):
    """Rows of alloy_vs_pure_diffusion.csv and the series by (element, key)."""
    table, series = [], {}
    for e in elements:
        s = {"pure": {float(r["T_K"]): float(r["D_star_m2s"]) * 1e4
                      for r in pure_pts if r["element"] == e}}
        for system in systems:
            p = tracer_points(rows, system, e, T_min)
            if p:
                s[system] = p
        for key, pts in s.items():
            T = np.array(list(pts))
            D = np.array([pts[t] for t in T])
            if key == "pure":
                D0, Q = (float(pure_fit[e]["D0_star_cm2s"]),
                         float(pure_fit[e]["Q_kJ_mol"]))
            else:
                D0, Q = fit_kj(T, D)
            table.append(dict(element=e, system=key, Q_kJ_mol=Q, D0_star_cm2s=D0,
                              **{f"D_star_{t:.0f}K_cm2s": v
                                 for t, v in zip(T, D)}))
            series[(e, key)] = pts
    return table, series


def write_alloy_vs_pure(path, table):
    head = ("element", "system", "Q_kJ_mol", "D0_star_cm2s")
    cols = sorted({c for r in table for c in r}, key=lambda c: (c not in head, c))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)


def element_shares(rows, systems, elements, T_min):
    out = []
    for system in systems:
        by = {}
        for r in rows:
            T = float(r["T_K"])
            if (r["alloy"] != system or T < T_min
                    or "subdiffusive" in (r.get("flags") or "")):
                continue
            n_at = int(r["n_sites"]) - 1
            for e in elements:
                v = r.get(f"D_{e}_A2s", "")
                if v not in ("", "nan"):
                    by.setdefault((e, T), []).append(
                        (n_at * float(v), float(r["D_vac_A2s"])))
        for (e, T), v in sorted(by.items()):
            nd, dv = np.array(v).T
            ratio = nd / dv
            out.append(dict(system=system, element=e, T_K=T, n_runs=len(v),
                            fX_over_f=gmean(nd) / gmean(dv),
                            min=float(ratio.min()), max=float(ratio.max())))
    return pd.DataFrame(out)


def element_ratios(series, pure_pts, d, elements, systems, temperatures):
    cv_pure = {(r["element"], float(r["T_K"])): float(r["c_v"]) for r in pure_pts}
    cv_alloy = d.groupby(["alloy", "T_K"]).c_v.first().to_dict()
    rows = []
    for T in temperatures:
        for e in elements:
            rec = dict(T_K=T, element=e,
                       D_star_pure_cm2s=series.get((e, "pure"), {}).get(T, np.nan))
            for s in systems:
                v = series.get((e, s), {}).get(T, np.nan)
                rec[f"D_star_{s}_cm2s"] = v
                rec[f"pure_over_{s}"] = rec["D_star_pure_cm2s"] / v
                if (s, T) in cv_alloy:
                    rec[f"c_v_pure_over_{s}"] = cv_pure.get((e, T), np.nan) / \
                        cv_alloy[(s, T)]
            if len(systems) == 2:
                a, b = systems
                rec[f"{a}_over_{b}"] = rec[f"D_star_{a}_cm2s"] / rec[f"D_star_{b}_cm2s"]
            rows.append(rec)
        for key in ["pure"] + list(systems):
            v = [series.get((e, key), {}).get(T, np.nan) for e in elements]
            v = np.array([x for x in v if np.isfinite(x)])
            if len(v) > 1:
                rows.append({"T_K": T, "element": f"spread_{key}",
                             "orders_of_magnitude": float(np.log10(v.max() / v.min()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# pure metals against experiment
# ---------------------------------------------------------------------------

def measured(lit, e):
    rows = [r for r in lit if r["element"] == e
            and r.get("show", "yes").strip().lower() != "no"]
    lt = next((r for r in rows if r["term"] in ("LT", "single")), None)
    return lt, rows


def arrhenius_cm2(D0, Q_kJ, T):
    return D0 * np.exp(-Q_kJ / (R_KJ * T))


def pure_vs_measured(pure_fit, lit, elements):
    out = []
    for e in elements:
        lt, terms = measured(lit, e)
        if lt is None or e not in pure_fit:
            continue
        f = pure_fit[e]
        for where, T in (("1100 K", 1100.0), ("T_min", float(lt["T_min_K"])),
                         ("T_max", float(lt["T_max_K"]))):
            chain = arrhenius_cm2(float(f["D0_star_cm2s"]), float(f["Q_kJ_mol"]), T)
            m_lt = arrhenius_cm2(float(lt["D0_cm2_s"]), float(lt["Q_kJ_mol"]), T)
            m_all = sum(arrhenius_cm2(float(r["D0_cm2_s"]), float(r["Q_kJ_mol"]), T)
                        for r in terms)
            out.append(dict(element=e, at=where, T_K=T, D_chain_cm2s=chain,
                            D_measured_low_T_term_cm2s=m_lt,
                            D_measured_sum_cm2s=m_all,
                            chain_over_low_T_term=chain / m_lt,
                            chain_over_sum=chain / m_all, reference=lt["bibkey"]))
    return pd.DataFrame(out)


def pure_summary(pure_fit, lit, pure_runs, elements):
    out = []
    for e in elements:
        if e not in pure_fit:
            continue
        f = pure_fit[e]
        lt, _ = measured(lit, e)
        fv = pure_runs[pure_runs.alloy == e].f_apparent
        out.append(dict(
            element=e, E_m_eV=float(f["Q_eV"]) - float(f["E_f_eV"]),
            E_m_model_eV=float(f["E_m_model_referee_eV"]),
            E_m_uma_eV=float(f["E_m_uma_eV"]),
            E_m_from_D_vac_eV=float(f["E_m_kmc_eV"]),
            E_f_eV=float(f["E_f_eV"]), Q_kJ_mol=float(f["Q_kJ_mol"]),
            Q_measured_kJ_mol=float(lt["Q_kJ_mol"]) if lt else np.nan,
            D0_star_cm2s=float(f["D0_star_cm2s"]),
            D0_measured_cm2s=float(lt["D0_cm2_s"]) if lt else np.nan,
            Q_deviation=(float(f["Q_kJ_mol"]) / float(lt["Q_kJ_mol"]) - 1.0
                         if lt else np.nan),
            measured_T_min_K=float(lt["T_min_K"]) if lt else np.nan,
            measured_T_max_K=float(lt["T_max_K"]) if lt else np.nan,
            reference=lt["bibkey"] if lt else "",
            f_vac_mean=float(fv.mean()), f_vac_sd=float(fv.std(ddof=0)),
            f_X_mean=float(f["f_X_mean"]), nu0_Hz=float(f["nu0_Hz"])))
    allf = pure_runs[pure_runs.alloy.isin(elements)].f_apparent
    out.append(dict(element="all", f_vac_mean=float(allf.mean()),
                    f_vac_sd=float(allf.std(ddof=0))))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------

def read_csv(path):
    with open(resolve(path), newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/analysis/diffusion_summary.yaml")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    systems = list(cfg["systems"])
    elements = list(cfg["elements"])
    temps = [float(t) for t in cfg["temperatures"]]
    T_fit = [float(t) for t in cfg["arrhenius_K"]]
    out = resolve(args.out_dir or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    camp = resolve(cfg["campaign"])
    runs = pd.read_csv(camp / "campaign_runs.csv")
    run_rows = read_csv(camp / "campaign_runs.csv")
    order = pd.read_csv(resolve(cfg["order_reached"]))
    head = (pd.read_csv(resolve(cfg["head_runs"])) if cfg.get("head_runs")
            else None)
    pure_fit = {r["element"]: r for r in read_csv(cfg["pure"]["diffusion"])}
    pure_pts = read_csv(cfg["pure"]["points"])
    pure_runs = pd.read_csv(resolve(cfg["pure"]["runs"]))
    lit = read_csv(cfg["literature"])

    d = alloy_runs(runs, systems, temps)
    table = kmc_table(d, order)
    fits, points = arrhenius_alloys(table, d, head, T_fit)
    ratios = alloy_ratios(table, *systems)
    avp, series = alloy_vs_pure(run_rows, pure_fit, pure_pts, elements, systems,
                                T_fit[0])
    write_alloy_vs_pure(out / "alloy_vs_pure_diffusion.csv", avp)
    print(f"-> {out / 'alloy_vs_pure_diffusion.csv'}")
    frames = (("kmc_table.csv", table), ("arrhenius_alloys.csv", fits),
              ("arrhenius_alloy_points.csv", points),
              ("alloy_ratios.csv", ratios),
              ("element_shares.csv",
               element_shares(run_rows, systems, elements, T_fit[0])),
              ("element_ratios.csv",
               element_ratios(series, pure_pts, d, elements, systems,
                              [t for t in temps if T_fit[0] <= t <= T_fit[1]])),
              ("pure_vs_measured.csv", pure_vs_measured(pure_fit, lit, elements)),
              ("pure_metals_summary.csv",
               pure_summary(pure_fit, lit, pure_runs, elements)))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    for name, frame in frames:
        frame.to_csv(out / name, index=False)
        print(f"\n-> {out / name}")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
