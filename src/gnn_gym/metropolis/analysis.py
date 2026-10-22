"""Every quantity the paper reads from the Metropolis chains.

    python -m gnn_gym.metropolis.analysis configs/metropolis/analysis.yaml
    python -m gnn_gym.metropolis.analysis configs/metropolis/analysis.yaml \
        --tord DIR --two-start DIR                 # read chains elsewhere

Inputs are the outputs of gnn_gym.metropolis.run: chain CSVs and snapshots,
and the per-sweep traces for the equilibration and two-start steps. A
directory given with --tord holds all three; in the configuration the
traces have their own directory. A step whose traces are absent is skipped.
Written to out_dir:

    metropolis_tord_summary.csv   leading pair of every chain of the T_ord
                                  campaign: alpha_1, eta, E, C_v
    equilibrium_<system>.csv      per T over the chains of the campaign: E and
                                  C_v (mean, min, max), largest spread of
                                  alpha_1 over the pairs, spread of E
    equilibrium_alpha_<system>.csv  per T and pair: alpha_1 mean, min, max
    ordering_temperature.json     T at the maximum of the mean C_v, and per
                                  chain
    alpha_eq_<system>.csv         equilibrium alpha_1 for the KMC: cooling
                                  branches, mean over seeds, from the lowest
                                  equilibrated temperature up
    alpha_eq_kmc.csv              the same interpolated linearly in T to the
                                  KMC temperatures
    equilibration_<system>.csv    tau_int, t_eq, N_eff, error of E and of the
                                  leading alpha_1 per chain and T
    equilibration_report.csv      the same for the fixed-T chains of the
                                  two-start test
    two_start.csv                 tail means of the two starts per T next to
                                  the range of the campaign chains at that T
    metropolis_tord_lro.csv       concentration waves of every snapshot of
                                  the campaign (psi, k, I/N, p, eta)
    lowT_table.csv                low-temperature state: alpha_1..3 (mean
                                  over the cooling seeds), largest eta, the
                                  strongest reflection and its intensity
                                  relative to complete order, per seed
    lowT_layers.csv, lowT_summary.csv, lowT_structure_factor.npz
                                  B2 domain maps of one double layer and
                                  the structure factor map (figure data)
    exafs_<T>K.csv                two-group neighbour fractions against the
                                  EXAFS measurement
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..lattice import Supercell
from ..model import read_checkpoint
from ..paths import load_yaml, resolve
from ..systems import get_system
from . import autocorr, exafs
from .observables import warren_cowley_shells
from .order import (b2_variant, concentration_waves, layer_sites,
                    local_b2_order, structure_factor_map, superstructure_mask)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_chain(path):
    """Chain CSV (gnn_gym.metropolis.run) as a DataFrame."""
    return pd.read_csv(path)


def read_snapshots(path):
    """{T: species codes} of a <stem>.snapshots.npz."""
    with np.load(path) as d:
        return {float(k[1:]): d[k].astype(np.int64) for k in d.files}


def cell_size(n_atoms, lattice):
    return int(round((n_atoms / len(lattice.basis)) ** (1.0 / 3.0)))


class Context:
    def __init__(self, cfg, overrides):
        """overrides: {runs key: directory or None}."""
        self.cfg = cfg
        runs = dict(cfg["runs"])
        runs.update({k: v for k, v in overrides.items() if v})
        self.runs = {k: Path(resolve(v)) for k, v in runs.items()}
        self.out = Path(resolve(cfg["out_dir"]))
        self.out.mkdir(parents=True, exist_ok=True)
        ck = read_checkpoint(resolve(cfg["model"]))
        self.model_elements = list(ck["model_config"]["elements"])
        self.code = {e: i for i, e in enumerate(self.model_elements)}
        self.cells = {}

    def system(self, name):
        return get_system(name)

    def cell(self, system, size):
        key = (system.lattice, size)
        if key not in self.cells:
            self.cells[key] = Supercell(system.lattice_obj, size)
        return self.cells[key]

    def pair_key(self, a, b):
        """'A-B' with A before B in the model element order."""
        a, b = sorted((a, b), key=self.code.get)
        return f"{a}-{b}"

    def tord_chains(self, system):
        """[(stem, mode, seed), ...] of the campaign chains of a system."""
        out = []
        for ch in self.cfg["tord_chains"]:
            mode, seed = ch.split("_s")
            out.append((f"{system}_{ch}", mode, int(seed)))
        return out

    def tord_frame(self, system):
        frames = []
        for stem, mode, seed in self.tord_chains(system):
            d = read_chain(self.runs["tord"] / f"{stem}.csv")
            d["system"], d["mode"], d["seed"] = system, mode, seed
            d["pair"] = [self.pair_key(h, o) for h, o in zip(d.host, d.other)]
            frames.append(d)
        return pd.concat(frames, ignore_index=True)

    def traces(self, key, stems):
        """Trace files of the stems in runs[key], or None (with a note) when
        none is there; an incomplete set is an error."""
        paths = [self.runs[key] / f"{s}.trace.csv" for s in stems]
        present = [p.exists() for p in paths]
        if not any(present):
            print(f"[analysis] no traces in {self.runs[key]}: skipped")
            return None
        if not all(present):
            missing = [p.name for p, ok in zip(paths, present) if not ok]
            raise FileNotFoundError(f"{self.runs[key]}: missing {missing}")
        return paths

    def write(self, name, rows_or_frame, float_format=None):
        path = self.out / name
        if isinstance(rows_or_frame, pd.DataFrame):
            rows_or_frame.to_csv(path, index=False, float_format=float_format)
        else:
            autocorr.write_csv(path, rows_or_frame)
        print(f"-> {path}")
        return path


# ---------------------------------------------------------------------------
# chain tables
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = ["system", "L", "mode", "seed", "T_K", "alpha_1",
                   "alpha_1_err", "eta_mean", "E_rel_meV_per_atom",
                   "E_err_meV_per_atom", "cv_fluct_kB", "cv_dEdT_kB",
                   "accept_rate", "n_eq", "n_samp"]


def summary(ctx):
    """Leading pair of every chain of the campaign."""
    chains = []
    for name, sc in ctx.cfg["systems"].items():
        for stem, mode, seed in ctx.tord_chains(name):
            chains.append((name, ctx.runs["tord"] / f"{stem}.csv", mode, seed,
                           sc["lead_pair"]))
    frames = []
    for name, path, mode, seed, lead in chains:
        d = read_chain(path)
        a, b = lead.split("-")
        d = d[((d.host == a) & (d.other == b)) | ((d.host == b) & (d.other == a))].copy()
        d["system"], d["mode"], d["seed"] = name, mode, seed
        d["L"] = [cell_size(n, ctx.system(name).lattice_obj) for n in d.n_atoms]
        frames.append(d)
    s = pd.concat(frames, ignore_index=True)
    if np.allclose(s.T_K, np.round(s.T_K)):
        s["T_K"] = s.T_K.round().astype(int)
    s = s.sort_values(["system", "L", "mode", "seed", "T_K"],
                      ascending=[True, True, True, True, False])
    # six significant digits: the precision of the chain CSVs
    ctx.write("metropolis_tord_summary.csv", s[SUMMARY_COLUMNS], "%.6g")


def equilibrium(ctx):
    """Mean over the campaign chains, spreads, T_ord and alpha_eq."""
    tord, alpha_eq_rows = {}, []
    for name, sc in ctx.cfg["systems"].items():
        mc = ctx.tord_frame(name)
        g = mc.groupby(["T_K", "pair"]).alpha_1
        al = g.agg(["mean", "min", "max", "size"]).reset_index()
        al.columns = ["T_K", "pair", "alpha_1_mean", "alpha_1_min",
                      "alpha_1_max", "n_chains"]
        ctx.write(f"equilibrium_alpha_{name}.csv", al)

        lead = mc[mc.pair == sc["lead_pair"]]
        per_T = lead.groupby("T_K").agg(
            E_mean=("E_rel_meV_per_atom", "mean"),
            E_min=("E_rel_meV_per_atom", "min"),
            E_max=("E_rel_meV_per_atom", "max"),
            cv_fluct_mean=("cv_fluct_kB", "mean"),
            cv_fluct_min=("cv_fluct_kB", "min"),
            cv_fluct_max=("cv_fluct_kB", "max"),
            cv_dEdT_mean=("cv_dEdT_kB", "mean"),
            n_chains=("cv_fluct_kB", "size")).reset_index()
        spread = (al.assign(s=al.alpha_1_max - al.alpha_1_min)
                  .groupby("T_K").s.max())
        per_T["alpha_1_spread_max"] = per_T.T_K.map(spread)
        per_T["E_spread"] = per_T.E_max - per_T.E_min
        ctx.write(f"equilibrium_{name}.csv", per_T)

        cv = lead.groupby("T_K").cv_fluct_kB.mean()
        cv_eq = cv[cv.index >= sc["equilibrated_from_K"]]
        per_chain = {f"{m}_s{s}": float(d.loc[d.cv_fluct_kB.idxmax(), "T_K"])
                     for (m, s), d in lead.groupby(["mode", "seed"])}
        tord[name] = dict(T_cv_max_K=float(cv.idxmax()),
                          cv_max_kB=float(cv.max()),
                          T_cv_max_equilibrated_K=float(cv_eq.idxmax()),
                          cv_max_equilibrated_kB=float(cv_eq.max()),
                          equilibrated_from_K=float(sc["equilibrated_from_K"]),
                          T_cv_max_per_chain_K=per_chain)

        cool = mc[(mc["mode"] == "cool") & (mc.T_K >= sc["equilibrated_from_K"])]
        eq = cool.groupby(["pair", "T_K"]).alpha_1.agg(["mean", "min", "max"])
        eq = eq.reset_index().sort_values(["pair", "T_K"])
        for pair, d in eq.groupby("pair", sort=False):
            host, other = pair.split("-")
            for _, r in d.iterrows():
                alpha_eq_rows.append(dict(system=name, host=host, other=other,
                                          T_K=r.T_K, alpha_1=r["mean"],
                                          alpha_1_min=r["min"],
                                          alpha_1_max=r["max"]))
    path = ctx.out / "ordering_temperature.json"
    path.write_text(json.dumps(tord, indent=2))
    print(f"-> {path}")

    eq = pd.DataFrame(alpha_eq_rows)
    for name in ctx.cfg["systems"]:
        ctx.write(f"alpha_eq_{name}.csv",
                  eq[eq.system == name].drop(columns="system"))
    rows = []
    for name in ctx.cfg["systems"]:
        d = eq[eq.system == name]
        for (host, other), g in d.groupby(["host", "other"], sort=False):
            g = g.sort_values("T_K")
            for T in ctx.cfg["kmc_temperatures"]:
                rows.append(dict(system=name, host=host, other=other, T_K=T,
                                 alpha_eq=float(np.interp(T, g.T_K.values,
                                                          g.alpha_1.values))))
    ctx.write("alpha_eq_kmc.csv", pd.DataFrame(rows))


def equilibration(ctx):
    """Autocorrelation and block test on the traces of every chain: energy
    and alpha_1 of the leading pair."""
    for name, sc in ctx.cfg["systems"].items():
        paths = ctx.traces("tord_traces",
                           [stem for stem, _, _ in ctx.tord_chains(name)])
        if paths is None:
            continue
        rows = autocorr.equilibration_table(autocorr.read_traces(paths),
                                            pair=sc["lead_pair"])
        ctx.write(f"equilibration_{name}.csv", rows)


def two_start(ctx):
    """Fixed-T chains from a cooled and from an ordered state."""
    ts = ctx.cfg["two_start"]
    stems = [f"{name}_eq_T{T}_{start}" for name, Ts in ts["temperatures"].items()
             for T in Ts for start in ts["starts"]]
    paths = ctx.traces("two_start", stems)
    if paths is None:
        return
    data = autocorr.read_traces(paths)
    rows = autocorr.equilibration_table(data)
    ctx.write("equilibration_report.csv", rows)

    summ = []
    for name, Ts in ts["temperatures"].items():
        lead = ctx.cfg["systems"][name]["lead_pair"]
        a, b = lead.split("-")
        mc = ctx.tord_frame(name)
        mc = mc[mc.pair == lead]
        for T in Ts:
            rec = dict(system=name, T_K=T, pair=lead)
            for start in ts["starts"]:
                f = f"{name}_eq_T{T}_{start}.trace.csv"
                cols = next(v for k, v in data.items() if k[0] == f)
                col = (f"alpha_{a}-{b}" if f"alpha_{a}-{b}" in cols
                       else f"alpha_{b}-{a}")
                for key, c in (("E", autocorr.ENERGY[1]), ("alpha_1", col)):
                    _, tail, _, err = autocorr.tail_statistics(cols[c])
                    rec[f"{key}_{start}"] = float(tail.mean())
                    rec[f"{key}_{start}_err"] = float(err)
            at = mc[mc.T_K == T]
            rec["E_campaign_min"] = float(at.E_rel_meV_per_atom.min())
            rec["E_campaign_max"] = float(at.E_rel_meV_per_atom.max())
            rec["alpha_1_campaign_min"] = float(at.alpha_1.min())
            rec["alpha_1_campaign_max"] = float(at.alpha_1.max())
            summ.append(rec)
    ctx.write("two_start.csv", pd.DataFrame(summ))


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------

def lro(ctx):
    """Concentration waves of every stored configuration of the campaign."""
    rows = []
    for name in sorted(ctx.cfg["systems"]):
        system = ctx.system(name)
        for stem, mode, seed in sorted(ctx.tord_chains(name)):
            snaps = read_snapshots(ctx.runs["tord"] / f"{stem}.snapshots.npz")
            size = cell_size(next(iter(snaps.values())).size, system.lattice_obj)
            cell = ctx.cell(system, size)
            mk = superstructure_mask(cell)
            for T in sorted(snaps, reverse=(mode == "cool")):
                r = concentration_waves(snaps[T], cell, len(ctx.model_elements), mk)
                for e, v in r.items():
                    rows.append(dict(system=name, mode=mode, seed=seed, T_K=T,
                                     element=ctx.model_elements[e], psi=v["psi"],
                                     eta=v["eta"], k_h=v["k"][0], k_k=v["k"][1],
                                     k_l=v["k"][2], I_over_N=v["I_over_N"],
                                     p_on_lead=v["p"],
                                     lead=ctx.model_elements[v["lead"]]))
    ctx.write("metropolis_tord_lro.csv", rows)
    return pd.DataFrame(rows)


def low_temperature_table(ctx, lro_frame):
    """alpha_1..3 over the cooling seeds, largest eta, strongest reflection."""
    rows = []
    for name, sc in ctx.cfg["systems"].items():
        lt = sc.get("lowT")
        if not lt:
            continue
        system = ctx.system(name)
        cool = [(stem, seed) for stem, mode, seed in ctx.tord_chains(name)
                if mode == "cool"]
        mc = ctx.tord_frame(name)
        for T in lt["temperatures"]:
            alphas, eta_max = [], 0.0
            for stem, seed in cool:
                z = read_snapshots(ctx.runs["tord"] / f"{stem}.snapshots.npz")[float(T)]
                cell = ctx.cell(system, cell_size(z.size, system.lattice_obj))
                alphas.append(warren_cowley_shells(z, cell, len(ctx.model_elements)))
                d = mc[(mc["mode"] == "cool") & (mc.seed == seed) & (mc.T_K == T)]
                eta_max = max(eta_max, float(d.eta_mean.iloc[0]))
            waves = lro_frame[(lro_frame.system == name) & (lro_frame["mode"] == "cool")
                              & (lro_frame.T_K == T)
                              & (lro_frame.element == lt.get("wave_element"))]
            for pair in lt["pairs"]:
                a, b = (ctx.code[x] for x in pair.split("-"))
                rec = dict(system=name, T_K=T, pair=pair)
                for s in (1, 2, 3):
                    rec[f"alpha_{s}"] = float(np.mean([al[s][a, b] for al in alphas]))
                rec["eta_max"] = eta_max
                for _, w in waves.sort_values("seed").iterrows():
                    rec[f"peak_s{w.seed}"] = f"({w.k_h:g} {w.k_k:g} {w.k_l:g})"
                    rec[f"I_rel_s{w.seed}"] = float(w.psi) ** 2
                rows.append(rec)
    ctx.write("lowT_table.csv", pd.DataFrame(rows))


def low_temperature_maps(ctx):
    """B2 domain maps of one double layer and the structure factor map."""
    m = ctx.cfg["lowT_maps"]
    T, z0, thresh, nmin = float(m["temperature"]), int(m["layer_z0"]), \
        float(m["thresh"]), int(m["nmin"])
    layers, summ = [], []
    for panel in m["panels"]:
        system = ctx.system(panel["system"])
        if panel.get("structure") == "b2":
            # perfect B2 of the pair, the reference drawn with the same encoding
            size = int(panel["size"])
            cell = ctx.cell(system, size)
            a, b = panel["pair"]
            z = np.where(cell.sublattice() == 0, ctx.code[a], ctx.code[b])
        else:
            stem = f"{panel['system']}_{m['chain']}"
            z = read_snapshots(ctx.runs["tord"] / f"{stem}.snapshots.npz")[T]
            cell = ctx.cell(system, cell_size(z.size, system.lattice_obj))
        pair = tuple(ctx.code[e] for e in panel["pair"])
        s, n = local_b2_order(z, cell, pair)
        var = b2_variant(s, n, thresh, nmin)
        sel = layer_sites(cell, z0)
        upa = cell.lattice.units_per_a
        for i in sel:
            layers.append(dict(panel=panel["name"], x_a=cell.coords[i, 0] / upa,
                               y_a=cell.coords[i, 1] / upa,
                               element=ctx.model_elements[z[i]],
                               variant=int(var[i]), s=float(s[i]), n_pair=int(n[i])))
        ok = n >= nmin
        summ.append(dict(panel=panel["name"], system=panel["system"],
                         pair="-".join(panel["pair"]),
                         layer_variant_plus=float((var[sel] == 1).mean()),
                         layer_variant_minus=float((var[sel] == -1).mean()),
                         layer_untinted=float((var[sel] == 0).mean()),
                         cell_mean_abs_s=float(np.abs(s[ok]).mean()),
                         cell_ordered_fraction=float(((np.abs(s) > thresh) & ok).mean())))
    ctx.write("lowT_layers.csv", pd.DataFrame(layers))
    ctx.write("lowT_summary.csv", pd.DataFrame(summ))

    sk = m["structure_factor"]
    system = ctx.system(sk["system"])
    z = read_snapshots(ctx.runs["tord"] / f"{sk['system']}_{m['chain']}.snapshots.npz")[T]
    cell = ctx.cell(system, cell_size(z.size, system.lattice_obj))
    proj, axis = structure_factor_map(z, cell, ctx.code[sk["element"]])
    pk = np.unravel_index(int(np.argmax(proj)), proj.shape)
    i0, i1 = int(np.argmin(np.abs(axis))), int(np.argmin(np.abs(axis + 1.0)))
    path = ctx.out / "lowT_structure_factor.npz"
    np.savez_compressed(path, S_hk=proj, k=axis,
                        peak_hk=np.array([axis[pk[0]], axis[pk[1]]]),
                        peak_S=proj.max(), S_100=proj[i1, i0])
    print(f"-> {path}  (strongest peak at (h, k) = ({axis[pk[0]]:+.3f}, "
          f"{axis[pk[1]]:+.3f}), S = {proj.max():.1f}; S at (-1 0 0) = "
          f"{proj[i1, i0]:.2f})")


def exafs_comparison(ctx):
    f = ctx.cfg["exafs"]
    system = ctx.system(f["system"])
    T = float(f["temperature"])
    configs = []
    for stem, _, _ in ctx.tord_chains(f["system"]):
        snaps = read_snapshots(ctx.runs["tord"] / f"{stem}.snapshots.npz")
        if T in snaps:
            configs.append(snaps[T])
    cell = ctx.cell(system, cell_size(configs[0].size, system.lattice_obj))
    rows = exafs.compare(configs, cell, ctx.model_elements, f["group"],
                         exafs.read_measurement(f["measurement"]))
    ctx.write(f"exafs_{T:.0f}K.csv", pd.DataFrame(rows))


STEPS = {"summary": summary, "equilibrium": equilibrium,
         "equilibration": equilibration, "two_start": two_start,
         "lowT_maps": low_temperature_maps, "exafs": exafs_comparison}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--tord", help="directory of the T_ord campaign chains "
                    "(CSVs, snapshots and traces)")
    ap.add_argument("--two-start", help="directory of the two-start chains")
    ap.add_argument("--out-dir", help="override out_dir")
    ap.add_argument("--only", nargs="+", choices=list(STEPS) + ["lro"],
                    help="run these steps only")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    ctx = Context(cfg, {"tord": args.tord, "tord_traces": args.tord,
                        "two_start": args.two_start})
    steps = args.only or list(STEPS) + ["lro"]
    for name in steps:
        if name == "lro":
            low_temperature_table(ctx, lro(ctx))
        else:
            STEPS[name](ctx)


if __name__ == "__main__":
    main()
