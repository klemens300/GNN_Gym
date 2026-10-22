"""Numbers read from single runs: where the vacancy goes, what distinguishes
the sites it is confined to, and how much order it builds within a given
physical time.

    python -m gnn_gym.analysis.run_details configs/analysis/run_details.yaml

sites        distinct vacancy sites within a window of steps
             [from_step, to_step) of a run (default: the whole run)
confinement  the configuration after to_step; the sites the vacancy visited
             in the window [from_step, to_step] (vacancy site included)
             against the rest of the cell: local order of a pair
             (gnn_gym.metropolis.order.local_pair_order: unlike minus like
             pair bonds over the first shell of the site and its
             neighbours; 0 for the random alloy) and the site energy of
             the site-energy model. The site energies are those of the
             vacancy-free cell, the vacancy site filled with each element
             in turn and averaged, and each is taken relative to the mean
             over the sites of its own element: the chemical offset of the
             elements is removed, and what remains is how favourable the
             environment of a site is (vacancy site: 0). Spread: standard
             deviation over all sites.
early_order  first-shell alpha_1 of a pair over the stored configurations
             of a run up to a physical time t_real_s (gnn_gym.realtime) or
             up to a step to_step, with the steps and the physical time
             covered

Written to `out_dir`: run_sites.csv, confinement.csv, early_order.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import realtime
from ..kmc.checkpoint import find_checkpoint, load_run
from ..kmc.observables import warren_cowley
from ..metropolis.order import local_pair_order
from ..model.predictor import VACANCY, JumpPredictor
from ..paths import load_yaml, resolve


def label(run_dir):
    p = Path(resolve(run_dir))
    return f"{p.name}@{p.resolve().parent.name}"


def sites(entry):
    r = load_run(find_checkpoint(resolve(entry["run"])), snapshots=False)
    a = int(entry.get("from_step", 0))
    b = min(int(entry.get("to_step", r.n_steps)), r.n_steps)
    vac = np.asarray(r.vacancy_indices[a:b + 1])
    return dict(run=label(entry["run"]), system=r.system, T_K=r.T_K,
                from_step=a, to_step=b, sites=int(len(np.unique(vac))),
                n_sites=r.n_sites)


def confinement(entry, models):
    r = load_run(find_checkpoint(resolve(entry["run"])), snapshots=False)
    a, b = int(entry["from_step"]), int(entry["to_step"])
    st = r.head(b).final_state
    sp = np.asarray(st.species, np.int64)
    visited = np.zeros(st.n_sites, bool)
    visited[np.asarray(r.vacancy_indices[a:b + 1])] = True
    els = list(r.elements)
    pair = tuple(els.index(x) for x in entry["pair"].split("-"))
    s = local_pair_order(sp, st.cell, pair)
    pred = JumpPredictor(str(resolve(models["de_model"])),
                         str(resolve(models["bsym_model"])), st.cell, els,
                         device="cpu", halo=False, verbose=False)
    eps = np.zeros(st.n_sites)
    with torch.no_grad():
        for k in range(len(els)):
            z = pred.species_map[np.where(sp == VACANCY, k, sp)]
            eps += pred.de_net.site_energies(torch.from_numpy(z[None]))[0] \
                .double().numpy() / len(els)
    rel = np.zeros(st.n_sites)
    for k in range(len(els)):
        m = sp == k
        rel[m] = 1e3 * (eps[m] - eps[m].mean())
    return dict(run=label(entry["run"]), system=r.system, T_K=r.T_K,
                from_step=a, to_step=b, pair=entry["pair"],
                n_visited=int(visited.sum()),
                local_order_visited=float(s[visited].mean()),
                local_order_rest=float(s[~visited].mean()),
                site_energy_visited_meV=float(rel[visited].mean()),
                site_energy_rest_meV=float(rel[~visited].mean()),
                site_energy_sd_meV=float(rel.std()))


def early_order(entry):
    r = load_run(find_checkpoint(resolve(entry["run"])))
    R = realtime.time_factors(r.system, r.T_K, r.n_sites,
                              r.attempt_frequency_Hz)["R"]
    if "t_real_s" in entry:
        t_target = float(entry["t_real_s"])
        steps = int(np.searchsorted(r.times_s * R, t_target, side="right"))
    else:
        steps = int(entry["to_step"])
        t_target = float(r.times_s[steps - 1] * R)
    keep = np.flatnonzero(r.snapshot_times_s * R <= t_target)
    els = list(r.elements)
    i, j = (els.index(x) for x in entry["pair"].split("-"))
    nb = r.final_state.neighbours
    a = np.array([warren_cowley(r.snapshot_species[k], nb, len(els))[i, j]
                  for k in keep])
    return dict(run=label(entry["run"]), system=r.system, T_K=r.T_K,
                pair=entry["pair"], t_real_s=t_target, steps_covered=steps,
                n_configurations=len(keep), alpha_max_abs=float(np.abs(a).max()),
                alpha_at_t=float(a[-1]))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/analysis/run_details.yaml")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    out = resolve(args.out_dir or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, fn, key in (("run_sites.csv", sites, "sites"),
                          ("confinement.csv", lambda e: confinement(e, cfg),
                           "confinement"),
                          ("early_order.csv", early_order, "early_order")):
        frame = pd.DataFrame([fn(e) for e in cfg.get(key, [])])
        frame.to_csv(out / name, index=False)
        print(f"-> {out / name}")
        print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
