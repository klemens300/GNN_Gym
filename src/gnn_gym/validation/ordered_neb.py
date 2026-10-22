"""Model barriers against uMLIP-NEB in the environments where the KMC is trapped.

The lattice model was trained on jumps in random cells. At low temperature
the vacancy orders its surroundings and is confined to a few sites; if the
model's barriers were wrong in those environments, the confinement could be
an artefact. This check recomputes the eight jumps open to the vacancy in
such states with the uMLIP recipe of the reference data and compares.

SOURCES
    KMC checkpoints (read directly from the npz): the state after a given
    number of steps of chosen runs and, as a control, the random initial
    state of one run. The control must reproduce the model error of random
    cells of the same subsystem.

WINDOWS
    Each jump becomes a periodic cell of the training size cut from the KMC
    cell, in the training convention: vacancy on the reference vacancy site,
    jumping atom at the reference jump vector, offsets centred on the jump.
    A jump along another first-shell direction is mapped onto the reference
    direction by reflecting the axes through the vacancy, a symmetry of the
    cubic lattice, so the environment is carried over exactly. The model and
    the uMLIP see the same cell. For the defect-free cell that sets the box,
    the vacancy site gets the species of the jumping atom.

REPORTED per state: errors of dE, B_sym and E_a, the first-neighbour
Warren-Cowley parameter of the ordering pair in the window and in the
neighbourhood the models read, and the lowest barrier according to the uMLIP
and to the model; per alloy the E_a error with a bootstrap interval against
the referee error of random cells of that subsystem; the mean error of high
barriers.

    python -m gnn_gym.validation.ordered_neb configs/validation/ordered_neb.yaml
    python -m gnn_gym.validation.ordered_neb <config> --model-only
    python -m gnn_gym.validation.ordered_neb <config> --summary-only
"""
import argparse
import csv
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..lattice import Supercell, get_lattice, hop_ball
from ..paths import load_yaml, resolve

COLUMNS = ["key", "source", "alloy", "T_K", "state", "slot", "dx", "dy", "dz",
           "parent_vac_site", "jump_species", "pair", "alpha_window",
           "alpha_local", "a_start_A", "a_relaxed_A",
           "uma_dE_eV", "uma_Bsym_eV", "uma_Efwd_eV", "uma_Ebwd_eV",
           "model_dE_eV", "model_Bsym_eV", "model_Efwd_eV",
           "saddle_image", "relax_conv", "neb_conv", "db_residual_eV",
           "t_s", "timestamp"]


# ---------------------------------------------------------------------------
# KMC states
# ---------------------------------------------------------------------------

def read_state(run_dir, which, step=None):
    """Species (-1 = vacancy), vacancy site and metadata of a KMC checkpoint.

    which = "initial" or "final"; with `step`, the final state is the one
    after that many steps. A run continued beyond `step` holds it as the
    configuration snapshot taken at the time of that step.
    """
    ck = sorted(Path(run_dir).glob("checkpoint_*.npz"))
    if not ck:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    d = np.load(ck[0], allow_pickle=True)
    n = int(d["n_steps"])
    if which == "initial":
        sp, v = d["initial_species"], int(d["initial_vacancy_index"])
    elif step is None or step == n:
        sp, v = d["current_species"], int(d["current_vacancy_index"])
    elif step < n:
        t = d["times_s"][step - 1]
        hit = np.flatnonzero(d["snapshot_times_s"] == t)
        if hit.size == 0:
            raise RuntimeError(f"{run_dir}: no snapshot after step {step}")
        sp = d["snapshot_species"][hit[-1]]
        v = int(d["vacancy_indices"][step])
    else:
        raise ValueError(f"{run_dir} has {n} steps, fewer than {step}")
    sp = np.asarray(sp, dtype=np.int64)
    if sp[v] != -1:
        raise RuntimeError(f"{run_dir} ({which}): site {v} is not the vacancy")
    return dict(size=int(d["supercell_size"]),
                els=[str(e) for e in d["element_symbols"].tolist()],
                species=sp, vac=v, T=float(d["T_K"]), n_steps=n,
                a=float(d["lattice_parameter_A"]))


def check_site_order(cell, a):
    """The KMC numbers its sites like ase's bulk(cubic=True).repeat()."""
    from ase.build import bulk
    pos = bulk("Fe", cell.lattice.name, a=a, cubic=True).repeat([cell.size] * 3).positions
    if not np.array_equal(np.rint(pos * cell.lattice.units_per_a / a).astype(int),
                          cell.coords):
        raise RuntimeError("KMC site order differs from the Supercell order")


def reflections(lattice):
    """Jump directions as axis reflections of the reference jump."""
    return [np.array(s) for s in itertools.product((1, -1), repeat=3)]


def cut_window(parent, state, win, d):
    """Species of the window around the jump vacancy -> vacancy + d * ref."""
    v, j = win.reference_jump()
    h = win.box // 2 - 1
    delta = (win.coords - win.coords[v] + h) % win.box - h
    sel = parent.index(parent.coords[state["vac"]] + delta * d)
    if len(np.unique(sel)) != len(sel):
        raise RuntimeError("window maps two sites onto one parent site")
    sp = state["species"][sel].copy()
    if sp[v] != -1 or sp[j] < 0:
        raise RuntimeError("window convention broken")
    return sp, int(sel[j])


def warren_cowley(species, nn, pair, region=None):
    """First-neighbour alpha of `pair` (species indices), bonds from `region`."""
    a, b = pair
    src = np.repeat(np.arange(len(species)), nn.shape[1])
    si, sj = species[src], species[nn.reshape(-1)]
    ok = (si >= 0) & (sj >= 0)
    if region is not None:
        ok &= np.asarray(region)[src]
    atoms = species[species >= 0]
    c_b = np.mean(atoms == b)
    bonds_a = ok & (si == a)
    if c_b == 0 or not bonds_a.any():
        return float("nan")
    return float(1.0 - np.mean(sj[bonds_a] == b) / c_b)


# ---------------------------------------------------------------------------
# Jobs: windows and model predictions
# ---------------------------------------------------------------------------

def build_jobs(cfg, runs_root):
    lattice = get_lattice(cfg["lattice"])
    win = Supercell(lattice, int(cfg["window_size"]))
    v, j = win.reference_jump()
    local = hop_ball(win.adjacency_lists(cfg["local_shells"]), [v, j],
                     int(cfg["local_hops"]))
    nn = win.neighbours(lattice.jump_shell)
    jobs = []
    for src in cfg["sources"]:
        st = read_state(Path(runs_root) / src["run"], src["state"], src.get("step"))
        parent = Supercell(lattice, st["size"])
        check_site_order(parent, st["a"])
        pair_names = cfg["order_pair"][src["alloy"]]
        pair = tuple(st["els"].index(e) for e in pair_names)
        for slot, d in enumerate(reflections(lattice)):
            sp, parent_j = cut_window(parent, st, win, d)
            syms = [st["els"][s] if s >= 0 else None for s in sp]
            jobs.append(dict(
                key=f"{src['label']}|{src['state']}|{slot}", source=src["label"],
                alloy=src["alloy"], T_K=st["T"], state=src["state"], slot=slot,
                d=d, symbols=syms, jump=syms[j], parent_vac=st["vac"],
                pair="-".join(pair_names),
                alpha=warren_cowley(sp, nn, pair),
                alpha_loc=warren_cowley(sp, nn, pair, local)))
    return win, jobs


def model_predictions(cfg, win, jobs, device):
    import torch
    from ..model.checkpoint import load_model
    from ..model.locality import readout_masks
    from ..training.evaluate import predict
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    de_net, de_cfg, de_ck = load_model(resolve(cfg["models"]["de"]), device)
    bs_net, bs_cfg, _ = load_model(resolve(cfg["models"]["bsym"]), device)
    if de_cfg.size != win.size:
        raise SystemExit("the window size must be the training cell size")
    code = {e: i for i, e in enumerate(de_cfg.elements)}
    v, j = win.reference_jump()
    z_i = np.array([[code[s] if s else de_cfg.vac_index for s in jb["symbols"]]
                    for jb in jobs], dtype=np.int8)
    z_f = z_i.copy()
    z_f[:, v] = [code[jb["jump"]] for jb in jobs]
    z_f[:, j] = de_cfg.vac_index
    arrays = dict(z_init=z_i, z_final=z_f, dE=np.zeros(len(jobs)))
    de_mask = readout_masks(de_cfg, de_ck["reach"], win)[0]
    bs_mask = readout_masks(bs_cfg, de_ck["reach"], win)[1]
    dE = predict(de_net, arrays, de_mask, device)
    bs = predict(bs_net, arrays, bs_mask, device)
    for jb, a, b in zip(jobs, dE, bs):
        jb["m_dE"], jb["m_bs"] = float(a), float(b)
        jb["m_fwd"] = float(b + 0.5 * a)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def bootstrap_mae(e, n_boot=20000, seed=0):
    """95 % bootstrap interval of the mean absolute error."""
    rng = np.random.default_rng(seed)
    m = [np.abs(rng.choice(e, len(e))).mean() for _ in range(n_boot)]
    return [float(x) for x in np.percentile(m, [2.5, 97.5])]


def referee_mae_by_subsystem(path):
    if not path or not resolve(path).exists():
        return {}
    err = {}
    with open(resolve(path), newline="") as fh:
        for r in csv.DictReader(fh):
            err.setdefault(r["subsystem"], []).append(
                (float(r["Ea_pred_eV"]) - float(r["Ea_true_eV"])) * 1e3)
    return {k: dict(n=len(v), mae_Ea=float(np.abs(v).mean()),
                    bias_Ea=float(np.mean(v))) for k, v in err.items()}


def summarise(cfg, csv_path, log=print):
    rows = [r for r in csv.DictReader(open(csv_path, newline=""))
            if r["neb_conv"] == "True" and r["relax_conv"] == "True"]
    groups = {}
    for r in rows:
        groups.setdefault((r["source"], r["state"]), []).append(r)
    out = {"groups": {}, "by_alloy": {}}
    log(f"{'source':14s} {'state':8s} {'n':>2s} {'a_loc':>6s} {'dE':>6s} "
        f"{'B_sym':>6s} {'E_a':>6s} {'bias':>6s}   lowest E_a: uMLIP / model (eV)")
    for (src, st), rs in sorted(groups.items()):
        def f(k):
            return np.array([float(r[k]) for r in rs])
        e_ea = (f("model_Efwd_eV") - f("uma_Efwd_eV")) * 1e3
        ea_u, ea_m = f("uma_Efwd_eV"), f("model_Efwd_eV")
        k = int(np.argmin(ea_u))
        g = dict(n=len(rs), alpha_window=float(np.nanmean(f("alpha_window"))),
                 alpha_local=float(np.nanmean(f("alpha_local"))),
                 mae_dE=float(np.abs(f("model_dE_eV") - f("uma_dE_eV")).mean() * 1e3),
                 mae_Bsym=float(np.abs(f("model_Bsym_eV") - f("uma_Bsym_eV")).mean() * 1e3),
                 mae_Ea=float(np.abs(e_ea).mean()), bias_Ea=float(e_ea.mean()),
                 lowest_Ea_uma=float(ea_u[k]), lowest_Ea_model=float(ea_m[k]),
                 same_lowest_jump=bool(int(np.argmin(ea_m)) == k))
        out["groups"][f"{src}|{st}"] = g
        log(f"{src:14s} {st:8s} {g['n']:2d} {g['alpha_local']:+6.2f} "
            f"{g['mae_dE']:6.1f} {g['mae_Bsym']:6.1f} {g['mae_Ea']:6.1f} "
            f"{g['bias_Ea']:+6.1f}   {g['lowest_Ea_uma']:.3f} / "
            f"{g['lowest_Ea_model']:.3f}  same jump: {g['same_lowest_jump']}")
    ref = referee_mae_by_subsystem(cfg.get("referee_predictions"))
    for alloy in sorted({r["alloy"] for r in rows}):
        for st in ("final", "initial"):
            rs = [r for r in rows if r["alloy"] == alloy and r["state"] == st]
            if not rs:
                continue
            e = np.array([(float(r["model_Efwd_eV"]) - float(r["uma_Efwd_eV"])) * 1e3
                          for r in rs])
            lo, hi = bootstrap_mae(e)
            out["by_alloy"][f"{alloy}|{st}"] = dict(
                n=int(len(e)), mae_Ea=float(np.abs(e).mean()), ci95=[lo, hi],
                bias_Ea=float(e.mean()), referee=ref.get(alloy, {}))
            rf = ref.get(alloy)
            log(f"{alloy:8s} {st:8s} n={len(e):2d}  E_a MAE {np.abs(e).mean():5.1f} meV "
                f"[95% {lo:.1f}-{hi:.1f}], bias {e.mean():+5.1f}   referee {alloy}: "
                + (f"n={rf['n']}, MAE {rf['mae_Ea']:.1f} meV" if rf else "n/a"))
    trapped = [r for r in rows if r["state"] == "final"]
    e = np.array([(float(r["model_Efwd_eV"]) - float(r["uma_Efwd_eV"])) * 1e3
                  for r in trapped])
    out["trapped_all"] = dict(n=int(len(e)), mae_Ea=float(np.abs(e).mean()),
                              bias_Ea=float(e.mean()))
    hi_cut = float(cfg["high_barrier_eV"])
    hi = np.array([(float(r["model_Efwd_eV"]) - float(r["uma_Efwd_eV"])) * 1e3
                   for r in rows if float(r["uma_Efwd_eV"]) > hi_cut])
    out["high_barriers"] = dict(cut_eV=hi_cut, n=int(len(hi)),
                                bias_Ea=float(hi.mean()) if len(hi) else None)
    worst = max(abs(g["lowest_Ea_model"] - g["lowest_Ea_uma"])
                for g in out["groups"].values())
    out["lowest_barrier_max_diff_meV"] = float(worst * 1e3)
    log(f"\ntrapped states: n={len(e)}, E_a MAE {np.abs(e).mean():.1f} meV, bias "
        f"{e.mean():+.1f} meV; barriers above {hi_cut} eV (n={len(hi)}): bias "
        f"{out['high_barriers']['bias_Ea']:+.1f} meV; lowest barrier uMLIP vs "
        f"model within {worst * 1e3:.0f} meV")
    return out


# ---------------------------------------------------------------------------

def read_done(path):
    if not Path(path).exists():
        return set()
    with open(path, newline="") as fh:
        return {r["key"] for r in csv.DictReader(fh)
                if str(r.get("timestamp") or "").strip()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--runs-root", default=".",
                    help="directory the run paths of the config are relative to")
    ap.add_argument("--only", default=None, help="substring filter on the job key")
    ap.add_argument("--model-only", action="store_true",
                    help="windows and model predictions, no uMLIP")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    out = resolve(args.out or cfg["out"])
    summary = resolve(args.summary or cfg["summary"])
    if args.summary_only:
        summary.write_text(json.dumps(summarise(cfg, out), indent=2))
        print(f"[ordered] -> {summary}")
        return 0

    win, jobs = build_jobs(cfg, resolve(args.runs_root))
    model_predictions(cfg, win, jobs, args.device)
    if args.only:
        jobs = [jb for jb in jobs if args.only in jb["key"]]
    print(f"[ordered] {len(jobs)} jumps")
    if args.model_only:
        for jb in jobs:
            print(f"{jb['key']:28s} {jb['jump']:>2s}  alpha {jb['alpha']:+.4f} "
                  f"local {jb['alpha_loc']:+.4f}  model dE {jb['m_dE']:+.6f} "
                  f"B_sym {jb['m_bs']:.6f} E_a {jb['m_fwd']:.6f}")
        return 0

    from ..oracle.jumps import compute_jump
    from ..oracle.structures import realised_fractions, vegard_lattice_parameter
    from ..oracle.uma import Uma
    recipe = load_yaml(cfg["recipe"])
    uma = Uma.from_recipe(recipe)
    v, j = win.reference_jump()
    done = read_done(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for jb in jobs:
        if jb["key"] in done:
            continue
        t0 = time.time()
        perfect = [s if s else jb["jump"] for s in jb["symbols"]]
        els = sorted(set(perfect))
        a0 = vegard_lattice_parameter(realised_fractions(perfect, els),
                                      recipe["vegard_A"])
        try:
            r = compute_jump(uma, recipe, win, perfect, v, j, a0)
        except Exception as exc:
            print(f"{jb['key']:28s} FAILED: {exc}", flush=True)
            continue
        if r["saddle_image"] in (0, r["n_images"] - 1):
            print(f"{jb['key']:28s} REJECTED: saddle on an endpoint", flush=True)
            continue
        row = dict(key=jb["key"], source=jb["source"], alloy=jb["alloy"],
                   T_K=jb["T_K"], state=jb["state"], slot=jb["slot"],
                   dx=jb["d"][0], dy=jb["d"][1], dz=jb["d"][2],
                   parent_vac_site=jb["parent_vac"], jump_species=jb["jump"],
                   pair=jb["pair"], alpha_window=f"{jb['alpha']:.4f}",
                   alpha_local=f"{jb['alpha_loc']:.4f}", a_start_A=f"{a0:.6f}",
                   a_relaxed_A=f"{r['a_relaxed']:.6f}",
                   uma_dE_eV=f"{r['dE']:.6f}", uma_Bsym_eV=f"{r['B_sym']:.6f}",
                   uma_Efwd_eV=f"{r['E_fwd']:.6f}", uma_Ebwd_eV=f"{r['E_bwd']:.6f}",
                   model_dE_eV=f"{jb['m_dE']:.6f}", model_Bsym_eV=f"{jb['m_bs']:.6f}",
                   model_Efwd_eV=f"{jb['m_fwd']:.6f}", saddle_image=r["saddle_image"],
                   relax_conv=r["relax_conv"], neb_conv=r["neb_conv"],
                   db_residual_eV=f"{r['db_residual']:.3e}",
                   t_s=f"{time.time() - t0:.1f}",
                   timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        new = not out.exists()
        with open(out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{jb['key']:28s} {jb['jump']:>2s}  E_a uMLIP {r['E_fwd']:.3f}  model "
              f"{jb['m_fwd']:.3f}  diff {(jb['m_fwd'] - r['E_fwd']) * 1e3:+6.1f} meV "
              f"{time.time() - t0:4.0f} s", flush=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(summarise(cfg, out), indent=2))
    print(f"[ordered] -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
