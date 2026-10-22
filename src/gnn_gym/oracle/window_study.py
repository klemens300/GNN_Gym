"""Cell-size convergence of the jump labels (choice of the 7^3 reference cell).

DESIGN
    One large parent cell (12^3) carries a random equiatomic decoration with
    the jump in its centre: vacancy on the first basis site of the central
    conventional cell, jumping atom at the reference jump vector. Blocks of
    n x n x n conventional cells around the jump are cut out, given periodic
    boundaries with box n a, and the same calculation is run in each: both
    endpoints relaxed from the ideal lattice at fixed a, NEB between them.
    Every window is a literal sub-block of the same configuration, so the
    local environment is nested exactly, and window membership is decided on
    integer cell indices, independent of a. The block is centred on the bond
    midpoint (as far as the integer grid allows).

ANALYSIS
    Paired: every environment is compared with itself at the reference, the
    mean over the plateau sizes n >= 8, which averages down the
    single-precision granularity of the large cells (the total energy step
    grows with the number of atoms; the residual E_fwd - E_bwd - dE, zero in
    exact arithmetic, is printed as the empirical floor). Per window size the
    mean signed error, MAE, RMS, maximum and the standard error of the mean
    are reported for dE, B_sym, E_fwd and the unrelaxed dE.

    python -m gnn_gym.oracle.window_study configs/oracle/window_study.yaml run
    python -m gnn_gym.oracle.window_study configs/oracle/window_study.yaml analyse
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..lattice import Supercell
from ..paths import load_yaml, resolve
from ..systems import get_system
from .structures import jump_endpoints, random_decoration
from .uma import neb_labels

QUANTITIES = [("dE", "dE_eV"), ("B_sym", "B_sym_eV"), ("E_fwd", "E_fwd_eV"),
              ("dE_ideal", "dE_ideal_eV")]
E_PER_ATOM_EV = 11.5          # typical |E| per atom, for the float32 step


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def parent_cell(system, size, seed):
    """Parent supercell, its decoration and the jump (v, j)."""
    cell = Supercell(system.lattice_obj, size)
    symbols = random_decoration(cell, system.composition, seed)
    v, j = cell.reference_jump()
    return cell, symbols, v, j


def cut_window(parent, symbols, n, v, j):
    """The n^3 block of conventional cells around the jump.

    Returns integer coordinates relative to the window origin (in the
    parent's site order), the symbols, and v, j in window indices.
    """
    edge = parent.lattice.cell_edge
    cellidx = parent.coords // edge
    lo = (cellidx[v] - n // 2) % parent.size
    offset = (cellidx - lo) % parent.size
    sel = np.flatnonzero(np.all(offset < n, axis=1))
    if sel.size != len(parent.lattice.basis) * n ** 3:
        raise RuntimeError(f"window {n}^3 holds {sel.size} sites")
    within = parent.coords[sel] - cellidx[sel] * edge
    win = offset[sel] * edge + within
    remap = {int(s): k for k, s in enumerate(sel)}
    return win, [symbols[i] for i in sel], remap[v], remap[j]


def window_atoms(lattice, coords, symbols, n, a):
    from ase import Atoms
    return Atoms(symbols=symbols, positions=coords * (a / lattice.units_per_a),
                 cell=np.eye(3) * (n * a), pbc=True)


def window_endpoints(cfg, system, env, n):
    """Ideal-lattice endpoints of window n of environment env."""
    parent, syms, v, j = parent_cell(system, int(cfg["parent_size"]),
                                     int(cfg["seed_base"]) + env)
    w, wsyms, vw, jw = cut_window(parent, syms, n, v, j)
    atoms = window_atoms(parent.lattice, w, wsyms, n, float(cfg["lattice_parameter_A"]))
    return jump_endpoints(atoms, vw, jw), syms[v], syms[j]


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------

COLUMNS = ["env", "parent_size", "n", "n_atoms", "a_A", "L_half_A",
           "vac_species", "jump_species", "E_initial_eV", "E_final_eV", "dE_eV",
           "E_saddle_eV", "E_fwd_eV", "E_bwd_eV", "B_sym_eV", "saddle_image",
           "E_initial_ideal_eV", "E_final_ideal_eV", "dE_ideal_eV",
           "relax_steps_i", "relax_steps_f", "relax_conv", "neb_steps",
           "neb_conv", "db_residual_eV", "t_s", "seed", "model",
           "neb_images", "relax_fmax", "neb_fmax", "timestamp"]


def run(cfg, out, envs=None, sizes=None):
    from .uma import Uma
    recipe = load_yaml(cfg["recipe"])
    system = get_system(cfg["system"])
    nb = cfg["neb"]
    a = float(cfg["lattice_parameter_A"])
    sizes = sizes or list(cfg["sizes"])
    envs = envs if envs is not None else range(int(cfg["n_env"]))
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            done = {(int(r["env"]), int(r["n"])) for r in csv.DictReader(fh)}
    uma = Uma.from_recipe(recipe)
    for env in envs:
        for n in sizes:
            if (env, n) in done:
                continue
            t0 = time.time()
            (init_id, fin_id), vs, js = window_endpoints(cfg, system, env, n)
            E_i0, E_f0 = uma.energy(init_id), uma.energy(fin_id)
            init_rel, E_i, c1, n1 = uma.relax(init_id, cfg["relax_fmax"])
            fin_rel, E_f, c2, n2 = uma.relax(fin_id, cfg["relax_fmax"])
            _im, energies, c3, n3 = uma.neb(init_rel, fin_rel, n_images=nb["n_images"],
                                            fmax=nb["fmax"], method=nb["method"],
                                            k=nb["k"], climb=nb["climb"],
                                            steps=nb["steps"])
            lab = neb_labels(energies, E_i, E_f)
            if lab["saddle_image"] in (0, len(energies) - 1):
                print(f"env {env} n={n}: saddle on an endpoint, skipped")
                continue
            row = {"env": env, "parent_size": cfg["parent_size"], "n": n,
                   "n_atoms": len(init_id), "a_A": f"{a:.9f}",
                   "L_half_A": f"{n * a / 2:.4f}", "vac_species": vs,
                   "jump_species": js, "E_initial_eV": f"{E_i:.6f}",
                   "E_final_eV": f"{E_f:.6f}", "dE_eV": f"{lab['dE']:.6f}",
                   "E_saddle_eV": f"{lab['E_saddle']:.6f}",
                   "E_fwd_eV": f"{lab['E_fwd']:.6f}", "E_bwd_eV": f"{lab['E_bwd']:.6f}",
                   "B_sym_eV": f"{lab['B_sym']:.6f}",
                   "saddle_image": lab["saddle_image"],
                   "E_initial_ideal_eV": f"{E_i0:.6f}",
                   "E_final_ideal_eV": f"{E_f0:.6f}",
                   "dE_ideal_eV": f"{E_f0 - E_i0:.6f}",
                   "relax_steps_i": n1, "relax_steps_f": n2,
                   "relax_conv": bool(c1 and c2), "neb_steps": n3, "neb_conv": c3,
                   "db_residual_eV": f"{lab['db_residual']:.3e}",
                   "t_s": f"{time.time() - t0:.1f}",
                   "seed": int(cfg["seed_base"]) + env, "model": uma.model,
                   "neb_images": nb["n_images"], "relax_fmax": cfg["relax_fmax"],
                   "neb_fmax": nb["fmax"],
                   "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            new = not out.exists()
            with open(out, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLUMNS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"env {env:2d} n={n:2d} N={len(init_id):5d}  dE "
                  f"{lab['dE'] * 1e3:+8.1f} meV  B_sym {lab['B_sym']:.4f} eV  "
                  f"{time.time() - t0:5.0f} s", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_rows(path):
    rows = defaultdict(dict)
    with open(resolve(path), newline="") as fh:
        for r in csv.DictReader(fh):
            rec = {k: float(r[c]) for k, c in QUANTITIES}
            rec["n_atoms"] = float(r["n_atoms"])
            rec["db"] = float(r["db_residual_eV"])
            rec["conv"] = r["relax_conv"] == "True" and r["neb_conv"] == "True"
            rows[int(r["env"])][int(r["n"])] = rec
    return rows


def analyse(cfg, log=print):
    rows = load_rows(cfg["data"])
    plateau = int(cfg["plateau_from"])
    sizes = sorted({n for e in rows for n in rows[e]})
    ref = {e: {k: float(np.mean([p[n][k] for n in p if n >= plateau]))
               for k, _ in QUANTITIES}
           for e, p in rows.items() if any(n >= plateau for n in p)}
    n_bad = sum(not rows[e][n]["conv"] for e in rows for n in rows[e])
    log(f"[window] {sum(len(v) for v in rows.values())} rows, {len(rows)} "
        f"environments, {n_bad} not converged; reference: mean over n >= {plateau}")
    out = {"reference": f"mean over n >= {plateau}", "sizes": {}}
    log(f"  {'n':>3} {'N':>6} {'|db| meV':>9} {'float32 meV':>12}")
    for n in sizes:
        na = np.mean([rows[e][n]["n_atoms"] for e in rows if n in rows[e]])
        db = np.mean([abs(rows[e][n]["db"]) for e in rows if n in rows[e]]) * 1e3
        out["sizes"][n] = {"n_atoms": float(na), "db_residual_meV": float(db),
                           "float32_step_meV": E_PER_ATOM_EV * na * 2 ** -23 * 1e3}
        log(f"  {n:>3} {int(na):>6} {db:>9.3f} "
            f"{out['sizes'][n]['float32_step_meV']:>12.3f}")
    for key, _ in QUANTITIES:
        log(f"\n  {key} [meV, paired]\n  {'n':>3} {'env':>4} {'mean':>8} {'MAE':>8} "
            f"{'RMS':>8} {'max':>8} {'sem(MAE)':>9}")
        for n in sizes:
            d = np.array([rows[e][n][key] - ref[e][key] for e in ref
                          if n in rows[e]]) * 1e3
            ad = np.abs(d)
            sem = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
            stats = {"n_env": int(d.size), "mean_meV": float(d.mean()),
                     "mae_meV": float(ad.mean()),
                     "rms_meV": float(np.sqrt((d ** 2).mean())),
                     "max_meV": float(ad.max()), "sem_meV": sem}
            out["sizes"][n][key] = stats
            log(f"  {n:>3} {d.size:>4} {d.mean():>8.1f} {ad.mean():>8.1f} "
                f"{stats['rms_meV']:>8.1f} {ad.max():>8.1f} {sem:>9.1f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("action", choices=["run", "analyse"])
    ap.add_argument("--envs", default=None, help="lo:hi environments (run)")
    ap.add_argument("--sizes", default=None, help="comma-separated sizes (run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.action == "run":
        envs = None
        if args.envs:
            lo, hi = args.envs.split(":")
            envs = range(int(lo), int(hi))
        sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
        out = resolve(args.out or cfg["data"])
        out.parent.mkdir(parents=True, exist_ok=True)
        run(cfg, out, envs, sizes)
    else:
        res = analyse(cfg)
        out = resolve(args.out or cfg["results"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"\n[window] -> {out}")


if __name__ == "__main__":
    main()
