"""Cubic elastic constants on the composition grid (input of the attempt frequency).

For every composition of a regular grid on the simplex (step 0.1: 286
compositions of Mo-Nb-Ta-W) several small cells (4^3, 128 sites) are
decorated at random, relaxed in cell shape and positions with the uMLIP, and
their elastic constants C11, C12, C44 are fitted from stress-strain curves
of the relaxed cell (matscipy, cubic symmetry, five strains up to +-2 %,
clamped ions). gnn_gym.realtime.nu0 averages them per composition and turns
them into the Debye frequency.

DECORATION OF ONE CELL (composition x, centre element c, seed s)
    counts round(x_e N) per element in alphabetical order, padded with the
    majority element or trimmed to N; the symbol list is shuffled with
    numpy's RandomState(s); finally the site at the centre of the cell gets
    element c. The cell relaxation starts from one fixed lattice parameter
    for every composition. Each composition is sampled with several centre
    elements and seeds (`centres` in the config: seeds count up per
    composition, 1, 2, ... in the order of that list).

The stored sweep (data/nu0/elastic_MoNbTaW.csv) lists every cell with its
centre element and seed; `--replay` recomputes chosen rows of it.

    python -m gnn_gym.oracle.elastic configs/oracle/elastic_MoNbTaW.yaml
    python -m gnn_gym.oracle.elastic configs/oracle/elastic_MoNbTaW.yaml \
        --replay data/nu0/elastic_MoNbTaW.csv --rows 0,1500 --out x.csv
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..lattice import Supercell, get_lattice
from ..paths import load_yaml, resolve
from .structures import to_atoms


def composition_grid(elements, step):
    """All compositions k_i / n with sum k_i = n, n = 1/step (exact integers)."""
    n = int(round(1.0 / step))
    out = []

    def rec(idx, remaining, cur):
        if idx == len(elements) - 1:
            out.append({e: k / n for e, k in zip(elements, cur + [remaining])})
            return
        for k in range(remaining + 1):
            rec(idx + 1, remaining - k, cur + [k])

    rec(0, n, [])
    return out


def composition_string(composition):
    return "".join(f"{e}{int(round(x * 100))}" for e, x in sorted(composition.items()))


def decorate_cell(cell, composition, centre, seed):
    """Symbols of one sweep cell (see module docstring)."""
    n = cell.n_sites
    symbols = []
    for e, x in sorted(composition.items()):
        symbols.extend([e] * int(round(x * n)))
    while len(symbols) < n:
        symbols.append(max(composition, key=composition.get))
    while len(symbols) > n:
        symbols.pop()
    np.random.RandomState(seed).shuffle(symbols)
    symbols[cell.index([cell.box // 2] * 3)] = centre
    return symbols


def elastic_constants(uma, atoms, n_steps, delta):
    """C11, C12, C44 (GPa) of a relaxed cubic cell, clamped ions."""
    from ase.units import GPa
    from matscipy.elasticity import Voigt_6x6_to_cubic, fit_elastic_constants
    a = atoms.copy()
    a.calc = uma.calc()
    Cij, _err = fit_elastic_constants(a, symmetry="cubic", N_steps=n_steps,
                                      delta=delta, optimizer=None, verbose=False)
    C11, C12, C44 = Voigt_6x6_to_cubic(Cij)
    a.calc = None
    return C11 / GPa, C12 / GPa, C44 / GPa


def compute_cell(uma, cfg, recipe, cell, composition, centre, seed):
    symbols = decorate_cell(cell, composition, centre, seed)
    a0 = float(cfg["a_start"])
    relaxed, E, conv, steps = uma.relax_cell(to_atoms(cell, symbols, a0),
                                             recipe["relax_fmax"],
                                             recipe["relax_steps"])
    C11, C12, C44 = elastic_constants(uma, relaxed, cfg["strain"]["n_steps"],
                                      cfg["strain"]["delta"])
    return dict(C11=C11, C12=C12, C44=C44, E=E, conv=conv, steps=steps, a0=a0,
                edge=float(np.mean(np.diagonal(relaxed.cell.array))),
                n_atoms=len(relaxed))


def planned_cells(cfg):
    """(composition, centre element, seed) of the configured sweep."""
    cells = []
    for comp in composition_grid(list(cfg["elements"]), float(cfg["grid_step"])):
        seed = 0
        for centre, runs in cfg["centres"]:
            for _ in range(int(runs)):
                seed += 1
                cells.append((comp, centre, seed))
    return cells


def replay_cells(path, cfg, rows):
    with open(resolve(path), newline="") as fh:
        table = list(csv.DictReader(fh))
    picked = [table[i] for i in rows] if rows is not None else table
    return [({e: float(r[e]) for e in cfg["elements"]}, r["centre_element"],
             int(r["seed"])) for r in picked]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--replay", default=None, help="recompute rows of this CSV")
    ap.add_argument("--rows", default=None, help="comma-separated row numbers")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    recipe = load_yaml(cfg["recipe"])
    els = list(cfg["elements"])
    cell = Supercell(get_lattice(cfg["lattice"]), int(cfg["size"]))
    rows = [int(x) for x in args.rows.split(",")] if args.rows else None
    cells = (replay_cells(args.replay, cfg, rows) if args.replay
             else planned_cells(cfg))
    out = resolve(args.out or cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = (["composition_string"] + els
               + ["centre_element", "seed", "C11_GPa", "C12_GPa", "C44_GPa",
                  "n_atoms", "lattice_parameter_A", "a_start_A", "E_eV",
                  "relax_conv", "relax_steps", "model", "task", "t_s",
                  "timestamp"])
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            done = {(r["composition_string"], r["centre_element"], int(r["seed"]))
                    for r in csv.DictReader(fh)}
    from .uma import Uma
    uma = Uma.from_recipe(recipe)
    print(f"[elastic] {len(cells)} cells, {cell.size}^3 = {cell.n_sites} sites")
    for comp, centre, seed in cells:
        key = (composition_string(comp), centre, seed)
        if key in done:
            continue
        t0 = time.time()
        r = compute_cell(uma, cfg, recipe, cell, comp, centre, seed)
        row = {"composition_string": key[0], **{e: comp.get(e, 0.0) for e in els},
               "centre_element": centre, "seed": seed,
               "C11_GPa": f"{r['C11']:.3f}", "C12_GPa": f"{r['C12']:.3f}",
               "C44_GPa": f"{r['C44']:.3f}", "n_atoms": r["n_atoms"],
               "lattice_parameter_A": f"{r['edge']:.6f}",
               "a_start_A": f"{r['a0']:.6f}", "E_eV": f"{r['E']:.6f}",
               "relax_conv": r["conv"], "relax_steps": r["steps"],
               "model": uma.model, "task": uma.task,
               "t_s": f"{time.time() - t0:.1f}",
               "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        new = not Path(out).exists()
        with open(out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"  {key[0]:16s} centre {centre:2s} seed {seed:2d}  C11/C12/C44 "
              f"{r['C11']:7.2f}/{r['C12']:7.2f}/{r['C44']:7.2f} GPa  edge "
              f"{r['edge']:.6f} A  {time.time() - t0:4.0f} s", flush=True)
    print(f"[elastic] -> {out}")


if __name__ == "__main__":
    main()
