"""Lattice parameter of a composition, measured with the uMLIP.

The barriers depend strongly on the lattice parameter (tens of meV per
percent), so the value used by the simulations is not entered by hand but
measured: small defect-free cells with random decorations of the target
composition are relaxed in cell shape and positions, starting from the
Vegard estimate, and a is the mean over decorations of the mean diagonal of
the relaxed cell divided by its size. The lattice parameter is an intensive
bulk quantity and converges quickly with cell size; the decoration spread
(about 1e-4 A) is reported. The largest relative off-diagonal element of
the relaxed cell is recorded as a check of cubic symmetry.

    python -m gnn_gym.oracle.calibrate configs/oracle/calibrate.yaml
    python -m gnn_gym.oracle.calibrate configs/oracle/calibrate.yaml --system TaW
"""
import argparse
import json
import time

import numpy as np

from ..lattice import Supercell
from ..paths import load_yaml, resolve
from ..systems import get_system
from .structures import random_decoration, to_atoms, vegard_lattice_parameter
from .uma import Uma


def calibrate(uma, system, size, seeds, recipe, log=print):
    cell = Supercell(system.lattice_obj, size)
    comp = dict(system.composition)
    a0 = vegard_lattice_parameter(comp, recipe["vegard_A"])
    per_seed, offdiag = [], []
    for seed in seeds:
        symbols = random_decoration(cell, comp, seed)
        t0 = time.time()
        atoms, _E, conv, _n = uma.relax_cell(to_atoms(cell, symbols, a0),
                                             recipe["relax_fmax"],
                                             recipe["relax_steps"])
        if not conv:
            raise RuntimeError(f"cell relaxation did not converge (seed {seed})")
        box = np.asarray(atoms.get_cell())
        diag = np.diagonal(box)
        a = float(diag.mean() / size)
        off = float(np.max(np.abs(box - np.diag(diag))) / diag.mean())
        per_seed.append(a)
        offdiag.append(off)
        log(f"  seed {seed}: a = {a:.9f} A  (start {a0:.4f}, off-diagonal "
            f"{off * 100:.3f} %, {time.time() - t0:.0f} s)")
    return {"a_A": float(np.mean(per_seed)), "a_std_A": float(np.std(per_seed)),
            "a_per_seed_A": per_seed, "a_start_A": a0,
            "max_offdiag_rel": float(max(offdiag)),
            "composition": {e: float(x) for e, x in sorted(comp.items())},
            "size": int(size), "n_sites": int(cell.n_sites),
            "seeds": list(seeds), "relax_fmax": recipe["relax_fmax"],
            "model": uma.model, "task": uma.task}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="configs/oracle/calibrate.yaml")
    ap.add_argument("--system", action="append", default=None,
                    help="calibrate only these systems (repeatable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    recipe = load_yaml(cfg["recipe"])
    out = resolve(args.out or cfg["out"])
    results = json.loads(out.read_text()) if out.exists() else {}
    uma = Uma.from_recipe(recipe)
    for name in args.system or cfg["systems"]:
        system = get_system(name)
        print(f"[calibrate] {name}: {cfg['size']}^3 cells, seeds {cfg['seeds']}")
        r = calibrate(uma, system, int(cfg["size"]), list(cfg["seeds"]), recipe)
        results[name] = r
        print(f"[calibrate] {name}: a = {r['a_A']:.9f} +- {r['a_std_A']:.6f} A")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"[calibrate] -> {out}")


def calibrated_lattice_parameter(name, path="results/oracle/lattice_parameters.json"):
    """The calibrated value of a system from the calibration output."""
    data = json.loads(resolve(path).read_text())
    return float(data[name]["a_A"])


if __name__ == "__main__":
    main()
