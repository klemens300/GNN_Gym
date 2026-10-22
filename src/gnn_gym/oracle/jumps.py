"""Reference vacancy jumps: uMLIP-NEB labels in periodic cells.

COMPOSITION
    A jump's composition is drawn without knowledge of the material: one of
    the 2^m - 1 non-empty subsets of the m elements with equal probability
    (unary, binary, ... cells equally frequent), then flat Dirichlet
    fractions within the subset. The cell is decorated at random with these
    fractions; composition and decoration are deterministic functions of the
    jump's seed, and the realised fractions and the seed are stored, so every
    structure is rebuilt from its row (gnn_gym.training.data does this).

GEOMETRY
    The vacancy sits on the first basis site of the central conventional cell
    and the jumping atom at the lattice's reference jump vector from it
    (`Supercell.reference_jump`); with periodic boundaries the absolute
    position carries no physics.

RECIPE (configs/oracle/recipe.yaml)
    1. the defect-free cell, started at the Vegard lattice parameter, is
       relaxed in cell shape and positions; this fixes the box;
    2. both endpoints are built in that box from the ideal positions and
       relaxed at fixed cell;
    3. climbing-image NEB between the relaxed endpoints.
    A jump is discarded only if the highest image is an endpoint (failed
    NEB). No barrier window is applied: it would remove exactly the fast
    jumps that dominate the KMC.

LABELS PER ROW
    dE = E(final) - E(initial), E_fwd, E_bwd, B_sym = (E_fwd + E_bwd)/2, the
    energy of the relaxed defect-free cell (vacancy formation energies,
    gnn_gym.oracle.formation) and the unrelaxed ideal-lattice energies of
    both endpoints.

    python -m gnn_gym.oracle.jumps configs/oracle/jumps_MoNbTaW.yaml --stream train
    python -m gnn_gym.oracle.jumps <config> --stream referee --indices 0:3 --out x.csv

Resume-safe: indices already complete in the output are skipped.
"""
import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..lattice import Supercell, get_lattice
from ..paths import load_yaml, resolve
from .structures import (jump_endpoints, random_decoration, realised_fractions,
                         to_atoms, vegard_lattice_parameter)
from .uma import Uma, neb_labels


def sample_composition(elements, rng):
    """Uniform over the non-empty element subsets, flat Dirichlet within."""
    m = len(elements)
    while True:
        mask = rng.integers(0, 2, m)
        if mask.any():
            break
    subset = [e for e, b in zip(elements, mask) if b]
    fracs = rng.dirichlet(np.ones(len(subset)))
    return {e: float(f) for e, f in zip(subset, fracs)}


def jump_cell(cell, elements, seed):
    """(composition drawn, symbols, vacancy site, jumping site) of one jump."""
    comp = sample_composition(elements, np.random.default_rng(seed))
    symbols = random_decoration(cell, comp, seed)
    v, j = cell.reference_jump()
    return comp, symbols, v, j


def compute_jump(uma, recipe, cell, symbols, v, j, a0):
    """The recipe on one decorated cell -> dict of results."""
    rf, nb = recipe["relax_fmax"], recipe["neb"]
    steps = recipe["relax_steps"]
    perfect, E_perf, c0, n0 = uma.relax_cell(to_atoms(cell, symbols, a0), rf, steps)
    box = np.asarray(perfect.get_cell()).copy()
    a_rel = float(np.diagonal(box).mean() / cell.size)

    win = to_atoms(cell, symbols, a0)
    win.set_cell(box, scale_atoms=True)
    init_id, fin_id = jump_endpoints(win, v, j)
    E_i0 = uma.energy(init_id)
    E_f0 = uma.energy(fin_id)
    init_rel, E_i, c1, n1 = uma.relax(init_id, rf, steps)
    fin_rel, E_f, c2, n2 = uma.relax(fin_id, rf, steps)
    _images, energies, c3, n3 = uma.neb(
        init_rel, fin_rel, n_images=nb["n_images"], fmax=nb["fmax"],
        steps=nb["steps"], k=nb["k"], climb=nb["climb"], method=nb["method"])
    lab = neb_labels(energies, E_i, E_f)
    lab.update(a_relaxed=a_rel, E_perfect=E_perf, E_initial=E_i, E_final=E_f,
               E_initial_ideal=E_i0, E_final_ideal=E_f0,
               relax_steps=(n0, n1, n2), relax_conv=bool(c0 and c1 and c2),
               neb_steps=n3, neb_conv=c3, n_images=len(energies))
    return lab


def columns_for(elements):
    return (["idx", "subsystem", "size", "n_sites"]
            + [f"frac_{e}" for e in elements]
            + ["vac_species", "jump_species",
               "a_start_A", "a_relaxed_A", "E_perfect_eV",
               "E_initial_eV", "E_final_eV", "dE_eV",
               "E_saddle_eV", "E_fwd_eV", "E_bwd_eV", "B_sym_eV",
               "saddle_image",
               "E_initial_ideal_eV", "E_final_ideal_eV", "dE_ideal_eV",
               "relax_steps_perfect", "relax_steps_i", "relax_steps_f",
               "relax_conv", "neb_steps", "neb_conv", "db_residual_eV",
               "t_s", "seed", "model", "task",
               "neb_images", "neb_fmax", "relax_fmax", "timestamp"])


def make_row(idx, seed, comp, symbols, v, j, a0, lab, cell, elements, recipe,
             t_s):
    rf = realised_fractions(symbols, elements)
    return {
        "idx": idx, "subsystem": "".join(sorted(comp)),
        "size": cell.size, "n_sites": len(symbols),
        **{f"frac_{e}": f"{rf[e]:.6f}" for e in elements},
        "vac_species": symbols[v], "jump_species": symbols[j],
        "a_start_A": f"{a0:.9f}", "a_relaxed_A": f"{lab['a_relaxed']:.9f}",
        "E_perfect_eV": f"{lab['E_perfect']:.6f}",
        "E_initial_eV": f"{lab['E_initial']:.6f}",
        "E_final_eV": f"{lab['E_final']:.6f}",
        "dE_eV": f"{lab['dE']:.6f}", "E_saddle_eV": f"{lab['E_saddle']:.6f}",
        "E_fwd_eV": f"{lab['E_fwd']:.6f}", "E_bwd_eV": f"{lab['E_bwd']:.6f}",
        "B_sym_eV": f"{lab['B_sym']:.6f}", "saddle_image": lab["saddle_image"],
        "E_initial_ideal_eV": f"{lab['E_initial_ideal']:.6f}",
        "E_final_ideal_eV": f"{lab['E_final_ideal']:.6f}",
        "dE_ideal_eV": f"{lab['E_final_ideal'] - lab['E_initial_ideal']:.6f}",
        "relax_steps_perfect": lab["relax_steps"][0],
        "relax_steps_i": lab["relax_steps"][1],
        "relax_steps_f": lab["relax_steps"][2],
        "relax_conv": lab["relax_conv"],
        "neb_steps": lab["neb_steps"], "neb_conv": lab["neb_conv"],
        "db_residual_eV": f"{lab['db_residual']:.3e}",
        "t_s": f"{t_s:.1f}", "seed": seed,
        "model": recipe["uma"]["model"], "task": recipe["uma"]["task"],
        "neb_images": recipe["neb"]["n_images"],
        "neb_fmax": recipe["neb"]["fmax"], "relax_fmax": recipe["relax_fmax"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def read_done(path):
    """Indices of complete rows (a truncated last line does not count)."""
    if not Path(path).exists():
        return set()
    done = set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                if str(r.get("timestamp") or "").strip():
                    done.add(int(r["idx"]))
            except (KeyError, ValueError):
                pass
    return done


def append_row(path, row, columns):
    header = not Path(path).exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if header:
            w.writeheader()
        w.writerow(row)


def parse_indices(spec, n):
    if spec is None:
        return range(n)
    lo, _, hi = spec.partition(":")
    return range(int(lo or 0), int(hi) if hi else n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--stream", required=True)
    ap.add_argument("--indices", default=None, help="range lo:hi of jump indices")
    ap.add_argument("--out", default=None, help="output CSV (default: config)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compositions and decorations only, no uMLIP")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    recipe = load_yaml(cfg["recipe"])
    st = cfg["streams"][args.stream]
    els = list(cfg["elements"])
    cell = Supercell(get_lattice(cfg["lattice"]), cfg["size"])
    seed_base, n_jumps = int(st["seed_base"]), int(st["n_jumps"])
    out = resolve(args.out or st["out"])
    columns = columns_for(els)
    indices = parse_indices(args.indices, n_jumps)
    print(f"{args.stream} stream: seeds {seed_base}+i, jumps {indices.start}.."
          f"{indices.stop - 1}, {cell.size}^3 = {cell.n_sites} sites, {els}")

    if args.dry_run:
        from collections import Counter
        hist = Counter()
        for idx in indices:
            comp, syms, v, j = jump_cell(cell, els, seed_base + idx)
            hist["".join(sorted(comp))] += 1
        for sub in sorted(hist, key=lambda s: (len(s), s)):
            print(f"  {sub:>12}: {hist[sub]}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    uma = Uma.from_recipe(recipe)
    stop = {"requested": False}

    def _sigint(_signum, _frame):
        if stop["requested"]:
            raise KeyboardInterrupt
        stop["requested"] = True
        print("\n[interrupt] finishing the current jump, then stopping",
              flush=True)

    signal.signal(signal.SIGINT, _sigint)
    t_start, n_done, n_failed = time.time(), 0, 0
    for idx in indices:
        if stop["requested"]:
            break
        if idx in done:
            continue
        t0 = time.time()
        seed = seed_base + idx
        comp, syms, v, j = jump_cell(cell, els, seed)
        a0 = vegard_lattice_parameter(realised_fractions(syms, els),
                                      recipe["vegard_A"])
        try:
            lab = compute_jump(uma, recipe, cell, syms, v, j, a0)
        except Exception as exc:
            n_failed += 1
            print(f"idx {idx:>6} FAILED: {exc}", flush=True)
            continue
        if lab["saddle_image"] in (0, lab["n_images"] - 1):
            n_failed += 1
            print(f"idx {idx:>6} REJECTED: saddle on an endpoint", flush=True)
            continue
        append_row(out, make_row(idx, seed, comp, syms, v, j, a0, lab, cell,
                                 els, recipe, time.time() - t0), columns)
        n_done += 1
        print(f"idx {idx:>6} {''.join(sorted(comp)):>12}  {syms[v]}->{syms[j]}"
              f"  dE={lab['dE'] * 1000:+8.1f} meV  B_sym={lab['B_sym']:.4f} eV"
              f"  a={lab['a_relaxed']:.5f}  {time.time() - t0:5.0f} s",
              flush=True)
    print(f"\n[written] {out}  ({n_done} rows, {n_failed} failed/rejected, "
          f"{(time.time() - t_start) / 3600:.2f} h)")
    return 130 if stop["requested"] else 0


if __name__ == "__main__":
    sys.exit(main())
