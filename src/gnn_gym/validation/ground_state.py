"""uMLIP energies of ordered candidates against the random alloy.

The site-energy model is trained on vacancy jumps only; whether it ranks
ordered arrangements correctly is tested against direct uMLIP energies of
the same candidates (the model side is computed with the Metropolis code).
All candidates have the composition of the system, so the comparison
(E - E_random per atom) needs no chemical potentials.

CANDIDATES (equiatomic, in a size^3 cell of the system's lattice)
    random      equal numbers of each element on random sites
    B2          the elements split into two equal groups, each group
                randomly on one of the two sublattices; every split that
                keeps the first element in the first group
    layered     whole (100) planes cycling a given element sequence along z
                (for Mo-Nb-Ta-W the Nb-Mo-Ta-W-W-Ta-Mo-Nb sequence of
                Kostiuchenko et al.); the cell edge must be a multiple of
                the period in conventional cells
    random and B2 are averaged over n_random decorations.

The structures are ionically relaxed at fixed cell (FIRE). An ordering
temperature estimate for each candidate below the random alloy is
|dE| / (k_B dS) with the ideal configurational entropy drop dS per atom
(ln 2 for B2, ln 4 for the layered candidate) -- an upper bound for a
transition into that candidate.

    python -m gnn_gym.validation.ground_state configs/validation/ground_state_MoNbTaW.yaml

Output: results/validation/uma_ground_state_<system>.json with
dE_meV_per_atom_vs_random[relax][candidate] and the decoration spread.
"""
import argparse
import json
from itertools import combinations

import numpy as np

from ..lattice import Supercell
from ..paths import load_yaml, resolve
from ..systems import get_system

K_B_EV_PER_K = 8.617333e-5


def fill_equiatomic(sites, elements, rng):
    """Shuffle `sites` and deal them out in equal blocks, one per element."""
    sites = np.array(sites)
    rng.shuffle(sites)
    per = len(sites) // len(elements)
    return {int(j): el for k, el in enumerate(elements)
            for j in sites[k * per:(k + 1) * per]}


def partitions(elements):
    """Even splits of the elements over two sublattices (first element in A)."""
    m = len(elements)
    if m % 2:
        return []
    return [(a, tuple(e for e in elements if e not in a))
            for a in combinations(elements, m // 2) if elements[0] in a]


def candidates(cell, elements, n_random, layer_sequence=None,
               random_seed=1000, b2_seed=100):
    """{name: ([symbol lists], entropy drop per atom / k_B)}."""
    n = cell.n_sites
    if n % len(elements):
        raise ValueError(f"{n} sites are not divisible by {len(elements)}")
    sub = cell.sublattice()

    def symbols(assign):
        return [assign[i] for i in range(n)]

    out = {"random": ([symbols(fill_equiatomic(
        np.arange(n), list(elements), np.random.default_rng(random_seed + s)))
        for s in range(n_random)], 0.0)}
    for ga, gb in partitions(list(elements)):
        cells = []
        for s in range(n_random):
            rng = np.random.default_rng(b2_seed + s)
            assign = fill_equiatomic(np.flatnonzero(sub == 0), list(ga), rng)
            assign.update(fill_equiatomic(np.flatnonzero(sub == 1), list(gb), rng))
            cells.append(symbols(assign))
        out["B2 {%s}|{%s}" % (",".join(ga), ",".join(gb))] = (cells, np.log(2.0))
    if layer_sequence and set(layer_sequence) == set(elements):
        period = len(layer_sequence)
        if (cell.box % period) != 0:
            raise ValueError(f"cell edge {cell.box} (in a/2) is not a multiple "
                             f"of the layer period {period}")
        plane = cell.coords[:, 2] % period
        n_distinct = len(set(layer_sequence))
        out["layered"] = ([[layer_sequence[p] for p in plane]],
                          np.log(float(n_distinct)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    recipe = load_yaml(cfg["recipe"])
    system = get_system(cfg["system"])
    elements = list(system.elements)
    a = cfg["lattice_parameter_A"]
    if a == "calibrated":
        from ..oracle.calibrate import calibrated_lattice_parameter
        a = calibrated_lattice_parameter(system.name, cfg["calibration"])
    a = float(a)
    cell = Supercell(system.lattice_obj, int(cfg["size"]))
    cand = candidates(cell, elements, int(cfg["n_random"]),
                      cfg.get("layer_sequence"))

    from ..oracle.structures import to_atoms
    from ..oracle.uma import Uma
    uma = Uma.from_recipe(recipe)
    n = cell.n_sites
    print(f"[ground state] {system.name}: {cell.size}^3 = {n} atoms, a = {a} A, "
          f"{cfg['n_random']} decorations per random/B2 candidate")

    result = dict(system=system.name, supercell=cell.size, natoms=n, a=a,
                  n_random=int(cfg["n_random"]), fmax=cfg["fmax"],
                  steps=cfg["steps"], model=uma.model, task=uma.task,
                  dE_meV_per_atom_vs_random={}, std_meV_per_atom={},
                  T_ord_K={}, entropy_drop_kB={k: float(v[1])
                                               for k, v in cand.items()})
    for relax in cfg["relax"]:
        rows = {}
        for name, (cells, _dS) in cand.items():
            e = []
            for syms in cells:
                atoms = to_atoms(cell, syms, a)
                if relax == "ionic":
                    _at, E, _c, _n = uma.relax(atoms, cfg["fmax"], cfg["steps"])
                elif relax == "full":
                    _at, E, _c, _n = uma.relax_cell(atoms, cfg["fmax"], cfg["steps"])
                else:
                    raise ValueError(f"unknown relaxation {relax!r}")
                e.append(E / n)
            e = np.array(e)
            rows[name] = (float(e.mean()), float(e.std()))
        e_rand = rows["random"][0]
        dE = {k: (v[0] - e_rand) * 1e3 for k, v in rows.items()}
        T = {k: (abs(d) * 1e-3 / (K_B_EV_PER_K * cand[k][1])
                 if cand[k][1] > 0 and d < 0 else None) for k, d in dE.items()}
        result["dE_meV_per_atom_vs_random"][relax] = dE
        result["std_meV_per_atom"][relax] = {k: v[1] * 1e3 for k, v in rows.items()}
        result["T_ord_K"][relax] = T
        print(f"\n  relax = {relax}   (E - E_random, meV/atom)")
        for name in sorted(dE, key=dE.get):
            t = f"T_ord ~ {T[name]:4.0f} K" if T[name] else ""
            print(f"    {name:24s} {dE[name]:+8.2f}  (+/- "
                  f"{rows[name][1] * 1e3:4.1f})  {t}")

    out = resolve(args.out or cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[ground state] -> {out}")


if __name__ == "__main__":
    main()
