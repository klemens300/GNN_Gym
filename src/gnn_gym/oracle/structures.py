"""Reference structures: decorated supercells as ASE Atoms, jump endpoints.

Site bookkeeping is integer (gnn_gym.lattice): which atom sits where never
depends on the lattice parameter, which enters only in `to_atoms`, when the
coordinates are handed to the calculator.

The integer counts of an arbitrary composition follow the largest-remainder
rule: floor(x_e N) atoms each, the remaining sites to the elements with the
largest fractional parts. Together with `gnn_gym.lattice.decorate` (sorted
symbols, one seeded shuffle) this is the convention of the jump datasets, so
a stored row (realised fractions and seed) rebuilds its structure exactly.
"""
import numpy as np

from ..lattice import decorate


def composition_counts(composition, n_sites):
    """Integer counts per element, largest-remainder rule."""
    els = sorted(composition)
    w = np.array([composition[e] for e in els], dtype=float)
    w = w / w.sum()
    counts = {e: int(np.floor(x * n_sites)) for e, x in zip(els, w)}
    rest = n_sites - sum(counts.values())
    by_remainder = sorted(els, key=lambda e: (w[els.index(e)] * n_sites) % 1.0,
                          reverse=True)
    for e in by_remainder[:rest]:
        counts[e] += 1
    return counts


def random_decoration(cell, composition, seed):
    """Symbols of a random decoration of `cell` with `composition`."""
    return decorate(composition_counts(composition, cell.n_sites), seed,
                    cell.n_sites)


def realised_fractions(symbols, elements):
    n = len(symbols)
    return {e: symbols.count(e) / n for e in elements}


def vegard_lattice_parameter(composition, table):
    """Fraction-weighted mean of the elemental lattice parameters."""
    missing = [e for e in composition if e not in table]
    if missing:
        raise ValueError(f"no Vegard reference for {missing}")
    return float(sum(x * table[e] for e, x in composition.items()))


def to_atoms(cell, symbols, a):
    """ASE Atoms of a decorated supercell at lattice parameter a."""
    from ase import Atoms
    return Atoms(symbols=list(symbols), positions=cell.positions(a),
                 cell=cell.cell(a), pbc=True)


def jump_endpoints(atoms, v, j):
    """Initial (vacancy on v) and final (atom of j moved onto v) structures.

    Both keep the same atom order, so the NEB interpolates atom by atom.
    """
    init = atoms.copy()
    del init[v]
    fin = atoms.copy()
    fin.positions[j] = fin.positions[v].copy()
    del fin[v]
    return init, fin
