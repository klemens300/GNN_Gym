"""Order parameters and statistics recorded along a Metropolis chain.

Warren-Cowley parameter of shell s for the pair (i, j),

    alpha_s(i, j) = 1 - P_s(j | i) / c_j,

with P_s(j | i) the fraction of the shell-s neighbours of an i atom that are
j atoms: 0 for the random alloy, negative where unlike neighbours are
preferred. The pair counts are integers, so the value does not depend on the
order in which the bonds are counted.

Sublattice order of element e on the two sublattices of a structure with a
two-site basis (bcc),

    eta_e = 2 |N_e(sublattice 0) / N_e - 1/2|,

1 for an element on one sublattice only, 0 for an even split. It is blind to
antiphase domains, which is why the order of the snapshots is also read in
reciprocal space (gnn_gym.metropolis.order).

Heat capacity per atom in units of k_B from the energy fluctuations of the
sampled sweeps,

    C_v / (N k_B) = Var(E) / (N k_B^2 T^2) = N Var(e) / (k_B T)^2,

with e = E / N the energy per atom; and, as a smooth cross-check, from the
slope dE/dT along a branch of temperatures.
"""
import numpy as np

K_B = 8.617333262e-5          # eV/K


def pair_counts(z, table, n_species):
    """(n_species, n_species) counts of directed bonds of a neighbour table."""
    src = np.repeat(np.arange(z.size), table.shape[1])
    dst = np.asarray(table).ravel()
    n_pairs = np.zeros((n_species, n_species), dtype=np.int64)
    np.add.at(n_pairs, (z[src], z[dst]), 1)
    return n_pairs


def warren_cowley(z, table, n_species):
    """alpha(i, j) for the shell whose neighbour table is given."""
    n_pairs = pair_counts(z, table, n_species)
    comp = np.bincount(z, minlength=n_species) / z.size
    bonds_i = n_pairs.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - (n_pairs / bonds_i[:, None]) / comp[None, :]


def warren_cowley_shells(z, cell, n_species, shells=(1, 2, 3)):
    """{shell: alpha matrix} on the neighbour tables of a supercell."""
    return {s: warren_cowley(z, cell.neighbours(s), n_species) for s in shells}


def eta_per_species(z, sublattice, n_species):
    """Sublattice order per species, 2 |N_e(sublattice 0) / N_e - 1/2|."""
    out = np.zeros(n_species)
    for e in range(n_species):
        n_e = int((z == e).sum())
        if n_e:
            out[e] = 2.0 * abs(((z == e) & (sublattice == 0)).sum() / n_e - 0.5)
    return out


def block_stats(x, n_blocks=5):
    """Mean and standard error from the scatter of n_blocks block means."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < n_blocks:
        return x.mean(axis=0), np.full(x.shape[1:], np.nan)
    blocks = np.array_split(x, n_blocks)
    means = np.array([b.mean(axis=0) for b in blocks])
    return x.mean(axis=0), means.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def heat_capacity_fluctuation(e_per_atom, n_atoms, T):
    """C_v per atom in k_B from sampled energies per atom (eV)."""
    e = np.asarray(e_per_atom, dtype=np.float64)
    return float(e.var(ddof=1) * n_atoms / (K_B ** 2 * T ** 2))


def heat_capacity_slope(T, e_meV_per_atom):
    """C_v per atom in k_B from dE/dT along a branch (central differences,
    one-sided at the ends); T ascending."""
    return np.gradient(np.asarray(e_meV_per_atom) * 1e-3, np.asarray(T)) / K_B
