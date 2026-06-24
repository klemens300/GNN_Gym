"""
Post-hoc observables computed from a KMCResult.

All functions here read the hop history (and optionally the periodic species
snapshots) of a finished run and produce derived quantities — they never
modify the run or trigger new BKL steps.

Provided observables:

- tracer_msd_per_element(result):
    Mean-squared displacement per element as a function of simulated time,
    obtained by replaying the hop history while tracking individual atom IDs.
    Each BKL step moves exactly one atom by the negative of the vacancy step
    vector; the per-element MSD is the mean of |r_atom(t) - r_atom(0)|^2 over
    all atoms of that element.

- warren_cowley_sro_snapshot(species, nn_table, element_symbols):
    1st-NN-shell Warren-Cowley short-range-order parameter alpha_ij computed
    from a single configuration. alpha_ij = 1 - P(j|i)/c_j where P(j|i) is the
    probability of finding a j-atom among the 1NN of an i-atom.

- warren_cowley_sro_trajectory(result):
    Time series of alpha_ij for every element pair, computed at every
    snapshot stored in result. Skips the calculation if the run had no
    snapshot recording.
"""

from typing import Dict, Tuple

import numpy as np

from KMC.state import VACANCY_SPECIES


# ---------------------------------------------------------------------------
# Tracer MSD per element
# ---------------------------------------------------------------------------

def tracer_msd_per_element(result) -> Dict[str, np.ndarray]:
    """Time-resolved tracer MSD averaged over all atoms of each element.

    The hop history is replayed once: for every BKL step we find which atom
    sat at the post-jump vacancy site, attribute the negative-vacancy step
    vector to that atom, and update an unfolded atom-position vector. The
    per-element MSD at time `result.times_with_zero_s[k]` is the average of
    |r_atom(k) - r_atom(0)|^2 over all atoms of that element.

    Args:
        result: KMCResult.

    Returns:
        Dict mapping each element symbol to an np.ndarray of shape
        [n_steps + 1] with MSD values in (Angstrom)^2. The vacancy is not an
        element and is omitted from the dict.
    """
    state = result.final_state
    element_symbols = state.element_symbols
    n_elements = len(element_symbols)

    # ----- Identify each atom's initial site -----
    initial_species = result.initial_species
    n_sites = initial_species.shape[0]
    non_vac_mask = initial_species != VACANCY_SPECIES
    non_vac_sites = np.where(non_vac_mask)[0]
    n_atoms = int(non_vac_sites.shape[0])

    # atom_id_at_site[s] = atom id currently at site s (-1 if vacancy)
    atom_id_at_site = np.full(n_sites, -1, dtype=np.int64)
    atom_id_at_site[non_vac_sites] = np.arange(n_atoms, dtype=np.int64)

    # Initial cartesian positions and species (constant per atom)
    site_positions = state.positions
    atom_pos_initial = site_positions[non_vac_sites].copy()
    atom_pos_unfolded = atom_pos_initial.copy()
    atom_species = initial_species[non_vac_sites].astype(np.int64)

    n_per_element = np.bincount(atom_species, minlength=n_elements).astype(np.int64)

    # ----- Replay hop history with incremental MSD update -----
    # We accumulate sum_e |Δr_atom|^2 over atoms of element e; per-atom MSD
    # is then sum_e / n_per_element[e]. We update incrementally because each
    # step changes only one atom's contribution.
    msd_sum_per_element = np.zeros(n_elements, dtype=np.float64)
    msd_per_element = np.zeros((result.n_steps + 1, n_elements), dtype=np.float64)

    vac_steps = np.diff(result.vacancy_positions_unfolded, axis=0)
    vacancy_indices = result.vacancy_indices

    for k in range(result.n_steps):
        v_before = int(vacancy_indices[k])
        v_after = int(vacancy_indices[k + 1])
        hopper_id = int(atom_id_at_site[v_after])

        # Atom moves opposite to the vacancy
        atom_step = -vac_steps[k]

        p_old = atom_pos_unfolded[hopper_id]
        p_new = p_old + atom_step

        # Incremental update of squared-displacement sum for this element
        d_old = p_old - atom_pos_initial[hopper_id]
        d_new = p_new - atom_pos_initial[hopper_id]
        sq_old = float(d_old @ d_old)
        sq_new = float(d_new @ d_new)
        el_idx = int(atom_species[hopper_id])
        msd_sum_per_element[el_idx] += (sq_new - sq_old)

        atom_pos_unfolded[hopper_id] = p_new
        atom_id_at_site[v_after] = -1
        atom_id_at_site[v_before] = hopper_id

        # Store per-element averages for this time point
        for e in range(n_elements):
            if n_per_element[e] > 0:
                msd_per_element[k + 1, e] = msd_sum_per_element[e] / n_per_element[e]

    return {
        symbol: msd_per_element[:, i] for i, symbol in enumerate(element_symbols)
    }


# ---------------------------------------------------------------------------
# Warren-Cowley short-range-order parameter (1st NN shell)
# ---------------------------------------------------------------------------

def warren_cowley_sro_snapshot(
    species: np.ndarray,
    nn_table: np.ndarray,
    element_symbols: list,
) -> Dict[Tuple[str, str], float]:
    """Warren-Cowley alpha_ij in the 1st-NN-shell from a single config.

    Definition:
        alpha_ij = 1 - P(j | i) / c_j
    where P(j | i) is the probability that a 1-NN of an i-atom is element j,
    measured as (number of i-j NN bonds) / (number of bonds out of i-atoms),
    and c_j is the composition fraction of j among the non-vacancy atoms.

    Args:
        species: int array shape [N_sites]; vacancy site has VACANCY_SPECIES.
        nn_table: int array shape [N_sites, 8]; the 8 NN site indices for
            every site.
        element_symbols: list[str], indexed by species value.

    Returns:
        Dict mapping (symbol_i, symbol_j) tuples to alpha_ij. NaN if a needed
        count is zero (e.g. element absent from the cell).
    """
    n_elements = len(element_symbols)
    n_sites = species.shape[0]

    non_vac_mask = species != VACANCY_SPECIES
    n_atoms_total = int(non_vac_mask.sum())
    if n_atoms_total == 0:
        return {(a, b): float("nan") for a in element_symbols for b in element_symbols}

    # Composition fractions among non-vacancy atoms
    n_per_element = np.bincount(
        species[non_vac_mask].astype(np.int64), minlength=n_elements
    )
    composition = n_per_element.astype(np.float64) / float(n_atoms_total)

    # Count NN-pairs (i, j) where both endpoints are non-vacancy.
    # Each unordered NN-bond is counted twice (once from each endpoint).
    n_pairs = np.zeros((n_elements, n_elements), dtype=np.int64)
    for site in range(n_sites):
        sp_i = species[site]
        if sp_i == VACANCY_SPECIES:
            continue
        i = int(sp_i)
        for nn_site in nn_table[site]:
            sp_j = species[int(nn_site)]
            if sp_j == VACANCY_SPECIES:
                continue
            j = int(sp_j)
            n_pairs[i, j] += 1

    # alpha_ij = 1 - P(j|i)/c_j = 1 - (n_pairs[i,j]/sum_k n_pairs[i,k]) / c_j
    sro: Dict[Tuple[str, str], float] = {}
    bonds_per_element = n_pairs.sum(axis=1)
    for i, sym_i in enumerate(element_symbols):
        for j, sym_j in enumerate(element_symbols):
            if bonds_per_element[i] == 0 or composition[j] == 0:
                sro[(sym_i, sym_j)] = float("nan")
                continue
            P_j_given_i = float(n_pairs[i, j]) / float(bonds_per_element[i])
            sro[(sym_i, sym_j)] = 1.0 - P_j_given_i / composition[j]

    return sro


def warren_cowley_sro_trajectory(result) -> Dict[Tuple[str, str], np.ndarray]:
    """Time series of alpha_ij at every recorded snapshot.

    Args:
        result: KMCResult — must have been created with snapshot_every_n_steps > 0.

    Returns:
        Dict mapping (symbol_i, symbol_j) -> np.ndarray of shape [n_snapshots]
        containing alpha_ij at each snapshot time.

    Raises:
        ValueError: if result has no recorded snapshots.
    """
    if result.snapshot_species is None:
        raise ValueError(
            "Result has no snapshot_species recorded; rerun with "
            "snapshot_every_n_steps > 0 to enable Warren-Cowley trajectories."
        )

    state = result.final_state
    nn_table = state.nn_table
    element_symbols = state.element_symbols

    n_snapshots = result.n_snapshots
    # Initialise with NaN, then fill row by row
    series: Dict[Tuple[str, str], np.ndarray] = {
        (a, b): np.full(n_snapshots, np.nan, dtype=np.float64)
        for a in element_symbols for b in element_symbols
    }

    for k in range(n_snapshots):
        snapshot = warren_cowley_sro_snapshot(
            result.snapshot_species[k], nn_table, element_symbols
        )
        for key, value in snapshot.items():
            series[key][k] = value

    return series
