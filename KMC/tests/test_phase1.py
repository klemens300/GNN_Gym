"""
Phase-1 sanity tests for the KMC engine.

Run from anywhere with pytest:

    cd /home/klemens/doctor/gnn_kmc/scipts
    pytest -v KMC/tests/test_phase1.py

Or directly as a script (also runs pytest internally):

    cd /home/klemens/doctor/gnn_kmc/scipts/KMC/tests
    python test_phase1.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `scipts/` importable so that `from KMC.* import ...` works
_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.state import KMCState, VACANCY_SPECIES
from KMC.barrier_predictor import MockBarrierPredictor, BarrierPredictor
from KMC.engine import bkl_step, K_B_EV_PER_K


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    """4x4x4 BCC -> 128 sites; small enough to keep tests fast."""
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=4,
        lattice_parameter_A=3.22,
        random_seed=42,
        temperature_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=10,
    )


# ---------------------------------------------------------------------------
# KMCConfig
# ---------------------------------------------------------------------------

def test_config_n_sites(small_config):
    assert small_config.n_sites == 2 * 4 ** 3 == 128


def test_config_composition_must_sum_to_one():
    with pytest.raises(ValueError):
        KMCConfig(composition={"Mo": 0.5, "Nb": 0.4})  # sum != 1.0


def test_config_unknown_element_in_composition():
    with pytest.raises(ValueError):
        KMCConfig(
            elements=["Mo", "Nb"],
            composition={"Mo": 0.5, "Cr": 0.5},  # Cr not in elements
        )


def test_config_explicit_vacancy_requires_xyz():
    """vacancy_initial_position='explicit' without vacancy_initial_xyz raises."""
    with pytest.raises(ValueError, match="vacancy_initial_xyz"):
        KMCConfig(vacancy_initial_position="explicit")


def test_config_explicit_vacancy_xyz_wrong_length():
    """vacancy_initial_xyz must be a length-3 tuple."""
    with pytest.raises(ValueError, match="3-tuple"):
        KMCConfig(
            vacancy_initial_position="explicit",
            vacancy_initial_xyz=(1.0, 2.0),  # only 2 components
        )


def test_config_explicit_vacancy_accepts_valid_xyz():
    """Valid (x, y, z) tuple passes __post_init__."""
    cfg = KMCConfig(
        vacancy_initial_position="explicit",
        vacancy_initial_xyz=(1.61, 1.61, 1.61),
    )
    assert cfg.vacancy_initial_xyz == (1.61, 1.61, 1.61)


# ---------------------------------------------------------------------------
# KMCState — geometry & initialization
# ---------------------------------------------------------------------------

def test_state_random_basic(small_config):
    state = KMCState.from_random_composition(small_config)
    assert state.n_sites == 128
    # Exactly one vacancy
    assert int((state.species == VACANCY_SPECIES).sum()) == 1
    assert int(state.species[state.vacancy_index]) == VACANCY_SPECIES
    # nn_table shape and content
    assert state.nn_table.shape == (state.n_sites, 8)
    for i in range(state.n_sites):
        nn = state.nn_table[i].tolist()
        assert len(set(nn)) == 8       # 8 distinct neighbours
        assert i not in nn              # no self-loop


def test_state_nn_distances_match_bcc(small_config):
    """Spot-check: every NN is at distance a*sqrt(3)/2 (with PBC)."""
    state = KMCState.from_random_composition(small_config)
    expected = small_config.lattice_parameter_A * np.sqrt(3.0) / 2.0
    cell = state.cell
    L = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
    for i in [0, 50, 100, 127]:
        for j in state.nn_table[i]:
            d = state.positions[j] - state.positions[i]
            for ax in range(3):
                d[ax] -= round(d[ax] / L[ax]) * L[ax]
            assert abs(np.linalg.norm(d) - expected) < 0.01


def test_state_random_composition_counts(small_config):
    """Equiatomic 4x4x4 BCC: 128 sites - 1 vacancy = 127 atoms ~= 32 each."""
    state = KMCState.from_random_composition(small_config)
    el_to_idx = {el: i for i, el in enumerate(small_config.elements)}
    counts = {el: int((state.species == idx).sum())
              for el, idx in el_to_idx.items()}
    total_atoms = sum(counts.values())
    assert total_atoms == 127
    # No element drifts more than 1 from the equiatomic count
    for el, c in counts.items():
        assert abs(c - 32) <= 1, f"{el}: {c}"


def test_state_bicrystal(small_config):
    small_config.initial_state_strategy = "bicrystal"
    small_config.bicrystal_elements = ("Ta", "W")
    state = KMCState.from_bicrystal(small_config)

    el_to_idx = {el: i for i, el in enumerate(small_config.elements)}
    Ta_idx, W_idx = el_to_idx["Ta"], el_to_idx["W"]
    non_vac = state.species != VACANCY_SPECIES
    # Only Ta and W appear (other elements absent in this bicrystal)
    assert set(state.species[non_vac].tolist()) == {Ta_idx, W_idx}
    n_ta = int((state.species == Ta_idx).sum())
    n_w = int((state.species == W_idx).sum())
    assert abs(n_ta - n_w) <= 1


def test_state_explicit_vacancy_position(small_config):
    """vacancy_initial_position='explicit' places the vacancy at the closest site."""
    # Pick a non-trivial target near the (1,1,1) corner of the supercell so
    # that we can assert against the actual nearest BCC site.
    target = (3.22, 3.22, 3.22)
    small_config.vacancy_initial_position = "explicit"
    small_config.vacancy_initial_xyz = target
    state = KMCState.from_random_composition(small_config)

    # Vacancy must be at exactly one site, and that site must be the
    # PBC-closest one to the requested xyz.
    assert int((state.species == VACANCY_SPECIES).sum()) == 1
    L = np.array([state.cell[0, 0], state.cell[1, 1], state.cell[2, 2]])
    deltas = state.positions - np.asarray(target)
    for ax in range(3):
        deltas[:, ax] -= np.round(deltas[:, ax] / L[ax]) * L[ax]
    expected_idx = int(np.argmin(np.linalg.norm(deltas, axis=1)))
    assert state.vacancy_index == expected_idx


def test_config_xyz_json_roundtrip(tmp_path):
    """vacancy_initial_xyz survives to_json/from_json as a tuple."""
    cfg = KMCConfig(
        vacancy_initial_position="explicit",
        vacancy_initial_xyz=(1.0, 2.0, 3.0),
    )
    path = tmp_path / "cfg.json"
    cfg.to_json(path)
    loaded = KMCConfig.from_json(path)
    assert loaded.vacancy_initial_xyz == (1.0, 2.0, 3.0)
    assert isinstance(loaded.vacancy_initial_xyz, tuple)


def test_state_from_symbol_array(small_config):
    """Custom initialisation reproduces the requested symbol assignment."""
    n = small_config.n_sites
    symbols = ["Mo"] * n
    symbols[5] = "W"
    symbols[10] = "Ta"
    state = KMCState.from_symbol_array(small_config, symbols, vacancy_index=0)

    el_to_idx = {el: i for i, el in enumerate(small_config.elements)}
    assert state.vacancy_index == 0
    assert int(state.species[0]) == VACANCY_SPECIES
    assert int(state.species[5]) == el_to_idx["W"]
    assert int(state.species[10]) == el_to_idx["Ta"]
    other_sites = [i for i in range(n) if i not in (0, 5, 10)]
    assert all(int(state.species[i]) == el_to_idx["Mo"] for i in other_sites)


# ---------------------------------------------------------------------------
# KMCState — mutation
# ---------------------------------------------------------------------------

def test_swap_vacancy_basic(small_config):
    state = KMCState.from_random_composition(small_config)
    nn = state.get_neighbor_atom_indices()
    target = int(nn[0])
    target_species = int(state.species[target])
    old_vac = state.vacancy_index

    state.swap_vacancy(target)

    assert state.vacancy_index == target
    assert int(state.species[target]) == VACANCY_SPECIES
    assert int(state.species[old_vac]) == target_species
    assert int((state.species == VACANCY_SPECIES).sum()) == 1


def test_swap_vacancy_into_vacancy_raises(small_config):
    state = KMCState.from_random_composition(small_config)
    with pytest.raises(ValueError):
        state.swap_vacancy(state.vacancy_index)


def test_state_copy_independence(small_config):
    state = KMCState.from_random_composition(small_config)
    other = state.copy()
    nn = state.get_neighbor_atom_indices()
    state.swap_vacancy(int(nn[0]))
    # `other` must not have moved
    assert other.vacancy_index != state.vacancy_index
    assert int((other.species == VACANCY_SPECIES).sum()) == 1


# ---------------------------------------------------------------------------
# KMCState — ASE conversion (visualization)
# ---------------------------------------------------------------------------

def test_to_atoms_without_vacancy(small_config):
    state = KMCState.from_random_composition(small_config)
    atoms = state.to_atoms(include_vacancy=False)
    assert len(atoms) == state.n_sites - 1
    syms = atoms.get_chemical_symbols()
    assert "X" not in syms
    assert set(syms).issubset(set(small_config.elements))


def test_to_atoms_with_vacancy_marker(small_config):
    state = KMCState.from_random_composition(small_config)
    atoms = state.to_atoms(include_vacancy=True, vacancy_symbol="X")
    assert len(atoms) == state.n_sites
    syms = atoms.get_chemical_symbols()
    assert syms.count("X") == 1
    # The X is at the vacancy site position
    np.testing.assert_allclose(
        atoms.positions[syms.index("X")],
        state.positions[state.vacancy_index],
    )


# ---------------------------------------------------------------------------
# MockBarrierPredictor
# ---------------------------------------------------------------------------

def test_mock_constant_predictor(small_config):
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.234)
    assert isinstance(predictor, BarrierPredictor)
    nn = state.get_neighbor_atom_indices()
    barriers = predictor.get_forward_barriers_batch(state, nn)
    assert barriers.shape == (8,)
    assert np.allclose(barriers, 1.234)


def test_mock_element_dependent_predictor(small_config):
    state = KMCState.from_random_composition(small_config)
    table = {"Mo": 0.8, "Nb": 0.9, "Ta": 1.0, "W": 1.1}
    predictor = MockBarrierPredictor(element_barriers_eV=table)
    nn = state.get_neighbor_atom_indices()
    barriers = predictor.get_forward_barriers_batch(state, nn)

    for k, atom_idx in enumerate(nn):
        sp = int(state.species[int(atom_idx)])
        symbol = state.element_symbols[sp]
        assert abs(barriers[k] - table[symbol]) < 1e-12


# ---------------------------------------------------------------------------
# bkl_step
# ---------------------------------------------------------------------------

def test_bkl_step_state_change(small_config):
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=123)

    nn_before = set(state.get_neighbor_atom_indices().tolist())
    old_vac = state.vacancy_index

    info = bkl_step(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        rng=rng,
    )

    # Vacancy moved to one of the original 8 NN atoms
    assert state.vacancy_index != old_vac
    assert state.vacancy_index in nn_before
    assert info["chosen_atom_index"] == state.vacancy_index
    assert info["delta_t_sim_s"] > 0.0
    assert int((state.species == VACANCY_SPECIES).sum()) == 1


def test_bkl_step_total_rate_constant_barrier(small_config):
    """For 8 jumps with identical barrier, R = 8 * nu * exp(-E/kT)."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=0)
    info = bkl_step(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        rng=rng,
    )

    kT = K_B_EV_PER_K * 1500.0
    R_expected = 8 * 1e13 * np.exp(-1.0 / kT)
    rel_err = abs(info["total_rate_Hz"] - R_expected) / R_expected
    assert rel_err < 1e-9


def test_bkl_step_dt_matches_analytical_mean(small_config):
    """E[dt] = 1/R for the residence-time scheme; verify via Monte Carlo."""
    predictor = MockBarrierPredictor(constant_eV=1.0)
    T = 1500.0
    nu = 1e13
    kT = K_B_EV_PER_K * T
    R_expected = 8 * nu * np.exp(-1.0 / kT)
    expected_mean_dt = 1.0 / R_expected   # E[-ln(u)/R] = 1/R for u ~ U(0,1)

    n_runs = 1000
    dts = np.empty(n_runs)
    base_state = KMCState.from_random_composition(small_config)
    for s in range(n_runs):
        # With constant barriers the rate is independent of configuration,
        # so we copy the base state and step it once for cleanliness.
        st = base_state.copy()
        rng = np.random.default_rng(seed=s)
        info = bkl_step(
            st, predictor, T_K=T, attempt_frequency_Hz=nu, rng=rng
        )
        dts[s] = info["delta_t_sim_s"]

    mean_dt = float(dts.mean())
    sem = expected_mean_dt / np.sqrt(n_runs)   # exponential SEM
    assert abs(mean_dt - expected_mean_dt) < 5 * sem, (
        f"mean dt = {mean_dt:.3e}, expected {expected_mean_dt:.3e} "
        f"(5-sigma envelope = {5*sem:.3e})"
    )


def test_bkl_step_invalid_temperature(small_config):
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=0)
    with pytest.raises(ValueError):
        bkl_step(state, predictor, T_K=0.0,
                 attempt_frequency_Hz=1e13, rng=rng)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
