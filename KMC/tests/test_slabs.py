"""
Tests for the multi-slab initial-state strategy.

Run from anywhere with pytest:

    cd /home/klemens/doctor/gnn_kmc/scipts
    pytest -v KMC/tests/test_slabs.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make `scipts/` importable so that `from KMC.* import ...` works
_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.state import KMCState, VACANCY_SPECIES
from KMC.runner import _build_initial_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def quad_slab_config():
    """6x6x6 BCC -> 432 sites, four equal slabs along x."""
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=6,
        lattice_parameter_A=3.22,
        random_seed=7,
        temperature_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=10,
        initial_state_strategy="slabs",
        slab_axis="x",
        slab_elements=["Mo", "Nb", "Ta", "W"],
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_slabs_requires_slab_elements():
    with pytest.raises(ValueError, match="slab_elements"):
        KMCConfig(
            initial_state_strategy="slabs",
            slab_elements=None,
        )


def test_slabs_requires_at_least_two_elements():
    with pytest.raises(ValueError, match="at least 2"):
        KMCConfig(
            initial_state_strategy="slabs",
            slab_elements=["Mo"],
        )


def test_slabs_elements_must_be_in_elements_list():
    with pytest.raises(ValueError, match="not in elements list"):
        KMCConfig(
            elements=["Mo", "Nb", "Ta", "W"],
            initial_state_strategy="slabs",
            slab_elements=["Mo", "Cr"],
        )


def test_slabs_config_json_roundtrip(quad_slab_config, tmp_path):
    path = tmp_path / "cfg.json"
    quad_slab_config.to_json(path)
    cfg2 = KMCConfig.from_json(path)
    assert cfg2.initial_state_strategy == "slabs"
    assert cfg2.slab_axis == "x"
    assert cfg2.slab_elements == ["Mo", "Nb", "Ta", "W"]


# ---------------------------------------------------------------------------
# Geometry of from_slabs
# ---------------------------------------------------------------------------

def test_from_slabs_total_count(quad_slab_config):
    state = KMCState.from_slabs(quad_slab_config)
    n_sites = quad_slab_config.n_sites
    assert state.species.shape == (n_sites,)
    # Exactly one vacancy
    assert int((state.species == VACANCY_SPECIES).sum()) == 1


def test_from_slabs_equal_partitioning(quad_slab_config):
    """Each of the four slabs should hold ~ n_sites / 4 atoms; ignore the
    vacancy site (which sits on a boundary and is one of those atoms)."""
    state = KMCState.from_slabs(quad_slab_config)
    n_sites = quad_slab_config.n_sites
    expected = n_sites / 4
    # Count per element index
    counts = {}
    for sp in state.species:
        if sp == VACANCY_SPECIES:
            continue
        counts[int(sp)] = counts.get(int(sp), 0) + 1
    # The vacancy steals exactly one site from one of the four slabs.
    for k, sym in enumerate(["Mo", "Nb", "Ta", "W"]):
        el_idx = state.element_symbols.index(sym)
        assert counts.get(el_idx, 0) in (int(expected), int(expected) - 1), (
            f"Slab {sym} has {counts.get(el_idx, 0)} atoms, "
            f"expected ~{expected}"
        )
    # Total atoms = n_sites - 1 (one vacancy)
    assert sum(counts.values()) == n_sites - 1


def test_from_slabs_assignment_along_axis(quad_slab_config):
    """All Mo atoms should sit in the first quarter of the box, W in the
    last quarter, etc."""
    state = KMCState.from_slabs(quad_slab_config)
    L = state.cell[0, 0]
    quarter = L / 4.0
    el_to_idx = {sym: i for i, sym in enumerate(state.element_symbols)}

    for slab_k, sym in enumerate(["Mo", "Nb", "Ta", "W"]):
        sp = el_to_idx[sym]
        mask = state.species == sp
        x_coords = state.positions[mask, 0]
        # Allow a tiny numerical tolerance for sites on a boundary
        assert np.all(x_coords >= slab_k * quarter - 1e-6)
        assert np.all(x_coords <= (slab_k + 1) * quarter + 1e-6)


def test_from_slabs_vacancy_on_central_boundary(quad_slab_config):
    """For 4 slabs the central boundary is at L/2."""
    state = KMCState.from_slabs(quad_slab_config)
    L = state.cell[0, 0]
    vac_x = state.positions[state.vacancy_index, 0]
    # The closest BCC site to L/2 will lie exactly on the L/2 plane in a
    # cubic supercell.
    assert abs(vac_x - L / 2.0) < 1e-6


def test_from_slabs_two_slabs_matches_bicrystal_geometry():
    """from_slabs with two elements reproduces from_bicrystal layout."""
    cfg_bi = KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        supercell_size=4,
        random_seed=0,
        n_steps=1,
        initial_state_strategy="bicrystal",
        bicrystal_axis="x",
        bicrystal_elements=("Ta", "W"),
    )
    cfg_sl = KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        supercell_size=4,
        random_seed=0,
        n_steps=1,
        initial_state_strategy="slabs",
        slab_axis="x",
        slab_elements=["Ta", "W"],
    )
    s_bi = KMCState.from_bicrystal(cfg_bi)
    s_sl = KMCState.from_slabs(cfg_sl)
    np.testing.assert_array_equal(s_bi.species, s_sl.species)
    assert s_bi.vacancy_index == s_sl.vacancy_index


def test_from_slabs_three_slabs():
    """Three slabs along y; sanity checks counts + central boundary."""
    cfg = KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        supercell_size=6,
        random_seed=0,
        n_steps=1,
        initial_state_strategy="slabs",
        slab_axis="y",
        slab_elements=["Mo", "Nb", "Ta"],
    )
    state = KMCState.from_slabs(cfg)
    n_sites = cfg.n_sites
    # Each slab gets ~ n_sites/3 atoms; vacancy steals one from one slab.
    el_to_idx = {sym: i for i, sym in enumerate(state.element_symbols)}
    for sym in ["Mo", "Nb", "Ta"]:
        cnt = int((state.species == el_to_idx[sym]).sum())
        assert abs(cnt - n_sites / 3) <= 1
    # Vacancy on the boundary at L * (3 // 2) / 3 = L/3
    L_y = state.cell[1, 1]
    vac_y = state.positions[state.vacancy_index, 1]
    assert abs(vac_y - L_y / 3.0) < 1e-6


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------

def test_runner_dispatches_slabs(quad_slab_config):
    rng = np.random.default_rng(0)
    state = _build_initial_state(quad_slab_config, rng)
    assert isinstance(state, KMCState)
    # Vacancy on central boundary plane (sanity)
    L = state.cell[0, 0]
    vac_x = state.positions[state.vacancy_index, 0]
    assert abs(vac_x - L / 2.0) < 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
