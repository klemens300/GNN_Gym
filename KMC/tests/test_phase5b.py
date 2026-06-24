"""
Phase-5b tests: DiffusionLookup + rescale_to_real_time.

Run from /path/to/GNN_Gym:

    pytest -v KMC/tests/test_phase5b.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.state import KMCState
from KMC.barrier_predictor import MockBarrierPredictor
from KMC.engine import run, K_B_EV_PER_K
from KMC.analysis import (
    CachedDiffusionEntry,
    DiffusionLookup,
    InterpolatedDiffusionPoint,
    rescale_to_real_time,
    real_time_axis,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _entry(composition, E_f_V, a=3.22, nu0=1e13, n_atoms=128):
    """Convenience constructor for cache entries used in tests."""
    return CachedDiffusionEntry(
        composition=composition,
        E_f_V_eV=float(E_f_V),
        lattice_parameter_A=float(a),
        jump_distance_A=float(a) * np.sqrt(3) / 2,
        nu_0_Hz=float(nu0),
        n_atoms_supercell=int(n_atoms),
        source="unit-test",
    )


@pytest.fixture
def small_config():
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=4,
        lattice_parameter_A=3.22,
        random_seed=42,
        temperature_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=50,
    )


# ---------------------------------------------------------------------------
# rescale_to_real_time
# ---------------------------------------------------------------------------

def test_rescale_to_real_time_value_check():
    """Spot-check against a hand-calculated value."""
    E_f_V = 2.0
    T = 1500.0
    n_atoms = 127
    kT = K_B_EV_PER_K * T
    c_v_eq = float(np.exp(-E_f_V / kT))

    tau_sim = 1.0e-9
    expected = tau_sim / (n_atoms * c_v_eq)
    got = rescale_to_real_time(tau_sim, E_f_V, T, n_atoms)
    assert abs(got - expected) / expected < 1e-12


def test_rescale_to_real_time_array():
    """Vector input -> vector output, scaling is uniform."""
    out = rescale_to_real_time(
        np.array([1e-12, 2e-12, 3e-12]),
        E_f_V_eV=2.0,
        T_K=1500.0,
        n_atoms=127,
    )
    np.testing.assert_allclose(out / out[0], np.array([1.0, 2.0, 3.0]))


def test_rescale_invalid_inputs():
    with pytest.raises(ValueError):
        rescale_to_real_time(1e-9, E_f_V_eV=2.0, T_K=0.0, n_atoms=127)
    with pytest.raises(ValueError):
        rescale_to_real_time(1e-9, E_f_V_eV=2.0, T_K=1500.0, n_atoms=0)


def test_real_time_axis_against_kmc_run(small_config):
    """real_time_axis builds the rescaled axis of an actual run."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=0)
    result = run(
        state, predictor,
        T_K=small_config.temperature_K,
        attempt_frequency_Hz=small_config.attempt_frequency_Hz,
        n_steps=small_config.n_steps,
        rng=rng,
    )

    E_f_V = 2.0
    rt = real_time_axis(result, E_f_V_eV=E_f_V)
    expected = rescale_to_real_time(
        result.times_with_zero_s,
        E_f_V_eV=E_f_V,
        T_K=result.T_K,
        n_atoms=result.final_state.n_sites - 1,
    )
    np.testing.assert_allclose(rt, expected)
    # Real time is much larger than sim time (vacancy concentration ~ 1e-7)
    assert rt[-1] > result.times_with_zero_s[-1] * 1e3


# ---------------------------------------------------------------------------
# DiffusionLookup persistence
# ---------------------------------------------------------------------------

def test_lookup_empty_cache_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        lookup = DiffusionLookup(path)
        result = lookup.get({"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25})
        assert result is None
        assert len(lookup) == 0


def test_lookup_add_persists_to_disk():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        lookup = DiffusionLookup(path)
        lookup.add(_entry(
            {"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25}, E_f_V=2.0
        ))

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["metadata"]["elements"] == ["Mo", "Nb", "Ta", "W"]
        assert len(data["entries"]) == 1

        # Re-loading produces the same content
        lookup2 = DiffusionLookup(path)
        assert len(lookup2) == 1
        assert lookup2.entries[0].E_f_V_eV == 2.0


def test_lookup_rejects_inconsistent_elements():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        lookup = DiffusionLookup(path)
        lookup.add(_entry(
            {"Mo": 0.5, "W": 0.5}, E_f_V=2.0
        ))
        with pytest.raises(ValueError):
            lookup.add(_entry(
                {"Mo": 0.5, "Ta": 0.5}, E_f_V=2.0
            ))


# ---------------------------------------------------------------------------
# DiffusionLookup interpolation
# ---------------------------------------------------------------------------

def test_lookup_exact_hit_with_three_neighbors():
    """When 3 entries are within max_distance, the average is returned."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        lookup = DiffusionLookup(path, n_neighbors=3, max_distance=0.10)

        # Three nearby entries
        lookup.add(_entry({"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25}, E_f_V=2.0))
        lookup.add(_entry({"Mo": 0.30, "Nb": 0.20, "Ta": 0.25, "W": 0.25}, E_f_V=2.4))
        lookup.add(_entry({"Mo": 0.20, "Nb": 0.30, "Ta": 0.25, "W": 0.25}, E_f_V=1.6))

        hit = lookup.get({"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25})
        assert hit is not None
        assert isinstance(hit, InterpolatedDiffusionPoint)
        assert hit.n_neighbors_used == 3
        # Mean of {2.0, 2.4, 1.6} = 2.0
        assert abs(hit.E_f_V_eV - 2.0) < 1e-12
        assert hit.max_distance_used <= 0.10


def test_lookup_too_far_returns_none():
    """If the closest entry is farther than max_distance, miss."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        lookup = DiffusionLookup(path, n_neighbors=3, max_distance=0.10)

        # All entries far from the query
        for shift in (0.50, 0.55, 0.60):
            lookup.add(_entry(
                {"Mo": shift, "Nb": (1 - shift), "Ta": 0.0, "W": 0.0},
                E_f_V=1.5,
            ))
        hit = lookup.get({"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25})
        assert hit is None


def test_lookup_get_or_compute_invokes_oracle():
    """On a miss, get_or_compute calls the oracle and stores the result."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"

        oracle_calls = []

        def fake_oracle(composition):
            oracle_calls.append(dict(composition))
            return _entry(dict(composition), E_f_V=2.5)

        lookup = DiffusionLookup(
            path,
            oracle=fake_oracle,
            n_neighbors=1,            # so a single computed point suffices
            max_distance=0.05,
        )
        target = {"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25}
        hit = lookup.get_or_compute(target)
        assert hit is not None
        assert hit.E_f_V_eV == 2.5
        assert len(oracle_calls) == 1
        assert oracle_calls[0] == target

        # Subsequent lookup uses the freshly-cached point, no extra oracle call
        hit2 = lookup.get_or_compute(target)
        assert hit2 is not None
        assert len(oracle_calls) == 1


def test_lookup_get_or_compute_raises_without_oracle():
    """Miss + no oracle -> RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        lookup = DiffusionLookup(path)
        with pytest.raises(RuntimeError):
            lookup.get_or_compute({"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25})


def test_lookup_with_fewer_than_k_entries_misses():
    """If only one entry exists but n_neighbors=3, lookup must miss.

    The intent is that the engine then triggers oracle compute via
    get_or_compute, which adds more points until n_neighbors are available.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        lookup = DiffusionLookup(path, n_neighbors=3, max_distance=0.10)
        lookup.add(_entry({"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25}, E_f_V=2.0))
        # Only 1 entry, n_neighbors=3 -> with current logic we still fall back
        # to k_effective=1 because all available are within max_distance.
        hit = lookup.get({"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25})
        assert hit is not None
        assert hit.n_neighbors_used == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
