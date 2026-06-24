"""
Phase-3 tests for GNNBarrierPredictor.

Verifies that the GNN-backed predictor:
- Loads the trained model and produces 8 finite barriers for the 8 NN jumps
- Is reproducible (same input -> same output)
- Is a true drop-in replacement for MockBarrierPredictor (BKL step works)
- Returns barriers in a physically reasonable range (~0.5-3 eV for MoNbTaW)

Run from /path/to/GNN_Gym:

    pytest -v KMC/tests/test_phase3.py

These tests skip automatically if torch / torch_geometric / fairchem are not
available, or if the trained model file is missing.
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
from KMC.barrier_predictor import (
    BarrierPredictor,
    MockBarrierPredictor,
)
from KMC.engine import bkl_step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_config():
    """4x4x4 BCC -> 128 sites; matches the GNN training cell size."""
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


@pytest.fixture(scope="module")
def gnn_predictor(small_config):
    """Module-scoped: load the model only once for all tests in this file."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    if not Path(small_config.gnn_model_path).exists():
        pytest.skip(
            f"GNN model not found at {small_config.gnn_model_path}; "
            "Phase-3 tests require the trained model."
        )

    from KMC.barrier_predictor import GNNBarrierPredictor
    return GNNBarrierPredictor(small_config)


@pytest.fixture
def state(small_config):
    return KMCState.from_random_composition(small_config)


# ---------------------------------------------------------------------------
# Construction & basic prediction
# ---------------------------------------------------------------------------

def test_gnn_predictor_implements_protocol(gnn_predictor):
    """GNNBarrierPredictor satisfies the BarrierPredictor protocol at runtime."""
    assert isinstance(gnn_predictor, BarrierPredictor)


def test_gnn_predictor_returns_finite_8_barriers(state, gnn_predictor):
    nn = state.get_neighbor_atom_indices()
    barriers = gnn_predictor.get_forward_barriers_batch(state, nn)
    assert barriers.shape == (8,)
    assert np.all(np.isfinite(barriers))


def test_gnn_barrier_values_are_physical(state, gnn_predictor):
    """For MoNbTaW the trained model lives in roughly 0.2-3.5 eV.

    Loose envelope; a barrier outside this would suggest a denormalisation
    bug or a wildly out-of-distribution input, not a real prediction.
    """
    nn = state.get_neighbor_atom_indices()
    barriers = gnn_predictor.get_forward_barriers_batch(state, nn)
    assert barriers.min() > -0.5, f"barriers = {barriers}"
    assert barriers.max() < 5.0, f"barriers = {barriers}"


def test_gnn_predictor_is_reproducible(state, gnn_predictor):
    """Same input state -> outputs identical to within cuDNN float-noise.

    The GNN training environment sets `torch.backends.cudnn.benchmark = True`
    and `cudnn.deterministic = False` (see scipts/GNN/utils.py:set_seed) for
    performance, so two repeated forward passes on the GPU can drift by a
    few * 1e-6 eV. That is many orders of magnitude below any physically
    meaningful barrier scale, so we assert near-equality, not bitwise
    equality.
    """
    nn = state.get_neighbor_atom_indices()
    b1 = gnn_predictor.get_forward_barriers_batch(state, nn)
    b2 = gnn_predictor.get_forward_barriers_batch(state, nn)
    np.testing.assert_allclose(b1, b2, atol=1e-5, rtol=1e-5)


def test_gnn_predictor_responds_to_state_change(state, gnn_predictor):
    """Different vacancy environment -> different barrier values.

    Compare barriers at the original vacancy position against barriers after
    one swap_vacancy: the local environment is different, so the eight
    predicted barriers should not all be identical.
    """
    nn_before = state.get_neighbor_atom_indices()
    b_before = gnn_predictor.get_forward_barriers_batch(state, nn_before)

    # Move vacancy to the first NN; the new neighbourhood differs from the old one
    state.swap_vacancy(int(nn_before[0]))
    nn_after = state.get_neighbor_atom_indices()
    b_after = gnn_predictor.get_forward_barriers_batch(state, nn_after)

    # At least some barriers should differ between the two environments
    assert not np.allclose(b_before, b_after, atol=1e-3), (
        f"Barriers unchanged after vacancy move: "
        f"before={b_before}, after={b_after}"
    )


# ---------------------------------------------------------------------------
# Drop-in replacement for MockBarrierPredictor
# ---------------------------------------------------------------------------

def test_bkl_step_runs_with_gnn_predictor(state, gnn_predictor):
    """A single BKL step using the GNN predictor must complete normally."""
    rng = np.random.default_rng(seed=0)
    initial_vac = int(state.vacancy_index)
    nn_before = set(state.get_neighbor_atom_indices().tolist())

    info = bkl_step(
        state, gnn_predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        rng=rng,
    )

    # Same invariants we required for the Mock-Predictor BKL step
    assert info["delta_t_sim_s"] > 0.0
    assert info["forward_barriers_eV"].shape == (8,)
    assert int(state.vacancy_index) != initial_vac
    assert int(state.vacancy_index) in nn_before
    assert int((state.species == VACANCY_SPECIES).sum()) == 1


# ---------------------------------------------------------------------------
# Optional: GNN vs Mock comparison (smoke check that both APIs match)
# ---------------------------------------------------------------------------

def test_gnn_and_mock_have_same_output_shape(state, gnn_predictor):
    """API-level compatibility: both predictors return shape (8,) float arrays."""
    nn = state.get_neighbor_atom_indices()
    gnn_b = gnn_predictor.get_forward_barriers_batch(state, nn)
    mock_b = MockBarrierPredictor(constant_eV=1.0).get_forward_barriers_batch(state, nn)
    assert gnn_b.shape == mock_b.shape == (8,)
    assert gnn_b.dtype == mock_b.dtype == np.float64


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
