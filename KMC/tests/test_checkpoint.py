"""
Tests for the rolling checkpoint mechanism.

Covers:
- The engine writes a checkpoint file at the configured cadence and at
  least one of those checkpoints survives a crash-equivalent abort
  (we just inspect the file after a normal-but-truncated run).
- The checkpoint can be reloaded as a full KMCResult.
- The reloaded KMCResult has the right shapes and consistent values
  (initial_species matches, hop history length matches the saved step
  count, snapshots survive).
- Atomic write: the .tmp file does not linger after a successful save.

Run from /home/klemens/doctor/gnn_kmc/scipts:

    pytest -v KMC/tests/test_checkpoint.py
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
from KMC.barrier_predictor import MockBarrierPredictor
from KMC.engine import run
from KMC.state import KMCState, VACANCY_SPECIES
from KMC.checkpoint import write_run_checkpoint, load_run_checkpoint
from KMC.runner import run_ensemble


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config(tmp_path):
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=4,
        lattice_parameter_A=3.22,
        random_seed=17,
        temperature_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=500,
        snapshot_every_n_steps=50,
        initial_state_strategy="random",
        output_dir=str(tmp_path),
        checkpoint_every_n_steps=100,
    )


# ---------------------------------------------------------------------------
# engine.run -> checkpoint file
# ---------------------------------------------------------------------------

def test_engine_writes_checkpoint(tmp_path, small_config):
    """engine.run must produce a .npz at the expected cadence."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)

    ckpt_path = tmp_path / "ckpt.npz"
    result = run(
        state,
        predictor,
        T_K=small_config.temperature_K,
        attempt_frequency_Hz=small_config.attempt_frequency_Hz,
        n_steps=small_config.n_steps,
        snapshot_every_n_steps=small_config.snapshot_every_n_steps,
        checkpoint_every_n_steps=small_config.checkpoint_every_n_steps,
        checkpoint_path=ckpt_path,
    )

    # File present and non-trivial, .tmp gone
    assert ckpt_path.exists(), "Checkpoint file was not written"
    assert ckpt_path.stat().st_size > 1024
    assert not ckpt_path.with_suffix(ckpt_path.suffix + ".tmp").exists()

    # Last write is at step 500 (multiple of 100), so saved n_steps == 500
    data = np.load(ckpt_path, allow_pickle=True)
    assert int(data["n_steps"]) == result.n_steps == 500


# ---------------------------------------------------------------------------
# Roundtrip: write -> load -> verify shape / scalar consistency
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_yields_full_result(tmp_path, small_config):
    """The reloaded KMCResult must contain all the data that was saved."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)

    ckpt_path = tmp_path / "ckpt.npz"
    original = run(
        state,
        predictor,
        T_K=small_config.temperature_K,
        attempt_frequency_Hz=small_config.attempt_frequency_Hz,
        n_steps=small_config.n_steps,
        snapshot_every_n_steps=small_config.snapshot_every_n_steps,
        checkpoint_every_n_steps=small_config.checkpoint_every_n_steps,
        checkpoint_path=ckpt_path,
    )

    reloaded = load_run_checkpoint(ckpt_path)

    # Scalar metadata
    assert reloaded.n_steps == original.n_steps
    assert reloaded.T_K == pytest.approx(original.T_K)
    assert reloaded.attempt_frequency_Hz == pytest.approx(
        original.attempt_frequency_Hz
    )
    assert reloaded.total_time_s == pytest.approx(original.total_time_s)

    # Per-step buffer shapes
    assert reloaded.times_s.shape == original.times_s.shape
    assert reloaded.delta_t_s.shape == original.delta_t_s.shape
    assert reloaded.barriers_eV.shape == original.barriers_eV.shape
    assert (
        reloaded.vacancy_positions_unfolded.shape
        == original.vacancy_positions_unfolded.shape
    )

    # Initial reference must match exactly
    np.testing.assert_array_equal(
        reloaded.initial_species, original.initial_species
    )
    assert reloaded.initial_vacancy_index == original.initial_vacancy_index

    # Per-step values must match exactly (these were just written/loaded)
    np.testing.assert_array_equal(reloaded.vacancy_indices, original.vacancy_indices)
    np.testing.assert_array_equal(reloaded.hopper_species, original.hopper_species)
    np.testing.assert_array_equal(
        reloaded.chosen_jump_local, original.chosen_jump_local
    )
    np.testing.assert_array_almost_equal(
        reloaded.times_s, original.times_s, decimal=12
    )

    # Snapshots survive
    assert reloaded.snapshot_species is not None
    assert reloaded.snapshot_species.shape == original.snapshot_species.shape
    np.testing.assert_array_equal(
        reloaded.snapshot_species, original.snapshot_species
    )


# ---------------------------------------------------------------------------
# Final state is recoverable
# ---------------------------------------------------------------------------

def test_reloaded_final_state_has_consistent_geometry(tmp_path, small_config):
    """final_state.positions / cell / nn_table must come back identical."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)

    ckpt_path = tmp_path / "ckpt.npz"
    original = run(
        state,
        predictor,
        T_K=small_config.temperature_K,
        attempt_frequency_Hz=small_config.attempt_frequency_Hz,
        n_steps=small_config.n_steps,
        snapshot_every_n_steps=small_config.snapshot_every_n_steps,
        checkpoint_every_n_steps=small_config.checkpoint_every_n_steps,
        checkpoint_path=ckpt_path,
    )
    reloaded = load_run_checkpoint(ckpt_path)

    # Same lattice geometry
    np.testing.assert_array_equal(
        reloaded.final_state.positions, original.final_state.positions
    )
    np.testing.assert_array_equal(
        reloaded.final_state.cell, original.final_state.cell
    )
    np.testing.assert_array_equal(
        reloaded.final_state.nn_table, original.final_state.nn_table
    )

    # Same final species + vacancy index
    np.testing.assert_array_equal(
        reloaded.final_state.species, original.final_state.species
    )
    assert reloaded.final_state.vacancy_index == (
        original.final_state.vacancy_index
    )

    # Element symbols and jump distance preserved
    assert reloaded.final_state.element_symbols == (
        original.final_state.element_symbols
    )
    assert reloaded.final_state.jump_distance_A == pytest.approx(
        original.final_state.jump_distance_A
    )


# ---------------------------------------------------------------------------
# Observables work on a reloaded checkpoint (sanity)
# ---------------------------------------------------------------------------

def test_observables_run_on_reloaded_result(tmp_path, small_config):
    """tracer_msd_per_element and warren_cowley_sro_trajectory must work."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)

    ckpt_path = tmp_path / "ckpt.npz"
    run(
        state,
        predictor,
        T_K=small_config.temperature_K,
        attempt_frequency_Hz=small_config.attempt_frequency_Hz,
        n_steps=small_config.n_steps,
        snapshot_every_n_steps=small_config.snapshot_every_n_steps,
        checkpoint_every_n_steps=small_config.checkpoint_every_n_steps,
        checkpoint_path=ckpt_path,
    )
    reloaded = load_run_checkpoint(ckpt_path)

    from KMC.observables import (
        tracer_msd_per_element,
        warren_cowley_sro_trajectory,
    )

    msd_per_el = tracer_msd_per_element(reloaded)
    assert set(msd_per_el.keys()) == {"Mo", "Nb", "Ta", "W"}
    for sym, arr in msd_per_el.items():
        assert arr.shape == (reloaded.n_steps + 1,), sym
        assert arr[0] == pytest.approx(0.0)

    sro = warren_cowley_sro_trajectory(reloaded)
    # Snapshots exist, so we expect a non-empty SRO trajectory dict
    assert len(sro) > 0


# ---------------------------------------------------------------------------
# Runner integration: per-realisation checkpoint paths
# ---------------------------------------------------------------------------

def test_runner_writes_per_realisation_checkpoints(tmp_path, small_config):
    """run_ensemble must emit one checkpoint per realisation."""
    cfg = KMCConfig(**{**{
        "elements": small_config.elements,
        "composition": small_config.composition,
        "supercell_size": small_config.supercell_size,
        "lattice_parameter_A": small_config.lattice_parameter_A,
        "random_seed": small_config.random_seed,
        "temperature_K": small_config.temperature_K,
        "attempt_frequency_Hz": small_config.attempt_frequency_Hz,
        "n_steps": small_config.n_steps,
        "snapshot_every_n_steps": small_config.snapshot_every_n_steps,
        "initial_state_strategy": small_config.initial_state_strategy,
        "output_dir": str(tmp_path),
        "checkpoint_every_n_steps": small_config.checkpoint_every_n_steps,
    }})
    predictor = MockBarrierPredictor(constant_eV=1.0)
    ensemble = run_ensemble(
        cfg, predictor, n_realizations=2,
        snapshot_every_n_steps=cfg.snapshot_every_n_steps,
    )
    assert ensemble.n_realizations == 2

    # Two distinct checkpoint files, both loadable
    for k in range(2):
        p = (
            tmp_path
            / f"checkpoint_T{cfg.temperature_K:.0f}K_real{k}.npz"
        )
        assert p.exists(), f"Missing checkpoint for realisation {k}: {p}"
        loaded = load_run_checkpoint(p)
        assert loaded.n_steps == cfg.n_steps


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
