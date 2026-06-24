"""
Phase-5a tests: ensemble runner.

Run from /path/to/GNN_Gym:

    pytest -v KMC/tests/test_phase5a.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.barrier_predictor import MockBarrierPredictor
from KMC.runner import run_ensemble, EnsembleResult
from KMC.engine import K_B_EV_PER_K
from KMC.result import KMCResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
        n_steps=500,
    )


@pytest.fixture
def ensemble(small_config):
    predictor = MockBarrierPredictor(constant_eV=1.0)
    return run_ensemble(
        small_config, predictor,
        n_realizations=5,
        snapshot_every_n_steps=50,
    )


# ---------------------------------------------------------------------------
# run_ensemble basics
# ---------------------------------------------------------------------------

def test_ensemble_basic(ensemble, small_config):
    assert isinstance(ensemble, EnsembleResult)
    assert ensemble.n_realizations == 5
    assert len(ensemble.results) == 5
    assert len(ensemble.seeds) == 5
    assert len(set(ensemble.seeds)) == 5  # distinct seeds

    for r in ensemble.results:
        assert isinstance(r, KMCResult)
        assert r.n_steps == small_config.n_steps


def test_ensemble_realisations_are_independent(ensemble):
    """Different seeds produce different trajectories."""
    first_jump = [int(r.vacancy_indices[1]) for r in ensemble.results]
    # With 5 seeds and 8 possible NN, "all identical" would be a tail event
    # of probability ~ 8^-4 ~ 2e-4 per fixed initial vacancy. Plus the
    # initial vacancy index is the same here (center of the cell), so the
    # first-jump distribution depends only on the engine rng. Still extremely
    # unlikely that all 5 land on the same NN.
    assert len(set(first_jump)) > 1


def test_ensemble_is_reproducible(small_config):
    """Same config + same predictor => identical ensemble output."""
    predictor = MockBarrierPredictor(constant_eV=1.0)
    e1 = run_ensemble(small_config, predictor, n_realizations=3)
    e2 = run_ensemble(small_config, predictor, n_realizations=3)
    assert e1.seeds == e2.seeds
    for r1, r2 in zip(e1.results, e2.results):
        np.testing.assert_array_equal(r1.vacancy_indices, r2.vacancy_indices)
        np.testing.assert_array_equal(r1.delta_t_s, r2.delta_t_s)


def test_ensemble_explicit_seeds(small_config):
    predictor = MockBarrierPredictor(constant_eV=1.0)
    seeds = [11, 22, 33]
    e = run_ensemble(small_config, predictor, n_realizations=3, seeds=seeds)
    assert e.seeds == seeds


def test_ensemble_seed_list_length_mismatch(small_config):
    predictor = MockBarrierPredictor(constant_eV=1.0)
    with pytest.raises(ValueError):
        run_ensemble(
            small_config, predictor, n_realizations=3, seeds=[1, 2]
        )


def test_ensemble_progress_callback(small_config):
    predictor = MockBarrierPredictor(constant_eV=1.0)
    calls = []

    def cb(idx, result):
        calls.append((idx, result.n_steps))

    run_ensemble(
        small_config, predictor,
        n_realizations=3,
        progress_callback=cb,
    )
    assert [c[0] for c in calls] == [0, 1, 2]
    assert all(c[1] == small_config.n_steps for c in calls)


# ---------------------------------------------------------------------------
# Aggregate methods
# ---------------------------------------------------------------------------

def test_vacancy_msd_ensemble_shapes(ensemble):
    out = ensemble.vacancy_msd_ensemble(max_lag_fraction=0.1)
    assert set(out.keys()) == {"lag_t_mean", "mean", "std", "per_seed"}
    n_lags = out["mean"].shape[0]
    assert out["lag_t_mean"].shape == (n_lags,)
    assert out["std"].shape == (n_lags,)
    assert out["per_seed"].shape == (ensemble.n_realizations, n_lags)
    # MSD is non-negative and increases with lag (on average)
    assert np.all(out["mean"] >= 0)
    assert out["mean"][-1] > out["mean"][1]


def test_diffusion_coefficient_ensemble_keys(ensemble):
    out = ensemble.vacancy_diffusion_coefficient_ensemble()
    assert set(out.keys()) == {"per_seed", "mean", "std", "median", "mad"}
    assert out["per_seed"].shape == (ensemble.n_realizations,)
    # Median and MAD should be finite
    assert np.isfinite(out["median"])
    assert np.isfinite(out["mad"])
    assert out["mad"] >= 0


def test_diffusion_coefficient_ensemble_matches_analytical(small_config):
    """Ensemble-averaged D from MSD slope matches D = nu * exp(-E/kT) * a^2.

    Single-trajectory D estimates from 1e4 KMC steps have ~15-20% intrinsic
    spread; with N realisations the standard error of the mean shrinks like
    1/sqrt(N). 10 realisations therefore give an expected ~5-7% scatter
    around the analytical value, which we wrap in a 10% tolerance to keep
    this test deterministically robust.
    """
    barrier_eV = 0.5
    predictor = MockBarrierPredictor(constant_eV=barrier_eV)

    cfg = KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=6,
        lattice_parameter_A=3.22,
        random_seed=0,
        temperature_K=2000.0,
        attempt_frequency_Hz=1e13,
        n_steps=10000,
    )
    e = run_ensemble(cfg, predictor, n_realizations=10)

    # Take the slope of the ensemble-averaged time-averaged MSD
    msd_pack = e.vacancy_msd_ensemble(max_lag_fraction=0.1)
    skip = max(1, int(0.05 * len(msd_pack["lag_t_mean"])))
    slope, _ = np.polyfit(
        msd_pack["lag_t_mean"][skip:],
        msd_pack["mean"][skip:],
        deg=1,
    )
    D_measured = slope / 6.0

    kT = K_B_EV_PER_K * cfg.temperature_K
    D_analytical = (
        cfg.attempt_frequency_Hz * np.exp(-barrier_eV / kT)
        * cfg.lattice_parameter_A ** 2
    )
    rel_err = abs(D_measured - D_analytical) / D_analytical
    assert rel_err < 0.10, (
        f"D_measured = {D_measured:.4e} A^2/s, "
        f"D_analytical = {D_analytical:.4e} A^2/s, rel_err = {rel_err:.3f}"
    )


def test_tracer_msd_ensemble_shapes(ensemble, small_config):
    out = ensemble.tracer_msd_per_element_ensemble()
    assert set(out.keys()) == set(small_config.elements)
    n = small_config.n_steps + 1
    for sym, sub in out.items():
        assert set(sub.keys()) == {"mean", "std", "per_seed"}
        assert sub["mean"].shape == (n,)
        assert sub["std"].shape == (n,)
        assert sub["per_seed"].shape == (ensemble.n_realizations, n)


def test_warren_cowley_sro_ensemble_shapes(ensemble, small_config):
    out = ensemble.warren_cowley_sro_ensemble()
    n_pairs = len(small_config.elements) ** 2
    assert len(out) == n_pairs
    n_snap = ensemble.results[0].n_snapshots
    for key, sub in out.items():
        assert set(sub.keys()) == {"mean", "std", "per_seed"}
        assert sub["mean"].shape == (n_snap,)
        assert sub["std"].shape == (n_snap,)
        assert sub["per_seed"].shape == (ensemble.n_realizations, n_snap)


def test_warren_cowley_sro_ensemble_requires_snapshots(small_config):
    """Without snapshots the WC aggregate must raise."""
    predictor = MockBarrierPredictor(constant_eV=1.0)
    e = run_ensemble(
        small_config, predictor,
        n_realizations=2,
        snapshot_every_n_steps=0,
    )
    with pytest.raises(ValueError):
        e.warren_cowley_sro_ensemble()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
