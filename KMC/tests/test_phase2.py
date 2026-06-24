"""
Phase-2 sanity tests for the KMC engine: full trajectory + analytical
verification of the BKL machinery.

Run:
    cd /home/klemens/doctor/gnn_kmc/scipts
    pytest -v KMC/tests/test_phase2.py

Or directly:
    cd /home/klemens/doctor/gnn_kmc/scipts/KMC/tests
    python test_phase2.py
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
from KMC.barrier_predictor import MockBarrierPredictor
from KMC.engine import run, bkl_step, K_B_EV_PER_K
from KMC.result import KMCResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    """4x4x4 BCC -> 128 sites; cheap enough for trajectory tests."""
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=4,
        lattice_parameter_A=3.22,
        random_seed=42,
        temperature_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=1000,
    )


@pytest.fixture
def med_config():
    """6x6x6 BCC -> 432 sites; better statistics for D-fit."""
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=6,
        lattice_parameter_A=3.22,
        random_seed=0,
        temperature_K=2000.0,
        attempt_frequency_Hz=1e13,
        n_steps=20000,
    )


# ---------------------------------------------------------------------------
# run(): basic invariants
# ---------------------------------------------------------------------------

def test_run_basic_n_steps(small_config):
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=42)

    result = run(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=500,
        rng=rng,
    )

    assert isinstance(result, KMCResult)
    assert result.n_steps == 500
    assert result.total_time_s > 0
    # Per-step arrays
    assert result.times_s.shape == (500,)
    assert result.delta_t_s.shape == (500,)
    assert result.hopper_species.shape == (500,)
    assert result.chosen_jump_local.shape == (500,)
    assert result.barriers_eV.shape == (500, 8)
    # Endpoint arrays (length n_steps + 1, including initial)
    assert result.vacancy_indices.shape == (501,)
    assert result.vacancy_positions_unfolded.shape == (501, 3)
    assert result.times_with_zero_s.shape == (501,)
    # Time monotonically increasing
    assert np.all(np.diff(result.times_s) > 0)
    assert np.all(result.delta_t_s > 0)
    # No snapshots when snapshot_every_n_steps not set
    assert result.snapshot_times_s is None
    assert result.snapshot_species is None
    # Vacancy moved (single vacancy preserved)
    assert int((result.final_state.species == VACANCY_SPECIES).sum()) == 1


def test_run_t_max_stops(small_config):
    """Stop on the time-based criterion before reaching n_steps."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=0.5)
    rng = np.random.default_rng(seed=0)

    # Set a very short time limit so the run stops on time, not steps
    result = run(
        state, predictor,
        T_K=2000.0,
        attempt_frequency_Hz=1e13,
        n_steps=10**6,
        t_max_sim_s=1e-12,
        rng=rng,
    )
    assert result.total_time_s >= 1e-12
    # Should have stopped well before n_steps
    assert result.n_steps < 10**6


def test_run_requires_stop_criterion(small_config):
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=0)
    with pytest.raises(ValueError):
        run(
            state, predictor,
            T_K=1500.0,
            attempt_frequency_Hz=1e13,
            rng=rng,  # neither n_steps nor t_max_sim_s
        )


def test_run_each_jump_into_neighbour(small_config):
    """Every recorded vacancy_indices[k+1] is in the NN of vacancy_indices[k]."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=1)
    result = run(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=300,
        rng=rng,
    )

    nn_table = result.final_state.nn_table
    for k in range(result.n_steps):
        v_before = int(result.vacancy_indices[k])
        v_after = int(result.vacancy_indices[k + 1])
        assert v_after in nn_table[v_before].tolist()


def test_run_unfolded_displacement_matches_jump(small_config):
    """Each step's unfolded delta has length jump_distance_A."""
    state = KMCState.from_random_composition(small_config)
    jd = state.jump_distance_A
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=2)
    result = run(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=200,
        rng=rng,
    )

    diffs = np.diff(result.vacancy_positions_unfolded, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    np.testing.assert_allclose(step_distances, jd, atol=1e-9)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def test_snapshots_recorded(small_config):
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=42)

    n_steps = 500
    every = 50
    result = run(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=n_steps,
        snapshot_every_n_steps=every,
        rng=rng,
    )

    expected_n_snap = 1 + n_steps // every  # +1 for the t=0 snapshot
    assert result.n_snapshots == expected_n_snap
    assert result.snapshot_species.shape == (expected_n_snap, state.n_sites)
    assert result.snapshot_vacancy_indices.shape == (expected_n_snap,)
    # First snapshot is the initial state
    np.testing.assert_array_equal(
        result.snapshot_species[0], result.initial_species
    )
    # Each snapshot has exactly one vacancy
    for k in range(expected_n_snap):
        assert int((result.snapshot_species[k] == VACANCY_SPECIES).sum()) == 1


# ---------------------------------------------------------------------------
# MSD linearity at constant barrier
# ---------------------------------------------------------------------------

def test_msd_is_linear_at_constant_barrier(med_config):
    """For a constant barrier, time-averaged MSD(tau) is linear in tau.

    Uses the sliding-window estimator: from one trajectory of length N we
    get N-m samples for lag m, which beats the O(MSD^2) variance of the
    single-trajectory MSD(t).
    """
    state = KMCState.from_random_composition(med_config)
    predictor = MockBarrierPredictor(constant_eV=0.5)
    rng = np.random.default_rng(seed=0)

    result = run(
        state, predictor,
        T_K=med_config.temperature_K,
        attempt_frequency_Hz=med_config.attempt_frequency_Hz,
        n_steps=med_config.n_steps,
        rng=rng,
    )

    msd, lag_t = result.time_averaged_vacancy_msd(max_lag_fraction=0.05)
    # Skip the first 5% of lags (sub-diffusive lattice transient)
    skip = max(1, int(0.05 * len(lag_t)))
    slope, intercept = np.polyfit(lag_t[skip:], msd[skip:], deg=1)
    fit = slope * lag_t[skip:] + intercept

    ss_res = np.sum((msd[skip:] - fit) ** 2)
    ss_tot = np.sum((msd[skip:] - msd[skip:].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.99, f"R^2 = {r2:.4f}, slope = {slope:.3e}"


# ---------------------------------------------------------------------------
# Diffusion coefficient vs analytical expectation
# ---------------------------------------------------------------------------

def test_diffusion_coefficient_matches_analytical(med_config):
    """At constant barrier:  D_vac = a^2 * nu * exp(-E/kT)  (BCC NN-only).

    Derivation:
        R = 8 * nu * exp(-E/kT)              (8 NN, all same barrier)
        <Δr^2>_per_step = (a*sqrt(3)/2)^2  = 3 a^2 / 4
        MSD / t = R * 3 a^2 / 4
        D_3D    = MSD / (6 t) = R a^2 / 8 = nu * exp(-E/kT) * a^2

    Uses a small ensemble of independent trajectories. A single-trajectory
    D-estimate has ~10–20 % intrinsic statistical variance for runs of order
    1e4 steps — that is a fundamental ergodicity-limited noise floor, not an
    engine bug. Averaging the time-averaged MSD across a 5-trajectory
    ensemble brings the estimator into the few-percent regime that Phase 5
    will use for all production runs.
    """
    barrier_eV = 0.5
    nu = med_config.attempt_frequency_Hz
    T = med_config.temperature_K
    a_A = med_config.lattice_parameter_A
    predictor = MockBarrierPredictor(constant_eV=barrier_eV)

    n_realizations = 5
    n_steps_each = 10000

    msd_sum = None
    lag_t_ref = None
    for seed in range(n_realizations):
        state = KMCState.from_random_composition(
            med_config,
            rng=np.random.default_rng(seed=seed),
        )
        run_rng = np.random.default_rng(seed=1000 + seed)
        result = run(
            state, predictor,
            T_K=T,
            attempt_frequency_Hz=nu,
            n_steps=n_steps_each,
            rng=run_rng,
        )
        msd, lag_t = result.time_averaged_vacancy_msd(max_lag_fraction=0.1)
        if msd_sum is None:
            msd_sum = msd.copy()
            lag_t_ref = lag_t.copy()
        else:
            msd_sum += msd
    msd_avg = msd_sum / n_realizations

    skip = max(1, int(0.05 * len(lag_t_ref)))
    slope, _ = np.polyfit(lag_t_ref[skip:], msd_avg[skip:], deg=1)
    D_measured = slope / 6.0

    kT = K_B_EV_PER_K * T
    D_analytical = nu * np.exp(-barrier_eV / kT) * (a_A ** 2)

    rel_err = abs(D_measured - D_analytical) / D_analytical
    assert rel_err < 0.05, (
        f"D_measured = {D_measured:.4e} A^2/s, "
        f"D_analytical = {D_analytical:.4e} A^2/s, rel_err = {rel_err:.3f}"
    )


# ---------------------------------------------------------------------------
# Detailed-balance / jump-direction symmetry
# ---------------------------------------------------------------------------

def test_jump_direction_distribution_uniform(small_config):
    """At constant barrier, all 8 NN directions are chosen with equal probability."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=42)

    n_steps = 8000
    result = run(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=n_steps,
        rng=rng,
    )

    counts = np.bincount(result.chosen_jump_local, minlength=8)
    expected = n_steps / 8.0
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    # Chi^2, 7 dof, 99.9% confidence cutoff = 24.32
    assert chi2 < 24.32, (
        f"chi2 = {chi2:.2f}, counts = {counts.tolist()}, expected ~ {expected:.1f}"
    )


# ---------------------------------------------------------------------------
# Element-dependent predictor: hopper_species is recorded correctly
# ---------------------------------------------------------------------------

def test_hopper_species_consistent_with_initial(small_config):
    """Reproduce what hopped from the recorded data and compare with engine output."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=7)

    initial_species = state.species.copy()
    result = run(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=200,
        rng=rng,
    )

    # The first hop's hopper_species must equal the species that was at
    # vacancy_indices[1] in the initial configuration.
    assert int(initial_species[result.vacancy_indices[1]]) == int(result.hopper_species[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
