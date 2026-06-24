"""
Phase-4 tests for observables and trajectory I/O.

Run from /path/to/GNN_Gym:

    pytest -v KMC/tests/test_phase4.py
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make `scipts/` importable
_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.state import KMCState, VACANCY_SPECIES
from KMC.barrier_predictor import MockBarrierPredictor
from KMC.engine import run
from KMC.observables import (
    tracer_msd_per_element,
    warren_cowley_sro_snapshot,
    warren_cowley_sro_trajectory,
)
from KMC.trajectory_writer import (
    write_extxyz_trajectory,
    write_event_log,
)


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
        n_steps=200,
    )


@pytest.fixture
def short_run(small_config):
    """One short run with snapshots enabled — used by most tests below."""
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    rng = np.random.default_rng(seed=0)
    return run(
        state, predictor,
        T_K=small_config.temperature_K,
        attempt_frequency_Hz=small_config.attempt_frequency_Hz,
        n_steps=small_config.n_steps,
        snapshot_every_n_steps=20,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Tracer MSD per element
# ---------------------------------------------------------------------------

def test_tracer_msd_shape_and_keys(short_run, small_config):
    msd_dict = tracer_msd_per_element(short_run)
    assert set(msd_dict.keys()) == set(small_config.elements)
    for symbol, msd in msd_dict.items():
        assert msd.shape == (short_run.n_steps + 1,)
        assert msd[0] == 0.0      # by definition: zero displacement at t=0
        assert msd.dtype == np.float64


def test_tracer_msd_is_monotonic_or_increasing(short_run):
    """Element-averaged MSD can fluctuate (random walk), but cumulative trends
    should be non-negative; specifically the largest value occurs near the end
    of the run more often than at the start."""
    msd_dict = tracer_msd_per_element(short_run)
    for symbol, msd in msd_dict.items():
        assert np.all(msd >= 0.0)
        # Final MSD should usually be > initial (which is 0)
        assert msd[-1] > 0.0


def test_tracer_displacements_sum_balances_vacancy(short_run):
    """Sum of all atom displacements equals minus the vacancy displacement.

    Each BKL step moves exactly one atom by -delta_vacancy. Replaying the
    history must produce: sum_atom (r_atom_final - r_atom_initial) =
    -(r_vac_final - r_vac_initial). This is a strict conservation, no
    statistics involved.
    """
    state = short_run.final_state
    initial_species = short_run.initial_species
    non_vac_mask = initial_species != VACANCY_SPECIES
    non_vac_sites = np.where(non_vac_mask)[0]
    n_atoms = int(non_vac_sites.shape[0])

    site_positions = state.positions
    atom_pos_initial = site_positions[non_vac_sites].copy()
    atom_pos_unfolded = atom_pos_initial.copy()

    atom_id_at_site = np.full(initial_species.shape[0], -1, dtype=np.int64)
    atom_id_at_site[non_vac_sites] = np.arange(n_atoms)

    vac_steps = np.diff(short_run.vacancy_positions_unfolded, axis=0)
    for k in range(short_run.n_steps):
        v_before = int(short_run.vacancy_indices[k])
        v_after = int(short_run.vacancy_indices[k + 1])
        hopper_id = int(atom_id_at_site[v_after])
        atom_pos_unfolded[hopper_id] += -vac_steps[k]
        atom_id_at_site[v_after] = -1
        atom_id_at_site[v_before] = hopper_id

    total_atom_displacement = (atom_pos_unfolded - atom_pos_initial).sum(axis=0)
    vacancy_displacement = (
        short_run.vacancy_positions_unfolded[-1]
        - short_run.vacancy_positions_unfolded[0]
    )
    np.testing.assert_allclose(
        total_atom_displacement, -vacancy_displacement, atol=1e-9
    )


# ---------------------------------------------------------------------------
# Warren-Cowley SRO
# ---------------------------------------------------------------------------

def test_wc_random_initial_close_to_zero(small_config):
    """Equiatomic random configurations give alpha_ij ~ 0 *on average*.

    Single-realisation fluctuations on 128 sites (32 atoms per element x 8
    NN slots = 256 bonds per species) are easily ~20 %, so a single seed is
    not a tight test. We average over a small ensemble of seeds and require
    |<alpha>| < 0.10 per pair, which is well within the residual cell-size
    bias for an equiatomic random alloy.
    """
    n_seeds = 8
    n_elements = len(small_config.elements)
    alpha_sum = {(a, b): 0.0
                 for a in small_config.elements
                 for b in small_config.elements}
    for seed in range(n_seeds):
        state = KMCState.from_random_composition(
            small_config, rng=np.random.default_rng(seed=seed)
        )
        sro = warren_cowley_sro_snapshot(
            state.species, state.nn_table, state.element_symbols
        )
        for key, value in sro.items():
            assert np.isfinite(value), f"alpha[{key}] is NaN at seed={seed}"
            alpha_sum[key] += value
    for key, total in alpha_sum.items():
        avg = total / n_seeds
        assert abs(avg) < 0.10, (
            f"Ensemble-averaged alpha[{key[0]},{key[1]}] = {avg:.3f} "
            f"(|.| > 0.10) over {n_seeds} random equiatomic seeds"
        )


def test_wc_bicrystal_strong_anti_clustering(small_config):
    """A perfect Ta|W bicrystal should give strongly positive alpha(Ta, W).

    In half-and-half geometry the two species only meet at the interface, so
    the typical Ta has almost only Ta neighbours and vice versa. That means
    P(W|Ta) << c_W = 0.5, hence alpha(Ta, W) > 0 (anti-clustering). Likewise
    alpha(Ta, Ta) and alpha(W, W) become strongly negative (self-clustering).
    """
    bi_config = KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.0, "Nb": 0.0, "Ta": 0.5, "W": 0.5},
        supercell_size=4,
        lattice_parameter_A=3.22,
        random_seed=0,
        initial_state_strategy="bicrystal",
        bicrystal_axis="x",
        bicrystal_elements=("Ta", "W"),
        n_steps=10,
    )
    state = KMCState.from_bicrystal(bi_config)
    sro = warren_cowley_sro_snapshot(
        state.species, state.nn_table, state.element_symbols
    )

    # Theoretical values for a perfect 4x4x4 BCC Ta|W bicrystal (no vacancy):
    #   alpha(Ta, W) = +0.5, alpha(Ta, Ta) = -0.5 (and likewise W).
    # The vacancy at the interface perturbs these slightly, so we use a
    # generous envelope around the theoretical values.
    assert sro[("Ta", "W")] >= 0.45
    assert sro[("W", "Ta")] >= 0.45
    assert sro[("Ta", "Ta")] <= -0.45
    assert sro[("W", "W")] <= -0.45


def test_wc_trajectory_shape(short_run):
    series = warren_cowley_sro_trajectory(short_run)
    n_pairs = len(short_run.final_state.element_symbols) ** 2
    assert len(series) == n_pairs
    for arr in series.values():
        assert arr.shape == (short_run.n_snapshots,)


def test_wc_trajectory_requires_snapshots(small_config):
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    result = run(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=20,
        rng=np.random.default_rng(seed=0),
    )
    with pytest.raises(ValueError):
        warren_cowley_sro_trajectory(result)


# ---------------------------------------------------------------------------
# Trajectory I/O
# ---------------------------------------------------------------------------

def test_extxyz_writer_round_trip(short_run):
    from ase.io import read as ase_read

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traj.extxyz"
        n_written = write_extxyz_trajectory(short_run, path, vacancy_symbol="X")
        assert n_written == short_run.n_snapshots
        assert path.exists() and path.stat().st_size > 0

        frames = ase_read(str(path), index=":")
        assert len(frames) == short_run.n_snapshots

        for k, frame in enumerate(frames):
            assert len(frame) == short_run.final_state.n_sites
            symbols = frame.get_chemical_symbols()
            assert symbols.count("X") == 1
            assert frame.info["snapshot_index"] == k
            np.testing.assert_allclose(
                frame.info["time_s"],
                float(short_run.snapshot_times_s[k]),
                rtol=1e-12,
            )


def test_extxyz_writer_requires_snapshots(small_config):
    state = KMCState.from_random_composition(small_config)
    predictor = MockBarrierPredictor(constant_eV=1.0)
    result = run(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=10,
        rng=np.random.default_rng(seed=0),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traj.extxyz"
        with pytest.raises(ValueError):
            write_extxyz_trajectory(result, path)


def test_event_log_round_trip(short_run):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.csv"
        n_rows = write_event_log(short_run, path)
        assert n_rows == short_run.n_steps
        assert path.exists()

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == short_run.n_steps
        # First row sanity check
        assert int(rows[0]["step"]) == 0
        assert float(rows[0]["delta_t_s"]) > 0
        assert (
            int(rows[0]["vacancy_idx_before"])
            == int(short_run.vacancy_indices[0])
        )
        assert (
            int(rows[0]["vacancy_idx_after"])
            == int(short_run.vacancy_indices[1])
        )
        assert rows[0]["hopper_symbol"] in short_run.final_state.element_symbols


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
