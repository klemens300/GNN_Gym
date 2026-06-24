"""
Phase-5c tests: Arrhenius fit, tau_order fit, KMCConfig JSON roundtrip,
run_temperature_sweep, and main.py end-to-end smoke test (Mock predictor).

Run from /path/to/GNN_Gym:

    pytest -v KMC/tests/test_phase5c.py
"""

import csv
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
from KMC.barrier_predictor import MockBarrierPredictor
from KMC.runner import run_temperature_sweep, run_ensemble
from KMC.engine import K_B_EV_PER_K
from KMC.analysis import (
    ArrheniusFit,
    TauOrderFit,
    arrhenius_fit_robust,
    arrhenius_fit_per_element,
    tau_order_from_alpha_curve,
    tau_order_matrix_from_ensemble,
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
        n_steps=300,
        n_realizations_per_T=3,
    )


# ---------------------------------------------------------------------------
# KMCConfig JSON roundtrip
# ---------------------------------------------------------------------------

def test_kmcconfig_json_roundtrip(small_config):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.json"
        small_config.to_json(path)
        loaded = KMCConfig.from_json(path)
        # All scalar fields must roundtrip identically
        assert loaded.temperature_K == small_config.temperature_K
        assert loaded.supercell_size == small_config.supercell_size
        assert loaded.composition == small_config.composition
        assert loaded.elements == small_config.elements
        assert loaded.lookup_n_neighbors == small_config.lookup_n_neighbors
        # Tuple-typed field is restored as a tuple
        assert loaded.bicrystal_elements == small_config.bicrystal_elements
        assert isinstance(loaded.bicrystal_elements, tuple)


def test_kmcconfig_json_with_sweep_field():
    cfg = KMCConfig(
        temperatures_K_sweep=[1200.0, 1500.0, 1800.0],
        n_realizations_per_T=10,
        n_steps=500,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sweep.json"
        cfg.to_json(path)
        loaded = KMCConfig.from_json(path)
        assert loaded.temperatures_K_sweep == [1200.0, 1500.0, 1800.0]
        assert loaded.n_realizations_per_T == 10


# ---------------------------------------------------------------------------
# run_temperature_sweep
# ---------------------------------------------------------------------------

def test_temperature_sweep_basic(small_config):
    predictor = MockBarrierPredictor(constant_eV=1.0)
    temps = [1200.0, 1500.0, 1800.0]
    sweep = run_temperature_sweep(
        small_config,
        predictor,
        temperatures_K=temps,
        n_realizations=2,
    )
    assert set(sweep.keys()) == set(temps)
    for T in temps:
        ens = sweep[T]
        assert ens.n_realizations == 2
        # Every result really used the per-T temperature
        for r in ens.results:
            assert abs(r.T_K - T) < 1e-9


def test_temperature_sweep_empty_raises(small_config):
    predictor = MockBarrierPredictor(constant_eV=1.0)
    with pytest.raises(ValueError):
        run_temperature_sweep(
            small_config,
            predictor,
            temperatures_K=[],
            n_realizations=2,
        )


def test_temperature_sweep_progress_callback(small_config):
    predictor = MockBarrierPredictor(constant_eV=1.0)
    calls = []

    def cb(k, T, ens):
        calls.append((k, T, ens.n_realizations))

    run_temperature_sweep(
        small_config,
        predictor,
        temperatures_K=[1200.0, 1500.0],
        n_realizations=2,
        progress_callback=cb,
    )
    assert [c[0] for c in calls] == [0, 1]
    assert [c[1] for c in calls] == [1200.0, 1500.0]


# ---------------------------------------------------------------------------
# Arrhenius fit (synthetic data with known D_0, Q)
# ---------------------------------------------------------------------------

def test_arrhenius_fit_recovers_known_parameters():
    """Build a synthetic D(T) from a known Arrhenius law and recover (D_0, Q)."""
    rng = np.random.default_rng(seed=0)
    D_0_true = 1e-6        # arbitrary units
    Q_true = 2.5           # eV
    temperatures = np.array([1200.0, 1400.0, 1600.0, 1800.0, 2000.0])

    D_per_T = []
    for T in temperatures:
        D_true = D_0_true * np.exp(-Q_true / (K_B_EV_PER_K * T))
        # Add ~10 % multiplicative noise across 8 realisations
        samples = D_true * np.exp(rng.normal(0.0, 0.10, size=8))
        D_per_T.append(samples)

    fit = arrhenius_fit_robust(temperatures, D_per_T)
    assert isinstance(fit, ArrheniusFit)
    assert fit.n_temperatures_used == len(temperatures)
    # 5 temperatures with ~10 % noise should pin Q to within ~0.05 eV
    assert abs(fit.Q_eV - Q_true) < 0.10
    # D_0 may have larger relative scatter; check within an order of magnitude
    assert 0.3 * D_0_true < fit.D_0 < 3.0 * D_0_true


def test_arrhenius_fit_robust_against_outlier():
    """A single 100x outlier among 5 realisations must not derail the fit.

    The robust median+MAD aggregation should suppress the outlier's effect
    on the per-temperature D estimate.
    """
    rng = np.random.default_rng(seed=1)
    D_0_true = 1e-6
    Q_true = 2.0
    temperatures = np.array([1300.0, 1500.0, 1700.0, 1900.0])

    D_per_T = []
    for k, T in enumerate(temperatures):
        D_true = D_0_true * np.exp(-Q_true / (K_B_EV_PER_K * T))
        samples = D_true * np.exp(rng.normal(0.0, 0.05, size=5))
        # Inject an outlier 100x too large at one of the temperatures
        if k == 1:
            samples[0] = 100.0 * D_true
        D_per_T.append(samples)

    fit = arrhenius_fit_robust(temperatures, D_per_T)
    assert abs(fit.Q_eV - Q_true) < 0.15


def test_arrhenius_fit_handles_negative_D_samples():
    """Negative or NaN per-realisation D values are silently dropped."""
    D_0_true = 1e-6
    Q_true = 2.5
    temperatures = np.array([1300.0, 1500.0, 1700.0])
    D_per_T = []
    for T in temperatures:
        D_true = D_0_true * np.exp(-Q_true / (K_B_EV_PER_K * T))
        samples = [D_true, D_true * 1.05, -1.0, np.nan, D_true * 0.95]
        D_per_T.append(np.array(samples))

    fit = arrhenius_fit_robust(temperatures, D_per_T)
    assert abs(fit.Q_eV - Q_true) < 0.20


def test_arrhenius_fit_too_few_temperatures_raises():
    with pytest.raises(ValueError):
        arrhenius_fit_robust([1500.0], [[1e-12]])


def test_arrhenius_predict_matches_fit():
    """The fitted curve passes near the input median points."""
    D_0_true = 1e-6
    Q_true = 2.0
    temperatures = np.array([1300.0, 1600.0, 1900.0])
    D_per_T = [
        [D_0_true * np.exp(-Q_true / (K_B_EV_PER_K * T))] for T in temperatures
    ]
    fit = arrhenius_fit_robust(temperatures, D_per_T)
    preds = fit.predict(temperatures)
    np.testing.assert_allclose(
        preds, fit.D_median, rtol=0.05, atol=0.0
    )


# ---------------------------------------------------------------------------
# tau_order fit (synthetic exponential decay)
# ---------------------------------------------------------------------------

def test_tau_order_recovers_known_tau():
    """Generate a clean exponential decay and recover tau."""
    tau_true = 1.5e-9
    a_0 = 0.6
    a_inf = 0.05

    times = np.linspace(0.0, 5 * tau_true, 30)
    alpha = a_inf + (a_0 - a_inf) * np.exp(-times / tau_true)

    fit = tau_order_from_alpha_curve(times, alpha)
    assert fit is not None
    assert isinstance(fit, TauOrderFit)
    assert abs(fit.tau_order_s - tau_true) / tau_true < 0.05
    assert abs(fit.alpha_0 - a_0) < 0.02
    assert abs(fit.alpha_inf - a_inf) < 0.02


def test_tau_order_too_few_points_returns_none():
    times = np.array([0.0, 1.0])
    alpha = np.array([0.5, 0.1])
    assert tau_order_from_alpha_curve(times, alpha, min_points=5) is None


def test_tau_order_all_nan_returns_none():
    times = np.linspace(0.0, 1.0, 20)
    alpha = np.full_like(times, np.nan)
    assert tau_order_from_alpha_curve(times, alpha) is None


def test_tau_order_matrix_from_ensemble(small_config):
    """End-to-end: run a Mock ensemble, fit tau_order on the WC trajectories."""
    predictor = MockBarrierPredictor(constant_eV=1.0)
    ens = run_ensemble(
        small_config,
        predictor,
        n_realizations=3,
        snapshot_every_n_steps=20,
    )
    taus = tau_order_matrix_from_ensemble(ens, min_points=5)
    n_pairs = len(small_config.elements) ** 2
    assert len(taus) == n_pairs
    # The values themselves are not particularly physical for a 300-step
    # constant-barrier mock run, but at least one fit should succeed.
    successful = [k for k, v in taus.items() if v is not None]
    assert len(successful) >= 1


# ---------------------------------------------------------------------------
# arrhenius_fit_per_element on a small synthetic sweep
# ---------------------------------------------------------------------------

def test_arrhenius_fit_per_element_from_sweep(small_config):
    """End-to-end sweep + fit smoke test with a Mock predictor."""
    predictor = MockBarrierPredictor(constant_eV=1.0)
    sweep = run_temperature_sweep(
        small_config,
        predictor,
        temperatures_K=[1500.0, 1750.0, 2000.0],
        n_realizations=3,
    )
    fits = arrhenius_fit_per_element(sweep)
    assert set(fits.keys()) == set(small_config.elements)
    for sym, fit in fits.items():
        assert isinstance(fit, ArrheniusFit)
        # Q must be finite and roughly the constant barrier (1.0 eV) for
        # the Mock predictor; we leave a generous envelope because BCC
        # geometry/PBC plus the small box add ~10 % bias.
        assert np.isfinite(fit.Q_eV)
        assert 0.8 < fit.Q_eV < 1.2


# ---------------------------------------------------------------------------
# main.py end-to-end with Mock predictor
# ---------------------------------------------------------------------------

def test_main_end_to_end_with_mock(monkeypatch):
    """Run main.py against a config that points at a non-existent GNN model.

    main.py falls back to MockBarrierPredictor in that case, runs a short
    T-sweep, and writes the four output CSVs.
    """
    from KMC import main as kmc_main

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        cfg = KMCConfig(
            elements=["Mo", "Nb", "Ta", "W"],
            composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
            supercell_size=4,
            lattice_parameter_A=3.22,
            random_seed=0,
            temperature_K=1500.0,
            attempt_frequency_Hz=1e13,
            n_steps=200,
            temperatures_K_sweep=[1500.0, 1750.0, 2000.0],
            n_realizations_per_T=2,
            snapshot_every_n_steps=20,
            gnn_model_path="/nonexistent/path/will/trigger/mock.pt",
            output_dir=str(out_dir),
        )
        cfg_path = Path(tmp) / "cfg.json"
        cfg.to_json(cfg_path)

        rc = kmc_main.main([str(cfg_path)])
        assert rc == 0

        # Outputs are present
        assert (out_dir / "config_used.json").exists()
        assert (out_dir / "diffusion_per_T.csv").exists()
        assert (out_dir / "diffusion_summary.csv").exists()
        assert (out_dir / "arrhenius_per_element.csv").exists()
        assert (out_dir / "tau_order_per_T.csv").exists()

        # Check rough shape of one CSV
        with open(out_dir / "diffusion_summary.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3   # one per T
        for row in rows:
            assert float(row["D_median_A2_per_s"]) > 0
            assert int(row["n_realisations"]) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
