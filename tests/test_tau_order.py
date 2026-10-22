"""Relaxation fit of alpha(t)."""
import numpy as np
import pytest

from gnn_gym.kmc.tau_order import fit_tau_order


def _curve(tau, a0, ainf, t_end, n=40):
    t = np.linspace(0.0, t_end, n)
    return t, ainf + (a0 - ainf) * np.exp(-t / tau)


def test_recovers_tau_of_a_converged_curve():
    t, a = _curve(1.5e-9, 0.6, 0.05, 5 * 1.5e-9)
    fit = fit_tau_order(t, a)
    assert fit.status == "converged" and fit.converged
    assert fit.tau_order_s == pytest.approx(1.5e-9, rel=1e-3)
    assert fit.alpha_0 == pytest.approx(0.6, abs=1e-4)
    assert fit.alpha_inf == pytest.approx(0.05, abs=1e-4)
    assert fit.fraction_completed > 0.99


def test_short_curve_is_step_limited_with_upper_bound():
    t, a = _curve(1.0, 0.0, -0.5, 0.3)
    fit = fit_tau_order(t, a)
    assert fit.status == "step_limited"
    assert fit.tau_upper_bound_s >= 1.0 * 0.999
    assert fit.fraction_completed < 1 - np.exp(-2.0)


def test_pinned_equilibrium_value():
    t, a = _curve(2.0, 0.0, -0.4, 1.0)
    fit = fit_tau_order(t, a, alpha_inf_fixed=-0.4)
    assert fit.alpha_inf_fixed and fit.alpha_inf == -0.4
    assert fit.tau_order_s == pytest.approx(2.0, rel=1e-4)


def test_flat_curve_has_no_trend():
    t = np.linspace(0, 1, 50)
    fit = fit_tau_order(t, np.full(50, 0.1))
    assert fit.status == "no_trend"


def test_noise_alone_is_bounded():
    """A curve without relaxation does not give alpha outside [-1, 1]."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 50)
    fit = fit_tau_order(t, 0.01 * rng.normal(size=50))
    assert -1 <= fit.alpha_inf <= 1 and -1 <= fit.alpha_0 <= 1
    assert fit.status != "converged"


def test_too_few_or_non_finite_points():
    assert fit_tau_order([0.0, 1.0], [0.5, 0.1], min_points=5) is None
    t = np.linspace(0.0, 1.0, 20)
    assert fit_tau_order(t, np.full_like(t, np.nan)) is None
