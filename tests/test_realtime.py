"""Simulated -> physical time: c_v, nu0 and the conversion factor.

Reference values are those of the paper pipeline (data/cv, data/nu0).
"""
import numpy as np
import pytest

from gnn_gym import realtime
from gnn_gym.realtime.nu0 import density, debye_frequency, load_table
from gnn_gym.realtime.vacancy import K_B_EV_PER_K

NU0_HZ = {"MoNbTaW": 6646284215669.7822, "TaW": 5921313252596.5684,
          "Mo": 9580938596194.4941, "Nb": 4218249241761.7793,
          "Ta": 4928504466774.7236, "W": 7593863031000.7803}

C_V = {("MoNbTaW", 700): 3.863390e-20, ("MoNbTaW", 1100): 1.631014e-13,
       ("MoNbTaW", 1500): 2.267814e-10, ("TaW", 700): 2.862767e-21,
       ("TaW", 1100): 2.514647e-14, ("TaW", 1500): 5.484314e-11,
       ("Mo", 1100): 1.998072e-13, ("Nb", 1100): 1.011515e-11,
       ("Ta", 1100): 3.645552e-13, ("W", 1100): 1.495344e-15}


@pytest.mark.parametrize("name", sorted(NU0_HZ))
def test_nu0(name):
    assert realtime.nu0_phys(name) == pytest.approx(NU0_HZ[name], rel=1e-12)


@pytest.mark.parametrize("key", sorted(C_V))
def test_c_v(key):
    assert realtime.c_v(*key) == pytest.approx(C_V[key], rel=2e-6)


def test_pure_element_is_arrhenius():
    # one formation energy: c_v = exp(S_f) exp(-E_f / kT)
    c1, c2 = realtime.c_v("W", 1000.0), realtime.c_v("W", 2000.0)
    E_f = -K_B_EV_PER_K * np.log(c2 / c1) / (1 / 2000.0 - 1 / 1000.0)
    assert E_f == pytest.approx(3.378, abs=1e-9)
    assert c1 == pytest.approx(np.exp(1.5) * np.exp(-3.378 / (K_B_EV_PER_K * 1000.0)))


def test_alloy_c_v_exceeds_mean_energy_value():
    # Boltzmann average over a distribution > value at its mean (convexity)
    T = 900.0
    from gnn_gym.realtime.vacancy import formation_energy_samples
    E = formation_energy_samples("data/cv/cv_7cube_MoNbTaW.json")
    at_mean = np.exp(1.5) * np.exp(-E.mean() / (K_B_EV_PER_K * T))
    assert realtime.c_v("MoNbTaW", T) > 1e3 * at_mean


def test_time_factors():
    f = realtime.time_factors("MoNbTaW", 1100.0, n_sites=3456, nu0_sim_Hz=1e13)
    R = (1e13 / f["nu0_phys_Hz"]) / (3455 * f["c_v"])
    assert f["R"] == R
    assert f["c_v"] == realtime.c_v("MoNbTaW", 1100.0)


def test_grid_and_fit():
    t = load_table("data/nu0/nu0_grid_MoNbTaW.csv", "data/nu0/nu0_fit_MoNbTaW.json")
    assert t.elastic({"W": 1.0})["source"] == "grid"
    assert t.elastic({"Mo": .25, "Nb": .25, "Ta": .25, "W": .25})["source"] == "fit"
    # the cubic fit stays close to the grid where both exist
    g = t.elastic({"Ta": 0.5, "W": 0.5})
    x = np.array([0, 0, .5, .5])
    D = np.array([np.prod(x[m]) for m in t.monomials])
    fit = {q: D @ np.array(t.fit["coefficients"][q]) for q in ("C11", "C12", "C44", "a")}
    nu_fit = debye_frequency(fit["C11"], fit["C12"], fit["C44"],
                             density(x, fit["a"], t.masses, 2), fit["a"], 2)
    assert nu_fit == pytest.approx(g["nu0"], rel=0.03)


def test_debye_isotropic_limit():
    # isotropic solid (C11 - C12 = 2 C44): G = C44, closed-form check
    C44, C12 = 80.0, 120.0
    C11 = C12 + 2 * C44
    rho, a = 1.0e4, 3.2
    nu = debye_frequency(C11, C12, C44, rho, a, 2)
    B = (C11 + 2 * C12) / 3 * 1e9
    G = C44 * 1e9
    vl, vt = np.sqrt((B + 4 * G / 3) / rho), np.sqrt(G / rho)
    vm = (1 / 3 * (1 / vl ** 3 + 2 / vt ** 3)) ** (-1 / 3)
    n = 2 / (a * 1e-10) ** 3
    assert nu == pytest.approx(vm / (2 * np.pi) * (6 * np.pi ** 2 * n) ** (1 / 3))
