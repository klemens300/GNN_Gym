"""Equilibrium vacancy concentration c_v(T).

In an alloy the vacancy formation energy E_f depends on the local
environment, so c_v is the Boltzmann average over the distribution of E_f,

    c_v(T) = exp(S_f / k_B) < exp(-E_f / k_B T) >,

not the value at the mean E_f: by the convexity of the exponential the
low-energy tail carries the vacancies. The distribution does not depend on
T, so c_v is evaluated directly at any temperature. The samples of E_f come
from the reference jump dataset (gnn_gym.oracle.formation); a pure element
has a single E_f. The vibrational formation entropy S_f is not computed and
is a constant of the configuration.
"""
import json
from functools import lru_cache

import numpy as np

from ..paths import resolve

K_B_EV_PER_K = 8.617333262e-5


@lru_cache(maxsize=None)
def formation_energy_samples(path):
    """E_f samples (eV) of an alloy spectrum file."""
    with open(resolve(path)) as fh:
        return np.asarray(json.load(fh)["E_f_samples_eV"], dtype=float)


@lru_cache(maxsize=None)
def pure_formation_energy(path, element):
    """E_f (eV) of a vacancy in a pure element."""
    with open(resolve(path)) as fh:
        return float(json.load(fh)["E_f_eV"][element])


def c_v_spectrum(E_f, T_K, S_f_kB):
    """c_v from an array of formation energies (Boltzmann average)."""
    val = float(np.mean(np.exp(-E_f / (K_B_EV_PER_K * T_K))))
    return val * float(np.exp(S_f_kB))


def c_v_single(E_f, T_K, S_f_kB):
    """c_v from a single formation energy."""
    val = float(np.exp(-float(E_f) / (K_B_EV_PER_K * T_K)))
    return val * float(np.exp(S_f_kB))
