"""Conversion of simulated KMC time to physical time.

The KMC cell holds one vacancy among N atoms and runs with a nominal attempt
frequency nu0_sim. Vacancy-mediated transport is linear in the vacancy
concentration and every rate is proportional to the attempt frequency, so
one factor converts the simulated clock:

    t_real = R t_sim,     R = (nu0_sim / nu0_phys) / (N c_v(T)).

c_v(T) is the equilibrium vacancy concentration (gnn_gym.realtime.vacancy),
nu0_phys the Debye frequency of the composition (gnn_gym.realtime.nu0). The
conversion is applied after the trajectory has been generated and changes
neither the sequence of jumps nor the order they build.

    from gnn_gym import realtime
    realtime.c_v("MoNbTaW", 1100.0)
    realtime.nu0_phys("TaW")
    realtime.time_factors("MoNbTaW", 1100.0, n_sites=3456, nu0_sim_Hz=1e13)
"""
from functools import lru_cache

from ..paths import load_yaml
from ..systems import get_system
from .vacancy import (c_v_single, c_v_spectrum, formation_energy_samples,
                      pure_formation_energy)

CONFIG = "configs/realtime.yaml"


@lru_cache(maxsize=None)
def _config(path):
    return load_yaml(path)


def c_v(system_name, T_K, config=CONFIG):
    """Equilibrium vacancy concentration of a system at temperature T_K."""
    cfg = _config(config)
    try:
        src = cfg["c_v"][system_name]
    except KeyError:
        raise ValueError(f"no c_v source for {system_name!r} in {config}") from None
    S_f = float(cfg["S_f_kB"])
    if "spectrum" in src:
        return c_v_spectrum(formation_energy_samples(src["spectrum"]), T_K, S_f)
    if "pure" in src:
        (element,) = get_system(system_name).elements
        return c_v_single(pure_formation_energy(src["pure"], element), T_K, S_f)
    raise ValueError(f"{system_name}: c_v source needs 'spectrum' or 'pure'")


def nu0_phys(system_name, config=CONFIG):
    """Physical attempt frequency (Hz): Debye frequency of the composition."""
    from .nu0 import load_table   # module also runs as a script
    c = _config(config)["nu0"]
    table = load_table(c["grid"], c["fit"], c["lattice"])
    return float(table.nu0(get_system(system_name).composition))


def time_factors(system_name, T_K, n_sites, nu0_sim_Hz, config=CONFIG):
    """c_v, nu0_phys and the factor R with t_real = R * t_sim.

    n_sites counts the lattice sites of the KMC cell including the vacancy,
    so the cell holds n_sites - 1 atoms.
    """
    cv = c_v(system_name, T_K, config)
    nu0 = nu0_phys(system_name, config)
    R = (float(nu0_sim_Hz) / nu0) / ((int(n_sites) - 1) * cv)
    return dict(c_v=cv, nu0_phys_Hz=nu0, R=R)


__all__ = ["c_v", "nu0_phys", "time_factors"]
