"""Barrier sources for the KMC tests (the package itself has none).

All follow the interface of gnn_gym.model.predictor.JumpPredictor:
`cell` and `barriers(species, vacancy, targets)`.
"""
import numpy as np

from gnn_gym.kmc.result import Trajectory
from gnn_gym.kmc.state import random_state
from gnn_gym.lattice import BCC, Supercell

ELEMENTS = ("Mo", "Nb", "Ta", "W")
EQUIATOMIC = {e: 0.25 for e in ELEMENTS}
A = 3.239121
NU = 1e13


class Constant:
    """The same barrier for every jump: an uncorrelated random walk."""

    def __init__(self, cell, value=1.0):
        self.cell, self.value = cell, float(value)

    def barriers(self, species, vacancy, targets):
        return np.full(len(targets), self.value)


class ByElement:
    """Barrier set by the species of the jumping atom."""

    def __init__(self, cell, table):
        self.cell, self.table = cell, np.asarray(table, float)

    def barriers(self, species, vacancy, targets):
        return self.table[np.asarray(species)[np.asarray(targets)]]


class TwoSiteTrap:
    """A low barrier between two sites, high barriers everywhere else."""

    def __init__(self, cell, site_a, site_b, low=0.2, high=1.2):
        self.cell = cell
        self.pair = {int(site_a), int(site_b)}
        self.low, self.high = low, high

    def barriers(self, species, vacancy, targets):
        return np.array([self.low if {int(vacancy), int(t)} == self.pair
                         else self.high for t in targets])


class Landscape:
    """Frozen random energy landscape that obeys detailed balance.

    E = sum_i eps[i, s_i] (vacancy = last column); a jump v <-> t has
    E_a = b0 + u_v + u_t + (E_B - E_A) / 2. The spread of eps traps the
    vacancy at low temperature, which exercises the escape layer.
    """

    def __init__(self, cell, n_species, spread=0.25, seed=0, b0=1.0):
        rng = np.random.default_rng(seed)
        self.cell = cell
        self.eps = rng.normal(0.0, spread, (cell.n_sites, n_species + 1))
        self.u = rng.uniform(0.0, 0.1, cell.n_sites)
        self.b0 = b0

    def barriers(self, species, vacancy, targets):
        sp = np.asarray(species)
        t = np.asarray(targets)
        s = sp[t]
        vac = -1
        dE = (self.eps[vacancy, s] + self.eps[t, vac]
              - self.eps[vacancy, vac] - self.eps[t, s])
        return self.b0 + self.u[vacancy] + self.u[t] + 0.5 * dE


def cell(size=4):
    return Supercell(BCC, size)


def simulate(predictor, T_K, n_steps, seed=0, escaper=None, snapshot_every=0,
             composition=EQUIATOMIC):
    """Run the engine directly; returns (state, trajectory)."""
    from gnn_gym.kmc.engine import run
    c = predictor.cell
    state = random_state(c, ELEMENTS, composition, np.random.default_rng(seed))
    traj = Trajectory(state.species, state.vacancy,
                      c.positions(A)[state.vacancy], state.targets().size)
    if snapshot_every:
        traj.snapshot(state)
    run(state, predictor, traj, T_K, NU, n_steps, A,
        np.random.default_rng(seed + 1), escaper=escaper,
        snapshot_every=snapshot_every)
    return state, traj
