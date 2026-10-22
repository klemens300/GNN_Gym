"""Swap Metropolis: detailed balance of the swap move, energy bookkeeping of
the sampler, order parameters on known structures, equilibration estimators."""
import itertools

import numpy as np
import pytest
import torch

from gnn_gym.lattice import BCC, Supercell
from gnn_gym.metropolis.autocorr import equilibration_onset, tau_int
from gnn_gym.metropolis.exafs import group_fractions
from gnn_gym.metropolis.observables import (block_stats, eta_per_species,
                                            warren_cowley, warren_cowley_shells)
from gnn_gym.metropolis.order import (concentration_waves, fold, local_b2_order,
                                      structure_factor, superstructure_mask)
from gnn_gym.metropolis.run import start_configuration
from gnn_gym.metropolis.sampler import BatchedSwapChain, SwapGeometry
from gnn_gym.paths import resolve

DE = resolve("models/MoNbTaW/de_L2_s0.pt")


# ---------------------------------------------------------------------------
# the swap move samples the Boltzmann distribution
# ---------------------------------------------------------------------------

class RingChain(BatchedSwapChain):
    """Two species on a ring, energy J per like nearest-neighbour bond. A swap
    at a changes the bonds of a only, so two swaps interact iff a site of one
    lies within one bond of a site of the other: that is the conflict."""

    def __init__(self, z, J, batch, seed, walk="until_conflict"):
        super().__init__(z, batch, seed, walk)
        self.J = J

    def energy(self, z):
        return self.J * float(np.sum(z == np.roll(z, 1)))

    def call(self, p, q):
        e0 = self.energy(self.z)
        dE = []
        for a, b in zip(p, q):
            z = self.z.copy()
            z[a], z[b] = z[b], z[a]
            dE.append(self.energy(z) - e0)
        s = np.stack([p, q], 1)
        d = np.abs(s[:, None, :, None] - s[None, :, None, :]) % self.n
        d = np.minimum(d, self.n - d)
        return np.array(dE), (d <= 1).any(-1).any(-1), None

    def apply(self, taken, p, q, payload):
        for r in taken:
            a, b = p[r], q[r]
            self.z[a], self.z[b] = self.z[b], self.z[a]


@pytest.mark.parametrize("batch, walk", [(1, "until_conflict"),
                                         (4, "until_accept")])
def test_single_swap_chain_samples_boltzmann(batch, walk):
    """The single-swap chain (batch 1, or any batch with until_accept): the
    energy distribution on an enumerable ring is the Boltzmann one
    (symmetric proposal, Metropolis acceptance)."""
    n, J, kT = 8, 1.0, 1.0
    states = []
    for ones in itertools.combinations(range(n), n // 2):
        z = np.zeros(n, dtype=np.int64)
        z[list(ones)] = 1
        states.append(z)
    chain = RingChain(states[0], J, batch=batch, seed=3, walk=walk)
    E = np.array([chain.energy(z) for z in states])
    w = np.exp(-E / kT)
    exact_mean = float((E * w).sum() / w.sum())
    levels = np.unique(E)
    exact_p = np.array([w[E == e].sum() for e in levels]) / w.sum()

    samples = []
    for k in range(20100):
        chain.sweep(kT)
        if k >= 100:
            samples.append(chain.energy(chain.z))
    samples = np.array(samples)
    mean, err = block_stats(samples, n_blocks=20)
    assert abs(mean - exact_mean) < 4 * err, (mean, err, exact_mean)
    p = np.array([(samples == e).mean() for e in levels])
    assert 0.5 * np.abs(p - exact_p).sum() < 0.02


def test_batch_decides_several_proposals_per_call():
    chain = RingChain(np.arange(24) % 2, 1.0, batch=6, seed=0)
    calls = sum(chain.sweep(2.0)[1] for _ in range(20))
    assert calls < 20 * 24


def test_until_accept_applies_one_move_per_call():
    chain = RingChain(np.arange(24) % 2, 1.0, batch=6, seed=0, walk="until_accept")
    taken = []
    apply = chain.apply
    chain.apply = lambda t, p, q, payload: (taken.append(len(t)),
                                            apply(t, p, q, payload))
    for _ in range(20):
        chain.sweep(2.0)
    assert taken and set(taken) == {1}
    with pytest.raises(ValueError):
        RingChain(np.arange(24) % 2, 1.0, batch=6, seed=0, walk="other")


# ---------------------------------------------------------------------------
# the sampler on the site-energy model
# ---------------------------------------------------------------------------

def test_geometry_bcc_reach2():
    cell = Supercell(BCC, 12)
    g = SwapGeometry(cell, [1, 2], reach=2, n_layers=2)
    assert (g.n_h, len(g.read_off), g.level_sizes) == (369, 65, [370, 176, 66])
    assert g.off[g.centre].tolist() == [0, 0, 0]


@pytest.mark.skipif(not DE.exists(), reason="model checkpoint not present")
def test_sampler_energies_match_full_cell():
    from gnn_gym.metropolis.sampler import SwapSampler
    from gnn_gym.model import load_model
    torch.manual_seed(0)
    net, cfg, ck = load_model(DE, "cpu", bind=False)
    cell = Supercell(BCC, 6)
    code = {e: i for i, e in enumerate(cfg.elements)}
    z = start_configuration("random", cell, cfg.elements, code,
                            np.random.default_rng(0))
    s = SwapSampler(net, cell, cfg.shells, int(ck["reach"]), z, "cpu",
                    batch=8, seed=1)
    # dE of a batch against the difference of two full-cell evaluations
    p, q = s.propose(8)
    dE, conf, _ = s.call(p, q)
    swapped = np.repeat(s.z[None], 8, axis=0)
    swapped[np.arange(8), p], swapped[np.arange(8), q] = s.z[q], s.z[p]
    eps = s.site_energies(np.vstack([s.z[None], swapped]))
    full = (eps[1:] - eps[:1]).sum(1)
    assert np.abs(dE - full).max() < 1e-5
    assert conf.diagonal().all()
    # running site energies after a sweep equal a full recomputation
    acc, calls = s.sweep(8.617333262e-5 * 2000.0)
    assert 0.3 < acc < 1.0 and calls < cell.n_sites
    running = s.eps_cpu().copy()
    assert np.array_equal(s.z_dev.cpu().numpy(), s.z)
    assert s.refresh() < 1e-5
    assert np.abs(running - s.site_energies(s.z[None])[0]).max() < 1e-5
    assert np.array_equal(np.bincount(s.z, minlength=4), np.bincount(z, minlength=4))


# ---------------------------------------------------------------------------
# start configurations
# ---------------------------------------------------------------------------

def test_start_configurations():
    cell = Supercell(BCC, 4)
    code = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
    rng = np.random.default_rng(0)
    z = start_configuration("random", cell, ["Mo", "W", "Nb", "Ta"], code, rng)
    assert np.bincount(z).tolist() == [32] * 4
    z = start_configuration("b2", cell, ["Mo", "W", "Nb", "Ta"], code, rng)
    sub = cell.sublattice()
    assert set(z[sub == 0].tolist()) == {0, 3} and set(z[sub == 1].tolist()) == {1, 2}
    seq = ["Nb", "Mo", "Ta", "W", "W", "Ta", "Mo", "Nb"]
    z = start_configuration("layered", cell, ["Mo", "Nb", "Ta", "W"], code, rng, seq)
    planes = cell.coords[:, 2]
    assert all(len(set(z[planes == k].tolist())) == 1 for k in range(cell.box))


# ---------------------------------------------------------------------------
# order parameters
# ---------------------------------------------------------------------------

def b2(cell, a=2, b=3):
    return np.where(cell.sublattice() == 0, a, b)


def test_warren_cowley_and_eta_perfect_b2():
    cell = Supercell(BCC, 6)
    z = b2(cell)
    al = warren_cowley_shells(z, cell, 4)
    assert al[1][2, 3] == -1.0 and al[2][2, 3] == 1.0 and al[3][2, 3] == 1.0
    assert al[1][2, 2] == 1.0
    assert eta_per_species(z, cell.sublattice(), 4)[[2, 3]].tolist() == [1.0, 1.0]


def test_random_alloy_is_disordered():
    cell = Supercell(BCC, 8)
    z = np.random.default_rng(1).permutation(np.repeat(np.arange(4), cell.n_sites // 4))
    al = warren_cowley(z, cell.neighbours(1), 4)
    assert np.abs(al).max() < 0.1
    r = concentration_waves(z, cell, 4)
    assert max(v["psi"] for v in r.values()) < 0.2


def test_concentration_waves_b2_and_antiphase_domains():
    cell = Supercell(BCC, 6)
    z = b2(cell)
    r = concentration_waves(z, cell, 4)
    assert abs(r[2]["psi"] - 1) < 1e-12 and r[2]["k"] == (1.0, 0.0, 0.0)
    assert abs(r[3]["p"] + 1) < 1e-12           # W on the trough of the Ta wave
    S = structure_factor(z, cell, 2)
    assert abs(S.max() - cell.n_sites / 4) < 1e-9
    half = cell.coords[:, 0] < cell.size
    z = np.where((cell.sublattice() == 0) ^ half, 2, 3)
    r = concentration_waves(z, cell, 4)
    assert r[2]["eta"] < 1e-12 and r[2]["psi"] > 0.5


def test_fold_and_mask():
    assert fold((-1.0, 0.0, 0.0), BCC) == (1.0, 0.0, 0.0)
    assert fold((0.5, -0.5, 0.5), BCC) == (0.5, 0.5, 0.5)
    assert fold((1.0, 1.0, 0.0), BCC) == (0.0, 0.0, 0.0)
    assert fold((5 / 6, 1 / 6, 0.0), BCC) == (0.8333, 0.1667, 0.0)
    cell = Supercell(BCC, 6)
    mask, k = superstructure_mask(cell)
    fundamental = k[~mask]                  # (000) and the three (110)-type points
    assert len(fundamental) == 4
    assert all(abs(np.round(v).sum()) % 2 == 0 for v in fundamental)


def test_local_b2_order():
    cell = Supercell(BCC, 6)
    s, n = local_b2_order(b2(cell), cell, (2, 3))
    assert np.all(s == 1.0) and np.all(n == 9)
    s, n = local_b2_order(b2(cell, 3, 2), cell, (2, 3))
    assert np.all(s == -1.0)


def test_group_fractions_b2():
    cell = Supercell(BCC, 4)
    els = ["Mo", "Nb", "Ta", "W"]
    rng = np.random.default_rng(0)
    z = start_configuration("b2", cell, ["Mo", "Nb", "Ta", "W"],
                            {e: i for i, e in enumerate(els)}, rng)
    f = group_fractions(z, cell, els, ["Mo", "Nb"])
    assert f[("Mo", 1)] == pytest.approx(0.0, abs=1e-12)
    assert f[("Mo", 2)] == pytest.approx(1.0, abs=1e-12)
    assert f[("W", 1)] == pytest.approx(1.0, abs=1e-12)
    assert f[("W", 2)] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# equilibration estimators
# ---------------------------------------------------------------------------

def test_tau_int():
    rng = np.random.default_rng(0)
    tau, _ = tau_int(rng.normal(size=20000))
    assert abs(tau - 0.5) < 0.1
    phi, x = 0.8, np.zeros(40000)
    e = rng.normal(size=x.size)
    for t in range(1, x.size):
        x[t] = phi * x[t - 1] + e[t]
    tau, _ = tau_int(x)
    exact = (1 + phi) / (2 * (1 - phi))     # 1/2 + sum_t phi^t
    assert abs(tau - exact) < 0.15 * exact


def test_equilibration_of_stationary_and_drifting_series():
    rng = np.random.default_rng(1)
    assert equilibration_onset(rng.normal(size=4000)) == 0
    # a drift dominates the autocorrelation: the block test then has no power
    # of its own, and the drift shows as a large tau_int and few samples
    x = rng.normal(size=4000) + np.linspace(10.0, 0.0, 4000)
    tau, _ = tau_int(x)
    assert tau > 200
