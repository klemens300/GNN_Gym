"""Tracer MSD, Warren-Cowley order and vacancy diffusion of stored runs."""
import numpy as np
import pytest

from gnn_gym.kmc.checkpoint import load_run
from gnn_gym.kmc.config import RunConfig
from gnn_gym.kmc.observables import tracer_msd, vacancy_diffusion, warren_cowley
from gnn_gym.kmc.runner import run_point
from gnn_gym.kmc.state import VACANCY, random_state
from kmc_mocks import ELEMENTS, EQUIATOMIC, Landscape, cell


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A short run with escapes on a 4^3 cell."""
    c = cell(4)
    cfg = RunConfig(system="MoNbTaW", T_K=500.0, seed=3, n_steps=600,
                    supercell_size=4, snapshot_every=20, checkpoint_every=200,
                    progress_every=0)
    d = tmp_path_factory.mktemp("obs")
    path = run_point(cfg, Landscape(c, 4, spread=0.25), d, log=lambda *a: None)
    return load_run(path)


def _replay(run):
    """Unfolded atom positions (lattice units) after the run, step by step."""
    c = run.cell
    owner = {int(s): i for i, s in
             enumerate(np.flatnonzero(run.initial_species != VACANCY))}
    pos = np.array([c.coords[s] for s in sorted(owner)], dtype=np.int64)
    start = pos.copy()
    vac = run.vacancy_indices
    msd = [np.zeros(len(run.elements))]
    sp = run.initial_species[run.initial_species != VACANCY]
    for k in range(run.n_steps):
        hops = (run.escape_micro_paths[k] if run.is_escape[k]
                else [vac[k + 1]])
        v = int(vac[k])
        for t in hops:
            a = owner.pop(int(t))
            pos[a] += c.min_image(c.coords[v] - c.coords[t])
            owner[v] = a
            v = int(t)
        d2 = ((pos - start) ** 2).sum(1)
        msd.append([d2[sp == e].mean() if (sp == e).any() else 0.0
                    for e in range(len(run.elements))])
    return np.asarray(msd) * (run.lattice_parameter_A / 2) ** 2, pos - start


def test_tracer_msd_equals_brute_force_replay(run):
    assert run.n_escapes > 0
    msd = tracer_msd(run)
    ref, _ = _replay(run)
    for i, e in enumerate(run.elements):
        assert msd[e].shape == (run.n_steps + 1,) and msd[e][0] == 0.0
        np.testing.assert_allclose(msd[e], ref[:, i], rtol=1e-12, atol=1e-12)


def test_atom_displacements_balance_the_vacancy(run):
    _, disp = _replay(run)
    vac_disp = run.vacancy_positions_unfolded[-1] - run.vacancy_positions_unfolded[0]
    np.testing.assert_allclose(disp.sum(0) * run.lattice_parameter_A / 2,
                               -vac_disp, atol=1e-9)


def _alpha_loop(species, neighbours, n_el):
    """Per-site reference implementation of the Warren-Cowley parameter."""
    pairs = np.zeros((n_el, n_el))
    for i, s in enumerate(species):
        if s == VACANCY:
            continue
        for j in neighbours[i]:
            if species[j] != VACANCY:
                pairs[s, species[j]] += 1
    atoms = species[species != VACANCY]
    comp = np.bincount(atoms, minlength=n_el) / atoms.size
    with np.errstate(divide="ignore", invalid="ignore"):
        a = 1 - pairs / pairs.sum(1)[:, None] / comp[None, :]
    a[~np.isfinite(a)] = np.nan
    return a


def test_warren_cowley_equals_per_site_loop(run):
    nb = run.final_state.neighbours
    for k in (0, len(run.snapshot_species) - 1):
        sp = run.snapshot_species[k]
        np.testing.assert_array_equal(warren_cowley(sp, nb, 4),
                                      _alpha_loop(sp, nb, 4))


def test_warren_cowley_random_alloy_near_zero():
    c = cell(4)
    acc = np.zeros((4, 4))
    for seed in range(8):
        st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(seed))
        a = warren_cowley(st.species, st.neighbours, 4)
        assert np.isfinite(a).all()
        acc += a
    assert np.abs(acc / 8).max() < 0.10


def test_warren_cowley_two_slabs_segregated():
    """Ta | W halves along x: unlike first neighbours only at the interfaces."""
    c = cell(4)
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(0))
    sp = np.where(c.coords[:, 0] < c.box // 2, 2, 3).astype(np.int8)
    sp[st.vacancy] = VACANCY
    a = warren_cowley(sp, st.neighbours, 4)
    assert a[2, 3] >= 0.45 and a[3, 2] >= 0.45
    assert a[2, 2] <= -0.45 and a[3, 3] <= -0.45
    assert np.isnan(a[0]).all()                       # no Mo in the cell


def test_vacancy_diffusion_exponent():
    t = np.linspace(0, 1, 2001)
    ballistic = np.stack([t, 0 * t, 0 * t], 1) * 10
    assert vacancy_diffusion(ballistic, t, 0, 1)[2] == pytest.approx(2, abs=0.05)
    # random walk with MSD = 3 t: D = 1/2, alpha = 1 (mean over 40 walks; a
    # single walk scatters by ~30 % at lags up to 10 % of its length)
    rng = np.random.default_rng(0)
    tw = np.arange(20001, dtype=float)
    res = [vacancy_diffusion(np.cumsum(rng.choice([-1, 1], (20001, 3)), axis=0)
                             .astype(float), tw, 0, tw[-1]) for _ in range(40)]
    assert np.mean([r[0] for r in res]) == pytest.approx(0.5, rel=0.1)
    assert np.mean([r[2] for r in res]) == pytest.approx(1.0, abs=0.05)
