"""BKL step and trajectory loop against analytic results."""
import numpy as np
import pytest

from gnn_gym.kmc.engine import K_B, bkl_step, select_jump
from gnn_gym.kmc.observables import vacancy_diffusion
from gnn_gym.kmc.state import VACANCY, random_state
from kmc_mocks import (A, ELEMENTS, EQUIATOMIC, NU, ByElement, Constant, cell,
                       simulate)


def test_select_jump_rate_and_residence_time():
    T, E = 1500.0, 1.0
    R = 8 * NU * np.exp(-E / (K_B * T))
    dts = [select_jump(np.full(8, E), T, NU, np.random.default_rng(s))[1]
           for s in range(2000)]
    # E[dt] = 1/R; the residence time is exponential, so the SEM is 1/R/sqrt(n)
    assert abs(np.mean(dts) - 1 / R) < 5 / R / np.sqrt(len(dts))


def test_bkl_step_moves_to_a_neighbour():
    c = cell(4)
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(0))
    nb = set(st.targets().tolist())
    target, slot, dt, barriers = bkl_step(st, Constant(c), 1500.0, NU,
                                          np.random.default_rng(1))
    assert target in nb and st.vacancy == target and dt > 0
    assert barriers.shape == (8,)
    assert int((st.species == VACANCY).sum()) == 1


def test_bkl_step_prefers_low_barriers():
    c = cell(4)
    pred = ByElement(c, [0.5, 2.0, 2.0, 2.0])         # only Mo jumps
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(3))
    for s in range(20):
        if (st.species[st.targets()] == 0).any():
            break
        st.swap(int(st.targets()[0]))
    moved = []
    for s in range(50):
        trial = st.copy()
        bkl_step(trial, pred, 600.0, NU, np.random.default_rng(s))
        moved.append(int(trial.species[st.vacancy]))
    assert set(moved) == {0}


def test_trajectory_invariants():
    pred = Constant(cell(4))
    st, tr = simulate(pred, 1500.0, 500, seed=0, snapshot_every=50)
    c = pred.cell
    n = tr.n_steps
    assert n == 500 and len(tr.vacancy) == n + 1
    assert np.all(np.diff(tr.times.data) > 0) and np.all(tr.delta_t.data > 0)
    vac = tr.vacancy.data
    for k in range(n):
        assert vac[k + 1] in st.neighbours[vac[k]]
    steps = np.linalg.norm(np.diff(tr.unfolded.data, axis=0), axis=1)
    np.testing.assert_allclose(steps, np.sqrt(3) / 2 * A, atol=1e-9)
    # first snapshot is the start configuration, one every 50 steps after
    assert len(tr.snapshot_times) == 1 + n // 50
    np.testing.assert_array_equal(tr.snapshot_species[0], tr.initial_species)
    # the first hopper is the atom that sat on the first target site
    assert tr.hopper.data[0] == tr.initial_species[vac[1]]
    assert not tr.is_escape.data.any()
    assert int((st.species == VACANCY).sum()) == 1
    assert c.n_sites == st.species.size


def test_jump_directions_uniform_at_constant_barrier():
    _, tr = simulate(Constant(cell(4)), 1500.0, 8000, seed=42)
    counts = np.bincount(tr.chosen.data, minlength=8)
    chi2 = float(((counts - 1000.0) ** 2 / 1000.0).sum())
    assert chi2 < 24.32                                   # 7 dof, 99.9 %


def test_vacancy_diffusion_matches_uncorrelated_walk():
    """Equal barriers: D_vac = Gamma d^2 / 6 = nu exp(-E/kT) a^2 (bcc)."""
    T, E = 2000.0, 0.5
    pred = Constant(cell(6), E)
    D = []
    for seed in range(10):
        _, tr = simulate(pred, T, 10000, seed=seed)
        t_pos = np.concatenate([[0.0], tr.times.data])
        D.append(vacancy_diffusion(tr.unfolded.data, t_pos, 0.0, t_pos[-1])[0])
    expected = NU * np.exp(-E / (K_B * T)) * A ** 2
    # a single 1e4-step walk scatters by ~30 % (lags up to 10 % of the run)
    assert abs(np.mean(D) / expected - 1) < 0.15


def test_vanishing_total_rate_raises():
    with pytest.raises(RuntimeError):
        select_jump(np.full(8, 1e6), 300.0, NU, np.random.default_rng(0))
