"""Absorbing-Markov-chain escape layer."""
import numpy as np
import pytest

from gnn_gym.kmc.config import EscapeConfig
from gnn_gym.kmc.engine import BARRIER_FALLBACK_EV, K_B
from gnn_gym.kmc.escape import (Basin, BasinDetector, Escaper, build_basin,
                                first_passage)
from gnn_gym.kmc.state import random_state
from kmc_mocks import (A, ELEMENTS, EQUIATOMIC, NU, Constant, TwoSiteTrap, cell,
                       simulate)


def _basin(A_, exits, entry=0):
    m = len(A_)
    return Basin([None] * m, [0] * m, np.asarray(A_, float), [[]] * m,
                 list(exits), entry, False)


def _gillespie(A_, exits, rng, n=20000):
    """Mean absorption time and exit frequencies of the same chain."""
    m = len(A_)
    trans = [[(j, A_[i][j]) for j in range(m) if j != i and A_[i][j] > 0]
             + [(("exit", e), r) for e, (s, _, r) in enumerate(exits) if s == i]
             for i in range(m)]
    times, chan = np.empty(n), np.empty(n, int)
    for k in range(n):
        i, t = 0, 0.0
        while True:
            rates = np.array([r for _, r in trans[i]])
            t += rng.exponential(1.0 / rates.sum())
            to = trans[i][rng.choice(len(rates), p=rates / rates.sum())][0]
            if isinstance(to, tuple):
                times[k], chan[k] = t, to[1]
                break
            i = to
    return times.mean(), np.bincount(chan, minlength=len(exits)) / n


def test_detector_flags_after_threshold_and_resets():
    det = BasinDetector(window=50, revisit_threshold=3)
    flags = [det.observe(k) for k in [(0, b"a"), (1, b"b")] * 4]
    assert flags[-2] and det.flagged and not any(flags[:6])
    det.reset()
    assert not det.flagged and not det.keys and not det.counts


def test_detector_window_forgets_old_visits():
    det = BasinDetector(window=4, revisit_threshold=3)
    for k in [(0, b"a")] * 3 + [(i, b"x") for i in range(1, 5)]:
        det.observe(k)
    assert (0, b"a") not in det.counts and not det.flagged


def test_two_state_mean_first_passage_closed_form():
    a, b, e0, e1 = 5e10, 4e10, 1e8, 2e8
    A_ = [[-(a + e0), a], [b, -(b + e1)]]
    tau, p = first_passage(_basin(A_, [(0, 100, e0), (1, 200, e1)]))
    det = (a + e0) * (b + e1) - a * b
    assert tau == pytest.approx((b + e1 + a) / det, rel=1e-12)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)


def test_single_state_is_inverse_exit_rate():
    tau, p = first_passage(_basin([[-7e9]], [(0, 5, 7e9)]))
    assert tau == pytest.approx(1 / 7e9, rel=1e-12) and p[0] == pytest.approx(1)


def test_first_passage_matches_stochastic_simulation():
    a, b, e0, e1 = 8e9, 6e9, 5e8, 3e8
    A_ = [[-(a + e0), a], [b, -(b + e1)]]
    exits = [(0, 100, e0), (1, 200, e1)]
    tau, p = first_passage(_basin(A_, exits))
    tau_mc, f_mc = _gillespie(A_, exits, np.random.default_rng(0))
    assert tau == pytest.approx(tau_mc, rel=0.05)
    np.testing.assert_allclose(p, f_mc, atol=0.02)


def _trap(size=4):
    c = cell(size)
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(1))
    return c, st, TwoSiteTrap(c, st.vacancy, int(st.targets()[0]))


def test_build_basin_finds_the_two_site_trap():
    c, st, pred = _trap()
    basin = build_basin(st, pred, 200.0, NU, internal_n_kT=4.0,
                        max_basin_size=50, fallback_eV=BARRIER_FALLBACK_EV)
    assert basin.size == 2 and not basin.leaks
    assert len(basin.exits) == 14                    # 2 x 7 high barriers
    tau, p = first_passage(basin)
    assert tau > 0 and p.sum() == pytest.approx(1.0)


def test_plain_bkl_stays_in_the_trap():
    c, st, pred = _trap()
    sites = {st.vacancy, int(st.targets()[0])}
    _, tr = simulate(pred, 200.0, 300, seed=1)
    assert set(tr.vacancy.data.tolist()) <= sites


def test_escape_layer_leaves_the_trap_consistently():
    c, st, pred = _trap()
    params = EscapeConfig(revisit_threshold=3, window=50, internal_n_kT=4.0,
                          max_basin_size=50, cooldown_steps=25)
    esc = Escaper(params, BARRIER_FALLBACK_EV)
    end, tr = simulate(pred, 200.0, 400, seed=1, escaper=esc, snapshot_every=10)
    flags = tr.is_escape.data
    assert flags.any() and len(set(tr.vacancy.data.tolist())) > 2
    vac = tr.vacancy.data
    for k in np.flatnonzero(flags):
        path = tr.micro_paths[int(k)]
        assert len(path) >= 1 and path[-1] == vac[k + 1]
        assert tr.chosen.data[k] == -1 and np.isnan(tr.barriers.data[k]).all()
        assert np.isfinite(tr.delta_t.data[k]) and tr.delta_t.data[k] > 0
        # the unfolded displacement equals the sum of the chain's jumps
        v, d = vac[k], np.zeros(3, int)
        for t in path:
            d += c.min_image(c.coords[t] - c.coords[v])
            v = t
        np.testing.assert_allclose(tr.unfolded.data[k + 1] - tr.unfolded.data[k],
                                   d * A / 2, atol=1e-9)
    # the clock never runs backwards (outside the trap a residence time is
    # ~1e16 s at 200 K, inside ~1e-8 s, below the resolution of the sum)
    assert np.all(np.diff(tr.times.data) >= 0)
    # snapshots: the start, every 10th step, and after every escape
    step = np.arange(1, tr.n_steps + 1)
    assert len(tr.snapshot_times) == 1 + int(((step % 10 == 0) | flags).sum())


def test_open_basin_is_left_to_bkl():
    c = cell(4)
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(0))
    params = EscapeConfig(max_basin_size=20, cooldown_steps=7)
    esc = Escaper(params, BARRIER_FALLBACK_EV)
    esc.detector.flagged = True
    before = st.key()
    assert esc.escape(st, Constant(c), 1000.0, NU, np.random.default_rng(0),
                      A / 2) is None
    assert st.key() == before and esc.cooldown == 7 and not esc.flagged


def test_detector_state_round_trip():
    c, st, pred = _trap()
    esc = Escaper(EscapeConfig(window=30), BARRIER_FALLBACK_EV)
    for t in [int(x) for x in st.targets()[:3]] * 4:
        s2 = st.copy()
        s2.swap(t)
        esc.observe(s2)
    esc.cooldown = 4
    arrays = esc.state_arrays(c.n_sites)
    other = Escaper(EscapeConfig(window=30), BARRIER_FALLBACK_EV)
    assert other.restore(arrays)
    assert list(other.detector.keys) == list(esc.detector.keys)
    assert other.detector.counts == esc.detector.counts
    assert other.flagged == esc.flagged and other.cooldown == 4
    assert not Escaper(EscapeConfig(), BARRIER_FALLBACK_EV).restore({})


def test_rates_use_boltzmann_factor():
    c, st, pred = _trap()
    basin = build_basin(st, pred, 300.0, NU, internal_n_kT=4.0,
                        max_basin_size=50, fallback_eV=BARRIER_FALLBACK_EV)
    k_low = NU * np.exp(-0.2 / (K_B * 300.0))
    assert basin.A[0, 1] == pytest.approx(k_low, rel=1e-12)
