"""Escape against plain KMC, order reached, Arrhenius fits of the summaries."""
import numpy as np
import pandas as pd
import pytest

from gnn_gym.analysis import diffusion_summary as ds
from gnn_gym.analysis import order_kinetics as ok
from gnn_gym.analysis.escape_vs_plain import compare
from gnn_gym.kmc.config import RunConfig
from gnn_gym.kmc.runner import run_point
from kmc_mocks import Landscape, cell


def test_escape_vs_plain_same_seed(tmp_path):
    pred = Landscape(cell(6), 4, spread=0.25, seed=5)
    for escape, n in ((True, 1000), (False, 800)):
        cfg = RunConfig(system="MoNbTaW", T_K=450.0, seed=11, n_steps=n,
                        supercell_size=6, escape=escape, snapshot_every=50,
                        checkpoint_every=100, progress_every=0)
        run_point(cfg, pred, tmp_path / str(escape), log=lambda *a: None)
    r = compare(tmp_path / "True", tmp_path / "False", 900)
    assert r["n_steps"] == 800 and r["same_start"]
    assert 0 <= r["first_escape_step"] <= r["first_difference_step"]
    assert r["time_ratio"] > 0.0 and r["time_ratio"] != 1.0
    assert np.isfinite(r["D_vac_ratio"])
    assert r["escape_frac_plain"] == 0.0 and r["escape_frac_escape"] > 0.0


def test_order_reached_and_curves():
    runs = pd.DataFrame(dict(run=["a", "b"], alloy=["TaW", "TaW"],
                             T_K=[900, 900], n_steps=[200, 100], flags=["", ""]))
    alpha = pd.concat([
        pd.DataFrame(dict(run="a", alloy="TaW", T_K=900, step=np.arange(0, 201, 10),
                          t_real_s=np.arange(0, 201, 10) * 2.0,
                          **{"Ta-W": -0.001 * np.arange(0, 201, 10)})),
        pd.DataFrame(dict(run="b", alloy="TaW", T_K=900, step=np.arange(0, 101, 4),
                          t_real_s=np.arange(0, 101, 4) * 3.0,
                          **{"Ta-W": -0.002 * np.arange(0, 101, 4)}))])
    eq = {"TaW": {("Ta", "W"): (np.array([800.0, 1000.0]), np.array([-0.3, -0.1]))}}
    tab = ok.order_reached(runs, alpha, eq, {"TaW": "Ta-W"}, [900.0], 0.05)
    assert tab.alpha_eq.tolist() == pytest.approx([-0.2, -0.2])
    # last 5 % of 21 points -> the mean of the last 3
    assert tab.alpha_end.iloc[0] == pytest.approx(-0.19)
    assert tab.share.iloc[1] == pytest.approx(np.mean([0.184, 0.192, 0.2]) / 0.2)
    curves, t_head = ok.kinetics_curves(tab, alpha, 100)
    # step grid of the run with fewer points (a), cut at step 100
    assert curves.step.tolist() == list(range(0, 101, 10))
    assert curves.share.iloc[-1] == pytest.approx((0.5 + 1.0) / 2)
    assert t_head[("TaW", 900.0)] == pytest.approx((200.0 + 300.0) / 2)


def test_arrhenius_fits_recover_parameters():
    T = np.array([1100.0, 1300.0, 1500.0])
    D = 3e-6 * np.exp(-1.7 / (ds.K_B * T))
    E, D0 = ds.fit_ev(T, D)
    assert E == pytest.approx(1.7, rel=1e-10) and D0 == pytest.approx(3e-6, rel=1e-9)
    D0k, Q = ds.fit_kj(T, D * 1e4)
    assert Q == pytest.approx(1.7 * ds.R_KJ / ds.K_B, rel=1e-9)
    assert D0k == pytest.approx(3e-2, rel=1e-9)
