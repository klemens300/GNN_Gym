"""A continued run is the uninterrupted run.

2N steps in one go must equal N steps, a restart from the checkpoint, and N
more steps: same vacancy sites, times, escapes and snapshots. With the escape
layer this requires the basin detector (window of recent configurations,
flag, cooldown) to be part of the checkpoint; without it a restart delays
the next detection and the trajectories part.
"""
import numpy as np
import pytest

from gnn_gym.kmc import checkpoint as ck
from gnn_gym.kmc.config import RunConfig
from gnn_gym.kmc.runner import run_point
from kmc_mocks import Landscape, cell

N = 400


def _cfg(n, escape):
    return RunConfig(system="MoNbTaW", T_K=450.0, seed=11, n_steps=n,
                     supercell_size=6, escape=escape, snapshot_every=50,
                     checkpoint_every=100, progress_every=0)


@pytest.fixture(scope="module")
def predictor():
    return Landscape(cell(6), 4, spread=0.25, seed=5)


def _run(pred, d, steps, escape):
    for n in steps:
        path = run_point(_cfg(n, escape), pred, d, log=lambda *a: None)
    return ck.load_run(path)


def _assert_same(a, b):
    np.testing.assert_array_equal(a.vacancy_indices, b.vacancy_indices)
    np.testing.assert_array_equal(a.times_s, b.times_s)
    np.testing.assert_array_equal(a.delta_t_s, b.delta_t_s)
    np.testing.assert_array_equal(a.is_escape, b.is_escape)
    assert a.escape_micro_paths.keys() == b.escape_micro_paths.keys()
    for k in a.escape_micro_paths:
        np.testing.assert_array_equal(a.escape_micro_paths[k],
                                      b.escape_micro_paths[k])
    np.testing.assert_array_equal(a.snapshot_times_s, b.snapshot_times_s)
    np.testing.assert_array_equal(a.snapshot_species, b.snapshot_species)
    np.testing.assert_array_equal(a.final_state.species, b.final_state.species)


@pytest.mark.parametrize("escape", [True, False])
def test_continued_run_equals_uninterrupted(tmp_path, predictor, escape):
    whole = _run(predictor, tmp_path / "whole", [2 * N], escape)
    parts = _run(predictor, tmp_path / "parts", [N, 2 * N], escape)
    _assert_same(whole, parts)
    if escape:
        assert whole.n_escapes > 0
        assert parts.is_escape[N:].any()             # escapes after the restart


@pytest.mark.parametrize("escape", [True, False])
def test_head_equals_shorter_run(tmp_path, predictor, escape):
    """The first N steps of a 2N-step run, occupation replayed, are the
    N-step run."""
    whole = _run(predictor, tmp_path / "whole", [2 * N], escape)
    short = _run(predictor, tmp_path / "short", [N], escape)
    head = whole.head(N)
    _assert_same(head, short)
    assert head.n_steps == N and head.total_time_s == short.total_time_s
    assert head.final_state.vacancy == short.final_state.vacancy
    for key in ("vacancy_positions_unfolded", "hopper_species",
                "chosen_jump_local", "barriers_eV"):
        np.testing.assert_array_equal(getattr(head, key), getattr(short, key))
    assert whole.head(3 * N) is whole


def test_restart_at_every_checkpoint(tmp_path, predictor):
    whole = _run(predictor, tmp_path / "whole", [2 * N], True)
    parts = _run(predictor, tmp_path / "parts",
                 list(range(100, 2 * N + 1, 100)), True)
    _assert_same(whole, parts)


def test_detector_state_is_stored(tmp_path, predictor):
    """The window is empty right after an escape or during a cooldown, so it
    is inspected at several step counts."""
    sizes = []
    for n in range(50, N + 1, 50):
        _run(predictor, tmp_path / "r", [n], True)
        with np.load(ck.checkpoint_path(tmp_path / "r", 450.0),
                     allow_pickle=True) as z:
            w = z["escape_window_vacancy"]
            assert z["escape_window_species"].shape == (w.size, 2 * 6 ** 3)
            assert {"escape_flagged", "escape_cooldown"} <= set(z.files)
            sizes.append(w.size)
    assert max(sizes) > 0


def test_restart_without_detector_state_departs(tmp_path, predictor):
    """Control: the same restart with the detector state removed (as a
    checkpoint that does not store it) gives a different trajectory, so the
    tests above are sensitive to it."""
    whole = _run(predictor, tmp_path / "whole", [2 * N], True)
    d = tmp_path / "parts"
    _run(predictor, d, [N], True)
    path = ck.checkpoint_path(d, 450.0)
    with np.load(path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files if not k.startswith("escape_window")
                and k not in ("escape_flagged", "escape_cooldown")}
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **data)
    parts = _run(predictor, d, [2 * N], True)
    assert not np.array_equal(whole.vacancy_indices, parts.vacancy_indices)
