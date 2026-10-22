"""Checkpoint files, run configs and campaign files."""
import glob

import numpy as np
import pytest

from gnn_gym.kmc import campaign, checkpoint as ck
from gnn_gym.kmc.config import RunConfig, load_campaign
from gnn_gym.kmc.runner import initial_state, run_point, seed_streams
from gnn_gym.paths import resolve
from kmc_mocks import Landscape, cell


def _cfg(**kw):
    base = dict(system="MoNbTaW", T_K=600.0, seed=17, n_steps=300,
                supercell_size=4, snapshot_every=50, checkpoint_every=100,
                progress_every=0)
    return RunConfig(**{**base, **kw})


def test_checkpoint_holds_the_run(tmp_path):
    cfg = _cfg()
    path = run_point(cfg, Landscape(cell(4), 4), tmp_path, log=lambda *a: None)
    assert path.name == "checkpoint_T600K_real0.npz"
    assert not path.with_suffix(".npz.tmp").exists()
    assert (tmp_path / "run.yaml").exists()
    r = ck.load_run(path)
    n = cfg.n_steps
    assert (r.system, r.T_K, r.n_steps) == ("MoNbTaW", 600.0, n)
    assert r.lattice_parameter_A == cfg.lattice_parameter_A
    assert r.jump_distance_A == pytest.approx(np.sqrt(3) / 2 * cfg.lattice_parameter_A)
    assert r.times_s.shape == (n,) and r.vacancy_indices.shape == (n + 1,)
    assert r.barriers_eV.shape == (n, 8) and r.barriers_eV.dtype == np.float32
    assert r.vacancy_positions_unfolded.shape == (n + 1, 3)
    assert r.escape_layer and r.is_escape.shape == (n,)
    assert r.total_time_s == r.times_s[-1]
    st = initial_state(cfg)
    np.testing.assert_array_equal(r.initial_species, st.species)
    assert r.initial_vacancy_index == st.vacancy
    assert r.final_state.vacancy == r.vacancy_indices[-1]
    assert r.snapshot_species.shape[0] >= 1 + n // 50
    assert ck.stored_steps(path) == n


def test_run_without_escape_layer(tmp_path):
    path = run_point(_cfg(escape=False), Landscape(cell(4), 4), tmp_path,
                     log=lambda *a: None)
    r = ck.load_run(path, snapshots=False)
    assert not r.escape_layer and not r.is_escape.any()
    assert r.snapshot_species is None


def test_reads_checkpoints_without_system_and_detector(tmp_path):
    path = run_point(_cfg(), Landscape(cell(4), 4), tmp_path, log=lambda *a: None)
    with np.load(path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files
                if k not in ("system", "lattice") and not k.startswith("escape_w")
                and k not in ("escape_flagged", "escape_cooldown")}
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **data)
    r = ck.load_run(path)
    assert r.system == "MoNbTaW" and r.n_steps == 300
    resumed = ck.load_resume(path)
    assert resumed["traj"].n_steps == 300 and resumed["system"] == "MoNbTaW"


def test_complete_run_is_not_rerun(tmp_path):
    pred = Landscape(cell(4), 4)
    path = run_point(_cfg(), pred, tmp_path, log=lambda *a: None)
    stamp = path.stat().st_mtime_ns
    run_point(_cfg(), pred, tmp_path, log=lambda *a: None)
    assert path.stat().st_mtime_ns == stamp


def test_continuing_with_another_temperature_is_refused(tmp_path):
    pred = Landscape(cell(4), 4)
    run_point(_cfg(), pred, tmp_path, log=lambda *a: None)
    (tmp_path / "checkpoint_T600K_real0.npz").rename(
        tmp_path / "checkpoint_T700K_real0.npz")
    with pytest.raises(ValueError):
        run_point(_cfg(T_K=700.0, n_steps=400), pred, tmp_path,
                  log=lambda *a: None)


def test_seed_streams_are_distinct_and_reproducible():
    a, b = seed_streams(42)
    x = np.random.default_rng(a).random(4)
    y = np.random.default_rng(b).random(4)
    assert not np.allclose(x, y)
    np.testing.assert_array_equal(x, np.random.default_rng(seed_streams(42)[0]).random(4))


@pytest.mark.parametrize("path", sorted(glob.glob(str(resolve("configs/kmc/*.yaml")))))
def test_campaign_files_load(path, capsys):
    camp = load_campaign(path)
    assert camp.points and len({p.name for p in camp.points}) == len(camp.points)
    for p in camp.points:
        assert isinstance(p.seed, int) and p.n_steps > 0
        assert p.lattice_parameter_A == p.material.lattice_parameter_A
    assert campaign.main([path, "--dry-run"]) == 0
