"""
Smoke-tests for the ``process_checkpoint`` module.

The tests run a tiny mock-predictor ensemble that writes checkpoints to
disk, then call ``process_path(...)`` to reproduce the standard end-of-run
outputs from those checkpoints. We only assert that the expected files
exist and are non-trivial in size; the format-level correctness of the
CSVs / PDF / ExtXYZ is covered by the dedicated tests for each writer.

Run from /home/klemens/doctor/gnn_kmc/scipts:

    pytest -v KMC/tests/test_process_checkpoint.py
"""

import json
import sys
from pathlib import Path

import pytest

# Make `scipts/` importable so `from KMC.* import ...` works
_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.barrier_predictor import MockBarrierPredictor
from KMC.runner import run_ensemble
from KMC.process_checkpoint import process_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_run_with_checkpoints(tmp_path: Path) -> Path:
    """Run a tiny ensemble that produces checkpoint files. Returns run_dir."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=4,
        lattice_parameter_A=3.22,
        random_seed=23,
        temperature_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=300,
        snapshot_every_n_steps=50,
        initial_state_strategy="slabs",
        slab_axis="x",
        slab_elements=["Mo", "Nb", "Ta", "W"],
        output_dir=str(run_dir),
        checkpoint_every_n_steps=100,
        write_summary_pdf=True,
        write_trajectory_extxyz=True,
        # Stay isolated from any real cache file on disk.
        diffusion_cache_path=str(tmp_path / "no_cache.json"),
    )
    # Persist the config so process_checkpoint can auto-find it.
    cfg.to_json(run_dir / "config.json")

    predictor = MockBarrierPredictor(constant_eV=1.0)
    run_ensemble(
        cfg, predictor,
        n_realizations=1,
        snapshot_every_n_steps=cfg.snapshot_every_n_steps,
    )

    # The runner writes checkpoint_T1500K_real0.npz; sanity-check that.
    expected = run_dir / "checkpoint_T1500K_real0.npz"
    assert expected.exists(), (
        f"Expected checkpoint not produced at {expected}. "
        f"Files in run_dir: {list(run_dir.iterdir())}"
    )
    return run_dir


def _expected_files_in(out_dir: Path):
    """Files that ``process_path`` should produce for a single-T run."""
    return [
        out_dir / "config_used.json",
        out_dir / "diffusion_per_T.csv",
        out_dir / "diffusion_summary.csv",
        out_dir / "tau_order_per_T.csv",
        out_dir / "trajectory_T1500K.extxyz",
        out_dir / "events_T1500K.csv",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_process_single_checkpoint_file(tmp_path):
    """Pointing at a single .npz produces the full end-of-run output set."""
    run_dir = _seed_run_with_checkpoints(tmp_path)
    ckpt = run_dir / "checkpoint_T1500K_real0.npz"

    out_dir = tmp_path / "partial"
    returned = process_path(ckpt, output_dir=out_dir)
    assert returned == out_dir

    for path in _expected_files_in(out_dir):
        assert path.exists(), f"Missing expected output: {path}"
        assert path.stat().st_size > 0, f"Output is empty: {path}"

    # PDF only present when matplotlib is available; check optionally.
    pdf = out_dir / "summary_T1500K.pdf"
    if pdf.exists():
        assert pdf.stat().st_size > 1024


def test_process_directory_of_checkpoints(tmp_path):
    """Passing a directory should auto-discover checkpoints."""
    run_dir = _seed_run_with_checkpoints(tmp_path)

    out_dir = tmp_path / "partial_from_dir"
    process_path(run_dir, output_dir=out_dir)

    for path in _expected_files_in(out_dir):
        assert path.exists(), f"Missing expected output: {path}"


def test_default_output_dir_is_partial_timestamped(tmp_path):
    """Without --out the script should create a _partial_<ts>/ subdir."""
    run_dir = _seed_run_with_checkpoints(tmp_path)
    ckpt = run_dir / "checkpoint_T1500K_real0.npz"

    out_dir = process_path(ckpt)
    assert out_dir.exists()
    assert out_dir.parent == run_dir
    assert out_dir.name.startswith("_partial_")


def test_outputs_use_loaded_step_count(tmp_path):
    """The diffusion CSV's row count reflects the saved partial step count.

    After 300 steps and checkpoints every 100, the latest checkpoint is at
    step 300 (the run finished). One realisation -> exactly 1 row in
    diffusion_per_T.csv (one row per realisation per temperature).
    """
    run_dir = _seed_run_with_checkpoints(tmp_path)
    ckpt = run_dir / "checkpoint_T1500K_real0.npz"

    out_dir = tmp_path / "partial_count"
    process_path(ckpt, output_dir=out_dir)

    csv_path = out_dir / "diffusion_per_T.csv"
    text = csv_path.read_text()
    # header + 1 data row
    assert text.count("\n") == 2, f"Unexpected row count in {csv_path}:\n{text}"


def test_explicit_config_arg(tmp_path):
    """--config <path> should override auto-detection."""
    run_dir = _seed_run_with_checkpoints(tmp_path)
    ckpt = run_dir / "checkpoint_T1500K_real0.npz"

    # Move the auto-discoverable config out of the way and pass an explicit
    # path from elsewhere. We just copy the existing config to a new home.
    explicit_cfg = tmp_path / "elsewhere" / "my_config.json"
    explicit_cfg.parent.mkdir(parents=True, exist_ok=True)
    explicit_cfg.write_text((run_dir / "config.json").read_text())
    (run_dir / "config.json").unlink()

    out_dir = tmp_path / "partial_explicit"
    process_path(ckpt, output_dir=out_dir, config_path=explicit_cfg)

    assert (out_dir / "config_used.json").exists()
    cfg_dump = json.loads((out_dir / "config_used.json").read_text())
    assert cfg_dump["temperature_K"] == 1500.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
