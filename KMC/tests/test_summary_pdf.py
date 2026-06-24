"""
Smoke-tests for the run-summary PDF generator.

We deliberately do not parse the rendered PDF — matplotlib's PdfPages output
varies across versions. The tests assert:
    * the file exists,
    * it is a valid PDF (starts with the %PDF magic and is non-trivial in
      size),
    * the number-of-pages return value is at least 1,
    * the entry point runs without exception both with and without a
      DiffusionLookup attached.

Run from anywhere with pytest:

    cd /path/to/GNN_Gym
    pytest -v KMC/tests/test_summary_pdf.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make `scipts/` importable so that `from KMC.* import ...` works
_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.barrier_predictor import MockBarrierPredictor
from KMC.runner import run_ensemble


# Skip the whole module if matplotlib is unavailable; the smoke test cannot
# run without it but the rest of the test suite is unaffected.
matplotlib = pytest.importorskip("matplotlib")
from KMC.summary_pdf import write_run_summary_pdf  # noqa: E402
from KMC.analysis import DiffusionLookup, CachedDiffusionEntry  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_ensemble_config():
    """4x4x4 BCC, equiatomic Mo-Nb-Ta-W, slabs initial state, snapshots on."""
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=4,
        lattice_parameter_A=3.22,
        random_seed=11,
        temperature_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=200,
        snapshot_every_n_steps=20,
        initial_state_strategy="slabs",
        slab_axis="x",
        slab_elements=["Mo", "Nb", "Ta", "W"],
        # Stay in the temporary directory so the test does not touch a real
        # diffusion cache on disk.
        diffusion_cache_path="/tmp/__test_no_cache_should_not_exist__.json",
    )


@pytest.fixture
def mock_ensemble(small_ensemble_config):
    """Run a tiny mock-predictor ensemble for the smoke test."""
    predictor = MockBarrierPredictor(constant_eV=1.0)
    return run_ensemble(
        small_ensemble_config,
        predictor,
        n_realizations=2,
        snapshot_every_n_steps=small_ensemble_config.snapshot_every_n_steps,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _assert_valid_pdf(path: Path):
    assert path.exists(), f"PDF was not created at {path}"
    assert path.stat().st_size > 1024, (
        f"PDF at {path} is suspiciously small ({path.stat().st_size} bytes)"
    )
    with open(path, "rb") as f:
        head = f.read(5)
    assert head == b"%PDF-", f"File at {path} does not look like a PDF"


def test_pdf_runs_without_diffusion_lookup(
    tmp_path, small_ensemble_config, mock_ensemble
):
    """Without a cache, all time axes stay in sim time; PDF still renders."""
    out = tmp_path / "summary_no_cache.pdf"
    n_pages = write_run_summary_pdf(
        mock_ensemble, out, small_ensemble_config, diffusion_lookup=None
    )
    _assert_valid_pdf(out)
    # Overview page is always emitted; SRO pages emit because snapshots > 0.
    assert n_pages >= 3, (
        f"Expected at least 3 pages with snapshots, got {n_pages}"
    )


def test_pdf_runs_with_diffusion_lookup(
    tmp_path, small_ensemble_config, mock_ensemble
):
    """A populated lookup should yield real-time rescaling without errors."""
    cache_path = tmp_path / "diffusion_cache.json"
    # Seed three slightly off-equiatomic neighbours so the kNN (k=3) lookup
    # has enough points to interpolate at the equiatomic query.
    composition = dict(small_ensemble_config.composition)
    base = CachedDiffusionEntry(
        composition=composition,
        E_f_V_eV=2.5,
        lattice_parameter_A=3.22,
        jump_distance_A=2.79,
        nu_0_Hz=1e13,
        n_atoms_supercell=128,
    )
    lookup = DiffusionLookup(
        cache_path,
        n_neighbors=1,
        max_distance=0.20,
    )
    lookup.add(base)

    out = tmp_path / "summary_with_cache.pdf"
    n_pages = write_run_summary_pdf(
        mock_ensemble, out, small_ensemble_config, diffusion_lookup=lookup
    )
    _assert_valid_pdf(out)
    assert n_pages >= 3


def test_pdf_handles_missing_snapshots(tmp_path, small_ensemble_config):
    """Without snapshots the SRO pages must be skipped, not crash."""
    predictor = MockBarrierPredictor(constant_eV=1.0)
    ensemble = run_ensemble(
        small_ensemble_config,
        predictor,
        n_realizations=1,
        snapshot_every_n_steps=0,  # no snapshots at all
    )
    out = tmp_path / "summary_no_snapshots.pdf"
    n_pages = write_run_summary_pdf(
        ensemble, out, small_ensemble_config, diffusion_lookup=None
    )
    _assert_valid_pdf(out)
    # Only the overview page is rendered when there are no snapshots.
    assert n_pages == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
