"""
KMC main entry point.

Usage:
    python -m KMC.main path/to/run_config.json

The config file is a JSON-serialised KMCConfig (see KMCConfig.to_json /
from_json). All run parameters live there: composition, supercell, temperatures
(single or sweep), GNN paths, output directory, sweep settings.

What this script does:
1. Loads the config and instantiates a GNNBarrierPredictor.
2. If `temperatures_K_sweep` is set: runs run_temperature_sweep, performs an
   Arrhenius fit per element, fits tau_order at each T, writes CSV tables
   to config.output_dir.
3. Otherwise: runs a single run_ensemble and writes summary tables.
"""

import os

# Set the PyTorch CUDA allocator config BEFORE the first torch import so it
# actually takes effect. `expandable_segments:True` reduces VRAM
# fragmentation on RTX 5090 / Blackwell during sustained GNN inference,
# where the per-step batch sizes vary slightly. Users can override the
# value by exporting PYTORCH_CUDA_ALLOC_CONF in the shell before invoking
# `python -m KMC.main`; setdefault leaves any existing value alone.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

# vGPU warmup: on some NVIDIA GRID profiles (e.g. GRID A100 / driver 525)
# the very first CUDA op must be a CPU->GPU copy and must happen BEFORE
# torch_geometric and any other CUDA-touching modules are imported.
# If it happens later (after torch_geometric init), model.to(device) and
# even torch.zeros(1).cuda() fail with
#   RuntimeError: CUDA driver error: operation not supported
# Doing the warmup here, before any heavy imports, fixes that. On
# bare-metal GPUs this is a no-op of ~1 ms.
import torch as _torch_warmup  # noqa: E402
if _torch_warmup.cuda.is_available():
    _ = _torch_warmup.zeros(1).cuda()
    _torch_warmup.cuda.synchronize()
del _torch_warmup

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# Make `scipts/` importable when running with `python -m KMC.main` from the
# repository root, or when invoked directly.
_SCIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.runner import (
    run_ensemble,
    run_temperature_sweep,
)
from KMC.barrier_predictor import (
    GNNBarrierPredictor,
    MockBarrierPredictor,
)
from KMC.analysis import (
    arrhenius_fit_per_element,
    tau_order_matrix_from_ensemble,
)
from KMC.trajectory_writer import (
    write_extxyz_trajectory,
    write_event_log,
)

# KMC.summary_pdf is imported lazily inside _write_summary_pdfs so that a
# missing matplotlib (rare on the cluster image) does not break main.py.


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_diffusion_table(
    out_dir: Path, sweep_results, max_lag=0.025, skip_lag=0.05
) -> Path:
    """One row per (T, realisation) with the per-realisation D in A^2/s."""
    path = out_dir / "diffusion_per_T.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "T_K", "realisation_idx", "seed", "D_A2_per_s"
        ])
        for T in sorted(sweep_results.keys()):
            ens = sweep_results[T]
            D_pack = ens.vacancy_diffusion_coefficient_ensemble(
                max_lag_fraction=max_lag,
                skip_lag_fraction=skip_lag,
            )
            for k, (seed, D) in enumerate(zip(ens.seeds, D_pack["per_seed"])):
                writer.writerow([f"{T:.6f}", k, seed, f"{D:.6e}"])
    return path


def _write_diffusion_summary(
    out_dir: Path, sweep_results, max_lag=0.025, skip_lag=0.05
) -> Path:
    """One row per T with median + MAD and mean + std D values."""
    path = out_dir / "diffusion_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "T_K", "n_realisations",
            "D_median_A2_per_s", "D_mad_A2_per_s",
            "D_mean_A2_per_s", "D_std_A2_per_s",
        ])
        for T in sorted(sweep_results.keys()):
            ens = sweep_results[T]
            D_pack = ens.vacancy_diffusion_coefficient_ensemble(
                max_lag_fraction=max_lag,
                skip_lag_fraction=skip_lag,
            )
            writer.writerow([
                f"{T:.6f}",
                ens.n_realizations,
                f"{D_pack['median']:.6e}",
                f"{D_pack['mad']:.6e}",
                f"{D_pack['mean']:.6e}",
                f"{D_pack['std']:.6e}",
            ])
    return path


def _write_arrhenius_table(out_dir: Path, fits) -> Path:
    """One row per element with D_0 and Q from the robust Arrhenius fit."""
    path = out_dir / "arrhenius_per_element.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "element",
            "D_0_A2_per_s", "D_0_uncertainty",
            "Q_eV", "Q_eV_uncertainty",
            "n_T_used",
        ])
        for sym, fit in fits.items():
            writer.writerow([
                sym,
                f"{fit.D_0:.6e}", f"{fit.D_0_uncertainty:.6e}",
                f"{fit.Q_eV:.6f}", f"{fit.Q_eV_uncertainty:.6f}",
                fit.n_temperatures_used,
            ])
    return path


def _write_trajectories(
    out_dir: Path, sweep_results, config: KMCConfig
) -> None:
    """For each temperature: write the first realisation's ExtXYZ trajectory
    and event log. Skipped silently if no snapshots were recorded."""
    if not config.write_trajectory_extxyz:
        return
    if config.snapshot_every_n_steps <= 0:
        print(
            "  [main] write_trajectory_extxyz=True but "
            "snapshot_every_n_steps=0 -> no trajectory written."
        )
        return
    for T in sorted(sweep_results.keys()):
        ens = sweep_results[T]
        if not ens.results:
            continue
        first_result = ens.results[0]
        # Tag filenames with the temperature so a sweep produces
        # distinguishable per-T trajectories.
        tag = f"T{T:.0f}K"
        traj_path = out_dir / f"trajectory_{tag}.extxyz"
        events_path = out_dir / f"events_{tag}.csv"
        try:
            n_frames = write_extxyz_trajectory(first_result, traj_path)
            print(f"  -> {traj_path}  ({n_frames} frames)")
        except ValueError as exc:
            print(f"  [main] Skipping ExtXYZ for T={T:.1f} K: {exc}")
            continue
        n_events = write_event_log(first_result, events_path)
        print(f"  -> {events_path}  ({n_events} events)")


def _maybe_load_diffusion_lookup(config: KMCConfig):
    """Try to instantiate a DiffusionLookup from the configured cache path.

    Returns ``None`` if the cache file does not exist or fails to load —
    that path is normal during early-stage runs where no per-composition
    diffusion data has been computed yet.
    """
    try:
        from KMC.analysis import DiffusionLookup
    except ImportError:
        return None
    cache_path = Path(config.diffusion_cache_path)
    if not cache_path.exists():
        return None
    try:
        return DiffusionLookup(
            cache_path,
            n_neighbors=config.lookup_n_neighbors,
            max_distance=config.lookup_max_distance,
        )
    except Exception as exc:
        print(
            f"  [main] WARNING: could not load DiffusionLookup from "
            f"{cache_path}: {exc}. Real-time rescaling will be skipped."
        )
        return None


def _write_summary_pdfs(
    out_dir: Path, sweep_results, config: KMCConfig
) -> None:
    """For each temperature: render a multi-page PDF run summary.

    Skipped silently when ``config.write_summary_pdf`` is False; logs a
    note if matplotlib is unavailable.
    """
    if not config.write_summary_pdf:
        return
    try:
        # Local import so a missing matplotlib only breaks PDF output, not
        # the rest of the run.
        from KMC.summary_pdf import write_run_summary_pdf as _write_pdf
    except ImportError as exc:
        print(
            f"  [main] WARNING: matplotlib not available ({exc}); "
            "skipping PDF summary."
        )
        return
    diffusion_lookup = _maybe_load_diffusion_lookup(config)
    for T in sorted(sweep_results.keys()):
        ens = sweep_results[T]
        if not ens.results:
            continue
        tag = f"T{T:.0f}K"
        pdf_path = out_dir / f"summary_{tag}.pdf"
        try:
            n_pages = _write_pdf(
                ens, pdf_path, config, diffusion_lookup=diffusion_lookup
            )
            print(f"  -> {pdf_path}  ({n_pages} pages)")
        except Exception as exc:
            print(f"  [main] PDF summary for T={T:.1f} K failed: {exc}")


def _write_tau_order_table(out_dir: Path, sweep_results) -> Path:
    """One row per (T, element-pair) with the fitted tau_order in seconds."""
    path = out_dir / "tau_order_per_T.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "T_K", "element_i", "element_j",
            "tau_order_s", "alpha_0", "alpha_inf", "rmse",
        ])
        for T in sorted(sweep_results.keys()):
            ens = sweep_results[T]
            try:
                taus = tau_order_matrix_from_ensemble(ens)
            except ValueError:
                continue
            for (sym_i, sym_j), fit in taus.items():
                if fit is None:
                    writer.writerow([
                        f"{T:.6f}", sym_i, sym_j, "", "", "", "",
                    ])
                else:
                    writer.writerow([
                        f"{T:.6f}", sym_i, sym_j,
                        f"{fit.tau_order_s:.6e}",
                        f"{fit.alpha_0:.6f}",
                        f"{fit.alpha_inf:.6f}",
                        f"{fit.rmse:.6e}",
                    ])
    return path


# ---------------------------------------------------------------------------
# Predictor selection
# ---------------------------------------------------------------------------

def _make_predictor(config: KMCConfig):
    """Decide between GNN and mock predictor based on whether the model exists."""
    model_path = Path(config.gnn_model_path)
    if model_path.exists():
        cache_mode = (
            "static-lattice cache (Phase 6 fast path)"
            if config.use_static_cache
            else "per-step rebuild (Phase 3 legacy path)"
        )
        print(f"  [predictor] GNN model: {model_path} | {cache_mode}")
        return GNNBarrierPredictor(
            config,
            use_static_cache=config.use_static_cache,
            inference_subbatch_size=config.inference_subbatch_size,
        )
    print(
        f"  [predictor] WARNING: GNN model not found at {model_path}; "
        "falling back to MockBarrierPredictor(constant_eV=1.0). "
        "This is only useful for smoke testing the runner."
    )
    return MockBarrierPredictor(constant_eV=1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a KMC simulation (single ensemble or T-sweep) "
                    "from a JSON config."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to a KMCConfig JSON file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each realisation from its existing per-realisation "
             "checkpoint (bit-identical continuation). Realisations "
             "without a matching checkpoint are started fresh.",
    )
    parser.add_argument(
        "--max_steps_this_session",
        type=int,
        default=None,
        help="Stop the BKL loop voluntarily after this many additional "
             "steps in the current Python process (final checkpoint is "
             "written before exit). Used by run_until_done.sh to bound "
             "a process's lifetime below the empirical crash window. "
             "Exit code 99 indicates a session-cap exit; 0 means the "
             "full config target was reached.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        return 2

    print(f"[main] Loading config from {config_path}")
    config = KMCConfig.from_json(config_path)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist the resolved config alongside the outputs
    config.to_json(out_dir / "config_used.json")

    print("[main] Building predictor...")
    predictor = _make_predictor(config)

    sweep = config.temperatures_K_sweep
    if sweep:
        # ---- Arrhenius sweep ----
        print(f"[main] Temperature sweep: {sweep} K, "
              f"{config.n_realizations_per_T} realisations / T")
        if args.resume:
            print(
                "  [main] WARNING: --resume is not supported in sweep mode "
                "yet; sweep starts fresh."
            )
        t0 = time.time()
        sweep_results = run_temperature_sweep(
            config,
            predictor,
            temperatures_K=sweep,
            n_realizations=config.n_realizations_per_T,
            snapshot_every_n_steps=config.snapshot_every_n_steps,
            progress_callback=(
                lambda k, T, ens: print(
                    f"  [sweep] T={T:.1f} K done, "
                    f"{ens.n_realizations} realisations"
                )
            ),
        )
        print(f"[main] Sweep finished in {time.time() - t0:.1f} s")

        p = _write_diffusion_table(out_dir, sweep_results)
        print(f"  -> {p}")
        p = _write_diffusion_summary(out_dir, sweep_results)
        print(f"  -> {p}")
        try:
            fits = arrhenius_fit_per_element(sweep_results)
            p = _write_arrhenius_table(out_dir, fits)
            print(f"  -> {p}")
        except ValueError as exc:
            print(f"  [main] Skipping Arrhenius fit: {exc}")
        p = _write_tau_order_table(out_dir, sweep_results)
        print(f"  -> {p}")
        _write_trajectories(out_dir, sweep_results, config)
        _write_summary_pdfs(out_dir, sweep_results, config)
        print("[main] Done.")
        return 0

    # ---- Single ensemble (no sweep) ----
    print(f"[main] Single ensemble: T={config.temperature_K:.1f} K, "
          f"{config.n_realizations_per_T} realisations"
          + (" (resume)" if args.resume else "")
          + (
              f" [session cap: {args.max_steps_this_session} steps]"
              if args.max_steps_this_session
              else ""
          ))
    t0 = time.time()
    ensemble = run_ensemble(
        config,
        predictor,
        n_realizations=config.n_realizations_per_T,
        snapshot_every_n_steps=config.snapshot_every_n_steps,
        resume=args.resume,
        max_steps_this_session=args.max_steps_this_session,
    )
    print(f"[main] Ensemble finished in {time.time() - t0:.1f} s")

    # Detect session-cap exit. When --max_steps_this_session is in use
    # and at least one realisation has not yet reached config.n_steps,
    # we exit with code 99 to signal the wrapper that more sessions are
    # needed. Code 0 means every realisation reached the target.
    if args.max_steps_this_session is not None and config.n_steps is not None:
        below_target = [
            r.n_steps < config.n_steps for r in ensemble.results
        ]
        if any(below_target):
            steps_done = [int(r.n_steps) for r in ensemble.results]
            print(
                f"[main] Session cap exit: realisations at "
                f"{steps_done} / target {config.n_steps}. "
                "Skipping output writers; the wrapper should re-invoke "
                "with --resume.",
                flush=True,
            )
            return 99

    # Wrap as a single-temperature "sweep" so the same writers can be reused
    sweep_results = {config.temperature_K: ensemble}
    p = _write_diffusion_table(out_dir, sweep_results)
    print(f"  -> {p}")
    p = _write_diffusion_summary(out_dir, sweep_results)
    print(f"  -> {p}")
    p = _write_tau_order_table(out_dir, sweep_results)
    print(f"  -> {p}")
    _write_trajectories(out_dir, sweep_results, config)
    _write_summary_pdfs(out_dir, sweep_results, config)
    print("[main] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
