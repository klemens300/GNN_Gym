"""
Inspect a partial-run checkpoint as if the run had finished.

Pointing this script at a checkpoint .npz (or at the output directory of a
running KMC simulation) reconstructs a full ``KMCResult`` from the latest
on-disk checkpoint, wraps it as a single-realisation ``EnsembleResult``, and
calls all the standard end-of-run writers from ``KMC.main``: the diffusion
CSVs, the tau_order CSV, the ExtXYZ trajectory, the event log and the
multi-page PDF summary.

The original run is undisturbed: the checkpoint file is opened read-only,
the running process keeps writing it via atomic tmp+rename, and the
generated outputs go into a separate ``_partial_<timestamp>/`` subdirectory
to avoid colliding with the run's eventual final outputs.

Usage:
    python -m KMC.process_checkpoint <checkpoint.npz>
    python -m KMC.process_checkpoint <run_directory>
    python -m KMC.process_checkpoint <path> --config <config.json>
    python -m KMC.process_checkpoint <path> --out <output_dir>

When called with a directory, every ``checkpoint_*.npz`` inside it is
processed; per-temperature ensembles are reconstructed from the
``checkpoint_T<...>K_real<k>.npz`` filename convention, so a multi-T or
multi-realisation run yields the same sweep_results layout that
``KMC.main`` produces at the end of a normal run.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make scipts/ importable when invoked with `python -m KMC.process_checkpoint`
_SCIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.checkpoint import load_run_checkpoint
from KMC.config import KMCConfig
from KMC.runner import EnsembleResult


# Filename pattern emitted by runner.run_ensemble
_CHECKPOINT_NAME_RE = re.compile(
    r"^checkpoint_T(?P<T>\d+(?:\.\d+)?)K_real(?P<real>\d+)\.npz$"
)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_checkpoints(arg_path: Path) -> List[Path]:
    """Return the list of checkpoint files implied by ``arg_path``.

    Accepts either a single .npz file or a directory containing multiple
    checkpoint files (one per realisation, possibly per temperature).
    """
    if arg_path.is_file():
        return [arg_path]
    if arg_path.is_dir():
        files = sorted(arg_path.glob("checkpoint_*.npz"))
        if not files:
            raise FileNotFoundError(
                f"No checkpoint files (checkpoint_*.npz) found in "
                f"{arg_path}."
            )
        return files
    raise FileNotFoundError(f"Path not found: {arg_path}")


def _resolve_config_path(
    explicit: Optional[Path], reference_checkpoint: Path
) -> Path:
    """Locate a KMCConfig JSON to associate with the checkpoint(s).

    Order of preference: explicit ``--config`` flag, then a sibling
    ``config_used.json`` (written by main.py at run start), then a sibling
    ``config.json`` (the user-authored input).
    """
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(
                f"--config path does not exist: {explicit}"
            )
        return explicit
    parent = reference_checkpoint.parent
    for candidate in ("config_used.json", "config.json"):
        p = parent / candidate
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not auto-detect a KMCConfig JSON for "
        f"{reference_checkpoint}. Pass --config <path> explicitly."
    )


def _parse_checkpoint_name(path: Path) -> Optional[Tuple[float, int]]:
    """Extract (temperature_K, realisation_idx) from a checkpoint filename.

    Returns None for files that do not follow the runner's naming
    convention; those are still loadable but cannot be grouped into a
    multi-realisation ensemble.
    """
    m = _CHECKPOINT_NAME_RE.match(path.name)
    if m is None:
        return None
    return float(m.group("T")), int(m.group("real"))


# ---------------------------------------------------------------------------
# Build sweep_results dict from checkpoint files
# ---------------------------------------------------------------------------

def _build_sweep_results(
    checkpoints: List[Path], config: KMCConfig
) -> Dict[float, EnsembleResult]:
    """Group checkpoints by temperature and assemble EnsembleResult objects.

    Files whose names cannot be parsed are treated as standalone
    realisations under the configured ``temperature_K``. This keeps the
    function permissive against custom user-renamed checkpoints.
    """
    by_T: Dict[float, List[Tuple[int, Path]]] = {}
    for p in checkpoints:
        parsed = _parse_checkpoint_name(p)
        if parsed is None:
            T = float(config.temperature_K)
            real = len(by_T.get(T, []))
        else:
            T, real = parsed
        by_T.setdefault(T, []).append((real, p))

    sweep_results: Dict[float, EnsembleResult] = {}
    for T, entries in by_T.items():
        # Sort within each temperature by realisation index for determinism
        entries.sort(key=lambda re_p: re_p[0])
        results = []
        seeds = []
        for real_idx, ckpt in entries:
            r = load_run_checkpoint(ckpt)
            results.append(r)
            # Seed value is informational here; we do not have the original
            # spawned seed for this realisation, so we tag with the
            # realisation index instead.
            seeds.append(int(real_idx))
            print(
                f"  [partial] {ckpt.name}: step {r.n_steps}, "
                f"sim t = {r.total_time_s:.3e} s"
            )
        # Each ensemble carries a config snapshot whose temperature_K is
        # set to the loaded value. KMCConfig is a dataclass; we use replace
        # so the original config is left intact.
        from dataclasses import replace
        cfg_T = replace(config, temperature_K=T)
        sweep_results[T] = EnsembleResult(
            config=cfg_T, results=results, seeds=seeds
        )
    return sweep_results


# ---------------------------------------------------------------------------
# Output writing (reuses the writers in KMC.main)
# ---------------------------------------------------------------------------

def _write_all_outputs(
    out_dir: Path,
    sweep_results: Dict[float, EnsembleResult],
    config: KMCConfig,
) -> None:
    """Run the same writer chain that KMC.main runs at the end of a run."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the (possibly time-overridden) config alongside the outputs
    # so the partial dump is self-contained.
    config.to_json(out_dir / "config_used.json")

    # Reuse the writers from main.py. They are private by convention but
    # stable in signature; importing them here keeps the output format
    # exactly aligned with end-of-run output.
    from KMC.main import (
        _write_diffusion_table,
        _write_diffusion_summary,
        _write_tau_order_table,
        _write_trajectories,
        _write_summary_pdfs,
    )

    p = _write_diffusion_table(out_dir, sweep_results)
    print(f"  -> {p}")
    p = _write_diffusion_summary(out_dir, sweep_results)
    print(f"  -> {p}")
    p = _write_tau_order_table(out_dir, sweep_results)
    print(f"  -> {p}")
    _write_trajectories(out_dir, sweep_results, config)
    _write_summary_pdfs(out_dir, sweep_results, config)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_path(
    arg_path: Path,
    output_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Path:
    """Discover checkpoints under ``arg_path`` and emit end-of-run outputs.

    Args:
        arg_path: a single checkpoint .npz file, or the directory holding
            one or more of them.
        output_dir: where to write the outputs. If None, a fresh
            ``_partial_<timestamp>/`` subdirectory next to the
            checkpoint(s) is used.
        config_path: explicit KMCConfig JSON; if None we look for
            ``config_used.json`` and then ``config.json`` next to the
            checkpoint.

    Returns:
        The output directory that received the partial run outputs.
    """
    arg_path = Path(arg_path)
    checkpoints = _resolve_checkpoints(arg_path)

    cfg_path = _resolve_config_path(
        Path(config_path) if config_path is not None else None,
        checkpoints[0],
    )
    config = KMCConfig.from_json(cfg_path)
    print(f"[partial] Loaded config from {cfg_path}")

    if output_dir is None:
        run_dir = (
            arg_path if arg_path.is_dir() else checkpoints[0].parent
        )
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = run_dir / f"_partial_{ts}"
    else:
        out_dir = Path(output_dir)
    print(f"[partial] Writing partial outputs to {out_dir}")

    sweep_results = _build_sweep_results(checkpoints, config)
    print(
        f"[partial] Reconstructed {sum(len(e.results) for e in sweep_results.values())} "
        f"realisation(s) across {len(sweep_results)} temperature(s)"
    )

    _write_all_outputs(out_dir, sweep_results, config)
    print("[partial] Done.")
    return out_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Process a KMC checkpoint into the standard end-of-run outputs "
            "(CSVs, ExtXYZ, event log, PDF summary)."
        )
    )
    parser.add_argument(
        "path",
        type=str,
        help=(
            "Path to a checkpoint .npz file or to a run directory "
            "containing one or more checkpoint_*.npz files."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the KMCConfig JSON. Default: auto-detect "
            "config_used.json or config.json next to the checkpoint."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output directory for partial outputs. Default: "
            "<run_dir>/_partial_<timestamp>/"
        ),
    )
    args = parser.parse_args(argv)

    process_path(
        Path(args.path),
        output_dir=Path(args.out) if args.out is not None else None,
        config_path=Path(args.config) if args.config is not None else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
