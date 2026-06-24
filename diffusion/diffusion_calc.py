"""
Composition-space sweep for vacancy diffusion in BCC alloys.

Enumerates the full simplex of compositions on a regular grid (step size
configurable), and for each moving atom in config.elements computes the
diffusion data at every composition point. Supports resume from CSV.

Usage:
    # Default: full sweep over all elements as moving atoms
    python diffusion_calc.py

    # 10% step instead of default 5%
    python diffusion_calc.py --step 0.10

    # Only specific moving atoms
    python diffusion_calc.py --moving-atoms W Mo

    # Force re-run from scratch
    python diffusion_calc.py --no-resume

    # Skip confirmation prompt
    python diffusion_calc.py --yes
"""

import csv
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from diffusion.config import DiffusionConfig
from diffusion.results import DiffusionResult
from diffusion.diffusion_oracle import DiffusionOracle


# ------------------------------------------------------------------ #
#  Composition simplex enumeration                                     #
# ------------------------------------------------------------------ #

def enumerate_composition_simplex(
    elements: List[str], step: float
) -> List[Dict[str, float]]:
    """
    Enumerate all compositions on a regular simplex grid.

    Uses exact integer partitions to avoid floating-point drift:
    with n = round(1/step), enumerate all (k_1, ..., k_E) where
    k_i >= 0 and sum(k_i) = n, then convert to fractions k_i / n.

    Parameters
    ----------
    elements : list of str
        Element symbols, e.g. ["Mo", "Nb", "Ta", "W"].
    step : float
        Grid spacing as a fraction (e.g. 0.05 for 5%).

    Returns
    -------
    list of dict
        Each dict maps element symbol to composition fraction.
        All fractions sum exactly to 1.0.
    """
    n = int(round(1.0 / step))
    n_elem = len(elements)
    compositions = []

    # Recursive integer-partition generator: distribute n units across n_elem bins
    def _recurse(idx: int, remaining: int, current: List[int]):
        if idx == n_elem - 1:
            # Last element gets whatever is left
            current.append(remaining)
            compositions.append(
                {el: k / n for el, k in zip(elements, current)}
            )
            current.pop()
            return
        for k in range(remaining + 1):
            current.append(k)
            _recurse(idx + 1, remaining - k, current)
            current.pop()

    _recurse(0, n, [])
    return compositions


def composition_to_string(composition: Dict[str, float]) -> str:
    """
    Convert composition dict to canonical string (matches oracle format).

    Example: {"Mo": 0.5, "W": 0.5, ...} -> "Mo50Nb0Ta0W50"
    Sorted alphabetically by element symbol for stable string keys.
    """
    return "".join(
        f"{el}{int(round(frac * 100))}"
        for el, frac in sorted(composition.items())
    )


# ------------------------------------------------------------------ #
#  Resume: read existing CSV                                           #
# ------------------------------------------------------------------ #

def load_completed_runs(csv_path: Path) -> Dict[Tuple[str, str], int]:
    """
    Count completed runs per (composition_string, moving_atom) from CSV.

    Returns
    -------
    dict
        Maps (comp_string, moving_atom) -> number of finished runs.
        Empty dict if CSV does not exist.
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    if not csv_path.exists():
        return counts

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["composition_string"], row["moving_atom"])
            counts[key] += 1
    return counts


def estimate_avg_runtime(csv_path: Path) -> Optional[float]:
    """
    Compute mean total runtime per run from CSV (in seconds).

    Returns None if CSV is missing or empty.
    """
    if not csv_path.exists():
        return None

    times = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                times.append(float(row["time_total_s"]))
            except (KeyError, ValueError):
                continue
    if not times:
        return None
    return sum(times) / len(times)


# ------------------------------------------------------------------ #
#  Pre-flight summary and confirmation                                 #
# ------------------------------------------------------------------ #

def _format_duration(seconds: float) -> str:
    """Render seconds as a compact human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


def _print_preflight(
    config: DiffusionConfig,
    moving_atoms: List[str],
    compositions: List[Dict[str, float]],
    completed: Dict[Tuple[str, str], int],
    n_runs: int,
) -> Tuple[int, int]:
    """
    Print pre-flight summary. Returns (total_planned, total_pending).
    """
    n_comp = len(compositions)
    total_planned = n_comp * len(moving_atoms) * n_runs

    # Count pending: for each (comp, moving_atom) pair, missing = max(0, n_runs - done)
    total_pending = 0
    for ma in moving_atoms:
        for comp in compositions:
            comp_str = composition_to_string(comp)
            done = completed.get((comp_str, ma), 0)
            total_pending += max(0, n_runs - done)

    total_done = total_planned - total_pending

    print("=" * 70)
    print("DIFFUSION COMPOSITION-SPACE SWEEP")
    print("=" * 70)
    print(config.summary())
    print("-" * 70)
    print(f"Moving atoms:       {moving_atoms}")
    print(f"Composition points: {n_comp}  (step = {config.composition_step*100:.1f}%)")
    print(f"Runs per point:     {n_runs}")
    print(f"Total runs planned: {total_planned}")
    print(f"  Already done:     {total_done}")
    print(f"  Pending:          {total_pending}")

    avg = estimate_avg_runtime(Path(config.csv_path))
    if avg is not None and total_pending > 0:
        eta = avg * total_pending
        print(f"Avg runtime/run:    {_format_duration(avg)}  (from CSV history)")
        print(f"Estimated ETA:      {_format_duration(eta)}")
    elif total_pending > 0:
        print(f"Avg runtime/run:    unknown (no CSV history yet)")
    print("=" * 70)

    return total_planned, total_pending


def _ask_confirmation(pending: int, threshold: int, auto_yes: bool) -> bool:
    """Prompt user if pending count exceeds threshold."""
    if auto_yes or pending <= threshold:
        return True
    print(
        f"\n⚠  {pending} pending runs exceeds confirmation threshold ({threshold})."
    )
    answer = input("Proceed? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


# ------------------------------------------------------------------ #
#  Main sweep                                                          #
# ------------------------------------------------------------------ #

def calculate_composition_space(
    moving_atoms: Optional[List[str]] = None,
    n_runs: Optional[int] = None,
    resume: bool = True,
    auto_yes: bool = False,
    **config_kwargs,
) -> List[DiffusionResult]:
    """
    Sweep the full composition simplex for each moving atom.

    Outer loop: moving atoms (so each element finishes completely before
    the next starts). Inner loop: composition points on the simplex grid.
    For each (composition, moving_atom) pair, runs n_runs independent
    configurations. Already-finished runs (read from CSV) are skipped
    when resume=True; only the missing runs are computed.

    Parameters
    ----------
    moving_atoms : list of str, optional
        Subset of elements to use as moving atoms. Default: all elements
        in config.elements.
    n_runs : int, optional
        Number of independent runs per (composition, moving_atom).
        Default: config.runs_per_composition.
    resume : bool
        If True, read existing CSV and only compute missing runs.
    auto_yes : bool
        Skip the confirmation prompt for large sweeps.
    **config_kwargs
        Override any DiffusionConfig parameter.

    Returns
    -------
    list of DiffusionResult
        All successful results from this invocation (does not include
        previously-finished runs loaded from CSV).
    """
    config = DiffusionConfig(**config_kwargs)

    if moving_atoms is None:
        moving_atoms = list(config.elements)
    for ma in moving_atoms:
        if ma not in config.elements:
            raise ValueError(
                f"Moving atom '{ma}' not in config.elements {config.elements}"
            )

    if n_runs is None:
        n_runs = config.runs_per_composition

    # Build the simplex grid
    compositions = enumerate_composition_simplex(
        config.elements, config.composition_step
    )

    # Load resume state
    completed = load_completed_runs(Path(config.csv_path)) if resume else {}

    # Pre-flight summary
    total_planned, total_pending = _print_preflight(
        config, moving_atoms, compositions, completed, n_runs
    )

    if total_pending == 0:
        print("\nNothing to do — all runs already complete.")
        return []

    if not _ask_confirmation(total_pending, config.confirm_threshold, auto_yes):
        print("Aborted by user.")
        return []

    # Run the sweep
    results: List[DiffusionResult] = []
    sweep_start = time.time()
    runs_done_this_session = 0

    try:
        with DiffusionOracle(config) as oracle:
            for ma_idx, moving_atom in enumerate(moving_atoms, start=1):
                print(f"\n{'#'*70}")
                print(
                    f"#  MOVING ATOM: {moving_atom}  "
                    f"({ma_idx}/{len(moving_atoms)})"
                )
                print(f"{'#'*70}")

                for comp_idx, composition in enumerate(compositions, start=1):
                    comp_str = composition_to_string(composition)
                    done = completed.get((comp_str, moving_atom), 0)
                    missing = max(0, n_runs - done)

                    if missing == 0:
                        # Fully complete — skip silently in dense sweeps
                        continue

                    print(
                        f"\n[{moving_atom} | {comp_idx}/{len(compositions)}] "
                        f"{comp_str}  "
                        f"(done: {done}/{n_runs}, todo: {missing})"
                    )

                    for r in range(missing):
                        result = oracle.calculate(
                            composition, moving_atom=moving_atom
                        )
                        runs_done_this_session += 1

                        if result is not None:
                            results.append(result)
                            if config.save_csv:
                                oracle.save_to_csv(result)
                            # Update in-memory completion counter
                            completed[(comp_str, moving_atom)] = (
                                completed.get((comp_str, moving_atom), 0) + 1
                            )

                        # Live global progress
                        elapsed = time.time() - sweep_start
                        avg = elapsed / runs_done_this_session
                        remaining_runs = total_pending - runs_done_this_session
                        eta = avg * remaining_runs
                        print(
                            f"    Session: {runs_done_this_session}/{total_pending} | "
                            f"OK: {len(results)} | "
                            f"Elapsed: {_format_duration(elapsed)} | "
                            f"ETA: {_format_duration(eta)}"
                        )

    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user. Partial progress saved to CSV.")
        print(f"   Completed this session: {len(results)} runs")
        print(f"   Resume by re-running the same command.")
        return results

    # Final summary
    total_time = time.time() - sweep_start
    print(f"\n{'='*70}")
    print("SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Successful runs:  {len(results)} / {runs_done_this_session} attempted")
    print(f"Total time:       {_format_duration(total_time)}")
    if runs_done_this_session > 0:
        print(f"Avg per run:      {_format_duration(total_time / runs_done_this_session)}")
    print(f"CSV:              {config.csv_path}")

    return results


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Sweep diffusion calculations over the full composition simplex"
    )
    parser.add_argument(
        "--step", type=float, default=None,
        help="Composition grid spacing as fraction (e.g. 0.05 for 5%%). "
             "Overrides config.composition_step.",
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Number of independent runs per (composition, moving_atom). "
             "Overrides config.runs_per_composition.",
    )
    parser.add_argument(
        "--moving-atoms", type=str, nargs="+", default=None,
        help="Subset of elements to use as moving atoms (default: all in config).",
    )
    parser.add_argument(
        "--elements", type=str, nargs="+", default=None,
        help="Override the element list in config (e.g. --elements Mo W).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not read existing CSV; recompute everything.",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt for large sweeps.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable CSV and structure saving (debug only).",
    )
    args = parser.parse_args()

    config_kwargs = {}
    if args.step is not None:
        config_kwargs["composition_step"] = args.step
    if args.runs is not None:
        config_kwargs["runs_per_composition"] = args.runs
    if args.elements is not None:
        config_kwargs["elements"] = args.elements
    if args.no_save:
        config_kwargs["save_csv"] = False
        config_kwargs["save_structures"] = False

    calculate_composition_space(
        moving_atoms=args.moving_atoms,
        n_runs=args.runs,
        resume=not args.no_resume,
        auto_yes=args.yes,
        **config_kwargs,
    )


if __name__ == "__main__":
    main()
