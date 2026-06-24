"""
Phase-6 micro-benchmark: legacy vs. static-cache GNN predictor.

Measures wall-clock per BKL step on a fixed random state, comparing:
  - GNNBarrierPredictor(use_static_cache=False)  (legacy per-step rebuild)
  - GNNBarrierPredictor(use_static_cache=True)   (Phase-6 fast path)

Run from /home/klemens/doctor/gnn_kmc/scipts:

    python -m KMC.demos.benchmark_predictor
    python -m KMC.demos.benchmark_predictor --supercells 4,6,8 --warmup 5 --measure 30

The output is printed in a single table; no files are written.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure `scipts/` is importable when run as a script
_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.state import KMCState


def _make_config(supercell_size: int) -> KMCConfig:
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=supercell_size,
        lattice_parameter_A=3.22,
        random_seed=42,
        temperature_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=1,
    )


def _bench_predictor(predictor, state, n_warmup: int, n_measure: int) -> dict:
    """Return per-step timings (mean / median / std) in seconds.

    For each measurement step, we feed the same neighbour-index list from
    the current vacancy. Between measurements we also call swap_vacancy so
    the species pattern around the vacancy actually changes - otherwise
    the GPU might cache results that are unrealistically reused.
    """
    rng = np.random.default_rng(0)

    # Warmup (kernel compilation, allocator priming, first cache build)
    for _ in range(n_warmup):
        nn = state.get_neighbor_atom_indices()
        predictor.get_forward_barriers_batch(state, nn)
        state.swap_vacancy(int(rng.choice(nn)))

    # GPU sync helper (no-op on CPU); avoids attributing async time wrongly.
    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    except Exception:
        sync = lambda: None

    timings = []
    for _ in range(n_measure):
        nn = state.get_neighbor_atom_indices()
        sync()
        t0 = time.perf_counter()
        predictor.get_forward_barriers_batch(state, nn)
        sync()
        timings.append(time.perf_counter() - t0)
        state.swap_vacancy(int(rng.choice(nn)))

    arr = np.asarray(timings)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "std_s": float(arr.std()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GNNBarrierPredictor: legacy vs static-cache."
    )
    parser.add_argument(
        "--supercells", type=str, default="4,6",
        help="Comma-separated supercell sizes to benchmark (default: 4,6).",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup steps before timing (default 5).",
    )
    parser.add_argument(
        "--measure", type=int, default=20,
        help="Number of timed steps per predictor (default 20).",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.supercells.split(",")]

    # Lazy import so the module can be imported without torch
    from KMC.barrier_predictor import GNNBarrierPredictor

    print(f"\nPhase-6 predictor benchmark "
          f"(warmup={args.warmup}, measure={args.measure})")
    print("=" * 78)
    header = (
        f"{'supercell':>9} | {'n_sites':>7} | "
        f"{'legacy ms/step':>14} | {'fast ms/step':>13} | {'speedup':>7}"
    )
    print(header)
    print("-" * 78)

    for supercell in sizes:
        cfg = _make_config(supercell_size=supercell)
        if not Path(cfg.gnn_model_path).exists():
            print(f"  skipping {supercell}: model missing at "
                  f"{cfg.gnn_model_path}")
            continue

        # Two predictors share no state - each gets its own GraphBuilder /
        # model load. We only need the fast one's cache to avoid pollution.
        legacy = GNNBarrierPredictor(cfg, use_static_cache=False)
        fast = GNNBarrierPredictor(cfg, use_static_cache=True)

        state_legacy = KMCState.from_random_composition(cfg)
        state_fast = state_legacy.copy()

        legacy_t = _bench_predictor(
            legacy, state_legacy, args.warmup, args.measure
        )
        fast_t = _bench_predictor(
            fast, state_fast, args.warmup, args.measure
        )

        speedup = legacy_t["median_s"] / max(fast_t["median_s"], 1e-12)
        print(
            f"{supercell:>9} | {cfg.n_sites:>7} | "
            f"{legacy_t['median_s']*1000:>14.2f} | "
            f"{fast_t['median_s']*1000:>13.2f} | "
            f"{speedup:>6.2f}x"
        )

        # Free the legacy predictor's GPU resources before the next size
        del legacy
        del fast
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
