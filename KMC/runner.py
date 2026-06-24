"""
Ensemble runner: independent KMC trajectories at the same temperature and
composition, combined into one EnsembleResult that carries ensemble
statistics for all observables.

For Arrhenius-style temperature sweeps, call run_ensemble multiple times
with different KMCConfig.temperature_K values and combine the results
externally (Phase 5c).

Default behaviour:
- Each realisation gets its own seed, derived from config.random_seed via
  numpy's SeedSequence.spawn(). The state-initialisation rng and the
  engine rng of one realisation are spawned from the same root, so a given
  ensemble seed reproduces exactly the same trajectory.
- Realisations are run sequentially; the GNN model (or any other
  predictor) is loaded once outside this function and reused.

A note on aggregation statistics:
- For derived diffusion coefficients we report BOTH mean+std AND
  median+MAD. HEA-barrier distributions are typically not Gaussian, so
  median+MAD are the more robust summary statistics; mean+std are kept
  for compatibility / quick sanity checks.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from KMC.config import KMCConfig
from KMC.engine import run
from KMC.observables import (
    tracer_msd_per_element,
    warren_cowley_sro_trajectory,
)
from KMC.result import KMCResult
from KMC.state import KMCState


# ---------------------------------------------------------------------------
# Initial-state factory
# ---------------------------------------------------------------------------

def _build_initial_state(
    config: KMCConfig,
    rng: np.random.Generator,
) -> KMCState:
    """Dispatch on config.initial_state_strategy."""
    strategy = config.initial_state_strategy
    if strategy == "random":
        return KMCState.from_random_composition(config, rng=rng)
    elif strategy == "bicrystal":
        return KMCState.from_bicrystal(config, rng=rng)
    elif strategy == "slabs":
        return KMCState.from_slabs(config, rng=rng)
    elif strategy == "custom":
        if config.custom_symbols is None or config.custom_vacancy_index is None:
            raise ValueError(
                "initial_state_strategy='custom' requires "
                "config.custom_symbols and config.custom_vacancy_index"
            )
        return KMCState.from_symbol_array(
            config, config.custom_symbols, config.custom_vacancy_index
        )
    else:
        raise ValueError(
            f"Unknown initial_state_strategy: {strategy!r}"
        )


# ---------------------------------------------------------------------------
# EnsembleResult container
# ---------------------------------------------------------------------------

@dataclass
class EnsembleResult:
    """Container holding all KMCResults of one ensemble run plus aggregates."""

    config: KMCConfig
    results: List[KMCResult]
    seeds: List[int] = field(default_factory=list)

    # ------- Properties -------

    @property
    def n_realizations(self) -> int:
        return len(self.results)

    @property
    def temperature_K(self) -> float:
        return self.config.temperature_K

    # ------- Vacancy-MSD aggregate -------

    def vacancy_msd_ensemble(
        self, max_lag_fraction: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """Time-averaged vacancy MSD averaged across the ensemble.

        Returns dict with keys:
            'lag_t_mean': mean lag-time array (across realisations)
            'mean':       MSD mean across realisations, shape [n_lags]
            'std':        MSD std across realisations
            'per_seed':   shape [n_realizations, n_lags] — raw per-seed data
        """
        msd_list = []
        lag_t_list = []
        for r in self.results:
            msd, lag_t = r.time_averaged_vacancy_msd(max_lag_fraction)
            msd_list.append(msd)
            lag_t_list.append(lag_t)
        msd_arr = np.stack(msd_list, axis=0)
        lag_t_arr = np.stack(lag_t_list, axis=0)
        return {
            "lag_t_mean": lag_t_arr.mean(axis=0),
            "mean": msd_arr.mean(axis=0),
            "std": msd_arr.std(axis=0),
            "per_seed": msd_arr,
        }

    # ------- D aggregate (mean/std + median/MAD per HEA recommendation) -------

    def vacancy_diffusion_coefficient_ensemble(
        self,
        max_lag_fraction: float = 0.025,
        skip_lag_fraction: float = 0.05,
    ) -> Dict[str, float]:
        """Per-seed D plus ensemble mean/std and median/MAD.

        For HEAs the D distribution across realisations is typically not
        Gaussian, so prefer the median + MAD entries when summarising.

        Returns dict with keys:
            'per_seed':   np.ndarray[n_realizations]
            'mean':       arithmetic mean
            'std':        sample standard deviation
            'median':     median across realisations
            'mad':        median absolute deviation from the median
        """
        D_list = [
            r.vacancy_diffusion_coefficient(
                max_lag_fraction=max_lag_fraction,
                skip_lag_fraction=skip_lag_fraction,
            )
            for r in self.results
        ]
        D_arr = np.asarray(D_list, dtype=np.float64)
        median = float(np.median(D_arr))
        return {
            "per_seed": D_arr,
            "mean": float(D_arr.mean()),
            "std": float(D_arr.std(ddof=0)),
            "median": median,
            "mad": float(np.median(np.abs(D_arr - median))),
        }

    # ------- Tracer-MSD per element aggregate -------

    def tracer_msd_per_element_ensemble(
        self,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Element-resolved tracer MSD averaged across the ensemble.

        Returns dict mapping each element symbol to a sub-dict:
            'mean':     np.ndarray[n_steps+1]
            'std':      np.ndarray[n_steps+1]
            'per_seed': np.ndarray[n_realizations, n_steps+1]
        """
        per_seed_msd_per_element: List[Dict[str, np.ndarray]] = [
            tracer_msd_per_element(r) for r in self.results
        ]

        out: Dict[str, Dict[str, np.ndarray]] = {}
        symbols = list(per_seed_msd_per_element[0].keys())
        for sym in symbols:
            stacked = np.stack(
                [d[sym] for d in per_seed_msd_per_element], axis=0
            )
            out[sym] = {
                "mean": stacked.mean(axis=0),
                "std": stacked.std(axis=0),
                "per_seed": stacked,
            }
        return out

    # ------- Warren-Cowley aggregate -------

    def warren_cowley_sro_ensemble(
        self,
    ) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
        """Element-pair-resolved WC-SRO trajectories averaged across the ensemble.

        Requires every realisation in the ensemble to have been recorded
        with snapshot_every_n_steps > 0; otherwise raises ValueError.

        Returns dict mapping (sym_i, sym_j) tuples to a sub-dict:
            'mean':     np.ndarray[n_snapshots]
            'std':      np.ndarray[n_snapshots]
            'per_seed': np.ndarray[n_realizations, n_snapshots]
        """
        per_seed_series: List[Dict[Tuple[str, str], np.ndarray]] = [
            warren_cowley_sro_trajectory(r) for r in self.results
        ]

        out: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        keys = list(per_seed_series[0].keys())
        for key in keys:
            stacked = np.stack(
                [d[key] for d in per_seed_series], axis=0
            )
            out[key] = {
                "mean": np.nanmean(stacked, axis=0),
                "std": np.nanstd(stacked, axis=0),
                "per_seed": stacked,
            }
        return out


# ---------------------------------------------------------------------------
# Ensemble-run entry point
# ---------------------------------------------------------------------------

def run_ensemble(
    config: KMCConfig,
    predictor,
    n_realizations: int = 5,
    seeds: Optional[Sequence[int]] = None,
    snapshot_every_n_steps: int = 0,
    progress_callback: Optional[Callable[[int, KMCResult], None]] = None,
    resume: bool = False,
    max_steps_this_session: Optional[int] = None,
) -> EnsembleResult:
    """Run n_realizations independent KMC trajectories with a shared predictor.

    Each realisation builds a fresh KMCState from the config (with its own
    rng for composition shuffling and vacancy placement) and then runs the
    full BKL loop with its own engine rng. The seed-to-state and seed-to-
    engine streams are spawned from a single SeedSequence so the whole
    ensemble is reproducible from the seed list.

    Args:
        config:    KMCConfig — defines composition, supercell, T, nu_attempt,
                   stop criterion (n_steps and/or t_max_sim_s), and the
                   initial-state strategy.
        predictor: BarrierPredictor — Mock or GNN; loaded once and reused.
        n_realizations: number of independent trajectories (default 5).
        seeds:     explicit seed list of length n_realizations. If None,
                   seeds are derived deterministically from
                   config.random_seed via SeedSequence.
        snapshot_every_n_steps: passed through to engine.run().
        progress_callback: optional callable(realisation_index, result)
                   invoked after each completed trajectory.
        resume: if True and a checkpoint file exists at the per-realisation
            checkpoint path, the run continues from that checkpoint
            bit-identically (RNG state included). Realisations without an
            existing checkpoint are started fresh, with a warning. The
            already-completed steps count against ``config.n_steps``, so
            an interrupted 12 000-of-150 000 run is finished by the
            remaining 138 000 steps.
        max_steps_this_session: if not None and > 0, every realisation
            stops voluntarily after this many additional steps in the
            current Python process. Combined with ``resume`` and an
            external auto-restart wrapper, this realises a recycle-style
            execution that bounds a process's lifetime well below the
            empirical crash window.

    Returns:
        EnsembleResult.
    """
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1, got {n_realizations}")

    # Derive seeds if not provided.
    # Note: SeedSequence.spawn(n) returns n distinct sequences, but they all
    # carry the same `.entropy` attribute (= the original seed). Reading
    # entropy here would give n duplicates and silently make every
    # realisation identical. We therefore use generate_state(n_realizations)
    # to draw n distinct uint32 seeds from the master.
    if seeds is None:
        master = np.random.SeedSequence(config.random_seed)
        seeds_array = master.generate_state(n_realizations)
        seeds_list = [int(s) for s in seeds_array.tolist()]
    else:
        if len(seeds) != n_realizations:
            raise ValueError(
                f"seeds has length {len(seeds)}, expected {n_realizations}"
            )
        seeds_list = [int(s) for s in seeds]

    results: List[KMCResult] = []
    for k, seed in enumerate(seeds_list):
        # Two independent rng streams from one root: state init + engine
        ss = np.random.SeedSequence(seed)
        state_seed, engine_seed = ss.spawn(2)
        state_rng = np.random.default_rng(state_seed)
        engine_rng = np.random.default_rng(engine_seed)

        # Per-realisation checkpoint path so distinct realisations never
        # overwrite each other's safety nets. Disabled when the config
        # field is unset / non-positive.
        checkpoint_path = None
        if (
            config.checkpoint_every_n_steps is not None
            and config.checkpoint_every_n_steps > 0
        ):
            checkpoint_path = (
                Path(config.output_dir)
                / (
                    f"checkpoint_T{config.temperature_K:.0f}K"
                    f"_real{k}.npz"
                )
            )

        # Optional resume from the matching per-realisation checkpoint.
        resume_state = None
        if resume:
            if checkpoint_path is None:
                print(
                    f"  [run_ensemble] resume requested but "
                    f"checkpoint_every_n_steps is unset; starting "
                    f"realisation {k} fresh."
                )
            elif not checkpoint_path.exists():
                print(
                    f"  [run_ensemble] resume requested but no checkpoint "
                    f"at {checkpoint_path}; starting realisation {k} fresh."
                )
            else:
                from KMC.checkpoint import load_resume_state
                try:
                    resume_state = load_resume_state(checkpoint_path)
                except ValueError as exc:
                    print(
                        f"  [run_ensemble] resume failed for realisation "
                        f"{k} ({exc}); starting fresh."
                    )
                    resume_state = None
                else:
                    # Sanity: temperature and attempt frequency must match
                    # the active config, otherwise the trajectory is not
                    # comparable and resume would silently corrupt stats.
                    if (
                        abs(resume_state["T_K"] - config.temperature_K) > 1e-6
                        or abs(
                            resume_state["attempt_frequency_Hz"]
                            - config.attempt_frequency_Hz
                        ) > 1e-6
                    ):
                        raise ValueError(
                            f"Checkpoint at {checkpoint_path} was written "
                            f"with T_K={resume_state['T_K']}, "
                            f"nu={resume_state['attempt_frequency_Hz']:g}, "
                            f"but current config has "
                            f"T_K={config.temperature_K}, "
                            f"nu={config.attempt_frequency_Hz:g}. "
                            "Refusing to resume."
                        )
                    print(
                        f"  [run_ensemble] resuming realisation {k} from "
                        f"step {resume_state['step']} "
                        f"(sim t = {resume_state['total_time_s']:.3e} s)."
                    )

        # Build the simulation state. On resume, the state comes straight
        # from the checkpoint (already at step N); on a fresh start, the
        # configured initial-state strategy is used.
        if resume_state is not None:
            state = resume_state["state"]
        else:
            state = _build_initial_state(config, state_rng)

        result = run(
            state,
            predictor,
            T_K=config.temperature_K,
            attempt_frequency_Hz=config.attempt_frequency_Hz,
            n_steps=config.n_steps,
            t_max_sim_s=config.t_max_sim_s,
            snapshot_every_n_steps=snapshot_every_n_steps,
            rng=engine_rng,
            progress_every_n_steps=config.progress_every_n_steps,
            checkpoint_every_n_steps=config.checkpoint_every_n_steps,
            checkpoint_path=checkpoint_path,
            empty_cuda_cache_every_n_steps=(
                config.empty_cuda_cache_every_n_steps
            ),
            resume_from_state=resume_state,
            max_steps_this_session=max_steps_this_session,
        )
        results.append(result)

        if progress_callback is not None:
            progress_callback(k, result)

    return EnsembleResult(config=config, results=results, seeds=seeds_list)


# ---------------------------------------------------------------------------
# Temperature sweep (Phase 5c)
# ---------------------------------------------------------------------------

def run_temperature_sweep(
    config: KMCConfig,
    predictor,
    temperatures_K: Sequence[float],
    n_realizations: int = 5,
    snapshot_every_n_steps: int = 0,
    progress_callback: Optional[Callable[[int, float, EnsembleResult], None]] = None,
) -> Dict[float, EnsembleResult]:
    """Run an ensemble at each requested temperature and return one
    EnsembleResult per temperature.

    A copy of the input config is made for each temperature with
    `temperature_K` overridden. All other fields (composition, supercell,
    n_steps, GNN paths, etc.) are inherited unchanged. Running is sequential.

    Args:
        config: base KMCConfig (its temperature_K is ignored; the listed
            temperatures override it).
        predictor: BarrierPredictor, loaded once and reused.
        temperatures_K: ordered iterable of temperatures to sweep.
        n_realizations: realisations per temperature.
        snapshot_every_n_steps: passed through to the engine for each run.
        progress_callback: optional callable(temperature_index, T, ensemble)
            invoked after each completed temperature.

    Returns:
        Dict mapping each temperature (float) to its EnsembleResult.
    """
    if len(list(temperatures_K)) == 0:
        raise ValueError("temperatures_K must contain at least one entry")

    out: Dict[float, EnsembleResult] = {}
    for k, T in enumerate(temperatures_K):
        T_float = float(T)
        cfg_T = replace(config, temperature_K=T_float)
        ens = run_ensemble(
            cfg_T,
            predictor,
            n_realizations=n_realizations,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )
        out[T_float] = ens
        if progress_callback is not None:
            progress_callback(k, T_float, ens)
    return out
