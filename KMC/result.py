"""
KMCResult: container for the data produced by a KMC run.

Stores per-step hop history plus optional periodic snapshots of the full
species array. From this, all downstream observables can be computed
post-hoc without re-running the engine:

- Vacancy MSD as a function of time (built-in helper).
- Tracer MSD per element (Phase 4): replay the hop history while tracking
  individual atom IDs, accumulate squared displacements per element.
- Warren-Cowley SRO as a function of time (Phase 4): use the periodic
  snapshots, or replay the hop history to reconstruct any intermediate
  configuration exactly.
- Sprung-Statistics (e.g. detailed-balance, jump-direction distribution):
  use chosen_jump_local and barriers_eV directly.

Per-step storage layout (n = number of steps):
    times_s                     [n]            cumulative simulated time
    delta_t_s                   [n]            time increment of step
    vacancy_indices             [n+1]          vacancy site BEFORE/AFTER each step
    hopper_species              [n]            element index of the atom that hopped
    chosen_jump_local           [n]            which of the 8 NN was selected (0..7)
    barriers_eV                 [n, 8]         forward barriers used in the step
    vacancy_positions_unfolded  [n+1, 3]       PBC-unfolded vacancy positions

Optional snapshots (only if snapshot_every_n_steps > 0 was requested):
    snapshot_times_s            [n_snap]
    snapshot_species            [n_snap, N_sites]
    snapshot_vacancy_indices    [n_snap]
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class KMCResult:
    """All data produced by one run() invocation."""

    # --- Run metadata ---
    n_steps: int
    total_time_s: float
    T_K: float                  # temperature this run was at (Phase 5b)
    attempt_frequency_Hz: float  # nu_0 used for the BKL rates (Phase 5b)
    final_state: object  # KMCState, but avoiding circular import

    # --- Initial reference (for replay) ---
    initial_species: np.ndarray              # [N_sites], with VACANCY_SPECIES at vacancy
    initial_vacancy_index: int

    # --- Hop history (per-step) ---
    times_s: np.ndarray                      # [n_steps]           float64
    delta_t_s: np.ndarray                    # [n_steps]           float64
    vacancy_indices: np.ndarray              # [n_steps+1]         int32
    hopper_species: np.ndarray               # [n_steps]           int8
    chosen_jump_local: np.ndarray            # [n_steps]           int8
    barriers_eV: np.ndarray                  # [n_steps, 8]        float32
    vacancy_positions_unfolded: np.ndarray   # [n_steps+1, 3]      float64

    # --- Optional periodic snapshots ---
    snapshot_times_s: Optional[np.ndarray] = None            # [n_snap]
    snapshot_species: Optional[np.ndarray] = None            # [n_snap, N_sites]
    snapshot_vacancy_indices: Optional[np.ndarray] = None    # [n_snap]

    # ------- Convenience accessors -------

    @property
    def times_with_zero_s(self) -> np.ndarray:
        """Time array including t=0 at the start, shape [n_steps+1]."""
        return np.concatenate(([0.0], self.times_s))

    @property
    def n_snapshots(self) -> int:
        if self.snapshot_times_s is None:
            return 0
        return int(self.snapshot_times_s.shape[0])

    # ------- Built-in observables -------

    def vacancy_msd(self) -> np.ndarray:
        """Single-trajectory squared displacement of the vacancy from t=0.

        For a single trajectory the variance of this estimator stays of order
        MSD(t)^2 at all times — a known property of random walks. Use this
        only for visualizing the raw walk; for quantitative D-extraction
        prefer time_averaged_vacancy_msd().

        Returns:
            np.ndarray of shape [n_steps+1] with squared displacement in
            (Angstrom)^2.
        """
        r0 = self.vacancy_positions_unfolded[0]
        delta = self.vacancy_positions_unfolded - r0
        return np.einsum("ij,ij->i", delta, delta)

    def time_averaged_vacancy_msd(
        self, max_lag_fraction: float = 0.05
    ) -> tuple:
        """Time-averaged MSD via sliding-window over the vacancy trajectory.

        Computes MSD(tau) = <|r(t+tau) - r(t)|^2>_t for lag tau in units of
        KMC steps. From a trajectory of length N this yields N-m sliding-
        window samples for lag m. They are NOT independent (windows overlap),
        and at lags approaching N the estimator becomes dominated by the
        single trajectory's local range rather than the ensemble average
        6D*tau. Standard practice is to keep the maximum lag well below the
        trajectory length; max_lag_fraction = 0.1 is the conservative default
        used here. Use a smaller fraction (~0.01) for very long runs and a
        larger one (~0.25) only when an ensemble average is also performed.

        Args:
            max_lag_fraction: fraction of the trajectory used as the maximum
                lag (default 0.1).

        Returns:
            (msd, mean_lag_times_s) where both arrays have shape [max_lag].
            msd[0] = 0 by definition; mean_lag_times_s[m] is the average
            wall-clock interval associated with a lag of m steps.
        """
        r = self.vacancy_positions_unfolded
        t = self.times_with_zero_s
        n = r.shape[0]
        max_lag = max(2, int(max_lag_fraction * n))

        msd = np.zeros(max_lag, dtype=np.float64)
        lag_times = np.zeros(max_lag, dtype=np.float64)
        for m in range(1, max_lag):
            diffs = r[m:] - r[:-m]
            msd[m] = float(np.einsum("ij,ij->i", diffs, diffs).mean())
            lag_times[m] = float((t[m:] - t[:-m]).mean())
        return msd, lag_times

    def vacancy_diffusion_coefficient(
        self,
        max_lag_fraction: float = 0.025,
        skip_lag_fraction: float = 0.05,
    ) -> float:
        """Estimate D_vacancy via linear regression of time-averaged MSD vs lag time.

        Uses the sliding-window estimator (time_averaged_vacancy_msd), which is
        the standard robust method. D = slope / 6 in 3D.

        Args:
            max_lag_fraction: maximum lag as a fraction of trajectory length
                (default 0.025; smaller than the visualization default of
                time_averaged_vacancy_msd because the slope estimate is more
                sensitive to single-trajectory ergodicity violations than the
                MSD curve itself).
            skip_lag_fraction: smallest lag fraction excluded from the fit to
                avoid sub-diffusive transients (default 0.05).

        Returns:
            D_vacancy in (Angstrom)^2 / s.
        """
        msd, lag_t = self.time_averaged_vacancy_msd(
            max_lag_fraction=max_lag_fraction
        )
        n = len(lag_t)
        skip = max(1, int(skip_lag_fraction * n))
        if n - skip < 2:
            raise ValueError(
                f"Too few lag points after skipping ({n - skip}); need >= 2"
            )
        slope, _ = np.polyfit(lag_t[skip:], msd[skip:], deg=1)
        return float(slope) / 6.0
