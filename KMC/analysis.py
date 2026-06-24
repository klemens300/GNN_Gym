"""
Analysis tools that bridge KMC simulation time to physical real time.

Two ingredients:

1. DiffusionLookup
   A JSON-persistent cache of composition-resolved diffusion-related
   quantities (E_f^V, lattice parameter, jump distance, Debye attempt
   frequency). Queries are resolved by k-nearest-neighbour interpolation
   on the composition simplex; if too few neighbours lie within
   max_distance, the lookup falls back to an oracle callable that runs a
   fresh DiffusionOracle / NEB job and stores its result in the cache.

2. rescale_to_real_time
   Maps simulated KMC time to physical real time using
       tau_real = tau_sim / (n_atoms * c_v_eq)
   where c_v_eq = exp(-E_f^V / kT) and n_atoms is the number of atoms in
   the KMC supercell (n_sites - 1, because one site is the vacancy).

   This rescaling absorbs the unphysical 1/n_atoms vacancy concentration
   built into a one-vacancy KMC supercell. It assumes vacancy-vacancy
   interactions in the real material are negligible (low c_v_eq regime),
   which is the standard assumption for refractory HEAs well below the
   solidus and is documented as a limitation in the paper Discussion.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# Boltzmann constant in eV/K (CODATA)
K_B_EV_PER_K = 8.617333262e-5


# ---------------------------------------------------------------------------
# Real-time rescaling
# ---------------------------------------------------------------------------

def rescale_to_real_time(
    times_sim_s: Union[float, np.ndarray],
    E_f_V_eV: float,
    T_K: float,
    n_atoms: int,
) -> Union[float, np.ndarray]:
    """Convert simulated KMC time to physical real time.

    A KMC supercell with `n_atoms` atoms and one vacancy carries an
    artificially high vacancy concentration of 1/n_atoms. The Boltzmann-
    equilibrium concentration in the real material is

        c_v_eq = exp(-E_f^V / kT).

    The two are reconciled by

        tau_real = tau_sim / (n_atoms * c_v_eq).

    Args:
        times_sim_s: simulated time(s) in seconds; scalar or ndarray.
        E_f_V_eV: vacancy formation energy in eV (composition-dependent;
            obtain via DiffusionLookup for non-trivial compositions).
        T_K: temperature in Kelvin.
        n_atoms: atoms in the KMC supercell (= n_sites - 1).

    Returns:
        Real time(s) in seconds, same shape as `times_sim_s`.
    """
    if T_K <= 0:
        raise ValueError(f"Temperature must be positive, got {T_K} K")
    if n_atoms <= 0:
        raise ValueError(f"n_atoms must be positive, got {n_atoms}")

    kT = K_B_EV_PER_K * T_K
    c_v_eq = float(np.exp(-E_f_V_eV / kT))
    return np.asarray(times_sim_s) / (n_atoms * c_v_eq)


def real_time_axis(
    result,
    E_f_V_eV: float,
    n_atoms: Optional[int] = None,
) -> np.ndarray:
    """Convenience: build the real-time axis matching result.times_with_zero_s.

    Uses result.T_K (recorded in Phase 5b) and defaults n_atoms to
    result.final_state.n_sites - 1 (one vacancy in the box).
    """
    if n_atoms is None:
        n_atoms = result.final_state.n_sites - 1
    return rescale_to_real_time(
        result.times_with_zero_s, E_f_V_eV, result.T_K, n_atoms
    )


# ---------------------------------------------------------------------------
# Cached diffusion entry
# ---------------------------------------------------------------------------

@dataclass
class CachedDiffusionEntry:
    """One composition-point of the diffusion lookup cache."""

    composition: Dict[str, float]
    E_f_V_eV: float
    lattice_parameter_A: float
    jump_distance_A: float
    nu_0_Hz: float                # Debye attempt frequency
    n_atoms_supercell: int        # cell size that produced this entry
    source: str = "DiffusionOracle"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CachedDiffusionEntry":
        return cls(**d)

    @classmethod
    def from_diffusion_result(cls, result) -> "CachedDiffusionEntry":
        """Build from a `DiffusionResult` (scipts/diffusion_coefficient/results.py).

        Computes the Debye attempt frequency from the elastic constants on
        the fly, so the cache is self-contained for KMC rescaling.
        """
        # Lazy import so users who never call this method do not need the
        # diffusion package's heavier dependencies (ase, matscipy) installed.
        # `diffusion` is a sibling package installed alongside KMC.
        from diffusion.diffusion_physics import debye_frequency

        nu_0 = debye_frequency(
            C11_GPa=result.C11_GPa,
            C12_GPa=result.C12_GPa,
            C44_GPa=result.C44_GPa,
            density_kg_m3=result.density_kg_m3,
            volume_A3=result.volume_A3,
            n_atoms=result.n_atoms,
        )

        return cls(
            composition=dict(result.composition),
            E_f_V_eV=float(result.vacancy_formation_energy_eV),
            lattice_parameter_A=float(result.lattice_parameter_A),
            jump_distance_A=float(result.jump_distance_A),
            nu_0_Hz=float(nu_0),
            n_atoms_supercell=int(result.n_atoms),
            source=f"{result.calculator}:{result.model}",
        )


# ---------------------------------------------------------------------------
# Interpolated lookup result
# ---------------------------------------------------------------------------

@dataclass
class InterpolatedDiffusionPoint:
    """k-NN-interpolated diffusion data for an arbitrary composition."""

    composition: Dict[str, float]
    E_f_V_eV: float
    lattice_parameter_A: float
    jump_distance_A: float
    nu_0_Hz: float

    n_neighbors_used: int = 0
    max_distance_used: float = 0.0
    source_compositions: List[Dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DiffusionLookup
# ---------------------------------------------------------------------------

class DiffusionLookup:
    """k-NN-interpolated, JSON-persistent diffusion-data lookup.

    The cache is loaded at construction time and re-saved every time a new
    entry is added. Queries return InterpolatedDiffusionPoint built as the
    arithmetic mean of the k closest cache entries (closeness measured by
    L2 distance in the composition vector).

    If fewer than k entries are within `max_distance`, `get(...)` returns
    None and `get_or_compute(...)` triggers the oracle callable (if
    available); otherwise both raise.

    Cache JSON layout:
        {
          "metadata": {"elements": ["Mo", "Nb", "Ta", "W"]},
          "entries": [
            {"composition": {"Mo": 0.25, ...}, "E_f_V_eV": ..., ...},
            ...
          ]
        }
    """

    def __init__(
        self,
        cache_path: Union[str, Path],
        oracle: Optional[Callable[[Dict[str, float]], object]] = None,
        n_neighbors: int = 3,
        max_distance: float = 0.10,
    ):
        """
        Args:
            cache_path: JSON file holding the cache.
            oracle: callable composition -> DiffusionResult (or
                CachedDiffusionEntry). Used by get_or_compute on cache miss.
            n_neighbors: k for the k-NN interpolation.
            max_distance: largest L2 distance (in composition fractions)
                at which a cache entry still counts as a usable neighbour.
        """
        self.cache_path = Path(cache_path)
        self.oracle = oracle
        self.n_neighbors = int(n_neighbors)
        self.max_distance = float(max_distance)
        self._entries: List[CachedDiffusionEntry] = []
        self._elements: Optional[List[str]] = None
        self._load()

    # ------- Persistence -------

    def _load(self) -> None:
        if not self.cache_path.exists():
            self._entries = []
            self._elements = None
            return
        with open(self.cache_path) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        self._elements = meta.get("elements")
        self._entries = [
            CachedDiffusionEntry.from_dict(e) for e in data.get("entries", [])
        ]

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"elements": self._elements} if self._elements else {}
        data = {
            "metadata": meta,
            "entries": [e.to_dict() for e in self._entries],
        }
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2)

    # ------- Element bookkeeping -------

    def _ensure_elements(self, composition: Dict[str, float]) -> List[str]:
        """Lock the element list at first contact; refuse mismatches later."""
        sorted_keys = tuple(sorted(composition.keys()))
        if self._elements is None:
            self._elements = list(sorted_keys)
        else:
            if tuple(sorted(self._elements)) != sorted_keys:
                raise ValueError(
                    f"Composition uses elements {sorted_keys}, but cache was "
                    f"built for {tuple(sorted(self._elements))}"
                )
        return self._elements

    def _composition_vector(self, composition: Dict[str, float]) -> np.ndarray:
        elements = self._ensure_elements(composition)
        return np.array(
            [float(composition.get(el, 0.0)) for el in elements],
            dtype=np.float64,
        )

    # ------- Adding to the cache -------

    def add(self, entry_or_result) -> CachedDiffusionEntry:
        """Add a CachedDiffusionEntry or a DiffusionResult to the cache."""
        if isinstance(entry_or_result, CachedDiffusionEntry):
            entry = entry_or_result
        else:
            entry = CachedDiffusionEntry.from_diffusion_result(entry_or_result)
        self._ensure_elements(entry.composition)
        self._entries.append(entry)
        self._save()
        return entry

    # ------- Query -------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[CachedDiffusionEntry]:
        return list(self._entries)

    def get(
        self, composition: Dict[str, float]
    ) -> Optional[InterpolatedDiffusionPoint]:
        """k-NN interpolated lookup. Returns None if cache too sparse / far."""
        if not self._entries:
            return None
        q = self._composition_vector(composition)

        distances = np.array(
            [
                np.linalg.norm(self._composition_vector(e.composition) - q)
                for e in self._entries
            ],
            dtype=np.float64,
        )
        order = np.argsort(distances)
        k = min(self.n_neighbors, len(self._entries))
        chosen_idx = order[:k]
        chosen_d = distances[chosen_idx]

        # All k neighbours must be within max_distance to count as usable.
        if not (chosen_d <= self.max_distance).all():
            return None

        chosen_entries = [self._entries[i] for i in chosen_idx]
        E_f_V = float(np.mean([e.E_f_V_eV for e in chosen_entries]))
        a = float(np.mean([e.lattice_parameter_A for e in chosen_entries]))
        jd = float(np.mean([e.jump_distance_A for e in chosen_entries]))
        nu0 = float(np.mean([e.nu_0_Hz for e in chosen_entries]))

        return InterpolatedDiffusionPoint(
            composition=dict(composition),
            E_f_V_eV=E_f_V,
            lattice_parameter_A=a,
            jump_distance_A=jd,
            nu_0_Hz=nu0,
            n_neighbors_used=k,
            max_distance_used=float(chosen_d.max()),
            source_compositions=[dict(e.composition) for e in chosen_entries],
        )

    def get_or_compute(
        self, composition: Dict[str, float]
    ) -> InterpolatedDiffusionPoint:
        """As `get`, but on cache miss runs the oracle and re-tries.

        Raises RuntimeError if no oracle is configured and the lookup misses,
        or if the oracle was used and the cache is still too sparse to satisfy
        n_neighbors / max_distance (e.g. when the oracle returned a single
        new point but the requested distance envelope is tighter).
        """
        hit = self.get(composition)
        if hit is not None:
            return hit

        if self.oracle is None:
            raise RuntimeError(
                f"DiffusionLookup miss for composition {composition} and no "
                "oracle is configured. Add cache entries first or pass "
                "oracle=... to the constructor."
            )

        result = self.oracle(composition)
        if isinstance(result, CachedDiffusionEntry):
            self.add(result)
        else:
            self.add(CachedDiffusionEntry.from_diffusion_result(result))

        hit = self.get(composition)
        if hit is None:
            raise RuntimeError(
                f"DiffusionLookup still misses after oracle compute for "
                f"composition {composition}. Cache size = {len(self)}; "
                f"need {self.n_neighbors} neighbours within "
                f"max_distance={self.max_distance}."
            )
        return hit


# ---------------------------------------------------------------------------
# Arrhenius fit (Phase 5c)
# ---------------------------------------------------------------------------

@dataclass
class ArrheniusFit:
    """Result of an Arrhenius regression: D(T) = D_0 * exp(-Q / kT)."""

    D_0: float
    Q_eV: float
    D_0_uncertainty: float
    Q_eV_uncertainty: float
    n_temperatures_used: int
    # Per-temperature aggregated D values used in the fit (median + MAD)
    temperatures_K: np.ndarray
    D_median: np.ndarray
    D_mad: np.ndarray

    def predict(self, T_K: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the fitted Arrhenius curve at temperature(s) T_K."""
        T = np.asarray(T_K, dtype=np.float64)
        return self.D_0 * np.exp(-self.Q_eV / (K_B_EV_PER_K * T))


def arrhenius_fit_robust(
    temperatures_K: Sequence[float],
    D_per_realization: Sequence[Sequence[float]],
) -> ArrheniusFit:
    """Robust Arrhenius fit using median + MAD aggregation per temperature.

    HEA barrier distributions are typically not Gaussian, so per-temperature
    aggregation uses median + median absolute deviation rather than mean +
    standard deviation. The fit itself is a weighted linear regression of
    log(D) against 1/T, with weights derived from the relative MAD at each
    point (small MAD -> high weight). Negative or non-finite D values
    contribute NaN at the realisation level and are skipped via
    np.nanmedian / np.nanmedian-of-deviations.

    Args:
        temperatures_K: list of temperatures (K), one per ensemble.
        D_per_realization: outer list indexed by temperature, inner list of
            per-realisation D values (in any consistent unit, e.g. A^2/s).

    Returns:
        ArrheniusFit. If fewer than 2 valid temperatures remain after
        filtering (D > 0 and finite), raises ValueError.
    """
    temps = np.asarray(temperatures_K, dtype=np.float64)
    n_T = temps.shape[0]
    if n_T != len(D_per_realization):
        raise ValueError(
            f"len(temperatures_K)={n_T} does not match len(D_per_realization)"
            f"={len(D_per_realization)}"
        )

    medians = np.empty(n_T, dtype=np.float64)
    mads = np.empty(n_T, dtype=np.float64)
    for k, samples in enumerate(D_per_realization):
        arr = np.asarray(samples, dtype=np.float64)
        # Mark unphysical / non-finite samples as NaN before aggregation
        bad = ~np.isfinite(arr) | (arr <= 0)
        arr = np.where(bad, np.nan, arr)
        med = float(np.nanmedian(arr)) if not np.all(bad) else np.nan
        mad = (
            float(np.nanmedian(np.abs(arr - med)))
            if not np.all(bad) else np.nan
        )
        medians[k] = med
        mads[k] = mad

    # Filter to temperatures with a usable median
    keep = np.isfinite(medians) & (medians > 0)
    if int(keep.sum()) < 2:
        raise ValueError(
            f"Arrhenius fit needs at least 2 valid temperatures, got "
            f"{int(keep.sum())}"
        )

    T_keep = temps[keep]
    D_keep = medians[keep]
    MAD_keep = mads[keep]

    # Weighted linear fit: log(D) = log(D_0) - Q/(kT) -> y = b + m*x with
    # x = 1/T, y = log(D), m = -Q/k_B, b = log(D_0).
    x = 1.0 / T_keep
    y = np.log(D_keep)

    # Weight ~ 1 / log-space sigma. log(D ± MAD) - log(D) ~ MAD/D,
    # so log-sigma ~ MAD/D. A floor avoids divide-by-zero for clean data.
    log_sigma = np.where(
        (MAD_keep > 0) & (D_keep > 0),
        MAD_keep / D_keep,
        1e-3,
    )
    weights = 1.0 / np.maximum(log_sigma, 1e-6)

    # numpy.polyfit accepts weights via the `w` parameter (deg=1)
    coeffs, cov = np.polyfit(x, y, deg=1, w=weights, cov=True)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    slope_var, intercept_var = float(cov[0, 0]), float(cov[1, 1])

    Q_eV = -slope * K_B_EV_PER_K
    D_0 = float(np.exp(intercept))
    Q_unc = float(np.sqrt(slope_var)) * K_B_EV_PER_K
    D_0_unc = D_0 * float(np.sqrt(intercept_var))

    return ArrheniusFit(
        D_0=D_0,
        Q_eV=Q_eV,
        D_0_uncertainty=D_0_unc,
        Q_eV_uncertainty=Q_unc,
        n_temperatures_used=int(keep.sum()),
        temperatures_K=T_keep,
        D_median=D_keep,
        D_mad=MAD_keep,
    )


def arrhenius_fit_per_element(
    sweep_results: Dict[float, object],
    max_lag_fraction: float = 0.025,
    skip_lag_fraction: float = 0.05,
) -> Dict[str, ArrheniusFit]:
    """Convenience: extract per-element Arrhenius fits from a temperature sweep.

    For now, the per-element D is approximated by the vacancy-MSD-based
    diffusion coefficient (one D per realisation per temperature). A full
    species-resolved D extraction (from tracer_msd_per_element) can be
    plugged in here later without changing this API.

    Args:
        sweep_results: dict[T -> EnsembleResult], typically the return of
            runner.run_temperature_sweep.
        max_lag_fraction, skip_lag_fraction: forwarded to
            EnsembleResult.vacancy_diffusion_coefficient_ensemble.

    Returns:
        Dict mapping each element symbol to an ArrheniusFit. For now the
        same vacancy-D fit is replicated under each element key; a future
        refinement can replace it with a true tracer fit per element.
    """
    temperatures = sorted(sweep_results.keys())
    if len(temperatures) < 2:
        raise ValueError(
            "Need at least 2 temperatures in sweep_results for an Arrhenius fit"
        )

    # Per-T list of per-realisation D values (vacancy diffusion coefficient)
    D_per_T: List[np.ndarray] = []
    elements: Optional[List[str]] = None
    for T in temperatures:
        ens = sweep_results[T]
        if elements is None:
            elements = list(ens.results[0].final_state.element_symbols)
        D_pack = ens.vacancy_diffusion_coefficient_ensemble(
            max_lag_fraction=max_lag_fraction,
            skip_lag_fraction=skip_lag_fraction,
        )
        D_per_T.append(D_pack["per_seed"])

    fit = arrhenius_fit_robust(temperatures, D_per_T)
    return {sym: fit for sym in (elements or [])}


# ---------------------------------------------------------------------------
# tau_order from WC-relaxation (Phase 5c)
# ---------------------------------------------------------------------------

@dataclass
class TauOrderFit:
    """Exponential-relaxation fit: alpha(t) = alpha_inf + (alpha_0 - alpha_inf) * exp(-t/tau)."""

    tau_order_s: float
    alpha_0: float
    alpha_inf: float
    n_points_used: int
    rmse: float


def tau_order_from_alpha_curve(
    times_s: np.ndarray,
    alpha_t: np.ndarray,
    min_points: int = 5,
) -> Optional[TauOrderFit]:
    """Fit alpha(t) = alpha_inf + (alpha_0 - alpha_inf) * exp(-t/tau).

    Numerical stability:
        Time is normalised to [0, 1] before fitting and the result is
        rescaled afterwards. The initial tau guess uses a half-life estimate
        (t such that alpha(t) sits at the midpoint of the visible decay).
        These two together make curve_fit converge to floating-point
        precision on noiseless data.

    Args:
        times_s: monotonically increasing time array.
        alpha_t: WC-SRO alpha values, same shape as times_s.
        min_points: smallest number of finite samples for which a fit is
            attempted; below this, returns None.

    Returns:
        TauOrderFit, or None if the input is too sparse / non-finite or
        scipy is unavailable / the fit failed.
    """
    t = np.asarray(times_s, dtype=np.float64)
    a = np.asarray(alpha_t, dtype=np.float64)

    mask = np.isfinite(t) & np.isfinite(a)
    t = t[mask]
    a = a[mask]
    if t.shape[0] < min_points:
        return None

    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    # Normalise the time axis to [0, 1] for numerical conditioning
    t_min = float(t[0])
    t_scale = float(t[-1] - t[0])
    if t_scale <= 0:
        return None
    t_norm = (t - t_min) / t_scale

    def model(tt, tau_norm, alpha_0, alpha_inf):
        return alpha_inf + (alpha_0 - alpha_inf) * np.exp(-tt / tau_norm)

    # Initial guesses
    a0_init = float(a[0])
    ainf_init = float(a[-1])
    midpoint = 0.5 * (a0_init + ainf_init)
    idx_half = int(np.argmin(np.abs(a - midpoint)))
    t_half_norm = float(t_norm[idx_half])
    # tau from t_half = tau * ln(2); fall back to 1/3 of the normalised range
    if t_half_norm > 1e-6:
        tau_norm_init = max(t_half_norm / float(np.log(2.0)), 1e-6)
    else:
        tau_norm_init = 1.0 / 3.0

    p0 = (tau_norm_init, a0_init, ainf_init)
    bounds = ([1e-12, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    try:
        popt, _ = curve_fit(
            model, t_norm, a,
            p0=p0, bounds=bounds,
            maxfev=10000, ftol=1e-12, xtol=1e-12,
        )
    except Exception:
        return None

    tau_norm, a0, ainf = float(popt[0]), float(popt[1]), float(popt[2])
    tau = tau_norm * t_scale
    residuals = a - model(t_norm, tau_norm, a0, ainf)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return TauOrderFit(
        tau_order_s=tau,
        alpha_0=a0,
        alpha_inf=ainf,
        n_points_used=int(t.shape[0]),
        rmse=rmse,
    )


def tau_order_matrix_from_ensemble(
    ensemble,
    times_axis: str = "snapshot",
    min_points: int = 5,
) -> Dict[Tuple[str, str], Optional[TauOrderFit]]:
    """Compute tau_order for every element-pair from an EnsembleResult.

    Args:
        ensemble: EnsembleResult with snapshots recorded.
        times_axis: 'snapshot' uses the snapshot times of the first
            realisation; could later be switched to ensemble-averaged times.
        min_points: forwarded to tau_order_from_alpha_curve.

    Returns:
        Dict mapping each (sym_i, sym_j) to a TauOrderFit (or None when the
        fit could not be performed).
    """
    series = ensemble.warren_cowley_sro_ensemble()
    # Use the snapshot times of the first realisation as the time axis
    first = ensemble.results[0]
    if first.snapshot_times_s is None:
        raise ValueError(
            "Ensemble has no snapshot times; cannot fit tau_order. Re-run "
            "with snapshot_every_n_steps > 0."
        )
    times_s = np.asarray(first.snapshot_times_s)

    out: Dict[Tuple[str, str], Optional[TauOrderFit]] = {}
    for key, sub in series.items():
        out[key] = tau_order_from_alpha_curve(
            times_s, sub["mean"], min_points=min_points
        )
    return out
