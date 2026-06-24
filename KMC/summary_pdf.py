"""
Multi-page PDF run summary for a single-temperature KMC ensemble.

The summary collects the three quantities that matter most after a run:
    1. Diffusion: vacancy MSD, per-element tracer MSD, extracted vacancy D.
    2. Order parameter: Warren-Cowley alpha_ij(t) per element pair.
    3. Real time: simulated time mapped to physical time via the
       composition's vacancy formation energy (loaded from a
       DiffusionLookup cache when available).

Entry point: ``write_run_summary_pdf(ensemble, output_path, config,
diffusion_lookup=None)``. Returns the number of pages written.

The PDF is intentionally built with matplotlib only — no extra deps.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

import matplotlib

# Use a non-interactive backend so the script works headlessly inside
# the conda env / on the cluster without an X server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from KMC.analysis import (
    DiffusionLookup,
    K_B_EV_PER_K,
    rescale_to_real_time,
)
from KMC.observables import warren_cowley_sro_snapshot
from KMC.state import VACANCY_SPECIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_seconds(t_s: float) -> str:
    """Human-readable seconds: scientific for very small/large, else direct."""
    if not np.isfinite(t_s) or t_s <= 0:
        return f"{t_s:.3e} s"
    if t_s < 1e-3 or t_s >= 1e3:
        return f"{t_s:.3e} s"
    return f"{t_s:.3f} s"


def _resolve_real_time_factor(
    ensemble,
    config,
    diffusion_lookup: Optional[DiffusionLookup],
) -> Optional[dict]:
    """Try to obtain the sim->real time conversion factor for this run.

    Returns a dict with keys ``E_f_V_eV``, ``T_K``, ``n_atoms``,
    ``c_v_eq``, ``factor`` (= 1 / (n_atoms * c_v_eq)), and ``source``
    describing where E_f_V came from. Returns None if no source could be
    resolved (no cache, lookup miss, etc.).
    """
    if diffusion_lookup is None:
        return None

    composition = dict(config.composition)
    try:
        hit = diffusion_lookup.get(composition)
    except Exception:
        return None
    if hit is None:
        return None

    T_K = float(ensemble.temperature_K)
    n_atoms = int(config.n_sites - 1)
    kT = K_B_EV_PER_K * T_K
    E_f_V = float(hit.E_f_V_eV)
    c_v_eq = float(np.exp(-E_f_V / kT))
    factor = 1.0 / (n_atoms * c_v_eq)
    return {
        "E_f_V_eV": E_f_V,
        "T_K": T_K,
        "n_atoms": n_atoms,
        "c_v_eq": c_v_eq,
        "factor": factor,
        "source": (
            f"DiffusionLookup ({hit.n_neighbors_used} neighbours, "
            f"max d={hit.max_distance_used:.3f})"
        ),
    }


def _initial_sro_matrix(result, element_symbols):
    """Compute alpha_ij at t=0 from the initial species array."""
    nn_table = result.final_state.nn_table
    return warren_cowley_sro_snapshot(
        result.initial_species, nn_table, element_symbols
    )


def _final_sro_matrix(result, element_symbols):
    """Compute alpha_ij at the final snapshot (or final hop-history state)."""
    nn_table = result.final_state.nn_table
    return warren_cowley_sro_snapshot(
        result.final_state.species, nn_table, element_symbols
    )


def _sro_dict_to_matrix(sro: dict, element_symbols):
    """Pack a {(sym_i, sym_j): alpha} dict into a square ndarray."""
    n = len(element_symbols)
    M = np.full((n, n), np.nan, dtype=np.float64)
    for i, si in enumerate(element_symbols):
        for j, sj in enumerate(element_symbols):
            M[i, j] = float(sro.get((si, sj), np.nan))
    return M


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _draw_overview_page(pdf, ensemble, config, real_time_info):
    """Page 1: header text, vacancy MSD, per-element tracer MSD, real time."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        f"KMC run summary  —  T = {ensemble.temperature_K:.0f} K, "
        f"{ensemble.n_realizations} realisation(s)",
        fontsize=13,
        fontweight="bold",
    )

    # Vacancy diffusion coefficient (median + MAD across the ensemble)
    D_pack = ensemble.vacancy_diffusion_coefficient_ensemble()

    # ----- Header text panel (top-left) -----
    ax_text = axes[0, 0]
    ax_text.axis("off")
    comp_str = ", ".join(
        f"{el}={frac:.3f}" for el, frac in config.composition.items()
    )
    sim_t_total = float(ensemble.results[0].total_time_s)
    if real_time_info is not None:
        real_t_total = sim_t_total * real_time_info["factor"]
        real_t_line = (
            f"Real time (total):       {_format_seconds(real_t_total)}\n"
            f"  E_f^V used:            {real_time_info['E_f_V_eV']:.3f} eV\n"
            f"  c_v_eq at this T:      {real_time_info['c_v_eq']:.3e}\n"
            f"  sim->real factor:      {real_time_info['factor']:.3e}\n"
            f"  source:                {real_time_info['source']}"
        )
    else:
        real_t_line = (
            "Real time (total):       not rescaled (no diffusion cache)"
        )

    summary_lines = [
        f"Composition:             {comp_str}",
        f"Initial-state strategy:  {config.initial_state_strategy}",
        f"Supercell size:          {config.supercell_size}^3 "
        f"= {config.n_sites} sites",
        f"Temperature:             {ensemble.temperature_K:.1f} K",
        f"Attempt frequency:       "
        f"{config.attempt_frequency_Hz:.3e} Hz",
        f"Steps:                   {config.n_steps}",
        f"Sim time (total):        {_format_seconds(sim_t_total)}",
        real_t_line,
        "",
        f"Vacancy D (median):      {D_pack['median']:.3e} A^2/s "
        f"(MAD {D_pack['mad']:.3e})",
        f"Vacancy D (mean):        {D_pack['mean']:.3e} A^2/s "
        f"(std {D_pack['std']:.3e})",
    ]
    ax_text.text(
        0.0, 1.0, "\n".join(summary_lines),
        family="monospace", fontsize=9, va="top", ha="left",
    )
    ax_text.set_title("Run parameters", fontsize=10)

    # ----- Vacancy MSD (top-right) -----
    ax_vac = axes[0, 1]
    msd_pack = ensemble.vacancy_msd_ensemble()
    lag_t = msd_pack["lag_t_mean"]
    msd_mean = msd_pack["mean"]
    msd_std = msd_pack["std"]
    ax_vac.plot(lag_t, msd_mean, color="C0", lw=1.5, label="ensemble mean")
    if ensemble.n_realizations > 1:
        ax_vac.fill_between(
            lag_t, msd_mean - msd_std, msd_mean + msd_std,
            color="C0", alpha=0.2, label="±1 std",
        )
    ax_vac.set_xlabel("lag time τ_sim (s)")
    ax_vac.set_ylabel("vacancy MSD (Å²)")
    ax_vac.set_title("Time-averaged vacancy MSD")
    ax_vac.grid(True, alpha=0.3)
    ax_vac.legend(loc="upper left", fontsize=8)

    # ----- Per-element tracer MSD (bottom-left) -----
    ax_tr = axes[1, 0]
    tracer_pack = ensemble.tracer_msd_per_element_ensemble()
    times_per_atom = ensemble.results[0].times_with_zero_s
    for sym, sub in tracer_pack.items():
        ax_tr.plot(times_per_atom, sub["mean"], lw=1.3, label=sym)
    ax_tr.set_xlabel("time t_sim (s)")
    ax_tr.set_ylabel("tracer MSD (Å²)")
    ax_tr.set_title("Per-element tracer MSD")
    ax_tr.grid(True, alpha=0.3)
    ax_tr.legend(fontsize=8, ncol=2)

    # ----- Real-time conversion (bottom-right) -----
    ax_rt = axes[1, 1]
    ax_rt.axis("off")
    if real_time_info is not None:
        rti = real_time_info
        rt_lines = [
            r"Real-time rescaling",
            "",
            r"τ_real = τ_sim / (N · c_v^eq)",
            r"     N           : atoms in supercell",
            r"     c_v^eq      = exp(-E_f^V / k_B T)",
            "",
            f"  N             = {rti['n_atoms']}",
            f"  T             = {rti['T_K']:.1f} K",
            f"  E_f^V         = {rti['E_f_V_eV']:.3f} eV",
            f"  c_v^eq        = {rti['c_v_eq']:.3e}",
            f"  N · c_v^eq    = {rti['n_atoms'] * rti['c_v_eq']:.3e}",
            f"  factor τ_real / τ_sim = {rti['factor']:.3e}",
            "",
            f"  source: {rti['source']}",
        ]
        ax_rt.text(
            0.0, 1.0, "\n".join(rt_lines),
            family="monospace", fontsize=9, va="top", ha="left",
        )
    else:
        ax_rt.text(
            0.5, 0.5,
            "No real-time rescaling.\n\n"
            "Configure diffusion_cache_path with a populated cache\n"
            "(or run the diffusion_coefficient/ workflow first) to\n"
            "enable τ_sim → τ_real conversion.",
            family="monospace", fontsize=9, ha="center", va="center",
            color="0.4",
        )
    ax_rt.set_title("Sim → real time conversion", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _draw_sro_timeseries_page(pdf, ensemble, real_time_info):
    """Page 2: alpha_ij(t) for every (i, j) pair, grouped by row element."""
    elements = list(ensemble.results[0].final_state.element_symbols)
    n = len(elements)
    if n < 2:
        return  # Nothing meaningful to show for a single-element run

    sro_series = ensemble.warren_cowley_sro_ensemble()
    times_snap = ensemble.results[0].snapshot_times_s
    if times_snap is None or len(times_snap) == 0:
        return  # No snapshots -> no SRO trajectory available

    if real_time_info is not None:
        times_plot = times_snap * real_time_info["factor"]
        time_label = "real time τ_real (s)"
    else:
        times_plot = times_snap
        time_label = "sim time τ_sim (s)"

    # Adaptive grid: at most 2 cols, n rows = n_elements rows for clarity.
    # Each subplot shows alpha_{i, j}(t) for fixed i, all j.
    ncols = 2 if n >= 2 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11, 3.0 * nrows), squeeze=False,
    )
    fig.suptitle(
        f"Warren-Cowley α(t)  —  T = {ensemble.temperature_K:.0f} K",
        fontsize=13, fontweight="bold",
    )

    for k, sym_i in enumerate(elements):
        row, col = divmod(k, ncols)
        ax = axes[row, col]
        for sym_j in elements:
            key = (sym_i, sym_j)
            if key not in sro_series:
                continue
            series = sro_series[key]["mean"]
            std = sro_series[key]["std"]
            ax.plot(times_plot, series, lw=1.3, label=sym_j)
            if ensemble.n_realizations > 1:
                ax.fill_between(
                    times_plot, series - std, series + std,
                    alpha=0.15,
                )
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel(time_label)
        ax.set_ylabel(f"α({sym_i}, j)")
        ax.set_title(f"i = {sym_i}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(title="j", fontsize=8, ncol=2, loc="best")

    # Hide any spare panels (when n is not a multiple of ncols)
    for k in range(n, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row, col].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _draw_sro_matrix_page(pdf, ensemble):
    """Page 3: heatmap of alpha_ij at the initial and final state."""
    first = ensemble.results[0]
    elements = list(first.final_state.element_symbols)
    if len(elements) < 2:
        return

    sro_init = _initial_sro_matrix(first, elements)
    sro_final = _final_sro_matrix(first, elements)
    M_init = _sro_dict_to_matrix(sro_init, elements)
    M_final = _sro_dict_to_matrix(sro_final, elements)

    # Symmetric colour scale around 0; clipped to a sensible WC-SRO range.
    finite_vals = np.concatenate(
        [M_init[np.isfinite(M_init)], M_final[np.isfinite(M_final)]]
    )
    if finite_vals.size:
        vmax = float(np.nanmax(np.abs(finite_vals)))
    else:
        vmax = 1.0
    vmax = max(vmax, 0.05)  # avoid degenerate near-zero scales

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Warren-Cowley α matrix  —  T = {ensemble.temperature_K:.0f} K",
        fontsize=13, fontweight="bold",
    )

    for ax, M, title in [
        (axes[0], M_init, "Initial state (t = 0)"),
        (axes[1], M_final, "Final state"),
    ]:
        im = ax.imshow(
            M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
        )
        ax.set_xticks(range(len(elements)))
        ax.set_yticks(range(len(elements)))
        ax.set_xticklabels(elements)
        ax.set_yticklabels(elements)
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_title(title, fontsize=10)
        # Annotate cells with the numeric value
        for i in range(len(elements)):
            for j in range(len(elements)):
                val = M[i, j]
                if np.isfinite(val):
                    ax.text(
                        j, i, f"{val:+.2f}",
                        ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(val) > 0.6 * vmax else "black",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def write_run_summary_pdf(
    ensemble,
    output_path: Union[str, Path],
    config,
    diffusion_lookup: Optional[DiffusionLookup] = None,
) -> int:
    """Build a multi-page PDF summary of one ensemble run.

    Args:
        ensemble: EnsembleResult to summarise.
        output_path: target .pdf file path.
        config: KMCConfig used for the run (for composition, supercell etc.).
        diffusion_lookup: optional DiffusionLookup for sim->real time
            rescaling. If omitted, all time axes stay in simulated time and
            the real-time panel reports the missing cache.

    Returns:
        Number of pages written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real_time_info = _resolve_real_time_factor(
        ensemble, config, diffusion_lookup
    )

    n_pages = 0
    with PdfPages(output_path) as pdf:
        _draw_overview_page(pdf, ensemble, config, real_time_info)
        n_pages += 1

        # Page 2/3 only when WC-SRO snapshots were recorded.
        first = ensemble.results[0]
        if first.snapshot_times_s is not None and first.n_snapshots > 0:
            _draw_sro_timeseries_page(pdf, ensemble, real_time_info)
            n_pages += 1
            _draw_sro_matrix_page(pdf, ensemble)
            n_pages += 1

    return n_pages
