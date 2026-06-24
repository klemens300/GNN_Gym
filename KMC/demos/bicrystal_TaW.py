"""
Demo: Ta|W bicrystal vacancy diffusion.

Sets up a 50-50 Ta|W bicrystal in an 8x8x8 BCC supercell (1024 sites), places
the vacancy at the planar interface, and runs 50000 BKL steps with the
trained MoNbTaW GNN. Snapshots every 200 steps -> 250 animation frames.

Outputs (under output_dir, default `kmc_outputs/bicrystal_TaW/`):
    trajectory.extxyz       Full snapshot trajectory (vacancy as 'X' for OVITO).
    event_log.csv           One row per BKL step (replayable).
    diagnostics.pdf         Multi-panel diagnostic figure:
                              - 3D scatter: initial vs final configuration
                              - c_Ta(x) profiles at 5 time points
                              - c_Ta(x, t) heatmap
                              - Warren-Cowley alpha vs time (Ta-W, Ta-Ta, W-W)
                              - Vacancy unfolded x-position vs time
    config_used.json        Run settings (for reproducibility).

Usage:
    cd /path/to/GNN_Gym
    conda activate fairchem
    python -m KMC.demos.bicrystal_TaW                # default: GNN, 50k steps
    python -m KMC.demos.bicrystal_TaW --mock         # 10-second smoke test
    python -m KMC.demos.bicrystal_TaW --n-steps 10000

The GNN run on 8x8x8 takes hours on a single GPU; start with --mock first
to confirm everything renders correctly.
"""

import argparse
import sys
import time as _wall_time
from pathlib import Path

import numpy as np

# Make scipts/ importable when called as a module
_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.state import KMCState, VACANCY_SPECIES
from KMC.barrier_predictor import (
    GNNBarrierPredictor,
    MockBarrierPredictor,
)
from KMC.engine import run
from KMC.observables import warren_cowley_sro_trajectory
from KMC.trajectory_writer import (
    write_extxyz_trajectory,
    write_event_log,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Ta|W bicrystal vacancy-diffusion demo."
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use the constant-barrier MockBarrierPredictor instead of the "
             "GNN. Smoke test only; recommended before launching the GNN run.",
    )
    p.add_argument(
        "--n-steps", type=int, default=50000,
        help="Number of BKL steps (default: 50000).",
    )
    p.add_argument(
        "--supercell", type=int, default=8,
        help="BCC supercell size N (-> 2*N^3 sites). Default 8 -> 1024 sites.",
    )
    p.add_argument(
        "--snapshot-every", type=int, default=200,
        help="Snapshot stride for trajectory + WC trajectory.",
    )
    p.add_argument(
        "--temperature", type=float, default=2000.0,
        help="Run temperature in Kelvin (default 2000 K).",
    )
    p.add_argument(
        "--output-dir", type=str, default="kmc_outputs/bicrystal_TaW",
        help="Where to write outputs.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build run configuration
# ---------------------------------------------------------------------------

def _build_config(args) -> KMCConfig:
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.0, "Nb": 0.0, "Ta": 0.5, "W": 0.5},
        supercell_size=args.supercell,
        lattice_parameter_A=3.22,
        initial_state_strategy="bicrystal",
        bicrystal_axis="x",
        bicrystal_elements=("Ta", "W"),
        vacancy_initial_position="center",
        temperature_K=args.temperature,
        attempt_frequency_Hz=1e13,
        n_steps=args.n_steps,
        random_seed=0,
        snapshot_every_n_steps=args.snapshot_every,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------------
# Diagnostics PDF
# ---------------------------------------------------------------------------

def _write_diagnostics_pdf(result, config: KMCConfig, pdf_path: Path) -> None:
    """Multi-panel diagnostic figure (PDF)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the projection)

    state = result.final_state
    element_symbols = state.element_symbols
    el_to_idx = {el: i for i, el in enumerate(element_symbols)}

    # ----- Helpers -----
    def _scatter_3d(ax, species, positions, vacancy_idx, title):
        """Plot Ta blue, W orange, vacancy yellow star."""
        Ta_idx = el_to_idx["Ta"]
        W_idx = el_to_idx["W"]
        mask_Ta = species == Ta_idx
        mask_W = species == W_idx
        ax.scatter(
            positions[mask_Ta, 0], positions[mask_Ta, 1], positions[mask_Ta, 2],
            c="tab:blue", s=18, alpha=0.55, label="Ta",
            depthshade=True,
        )
        ax.scatter(
            positions[mask_W, 0], positions[mask_W, 1], positions[mask_W, 2],
            c="tab:orange", s=18, alpha=0.55, label="W",
            depthshade=True,
        )
        vac_pos = positions[vacancy_idx]
        ax.scatter(
            [vac_pos[0]], [vac_pos[1]], [vac_pos[2]],
            c="gold", s=220, marker="*",
            edgecolor="black", linewidth=0.8,
            label="vacancy",
        )
        ax.set_xlabel("x [Å]")
        ax.set_ylabel("y [Å]")
        ax.set_zlabel("z [Å]")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

    # ----- Bin atom positions along x for c_Ta(x) profiles -----
    site_positions = state.positions
    L_x = float(state.cell[0, 0])
    n_bins = 2 * config.supercell_size  # one bin per BCC layer along x

    bin_edges = np.linspace(0.0, L_x, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    Ta_idx = el_to_idx["Ta"]

    n_snap = result.n_snapshots
    c_Ta_grid = np.zeros((n_snap, n_bins), dtype=np.float64)
    for k in range(n_snap):
        species_k = result.snapshot_species[k]
        non_vac_mask = species_k != VACANCY_SPECIES
        x = site_positions[non_vac_mask, 0]
        sp = species_k[non_vac_mask]
        bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            in_bin = bin_idx == b
            n_total = int(in_bin.sum())
            if n_total == 0:
                c_Ta_grid[k, b] = np.nan
                continue
            n_Ta = int((sp[in_bin] == Ta_idx).sum())
            c_Ta_grid[k, b] = n_Ta / n_total

    # ----- Warren-Cowley trajectory -----
    wc_series = warren_cowley_sro_trajectory(result)
    snapshot_times = result.snapshot_times_s

    # ----- Build the figure -----
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"Ta|W bicrystal vacancy diffusion - "
        f"{config.supercell_size}×{config.supercell_size}×"
        f"{config.supercell_size} BCC, T = {config.temperature_K:.0f} K, "
        f"{result.n_steps} BKL steps, "
        f"t_sim = {result.total_time_s:.2e} s",
        fontsize=12, y=0.995,
    )

    # Top row: two 3D scatters
    ax_init = fig.add_subplot(2, 3, 1, projection="3d")
    _scatter_3d(
        ax_init,
        result.initial_species,
        site_positions,
        result.initial_vacancy_index,
        title="Initial configuration (t = 0)",
    )

    ax_final = fig.add_subplot(2, 3, 2, projection="3d")
    _scatter_3d(
        ax_final,
        result.snapshot_species[-1] if n_snap > 0 else result.initial_species,
        site_positions,
        int(result.snapshot_vacancy_indices[-1])
        if n_snap > 0 else result.initial_vacancy_index,
        title=f"Final configuration (t = {result.total_time_s:.2e} s)",
    )

    # Top right: vacancy x-position vs time
    ax_vac = fig.add_subplot(2, 3, 3)
    times_full = result.times_with_zero_s
    ax_vac.plot(
        times_full,
        result.vacancy_positions_unfolded[:, 0],
        color="black", lw=0.8,
    )
    ax_vac.axhline(L_x / 2.0, color="grey", ls="--", lw=0.8,
                   label="initial interface")
    ax_vac.set_xlabel("t [s]")
    ax_vac.set_ylabel("vacancy x (unfolded) [Å]")
    ax_vac.set_title("Vacancy x-position over time")
    ax_vac.legend(fontsize=8)
    ax_vac.grid(alpha=0.3)

    # Bottom-left: c_Ta(x) profiles at five times
    ax_prof = fig.add_subplot(2, 3, 4)
    if n_snap >= 2:
        idx_pick = np.linspace(0, n_snap - 1, num=min(5, n_snap), dtype=int)
        cmap = plt.get_cmap("viridis")
        for j, k in enumerate(idx_pick):
            color = cmap(j / max(len(idx_pick) - 1, 1))
            ax_prof.plot(
                bin_centers, c_Ta_grid[k],
                marker="o", color=color, lw=1.2,
                label=f"t = {snapshot_times[k]:.2e} s",
            )
        ax_prof.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax_prof.axvline(L_x / 2.0, color="grey", ls="--", lw=0.8,
                        label="initial interface")
    ax_prof.set_xlabel("x [Å]")
    ax_prof.set_ylabel("c_Ta")
    ax_prof.set_ylim(-0.05, 1.05)
    ax_prof.set_title("c_Ta(x) profiles")
    ax_prof.legend(fontsize=7, loc="best")
    ax_prof.grid(alpha=0.3)

    # Bottom-middle: c_Ta(x, t) heatmap
    ax_hm = fig.add_subplot(2, 3, 5)
    if n_snap >= 2:
        im = ax_hm.imshow(
            c_Ta_grid,
            aspect="auto", origin="lower",
            extent=[0.0, L_x, 0.0, snapshot_times[-1]],
            cmap="RdBu_r", vmin=0.0, vmax=1.0,
        )
        cb = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cb.set_label("c_Ta")
        ax_hm.axvline(L_x / 2.0, color="black", ls="--", lw=0.8)
    ax_hm.set_xlabel("x [Å]")
    ax_hm.set_ylabel("t [s]")
    ax_hm.set_title("c_Ta(x, t)")

    # Bottom-right: Warren-Cowley alpha vs time
    ax_wc = fig.add_subplot(2, 3, 6)
    pairs = [
        (("Ta", "W"), "tab:purple", "α(Ta, W)"),
        (("Ta", "Ta"), "tab:blue", "α(Ta, Ta)"),
        (("W", "W"), "tab:orange", "α(W, W)"),
    ]
    for key, color, label in pairs:
        if key in wc_series:
            ax_wc.plot(
                snapshot_times, wc_series[key],
                color=color, lw=1.2, label=label,
            )
    ax_wc.axhline(0.0, color="grey", ls=":", lw=0.8)
    ax_wc.set_xlabel("t [s]")
    ax_wc.set_ylabel("Warren-Cowley α")
    ax_wc.set_title("Short-range order vs time")
    ax_wc.legend(fontsize=8)
    ax_wc.grid(alpha=0.3)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    config = _build_config(args)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(out_dir / "config_used.json")

    n_sites = config.n_sites
    print(f"[demo] Bicrystal Ta|W")
    print(f"  supercell:  {config.supercell_size}^3 BCC -> {n_sites} sites "
          f"(N = {n_sites - 1} atoms)")
    print(f"  T:          {config.temperature_K:.1f} K")
    print(f"  n_steps:    {config.n_steps}")
    print(f"  snapshots:  every {config.snapshot_every_n_steps} -> "
          f"{config.n_steps // config.snapshot_every_n_steps + 1} frames")
    print(f"  output_dir: {out_dir.resolve()}")

    if args.mock:
        print("[demo] Predictor: MockBarrierPredictor(constant_eV=1.0) "
              "(--mock; engine smoke test only, not physical)")
        predictor = MockBarrierPredictor(constant_eV=1.0)
    else:
        print(f"[demo] Predictor: GNNBarrierPredictor(model={config.gnn_model_path})")
        print("[demo] WARNING: a GNN run on this box size + 50k steps takes "
              "hours on a single GPU. Use --mock first to verify the pipeline.")
        predictor = GNNBarrierPredictor(config)

    print("[demo] Building initial bicrystal state...")
    state = KMCState.from_bicrystal(config)

    # Aim for ~200 status lines across the whole run, but cap at 50 steps so
    # users get the first ETA within ~1 minute even on slow GNN configs.
    progress_every = max(1, min(50, config.n_steps // 200))
    print(f"[demo] Starting BKL run (progress update every {progress_every} steps)...")
    t0 = _wall_time.time()
    result = run(
        state, predictor,
        T_K=config.temperature_K,
        attempt_frequency_Hz=config.attempt_frequency_Hz,
        n_steps=config.n_steps,
        snapshot_every_n_steps=config.snapshot_every_n_steps,
        rng=np.random.default_rng(seed=config.random_seed),
        progress_every_n_steps=progress_every,
    )
    wall_s = _wall_time.time() - t0
    print(f"[demo] Run finished in {wall_s:.1f} s "
          f"({wall_s / max(result.n_steps, 1) * 1e3:.2f} ms/step)")
    print(f"[demo] Simulated time: {result.total_time_s:.3e} s")

    # --- Outputs ---
    print("[demo] Writing trajectory.extxyz ...")
    n_frames = write_extxyz_trajectory(
        result, out_dir / "trajectory.extxyz", vacancy_symbol="X"
    )
    print(f"  -> {n_frames} frames written to {(out_dir / 'trajectory.extxyz').resolve()}")

    print("[demo] Writing event_log.csv ...")
    n_rows = write_event_log(result, out_dir / "event_log.csv")
    print(f"  -> {n_rows} rows written to {(out_dir / 'event_log.csv').resolve()}")

    print("[demo] Writing diagnostics.pdf ...")
    _write_diagnostics_pdf(result, config, out_dir / "diagnostics.pdf")
    print(f"  -> {(out_dir / 'diagnostics.pdf').resolve()}")

    print()
    print("[demo] Done.")
    print("      To view the trajectory in OVITO:")
    print(f"          ovito {(out_dir / 'trajectory.extxyz').resolve()}")
    print("      OVITO will recognise 'X' atoms as the vacancy; recolour them "
          "(yellow/gold) for clarity, and consider bond-rendering off plus a "
          "Common-Neighbour-Analysis modifier to highlight the BCC lattice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
