"""
Diagnostic v2: pin down where the time-averaged MSD plateaus, and check
whether the unfolded vacancy trajectory has a systematic drift.

Run:
    cd /home/klemens/doctor/gnn_kmc/scipts/KMC/tests
    python debug_msd.py
"""

import sys
from pathlib import Path

import numpy as np

_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.state import KMCState
from KMC.barrier_predictor import MockBarrierPredictor
from KMC.engine import run, K_B_EV_PER_K


def main():
    np.set_printoptions(precision=4, suppress=False)

    cfg = KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=6,
        lattice_parameter_A=3.22,
        random_seed=0,
        temperature_K=2000.0,
        attempt_frequency_Hz=1e13,
        n_steps=20000,
    )
    barrier_eV = 0.5
    nu = cfg.attempt_frequency_Hz
    T = cfg.temperature_K
    a_A = cfg.lattice_parameter_A

    state = KMCState.from_random_composition(cfg)
    predictor = MockBarrierPredictor(constant_eV=barrier_eV)
    rng = np.random.default_rng(seed=0)
    result = run(
        state, predictor,
        T_K=T,
        attempt_frequency_Hz=nu,
        n_steps=cfg.n_steps,
        rng=rng,
    )

    # --- (1) Mean step vector — should be (0, 0, 0) for unbiased RW ---
    diffs = np.diff(result.vacancy_positions_unfolded, axis=0)
    print("=" * 70)
    print("(1) Step-vector statistics (should be unbiased)")
    print("=" * 70)
    print(f"  n_steps                = {len(diffs)}")
    print(f"  mean step vector       = {diffs.mean(axis=0)}")
    print(f"  std  step vector       = {diffs.std(axis=0)}")
    print(f"  expected component value: ±{a_A/2:.4f}")
    expected_std = a_A / 2  # |component| = a/2 always; mean=0; std = a/2
    print(f"  expected std           = {expected_std:.4f}")
    # Per-component histogram
    for ax_name, ax in zip("xyz", range(3)):
        comp = diffs[:, ax]
        n_pos = int((comp > 0).sum())
        n_neg = int((comp < 0).sum())
        n_zero = int((comp == 0).sum())
        print(f"  axis {ax_name}: pos={n_pos}, neg={n_neg}, zero={n_zero}")

    # --- (2) Where does the unfolded trajectory go? ---
    print()
    print("=" * 70)
    print("(2) Unfolded trajectory range (each axis)")
    print("=" * 70)
    r = result.vacancy_positions_unfolded
    print(f"  initial position = {r[0]}")
    print(f"  final  position  = {r[-1]}")
    print(f"  displacement     = {r[-1] - r[0]}  ||...|| = {np.linalg.norm(r[-1] - r[0]):.4f}")
    print(f"  per-axis min/max:")
    for ax_name, ax in zip("xyz", range(3)):
        print(f"    {ax_name}: min={r[:, ax].min():.3f}  max={r[:, ax].max():.3f}  "
              f"range={r[:, ax].max() - r[:, ax].min():.3f}")
    print(f"  Box length L = {state.cell[0,0]:.3f}")

    # --- (3) RMS unfolded displacement vs time-step ---
    print()
    print("=" * 70)
    print("(3) Single-trajectory unfolded RMS displacement |r(k)-r(0)|")
    print("=" * 70)
    sq = np.einsum("ij,ij->i", r - r[0], r - r[0])
    rms = np.sqrt(sq)
    for k in [10, 100, 1000, 5000, 10000, 15000, 20000]:
        if k < len(rms):
            expected_rms = np.sqrt(k * (a_A * np.sqrt(3) / 2) ** 2)
            print(f"  k={k:6d}: rms={rms[k]:8.3f}  "
                  f"(expect ~ sqrt(k * jd^2) = {expected_rms:8.3f})")

    # --- (4) Are the species-occupancy-pre/post-swap correct? ---
    print()
    print("=" * 70)
    print("(4) Diagnostic: vacancy_indices first 20 steps + step vectors")
    print("=" * 70)
    for k in range(min(20, result.n_steps)):
        v_b = int(result.vacancy_indices[k])
        v_a = int(result.vacancy_indices[k + 1])
        step = diffs[k]
        # Find this step in the nn_table
        is_nn = v_a in result.final_state.nn_table[v_b]
        print(f"  step {k:3d}: v={v_b:4d} -> {v_a:4d}  "
              f"step=({step[0]:+.3f}, {step[1]:+.3f}, {step[2]:+.3f})  "
              f"|step|={np.linalg.norm(step):.4f}  is_nn={is_nn}")


if __name__ == "__main__":
    main()
