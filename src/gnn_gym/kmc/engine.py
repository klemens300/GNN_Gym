"""Rejection-free kinetic Monte Carlo (Bortz-Kalos-Lebowitz).

One step: the predictor returns the barriers E_j of all jumps of the vacancy
to its jump-shell neighbours, the rates are r_j = nu0 exp(-E_j / k_B T), the
clock advances by dt = -ln(u) / sum_j r_j with u uniform in (0, 1], and jump
j is chosen with probability r_j / sum_k r_k. The random numbers are drawn in
this order from one numpy Generator, so a run is a deterministic function of
its seed and its barriers.

With an Escaper attached, a flagged basin is left analytically in one step
(gnn_gym.kmc.escape) instead of by repeated BKL steps.
"""
import time

import numpy as np

from .escape import sanitise

K_B = 8.617333262e-5        # eV/K
# A non-finite or negative barrier from the predictor is replaced by this
# value, which suppresses the jump at every temperature of interest.
BARRIER_FALLBACK_EV = 5.0


def select_jump(barriers, T_K, attempt_frequency_Hz, rng):
    """(chosen candidate, residence time) of one BKL step."""
    rates = attempt_frequency_Hz * np.exp(-barriers / (K_B * T_K))
    total = float(rates.sum())
    if not total > 0:
        raise RuntimeError(f"total rate {total} for barriers {barriers}")
    u = max(float(rng.random()), np.finfo(float).tiny)
    dt = -np.log(u) / total
    chosen = int(rng.choice(len(rates), p=rates / total))
    return chosen, dt


def bkl_step(state, predictor, T_K, attempt_frequency_Hz, rng):
    """One BKL step; mutates `state`. Returns (target, slot, dt, barriers)."""
    targets = state.targets()
    barriers = sanitise(predictor.barriers(state.species, state.vacancy,
                                           targets), BARRIER_FALLBACK_EV)
    chosen, dt = select_jump(barriers, T_K, attempt_frequency_Hz, rng)
    target = int(targets[chosen])
    state.swap(target)
    return target, chosen, dt, barriers


def run(state, predictor, traj, T_K, attempt_frequency_Hz, n_steps,
        lattice_parameter_A, rng, escaper=None, snapshot_every=0,
        checkpoint=None, checkpoint_every=0, progress_every=0):
    """Advance `state` and `traj` until traj holds n_steps steps.

    checkpoint   callable() writing the current run to disk; called every
                 `checkpoint_every` completed steps and after the last one
    """
    cell = state.cell
    step_length = lattice_parameter_A / cell.lattice.units_per_a
    nan_row = np.full(traj.n_candidates, np.nan)
    start, t_wall = traj.n_steps, time.time()

    while traj.n_steps < n_steps:
        v_before = state.vacancy
        escaped = False
        if escaper is not None and escaper.flagged:
            out = escaper.escape(state, predictor, T_K, attempt_frequency_Hz,
                                 rng, step_length)
            if out is not None:
                traj.record(out.tau_s, state.vacancy, out.displacement,
                            state.species[v_before], -1, nan_row,
                            micro_path=out.micro_path)
                escaped = True
        if not escaped:
            target, slot, dt, barriers = bkl_step(
                state, predictor, T_K, attempt_frequency_Hz, rng)
            d = cell.min_image(cell.coords[target] - cell.coords[v_before])
            traj.record(dt, target, d * step_length, state.species[v_before],
                        slot, barriers)
            if escaper is not None:
                escaper.observe(state)

        step = traj.n_steps
        if snapshot_every > 0 and (step % snapshot_every == 0 or escaped):
            traj.snapshot(state)
        if checkpoint is not None and step % checkpoint_every == 0:
            checkpoint()
        if progress_every > 0 and (step % progress_every == 0
                                   or step == start + 1):
            ms = (time.time() - t_wall) / (step - start) * 1e3
            print(f"[kmc] {step}/{n_steps}  t_sim {traj.total_time:.4e} s  "
                  f"{ms:.2f} ms/step", flush=True)

    if checkpoint is not None and traj.n_steps % checkpoint_every != 0:
        checkpoint()
    return traj
