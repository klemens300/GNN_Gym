"""
BKL (Bortz-Kalos-Lebowitz) residence-time KMC engine.

Two entry points:
- bkl_step(...): a single step (used internally by run() and tested in Phase 1).
- run(...):     the full trajectory loop with hop-history recording and optional
                periodic snapshots.
"""

import sys
import time as _wallt
from typing import Optional

import numpy as np


# Boltzmann constant in eV/K (CODATA)
K_B_EV_PER_K = 8.617333262e-5

# Sentinel value used when sanitising pathological predictor outputs.
# Any NaN/Inf or unphysically negative barrier is replaced by this value
# so the corresponding hop is effectively suppressed in the BKL roulette
# wheel (exp(-5 eV / kT) at 1500 K is ~5e-17 -> rate ~ 0). Picked high
# enough to dominate over realistic barriers (typically 0.5-3 eV in the
# MoNbTaW system) without causing exp() underflow to zero everywhere.
BARRIER_FALLBACK_EV = 5.0


# ---------------------------------------------------------------------------
# Single step
# ---------------------------------------------------------------------------

def bkl_step(
    state,
    predictor,
    T_K: float,
    attempt_frequency_Hz: float,
    rng: np.random.Generator,
) -> dict:
    """Execute one BKL step on a vacancy-mediated diffusion KMC.

    Algorithm:
        1. Enumerate the 8 candidate jumps (NN atoms around the vacancy)
        2. Query the predictor for the 8 forward barriers
        3. Compute rates r_k = nu_attempt * exp(-E_k / kT)
        4. Total rate R = sum(r_k)
        5. Time advance dt = -ln(u) / R, with u ~ U(0, 1)
        6. Roulette-wheel selection: pick jump k with probability r_k / R
        7. Mutate state in-place (swap vacancy with chosen NN atom)

    Args:
        state: KMCState (mutated in place).
        predictor: BarrierPredictor implementation.
        T_K: temperature in Kelvin (must be > 0).
        attempt_frequency_Hz: nu_0 (attempt frequency).
        rng: numpy random Generator (for reproducibility).

    Returns:
        dict with diagnostics for the step:
            'delta_t_sim_s'           (float)         simulation time advance
            'chosen_atom_index'       (int)           site index of the jumper
            'chosen_jump_index_local' (int)           which of the 8 NN was picked
            'forward_barriers_eV'     (np.ndarray[8]) barriers from the predictor
            'rates_Hz'                (np.ndarray[8]) per-jump rates
            'total_rate_Hz'           (float)         R = sum(rates)
    """
    if T_K <= 0:
        raise ValueError(f"Temperature must be positive, got {T_K} K")

    nn_atom_indices = state.get_neighbor_atom_indices()
    barriers_eV = predictor.get_forward_barriers_batch(state, nn_atom_indices)
    barriers_eV = np.asarray(barriers_eV, dtype=np.float64)

    if barriers_eV.shape != (len(nn_atom_indices),):
        raise ValueError(
            f"Predictor returned shape {barriers_eV.shape}, expected "
            f"({len(nn_atom_indices)},)"
        )

    # Defensive sanitisation. Pathological predictor outputs (NaN, Inf,
    # or unphysically negative barriers) cause downstream overflow in
    # exp(-barriers/kT) -> Inf rates -> Inf/Inf = NaN probabilities ->
    # rng.choice failure. Such outputs occur in two known regimes:
    #   (a) the simulation drifts into a configuration sufficiently far
    #       from the GNN's training distribution to make the prediction
    #       collapse;
    #   (b) hardware-induced corruption of the forward-pass tensors,
    #       e.g. marginal VRAM, surfacing as a single bit-flip in the
    #       output value.
    # We replace the offending entries with BARRIER_FALLBACK_EV (an
    # effectively prohibitive barrier) so that the corresponding hop is
    # not selected, and log the event with full context for offline
    # analysis. The KMC trajectory continues with the remaining viable
    # hops; if all 8 are bad, a normal RuntimeError downstream catches
    # it (R_total = 0).
    bad_mask = ~np.isfinite(barriers_eV) | (barriers_eV < 0.0)
    if bad_mask.any():
        print(
            f"  [bkl_step] WARNING: {int(bad_mask.sum())}/"
            f"{len(barriers_eV)} barriers were non-finite or negative; "
            f"raw = {barriers_eV.tolist()}; replacing with "
            f"{BARRIER_FALLBACK_EV} eV.",
            flush=True,
        )
        barriers_eV = np.where(bad_mask, BARRIER_FALLBACK_EV, barriers_eV)

    # Boltzmann rates
    kT = K_B_EV_PER_K * T_K
    rates = attempt_frequency_Hz * np.exp(-barriers_eV / kT)
    R_total = float(rates.sum())
    if R_total <= 0:
        raise RuntimeError(
            f"Total rate is non-positive ({R_total}); barriers may be "
            f"non-finite or extremely large. barriers_eV = {barriers_eV}"
        )

    # Residence-time advance
    u = float(rng.random())
    # Avoid log(0); rng.random() draws from [0, 1) so u==0 is rare but possible
    u = max(u, np.finfo(float).tiny)
    delta_t = -np.log(u) / R_total

    # Roulette-wheel selection over the 8 candidate jumps
    probs = rates / R_total
    choice_local = int(rng.choice(len(nn_atom_indices), p=probs))
    chosen_atom_idx = int(nn_atom_indices[choice_local])

    # Mutate state in place
    state.swap_vacancy(chosen_atom_idx)

    return {
        "delta_t_sim_s": delta_t,
        "chosen_atom_index": chosen_atom_idx,
        "chosen_jump_index_local": choice_local,
        "forward_barriers_eV": barriers_eV,
        "rates_Hz": rates,
        "total_rate_Hz": R_total,
    }


# ---------------------------------------------------------------------------
# Full trajectory loop
# ---------------------------------------------------------------------------

def _fmt_seconds(seconds: float) -> str:
    """Format an elapsed wall-clock interval as H:MM:SS."""
    seconds = int(round(max(seconds, 0.0)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def run(
    state,
    predictor,
    T_K: float,
    attempt_frequency_Hz: float,
    n_steps: Optional[int] = None,
    t_max_sim_s: Optional[float] = None,
    snapshot_every_n_steps: int = 0,
    rng: Optional[np.random.Generator] = None,
    progress_every_n_steps: Optional[int] = None,
    checkpoint_every_n_steps: Optional[int] = None,
    checkpoint_path=None,
    empty_cuda_cache_every_n_steps: Optional[int] = None,
    resume_from_state: Optional[dict] = None,
    max_steps_this_session: Optional[int] = None,
):
    """Run a KMC trajectory until n_steps OR t_max_sim_s is reached.

    The loop stops at whichever criterion is hit first; at least one of
    n_steps and t_max_sim_s must be specified.

    Args:
        state: KMCState (mutated in place during the run).
        predictor: BarrierPredictor implementation.
        T_K: temperature (K).
        attempt_frequency_Hz: nu_0 (attempt frequency).
        n_steps: stop after this many steps (optional).
        t_max_sim_s: stop after total simulated time exceeds this (optional).
        snapshot_every_n_steps: if > 0, save a full species snapshot at step 0
            and every k steps thereafter. Default 0 (no snapshots).
        rng: numpy random Generator (default: a fresh np.random.default_rng()).
        progress_every_n_steps: if > 0, print a one-line status update to
            stdout every k steps with current step count, simulated time,
            wall-clock elapsed, ETA (only when n_steps is known), and the
            running ms-per-step. Default None disables the print.
        checkpoint_every_n_steps: if > 0 and ``checkpoint_path`` is set,
            atomically write a partial-run .npz to ``checkpoint_path`` every
            k completed steps. Each write overwrites the previous file via
            a tmp-and-rename, so a crash can never leave a corrupted
            checkpoint behind. Reload the partial result with
            ``KMC.checkpoint.load_run_checkpoint(path)``.
        checkpoint_path: target .npz path; required for checkpointing to
            actually happen. Default None disables checkpointing.
        empty_cuda_cache_every_n_steps: if > 0, call torch.cuda.empty_cache()
            every k completed steps to keep the VRAM footprint flat across
            multi-hour runs (counters gradual fragmentation). Costs roughly
            50-100 ms per call, so use sparingly (e.g. 1000). Silently
            skipped if torch is unavailable or CUDA is not active. Default
            None disables it.
        resume_from_state: optional dict produced by
            ``KMC.checkpoint.load_resume_state(path)``. When provided, the
            engine skips fresh-state initialisation and continues the BKL
            loop from the saved step using the saved RNG state. ``state``
            should already have been reconstructed from the checkpoint by
            the caller (the runner does this); the dict supplies the per-
            step buffers, the t=0 reference, simulated-time progress, and
            the engine RNG state. Existing ``rng`` is overwritten by the
            saved state when present. Default None starts a fresh run.
        max_steps_this_session: if not None and > 0, the loop stops
            voluntarily after this many *additional* steps in the current
            invocation, regardless of whether ``n_steps`` is reached.
            Used by the auto-recycle wrapper to bound a single Python
            process's lifetime, so accumulated GPU/driver state never
            reaches the empirically observed crash window. The next
            ``--resume`` invocation continues bit-identically. The final
            checkpoint is always written before returning.

    Returns:
        KMCResult with the hop history, optional snapshots, and final state.
        Note: when stopped via ``max_steps_this_session``, ``n_steps`` in
        the result reflects the steps actually completed, which may be
        less than the input ``n_steps`` argument.
    """
    # Local import to keep engine module importable on its own (no circular dep)
    from KMC.result import KMCResult

    if n_steps is None and t_max_sim_s is None:
        raise ValueError(
            "At least one of n_steps or t_max_sim_s must be specified"
        )
    if rng is None:
        rng = np.random.default_rng()

    # --- Cell vector lengths for PBC unfolding (assumes orthogonal cubic cell) ---
    cell = state.cell
    L = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])

    if resume_from_state is None:
        # --- Fresh start: reference state for replay ---
        initial_species = state.species.copy()
        initial_vacancy_index = int(state.vacancy_index)
        initial_pos = state.positions[state.vacancy_index].copy()

        # --- Per-step buffers (lists; converted to ndarrays at end) ---
        times_list = []
        delta_t_list = []
        vacancy_indices_list = [initial_vacancy_index]
        vacancy_positions_unfolded_list = [initial_pos.copy()]
        hopper_species_list = []
        chosen_jump_local_list = []
        barriers_list = []

        # --- Snapshot buffers ---
        snapshots_times = []
        snapshots_species = []
        snapshots_vacancy = []
        if snapshot_every_n_steps > 0:
            snapshots_times.append(0.0)
            snapshots_species.append(initial_species.copy())
            snapshots_vacancy.append(initial_vacancy_index)

        # --- Main loop progress markers ---
        current_unfolded = initial_pos.copy()
        total_time = 0.0
        step = 0
    else:
        # --- Resume from checkpoint: take buffers + progress markers from
        #     the saved state. ``state`` is assumed to already represent the
        #     simulation after ``step`` steps (the runner reconstructs it).
        rs = resume_from_state
        initial_species = np.asarray(rs["initial_species"]).copy()
        initial_vacancy_index = int(rs["initial_vacancy_index"])

        # Lists are taken as-is (load_resume_state returns plain lists).
        times_list = list(rs["times_list"])
        delta_t_list = list(rs["delta_t_list"])
        vacancy_indices_list = list(rs["vacancy_indices_list"])
        vacancy_positions_unfolded_list = list(
            rs["vacancy_positions_unfolded_list"]
        )
        hopper_species_list = list(rs["hopper_species_list"])
        chosen_jump_local_list = list(rs["chosen_jump_local_list"])
        barriers_list = list(rs["barriers_list"])

        snapshots_times = list(rs["snapshots_times"])
        snapshots_species = list(rs["snapshots_species"])
        snapshots_vacancy = list(rs["snapshots_vacancy"])

        step = int(rs["step"])
        total_time = float(rs["total_time_s"])
        # Latest unfolded vacancy position is the most recently appended
        # element of vacancy_positions_unfolded_list (length = step + 1).
        current_unfolded = np.asarray(
            vacancy_positions_unfolded_list[-1], dtype=np.float64
        ).copy()

        # Restore the engine RNG state for bit-identical continuation.
        rng.bit_generator.state = rs["engine_rng_state"]

    wall_t_start = _wallt.time()
    session_start_step = step

    progress_active = (
        progress_every_n_steps is not None and progress_every_n_steps > 0
    )
    checkpoint_active = (
        checkpoint_every_n_steps is not None
        and checkpoint_every_n_steps > 0
        and checkpoint_path is not None
    )
    empty_cache_active = (
        empty_cuda_cache_every_n_steps is not None
        and empty_cuda_cache_every_n_steps > 0
    )
    session_cap_active = (
        max_steps_this_session is not None and max_steps_this_session > 0
    )
    if progress_active:
        n_steps_label = (
            f"n_steps={n_steps}" if n_steps is not None
            else f"t_max_sim_s={t_max_sim_s}"
        )
        print(
            f"[run] starting BKL loop (T={T_K:.1f} K, {n_steps_label}, "
            f"N_sites={state.n_sites})",
            flush=True,
        )
        sys.stdout.flush()

    while True:
        # Stop conditions (checked BEFORE the step so we don't overshoot)
        if n_steps is not None and step >= n_steps:
            break
        if t_max_sim_s is not None and total_time >= t_max_sim_s:
            break
        # Voluntary session-cap: stop this Python process after a fixed
        # number of additional steps so the wrapper can recycle a fresh
        # process. Independent of n_steps.
        if (
            session_cap_active
            and (step - session_start_step) >= max_steps_this_session
        ):
            if progress_active:
                print(
                    f"[run] session step cap reached "
                    f"({max_steps_this_session} steps this session); "
                    f"writing final checkpoint and returning.",
                    flush=True,
                )
            break

        v_before = state.vacancy_index
        info = bkl_step(state, predictor, T_K, attempt_frequency_Hz, rng)
        v_after = state.vacancy_index

        # PBC-unfolded vacancy displacement for this step
        delta = state.positions[v_after] - state.positions[v_before]
        for ax in range(3):
            delta[ax] -= round(delta[ax] / L[ax]) * L[ax]
        current_unfolded = current_unfolded + delta

        # After swap_vacancy, the species that hopped now lives at v_before
        hopper_species = int(state.species[v_before])

        total_time += info["delta_t_sim_s"]
        times_list.append(total_time)
        delta_t_list.append(info["delta_t_sim_s"])
        vacancy_indices_list.append(v_after)
        vacancy_positions_unfolded_list.append(current_unfolded.copy())
        hopper_species_list.append(hopper_species)
        chosen_jump_local_list.append(info["chosen_jump_index_local"])
        barriers_list.append(info["forward_barriers_eV"])

        step += 1

        if snapshot_every_n_steps > 0 and step % snapshot_every_n_steps == 0:
            snapshots_times.append(total_time)
            snapshots_species.append(state.species.copy())
            snapshots_vacancy.append(int(state.vacancy_index))

        # Periodic VRAM-cache flush: counters gradual fragmentation on
        # multi-hour GPU runs. No-op when torch / CUDA are not available.
        if empty_cache_active and step % empty_cuda_cache_every_n_steps == 0:
            try:
                import torch as _torch_for_cache
                if _torch_for_cache.cuda.is_available():
                    _torch_for_cache.cuda.empty_cache()
            except ImportError:
                pass

        # Rolling checkpoint: dump partial-run state every k steps so a
        # crash leaves us with the latest cleanly-written .npz.
        if checkpoint_active and step % checkpoint_every_n_steps == 0:
            try:
                from KMC.checkpoint import write_run_checkpoint
                # Snapshot the engine RNG state alongside the buffers so a
                # later --resume continues bit-identically.
                rng_state_snapshot = dict(rng.bit_generator.state)
                write_run_checkpoint(
                    checkpoint_path,
                    step=step,
                    total_time_s=total_time,
                    T_K=T_K,
                    attempt_frequency_Hz=attempt_frequency_Hz,
                    state=state,
                    initial_species=initial_species,
                    initial_vacancy_index=initial_vacancy_index,
                    times_list=times_list,
                    delta_t_list=delta_t_list,
                    vacancy_indices_list=vacancy_indices_list,
                    hopper_species_list=hopper_species_list,
                    chosen_jump_local_list=chosen_jump_local_list,
                    barriers_list=barriers_list,
                    vacancy_positions_unfolded_list=(
                        vacancy_positions_unfolded_list
                    ),
                    snapshots_times=snapshots_times,
                    snapshots_species=snapshots_species,
                    snapshots_vacancy=snapshots_vacancy,
                    engine_rng_state=rng_state_snapshot,
                )
                if progress_active:
                    print(
                        f"[run] checkpoint saved at step {step} "
                        f"-> {checkpoint_path}",
                        flush=True,
                    )
                    sys.stdout.flush()
            except Exception as exc:
                # Failing to checkpoint must never abort the run; log and
                # carry on. The previous (good) checkpoint still exists.
                print(
                    f"[run] WARNING: checkpoint write failed at step "
                    f"{step}: {exc}",
                    flush=True,
                )

        # Print after every k steps; also print after the very first step so
        # the user sees a confirmation that the engine is alive without
        # waiting for the full first interval (k=200 can take many minutes
        # on 8^3 with the GNN predictor).
        if (progress_active
                and (step % progress_every_n_steps == 0 or step == 1)):
            elapsed = _wallt.time() - wall_t_start
            # Rate is measured over the steps done IN THIS SESSION only.
            # ``step`` is the absolute counter, which after a --resume
            # already includes all steps from previous sessions; dividing
            # session elapsed by the absolute step would dilute the rate
            # (and the ETA) by the already-completed work.
            steps_this_session = max(step - session_start_step, 1)
            s_per_step = elapsed / steps_this_session
            ms_per_step = s_per_step * 1000.0
            sim_t_str = f"{total_time:.3e} s"
            if n_steps is not None and n_steps > 0:
                frac = step / n_steps
                eta_s = max(n_steps - step, 0) * s_per_step
                msg = (
                    f"[run] {step}/{n_steps} ({frac*100:5.1f}%) | "
                    f"sim t = {sim_t_str} | "
                    f"elapsed {_fmt_seconds(elapsed)} | "
                    f"ETA {_fmt_seconds(eta_s)} | "
                    f"{ms_per_step:6.2f} ms/step"
                )
            else:
                msg = (
                    f"[run] step {step} | "
                    f"sim t = {sim_t_str} | "
                    f"elapsed {_fmt_seconds(elapsed)} | "
                    f"{ms_per_step:6.2f} ms/step"
                )
            print(msg, flush=True)
            sys.stdout.flush()

    # --- Final checkpoint on session-cap exit ---
    # When the loop stops because of the session cap (rather than n_steps
    # or t_max_sim_s), make sure the very latest step is persisted, even
    # if it is not on a checkpoint_every_n_steps boundary. This way the
    # auto-recycle wrapper never loses a partial chunk.
    if checkpoint_active and session_cap_active and step > session_start_step:
        try:
            from KMC.checkpoint import write_run_checkpoint
            rng_state_snapshot = dict(rng.bit_generator.state)
            write_run_checkpoint(
                checkpoint_path,
                step=step,
                total_time_s=total_time,
                T_K=T_K,
                attempt_frequency_Hz=attempt_frequency_Hz,
                state=state,
                initial_species=initial_species,
                initial_vacancy_index=initial_vacancy_index,
                times_list=times_list,
                delta_t_list=delta_t_list,
                vacancy_indices_list=vacancy_indices_list,
                hopper_species_list=hopper_species_list,
                chosen_jump_local_list=chosen_jump_local_list,
                barriers_list=barriers_list,
                vacancy_positions_unfolded_list=(
                    vacancy_positions_unfolded_list
                ),
                snapshots_times=snapshots_times,
                snapshots_species=snapshots_species,
                snapshots_vacancy=snapshots_vacancy,
                engine_rng_state=rng_state_snapshot,
            )
            if progress_active:
                print(
                    f"[run] final session checkpoint at step {step} "
                    f"-> {checkpoint_path}",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[run] WARNING: final session checkpoint write failed "
                f"at step {step}: {exc}",
                flush=True,
            )

    # --- Pack the result ---
    snap_t = (np.asarray(snapshots_times, dtype=np.float64)
              if snapshots_times else None)
    snap_sp = (np.stack(snapshots_species, axis=0)
               if snapshots_species else None)
    snap_v = (np.asarray(snapshots_vacancy, dtype=np.int32)
              if snapshots_vacancy else None)

    return KMCResult(
        n_steps=step,
        total_time_s=total_time,
        T_K=float(T_K),
        attempt_frequency_Hz=float(attempt_frequency_Hz),
        final_state=state,
        initial_species=initial_species,
        initial_vacancy_index=initial_vacancy_index,
        times_s=np.asarray(times_list, dtype=np.float64),
        delta_t_s=np.asarray(delta_t_list, dtype=np.float64),
        vacancy_indices=np.asarray(vacancy_indices_list, dtype=np.int32),
        hopper_species=np.asarray(hopper_species_list, dtype=np.int8),
        chosen_jump_local=np.asarray(chosen_jump_local_list, dtype=np.int8),
        barriers_eV=np.asarray(barriers_list, dtype=np.float32),
        vacancy_positions_unfolded=np.asarray(
            vacancy_positions_unfolded_list, dtype=np.float64
        ),
        snapshot_times_s=snap_t,
        snapshot_species=snap_sp,
        snapshot_vacancy_indices=snap_v,
    )
