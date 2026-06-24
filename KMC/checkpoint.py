"""
Rolling checkpoint I/O for long KMC runs.

Goal: never lose more than ``checkpoint_every_n_steps`` steps of progress to
a crash, OOM, accidental Ctrl-C, or session timeout. The checkpoint is a
single compressed .npz file that holds enough state to rebuild a full
``KMCResult`` after the fact, so all existing observables (MSD, Warren-
Cowley SRO, PDF summary, ...) work on a partially-completed run.

Design:
- One file per realisation; subsequent checkpoints overwrite the same path.
- Atomic write: data goes into ``<path>.tmp`` first and is renamed in one
  step via os.replace, so an interrupted write cannot corrupt the previous
  good checkpoint.
- The checkpoint stores the per-step buffers up to the current step, plus
  the periodic snapshots collected so far, plus enough geometry metadata
  (supercell size, lattice parameter, element symbols) to rebuild a fresh
  ``KMCState`` and therefore a fresh ``KMCResult`` at load time.
- The numpy random Generator state of the engine RNG is also stored when
  available, which makes the checkpoint usable as a true resume point
  via :func:`load_resume_state`. The downstream trajectory after resume
  is bit-identical to an uninterrupted run from the same seed.

What is NOT stored:
- The predictor's static-lattice graph cache. It is rebuilt on the first
  BKL step after resume in well under a second, so persisting it would
  add complexity for no real gain.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_run_checkpoint(
    path: Union[str, Path],
    *,
    step: int,
    total_time_s: float,
    T_K: float,
    attempt_frequency_Hz: float,
    state,
    initial_species: np.ndarray,
    initial_vacancy_index: int,
    times_list: Sequence[float],
    delta_t_list: Sequence[float],
    vacancy_indices_list: Sequence[int],
    hopper_species_list: Sequence[int],
    chosen_jump_local_list: Sequence[int],
    barriers_list: Sequence[np.ndarray],
    vacancy_positions_unfolded_list: Sequence[np.ndarray],
    snapshots_times: Sequence[float],
    snapshots_species: Sequence[np.ndarray],
    snapshots_vacancy: Sequence[int],
    engine_rng_state: Optional[dict] = None,
) -> Path:
    """Atomically write a partial-run checkpoint to ``path``.

    Args:
        path: target .npz file. Parent directory is created if missing.
        step: number of completed BKL steps at the time of the checkpoint.
        total_time_s: cumulative simulated time after ``step`` steps.
        T_K, attempt_frequency_Hz: run parameters (kept for KMCResult
            reconstruction).
        state: current KMCState (its species + vacancy index are saved).
        initial_species, initial_vacancy_index: state at t=0 (kept for
            KMCResult.initial_*).
        times_list, delta_t_list, ...: the per-step buffers as used inside
            engine.run. Lengths follow the same conventions as the final
            KMCResult (most are length ``step``; the *_indices and
            *_positions_unfolded buffers are length ``step + 1``).
        snapshots_*: the periodic-snapshot buffers (each length is the
            number of snapshots collected so far).
        engine_rng_state: state dict of the engine's numpy random Generator
            (``rng.bit_generator.state``). When present, the checkpoint
            can be used to resume the run bit-identically; when absent
            (e.g. legacy checkpoints), :func:`load_resume_state` will
            raise.

    Returns:
        The final path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Derive geometry parameters from the state (it does not store them
    # explicitly because the geometry cache in KMC.state is keyed by them).
    n_sites = int(state.n_sites)
    supercell_size = int(round((n_sites / 2) ** (1.0 / 3.0)))
    if 2 * supercell_size ** 3 != n_sites:
        raise ValueError(
            f"Cannot infer supercell_size from n_sites={n_sites}; "
            "expected n_sites = 2 * supercell_size^3 (BCC convention)."
        )
    lattice_parameter_A = float(state.cell[0, 0]) / float(supercell_size)

    # Pack snapshot buffers; if no snapshots were recorded yet, write tiny
    # zero-length arrays so the .npz still has the same key set.
    if snapshots_species:
        snap_species_arr = np.stack(snapshots_species, axis=0)
    else:
        snap_species_arr = np.zeros((0, n_sites), dtype=np.int8)
    snap_times_arr = np.asarray(snapshots_times, dtype=np.float64)
    snap_vac_arr = np.asarray(snapshots_vacancy, dtype=np.int32)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    # IMPORTANT: pass an open file handle to np.savez_compressed, not a path.
    # When given a string/Path that does not already end in '.npz', numpy
    # silently appends '.npz' to the filename. Our tmp path ends in '.npz.tmp'
    # (so that the final rename target keeps the '.npz' suffix), which would
    # cause numpy to actually write to '<path>.npz.tmp.npz' and the os.replace
    # below would fail with FileNotFoundError. Writing through an open handle
    # bypasses the auto-extension logic entirely.
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            # Geometry
            supercell_size=np.asarray(supercell_size, dtype=np.int64),
            lattice_parameter_A=np.asarray(
                lattice_parameter_A, dtype=np.float64
            ),
            element_symbols=np.asarray(
                list(state.element_symbols), dtype=object
            ),
            jump_distance_A=np.asarray(
                state.jump_distance_A, dtype=np.float64
            ),
            # Run parameters
            T_K=np.asarray(T_K, dtype=np.float64),
            attempt_frequency_Hz=np.asarray(
                attempt_frequency_Hz, dtype=np.float64
            ),
            n_steps=np.asarray(step, dtype=np.int64),
            total_time_s=np.asarray(total_time_s, dtype=np.float64),
            # Initial reference
            initial_species=np.asarray(initial_species, dtype=np.int8),
            initial_vacancy_index=np.asarray(
                initial_vacancy_index, dtype=np.int64
            ),
            # Current state
            current_species=np.asarray(state.species, dtype=np.int8),
            current_vacancy_index=np.asarray(
                state.vacancy_index, dtype=np.int64
            ),
            # Hop history
            times_s=np.asarray(times_list, dtype=np.float64),
            delta_t_s=np.asarray(delta_t_list, dtype=np.float64),
            vacancy_indices=np.asarray(
                vacancy_indices_list, dtype=np.int32
            ),
            hopper_species=np.asarray(hopper_species_list, dtype=np.int8),
            chosen_jump_local=np.asarray(
                chosen_jump_local_list, dtype=np.int8
            ),
            barriers_eV=(
                np.stack(barriers_list, axis=0).astype(np.float32)
                if barriers_list else np.zeros((0, 8), dtype=np.float32)
            ),
            vacancy_positions_unfolded=(
                np.stack(vacancy_positions_unfolded_list, axis=0).astype(
                    np.float64
                )
                if vacancy_positions_unfolded_list
                else np.zeros((0, 3), dtype=np.float64)
            ),
            # Snapshots (zero-length arrays if no snapshots were captured)
            snapshot_times_s=snap_times_arr,
            snapshot_species=snap_species_arr,
            snapshot_vacancy_indices=snap_vac_arr,
            # Engine RNG state for resume. Stored as a 0-d object array so
            # numpy's pickle path serialises the state dict verbatim.
            # If not provided, an explicit None marker keeps the key set
            # consistent across writes; load_resume_state checks for this.
            engine_rng_state=np.array(
                engine_rng_state if engine_rng_state is not None else {},
                dtype=object,
            ),
            engine_rng_state_present=np.asarray(
                engine_rng_state is not None, dtype=bool
            ),
        )
    # Atomic rename. os.replace is atomic on POSIX and on Windows >= 3.3.
    os.replace(tmp_path, path)
    return path


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_run_checkpoint(path: Union[str, Path]):
    """Load a checkpoint file and rebuild a full ``KMCResult``.

    The reconstructed KMCResult.final_state is the ``current_state`` at the
    time of the checkpoint (equivalent to what would be returned at the end
    of the run if the checkpoint marked the end). All per-step buffers and
    snapshots are populated from the saved arrays, so any observable that
    consumes a KMCResult works directly.
    """
    # Local imports to avoid pulling KMCState / KMCResult on module import.
    from KMC.result import KMCResult
    from KMC.state import KMCState, _build_bcc_geometry

    path = Path(path)
    data = np.load(path, allow_pickle=True)

    supercell_size = int(data["supercell_size"])
    lattice_parameter_A = float(data["lattice_parameter_A"])
    # element_symbols was stored as a numpy object array; coerce back to
    # a clean list[str].
    element_symbols = [str(s) for s in data["element_symbols"].tolist()]
    jump_distance_A = float(data["jump_distance_A"])

    positions, cell, nn_table = _build_bcc_geometry(
        supercell_size, lattice_parameter_A
    )

    final_state = KMCState(
        positions=positions,
        cell=cell,
        nn_table=nn_table,
        species=np.asarray(data["current_species"]).copy(),
        vacancy_index=int(data["current_vacancy_index"]),
        element_symbols=element_symbols,
        jump_distance_A=jump_distance_A,
    )

    snap_times = np.asarray(data["snapshot_times_s"])
    snap_species = np.asarray(data["snapshot_species"])
    snap_vac = np.asarray(data["snapshot_vacancy_indices"])
    if snap_times.size == 0:
        snap_times_out = None
        snap_species_out = None
        snap_vac_out = None
    else:
        snap_times_out = snap_times.copy()
        snap_species_out = snap_species.copy()
        snap_vac_out = snap_vac.copy()

    return KMCResult(
        n_steps=int(data["n_steps"]),
        total_time_s=float(data["total_time_s"]),
        T_K=float(data["T_K"]),
        attempt_frequency_Hz=float(data["attempt_frequency_Hz"]),
        final_state=final_state,
        initial_species=np.asarray(data["initial_species"]).copy(),
        initial_vacancy_index=int(data["initial_vacancy_index"]),
        times_s=np.asarray(data["times_s"]).copy(),
        delta_t_s=np.asarray(data["delta_t_s"]).copy(),
        vacancy_indices=np.asarray(data["vacancy_indices"]).copy(),
        hopper_species=np.asarray(data["hopper_species"]).copy(),
        chosen_jump_local=np.asarray(data["chosen_jump_local"]).copy(),
        barriers_eV=np.asarray(data["barriers_eV"]).copy(),
        vacancy_positions_unfolded=np.asarray(
            data["vacancy_positions_unfolded"]
        ).copy(),
        snapshot_times_s=snap_times_out,
        snapshot_species=snap_species_out,
        snapshot_vacancy_indices=snap_vac_out,
    )


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------

def load_resume_state(path: Union[str, Path]) -> dict:
    """Load a checkpoint and return everything ``engine.run`` needs to resume.

    Unlike :func:`load_run_checkpoint`, which packages a partial run as a
    completed ``KMCResult`` (for inspection only), this function returns
    the live engine state that allows the BKL loop to continue from the
    checkpointed step bit-identically. Concretely the returned dict
    contains:

    - ``state`` (KMCState): the simulation state *after* ``step`` steps.
    - ``initial_species``, ``initial_vacancy_index``: the t=0 reference
      needed for replay-based observables.
    - ``step``, ``total_time_s``: progress markers for the loop.
    - ``T_K``, ``attempt_frequency_Hz``: must match the active config.
    - The full per-step buffers as **lists** (so the engine can keep
      ``.append``-ing): ``times_list``, ``delta_t_list``, ...
    - The snapshot buffers as lists.
    - ``engine_rng_state``: the numpy bit-generator state dict, ready to
      be assigned back via ``rng.bit_generator.state = ...``.

    Raises ValueError if the checkpoint was written without an engine RNG
    state (e.g. an older checkpoint format), because in that case a
    bit-identical resume is impossible.
    """
    from KMC.state import KMCState, _build_bcc_geometry

    path = Path(path)
    data = np.load(path, allow_pickle=True)

    # Engine RNG presence check: needed for true resume.
    if "engine_rng_state_present" not in data.files or not bool(
        data["engine_rng_state_present"]
    ):
        raise ValueError(
            f"Checkpoint at {path} was written without an engine RNG "
            "state, so a bit-identical resume is not possible. Use "
            "load_run_checkpoint(...) for inspection-only access, or "
            "start a fresh run."
        )
    engine_rng_state = data["engine_rng_state"].item()
    if not isinstance(engine_rng_state, dict) or len(engine_rng_state) == 0:
        raise ValueError(
            f"Checkpoint at {path} contains an empty engine RNG state."
        )

    supercell_size = int(data["supercell_size"])
    lattice_parameter_A = float(data["lattice_parameter_A"])
    element_symbols = [str(s) for s in data["element_symbols"].tolist()]
    jump_distance_A = float(data["jump_distance_A"])

    positions, cell, nn_table = _build_bcc_geometry(
        supercell_size, lattice_parameter_A
    )

    state = KMCState(
        positions=positions,
        cell=cell,
        nn_table=nn_table,
        species=np.asarray(data["current_species"]).copy(),
        vacancy_index=int(data["current_vacancy_index"]),
        element_symbols=element_symbols,
        jump_distance_A=jump_distance_A,
    )

    # Snapshots: stored as stacked arrays. Convert to list-of-arrays so
    # the engine can continue to append per-step.
    snap_times_arr = np.asarray(data["snapshot_times_s"])
    snap_species_arr = np.asarray(data["snapshot_species"])
    snap_vac_arr = np.asarray(data["snapshot_vacancy_indices"])
    if snap_times_arr.size == 0:
        snapshots_times = []
        snapshots_species = []
        snapshots_vacancy = []
    else:
        snapshots_times = list(snap_times_arr.tolist())
        snapshots_species = [
            snap_species_arr[i].copy() for i in range(snap_species_arr.shape[0])
        ]
        snapshots_vacancy = [int(v) for v in snap_vac_arr.tolist()]

    barriers_arr = np.asarray(data["barriers_eV"])
    if barriers_arr.size == 0:
        barriers_list = []
    else:
        barriers_list = [
            barriers_arr[i].copy() for i in range(barriers_arr.shape[0])
        ]

    vac_unf_arr = np.asarray(data["vacancy_positions_unfolded"])
    if vac_unf_arr.size == 0:
        vac_unf_list = []
    else:
        vac_unf_list = [
            vac_unf_arr[i].copy() for i in range(vac_unf_arr.shape[0])
        ]

    return {
        "state": state,
        "initial_species": np.asarray(data["initial_species"]).copy(),
        "initial_vacancy_index": int(data["initial_vacancy_index"]),
        "step": int(data["n_steps"]),
        "total_time_s": float(data["total_time_s"]),
        "T_K": float(data["T_K"]),
        "attempt_frequency_Hz": float(data["attempt_frequency_Hz"]),
        "times_list": list(np.asarray(data["times_s"]).tolist()),
        "delta_t_list": list(np.asarray(data["delta_t_s"]).tolist()),
        "vacancy_indices_list": [
            int(v) for v in np.asarray(data["vacancy_indices"]).tolist()
        ],
        "hopper_species_list": [
            int(v) for v in np.asarray(data["hopper_species"]).tolist()
        ],
        "chosen_jump_local_list": [
            int(v) for v in np.asarray(data["chosen_jump_local"]).tolist()
        ],
        "barriers_list": barriers_list,
        "vacancy_positions_unfolded_list": vac_unf_list,
        "snapshots_times": snapshots_times,
        "snapshots_species": snapshots_species,
        "snapshots_vacancy": snapshots_vacancy,
        "engine_rng_state": engine_rng_state,
    }
