"""Checkpoint files of KMC runs.

One compressed .npz per trajectory holds the complete record of the run so
far (gnn_gym.kmc.result), the current occupation, the state of the BKL
random-number generator and the state of the escape-layer detector. It is
both the output of the run and the point from which an interrupted or
extended run continues; continuing reproduces the uninterrupted trajectory
exactly. Each write goes to a temporary file that is then renamed, so an
interruption never leaves a damaged checkpoint.

Checkpoints without the keys `system`, `lattice` and `escape_*` are read as
well: the system is then identified by its element list in
configs/systems.yaml, and a run continued from such a file starts with an
empty basin detector.
"""
import os
from pathlib import Path

import numpy as np

from ..lattice import Supercell, get_lattice
from ..systems import load_systems
from .result import KMCRun, Trajectory
from .state import LatticeState


def checkpoint_path(run_dir, T_K):
    return Path(run_dir) / f"checkpoint_T{float(T_K):.0f}K_real0.npz"


def find_checkpoint(run_dir):
    found = sorted(Path(run_dir).glob("checkpoint_*.npz"))
    return found[0] if found else None


def save(path, *, system, T_K, attempt_frequency_Hz, lattice_parameter_A,
         state, traj, rng, escaper=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = state.cell
    lat = cell.lattice
    paths = traj.micro_paths
    steps = sorted(paths)
    arrays = dict(
        system=np.asarray(system),
        lattice=np.asarray(lat.name),
        supercell_size=np.asarray(cell.size, np.int64),
        lattice_parameter_A=np.asarray(lattice_parameter_A, np.float64),
        element_symbols=np.asarray(list(state.elements), dtype=object),
        jump_distance_A=np.asarray(
            lat.shell_distance(lat.jump_shell, lattice_parameter_A)),
        T_K=np.asarray(T_K, np.float64),
        attempt_frequency_Hz=np.asarray(attempt_frequency_Hz, np.float64),
        n_steps=np.asarray(traj.n_steps, np.int64),
        total_time_s=np.asarray(traj.total_time, np.float64),
        initial_species=traj.initial_species,
        initial_vacancy_index=np.asarray(traj.initial_vacancy, np.int64),
        current_species=np.asarray(state.species, np.int8),
        current_vacancy_index=np.asarray(state.vacancy, np.int64),
        times_s=traj.times.data,
        delta_t_s=traj.delta_t.data,
        vacancy_indices=traj.vacancy.data,
        hopper_species=traj.hopper.data,
        chosen_jump_local=traj.chosen.data,
        barriers_eV=traj.barriers.data,
        vacancy_positions_unfolded=traj.unfolded.data,
        snapshot_times_s=np.asarray(traj.snapshot_times, np.float64),
        snapshot_species=(np.stack(traj.snapshot_species)
                          if traj.snapshot_species
                          else np.zeros((0, cell.n_sites), np.int8)),
        snapshot_vacancy_indices=np.asarray(traj.snapshot_vacancy, np.int32),
        engine_rng_state=np.array(dict(rng.bit_generator.state), dtype=object),
        engine_rng_state_present=np.asarray(True),
        # empty when the run has no escape layer
        is_escape=(traj.is_escape.data if escaper is not None
                   else np.zeros(0, bool)),
        escape_path_steps=np.asarray(steps, np.int64),
        escape_path_lengths=np.asarray([len(paths[s]) for s in steps],
                                       np.int64),
        escape_path_concat=(np.concatenate([paths[s] for s in steps])
                            .astype(np.int64) if steps
                            else np.zeros(0, np.int64)),
    )
    if escaper is not None:
        arrays.update(escaper.state_arrays(cell.n_sites))
    tmp = path.with_suffix(path.suffix + ".tmp")
    # an open handle keeps numpy from appending ".npz" to the temporary name
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)
    return path


def stored_steps(path):
    """Number of steps in a checkpoint (reads only that entry)."""
    with np.load(path, allow_pickle=True) as z:
        return int(z["n_steps"])


def _system_of(z):
    if "system" in z.files:
        return str(z["system"])
    els = [str(e) for e in z["element_symbols"].tolist()]
    match = [n for n, s in load_systems().items() if list(s.elements) == els]
    if len(match) != 1:
        raise ValueError(f"cannot identify the system of elements {els}")
    return match[0]


def _micro_paths(z):
    out, off = {}, 0
    concat = np.asarray(z["escape_path_concat"], np.int64)
    for s, n in zip(z["escape_path_steps"].tolist(),
                    z["escape_path_lengths"].tolist()):
        out[int(s)] = concat[off:off + int(n)].copy()
        off += int(n)
    return out


def _arrays(z, snapshots=True):
    """The checkpoint as a dict of arrays in the current layout."""
    a = {k: z[k] for k in z.files
         if k != "snapshot_species" and not k.startswith("escape_path")}
    n = int(a["n_steps"])
    a["escape_layer"] = np.asarray(a["is_escape"]).size > 0
    if not a["escape_layer"]:
        a["is_escape"] = np.zeros(n, bool)
    a["escape_micro_paths"] = _micro_paths(z)
    if snapshots:
        a["snapshot_species"] = z["snapshot_species"]
    return a


def _final_state(a):
    lattice = str(a["lattice"]) if "lattice" in a else "bcc"
    cell = Supercell(get_lattice(lattice), int(a["supercell_size"]))
    els = [str(e) for e in a["element_symbols"].tolist()]
    return LatticeState(cell, els, np.asarray(a["current_species"]).copy(),
                        int(a["current_vacancy_index"]))


def load_run(path, snapshots=True):
    """KMCRun from a checkpoint; snapshots=False skips the occupations."""
    with np.load(path, allow_pickle=True) as z:
        a = _arrays(z, snapshots)
        system = _system_of(z)
        state = _final_state(a)
    snap = None
    if snapshots and len(a["snapshot_times_s"]):
        snap = (np.asarray(a["snapshot_times_s"]),
                np.asarray(a["snapshot_species"]),
                np.asarray(a["snapshot_vacancy_indices"]))
    return KMCRun(
        system=system, T_K=float(a["T_K"]),
        attempt_frequency_Hz=float(a["attempt_frequency_Hz"]),
        lattice_parameter_A=float(a["lattice_parameter_A"]),
        jump_distance_A=float(a["jump_distance_A"]),
        final_state=state, n_steps=int(a["n_steps"]),
        total_time_s=float(a["total_time_s"]),
        initial_species=np.asarray(a["initial_species"]),
        initial_vacancy_index=int(a["initial_vacancy_index"]),
        times_s=np.asarray(a["times_s"]), delta_t_s=np.asarray(a["delta_t_s"]),
        vacancy_indices=np.asarray(a["vacancy_indices"]),
        hopper_species=np.asarray(a["hopper_species"]),
        chosen_jump_local=np.asarray(a["chosen_jump_local"]),
        barriers_eV=np.asarray(a["barriers_eV"]),
        vacancy_positions_unfolded=np.asarray(a["vacancy_positions_unfolded"]),
        escape_layer=bool(a["escape_layer"]),
        is_escape=np.asarray(a["is_escape"], bool),
        escape_micro_paths=a["escape_micro_paths"],
        snapshot_times_s=None if snap is None else snap[0],
        snapshot_species=None if snap is None else snap[1],
        snapshot_vacancy_indices=None if snap is None else snap[2])


def load_resume(path, extra_steps=0):
    """Everything needed to continue a run: a dict with system, T_K,
    attempt_frequency_Hz, lattice_parameter_A, state, traj, rng_state and
    the raw arrays (for the escape detector)."""
    with np.load(path, allow_pickle=True) as z:
        a = _arrays(z)
        system = _system_of(z)
        state = _final_state(a)
    if not bool(a.get("engine_rng_state_present", False)):
        raise ValueError(f"{path} holds no generator state; cannot continue")
    return dict(system=system, T_K=float(a["T_K"]),
                attempt_frequency_Hz=float(a["attempt_frequency_Hz"]),
                lattice_parameter_A=float(a["lattice_parameter_A"]),
                state=state, traj=Trajectory.from_arrays(a, extra_steps),
                rng_state=a["engine_rng_state"].item(), arrays=a)
