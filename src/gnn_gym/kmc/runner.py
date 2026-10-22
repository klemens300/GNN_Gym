"""One KMC run: start or continue a trajectory and write its checkpoint.

The seed of a run is expanded with numpy's SeedSequence into two
independent streams, one for the random start configuration and one for the
BKL loop (and the exit choice of the escape layer):

    s = SeedSequence(SeedSequence(seed).generate_state(1)[0])
    decoration, engine = s.spawn(2)

If the run directory already holds a checkpoint, the run continues from it
up to cfg.n_steps (an extended step target extends the trajectory); a
checkpoint that already holds n_steps steps is left as it is.
"""
import numpy as np
import yaml

from ..lattice import Supercell
from ..model.predictor import JumpPredictor
from . import checkpoint as ck
from .engine import BARRIER_FALLBACK_EV, run
from .escape import Escaper
from .result import Trajectory
from .state import random_state


def seed_streams(seed):
    """(decoration, engine) SeedSequences of a run seed."""
    root = np.random.SeedSequence(int(seed)).generate_state(1)[0]
    return np.random.SeedSequence(int(root)).spawn(2)


def make_cell(cfg):
    return Supercell(cfg.material.lattice_obj, cfg.supercell_size)


def initial_state(cfg, cell=None):
    """Random start configuration of a run, from its seed."""
    cell = make_cell(cfg) if cell is None else cell
    mat = cfg.material
    rng = np.random.default_rng(seed_streams(cfg.seed)[0])
    return random_state(cell, mat.elements, mat.composition, rng)


def make_predictor(cfg, de_model, bsym_model, device="auto", cell=None,
                   halo=True):
    cell = make_cell(cfg) if cell is None else cell
    return JumpPredictor(de_model, bsym_model, cell, cfg.material.elements,
                         device=device, halo=halo)


def run_point(cfg, predictor, run_dir, log=print):
    """Run (or continue) one trajectory into run_dir; returns its checkpoint."""
    if (predictor.cell.size != cfg.supercell_size
            or predictor.cell.lattice != cfg.material.lattice_obj):
        raise ValueError("the predictor is bound to a different cell")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = ck.checkpoint_path(run_dir, cfg.T_K)
    escaper = (Escaper(cfg.escape_params, BARRIER_FALLBACK_EV)
               if cfg.escape else None)
    rng = np.random.default_rng(seed_streams(cfg.seed)[1])

    if path.exists():
        done = ck.stored_steps(path)
        if done >= cfg.n_steps:
            log(f"[kmc] {cfg.name}: {done} steps stored, nothing to do")
            return path
        r = ck.load_resume(path, extra_steps=cfg.n_steps - done)
        for key, want in (("system", cfg.system), ("T_K", float(cfg.T_K)),
                          ("attempt_frequency_Hz", cfg.attempt_frequency_Hz),
                          ("lattice_parameter_A", cfg.lattice_parameter_A)):
            if (r[key] != want if isinstance(want, str)
                    else abs(r[key] - want) > 1e-9 * abs(want)):
                raise ValueError(f"{path}: {key} = {r[key]}, config {want}")
        state, traj = r["state"], r["traj"]
        rng.bit_generator.state = r["rng_state"]
        restored = escaper is not None and escaper.restore(r["arrays"])
        log(f"[kmc] {cfg.name}: continuing from step {traj.n_steps}"
            + (" (basin detector restored)" if restored else ""))
    else:
        state = initial_state(cfg, predictor.cell)
        cell = state.cell
        traj = Trajectory(state.species, state.vacancy,
                          cell.positions(cfg.lattice_parameter_A)[state.vacancy],
                          state.targets().size)
        if cfg.snapshot_every > 0:
            traj.snapshot(state)
        log(f"[kmc] {cfg.name}: new run, seed {cfg.seed}")

    (run_dir / "run.yaml").write_text(yaml.safe_dump(cfg.to_dict(),
                                                     sort_keys=False))

    def save():
        ck.save(path, system=cfg.system, T_K=cfg.T_K,
                attempt_frequency_Hz=cfg.attempt_frequency_Hz,
                lattice_parameter_A=cfg.lattice_parameter_A,
                state=state, traj=traj, rng=rng, escaper=escaper)

    run(state, predictor, traj, cfg.T_K, cfg.attempt_frequency_Hz,
        cfg.n_steps, cfg.lattice_parameter_A, rng, escaper=escaper,
        snapshot_every=cfg.snapshot_every, checkpoint=save,
        checkpoint_every=cfg.checkpoint_every,
        progress_every=cfg.progress_every)
    log(f"[kmc] {cfg.name}: {traj.n_steps} steps, t_sim "
        f"{traj.total_time:.4e} s, {int(np.count_nonzero(traj.is_escape.data))}"
        f" escapes -> {path}")
    return path
