"""The record of a KMC run.

Per step k = 0 .. n-1 (n = number of steps):

    times_s[k]                    simulated time after step k
    delta_t_s[k]                  residence time of step k
    vacancy_indices[k]            vacancy site before step k ([n] = final)
    vacancy_positions_unfolded[k] unfolded vacancy position before step k (A)
    hopper_species[k]             species of the atom that moved
    chosen_jump_local[k]          chosen candidate (-1 for an escape)
    barriers_eV[k]                barriers of all candidates (NaN for an escape)
    is_escape[k]                  step k is an analytic escape
    escape_micro_paths[k]         for an escape: the sites the vacancy was
                                  exchanged with, in order

Snapshots of the full occupation are stored every `snapshot_every` steps and
after every escape, with their simulated time and vacancy site. Everything
else (diffusion, order, trapping diagnostics) is computed from this record
after the run.

`Trajectory` grows during the run and is what a checkpoint stores; `KMCRun`
is a finished or checkpointed run as read back from disk. `KMCRun.head(n)`
is the run cut after its first n steps, with the occupation at that point
replayed from the initial one; a run and its continuation can thereby be
compared over the same number of steps.
"""
import dataclasses
from dataclasses import dataclass

import numpy as np

from .state import LatticeState


class _Growable:
    """Append-only numpy buffer with amortised doubling."""

    def __init__(self, dtype, shape=(), data=None, capacity=1024):
        data = np.zeros((0,) + shape, dtype) if data is None else data
        self._a = np.empty((max(capacity, len(data)),) + shape, dtype)
        self._a[:len(data)] = data
        self._n = len(data)

    def append(self, value):
        if self._n == len(self._a):
            grown = np.empty((2 * len(self._a),) + self._a.shape[1:],
                             self._a.dtype)
            grown[:self._n] = self._a[:self._n]
            self._a = grown
        self._a[self._n] = value
        self._n += 1

    def __len__(self):
        return self._n

    @property
    def data(self):
        return self._a[:self._n]


class Trajectory:
    """Per-step buffers of a running simulation."""

    def __init__(self, initial_species, initial_vacancy, initial_position,
                 n_candidates):
        self.initial_species = np.asarray(initial_species, np.int8).copy()
        self.initial_vacancy = int(initial_vacancy)
        self.n_candidates = int(n_candidates)
        self.total_time = 0.0
        self.times = _Growable(np.float64)
        self.delta_t = _Growable(np.float64)
        self.vacancy = _Growable(np.int32, data=np.array([initial_vacancy]))
        self.unfolded = _Growable(np.float64, (3,),
                                  data=np.asarray([initial_position], float))
        self.hopper = _Growable(np.int8)
        self.chosen = _Growable(np.int8)
        self.barriers = _Growable(np.float32, (n_candidates,))
        self.is_escape = _Growable(bool)
        self.micro_paths = {}
        self.snapshot_times = []
        self.snapshot_species = []
        self.snapshot_vacancy = []

    @property
    def n_steps(self):
        return len(self.delta_t)

    @property
    def position(self):
        """Current unfolded vacancy position."""
        return self.unfolded.data[-1]

    def record(self, dt, vacancy, displacement, hopper, chosen, barriers,
               micro_path=None):
        """Append one step; `micro_path` marks an escape."""
        if micro_path is not None:
            self.micro_paths[self.n_steps] = np.asarray(micro_path, np.int64)
        self.total_time += dt
        self.times.append(self.total_time)
        self.delta_t.append(dt)
        self.vacancy.append(vacancy)
        self.unfolded.append(self.position + displacement)
        self.hopper.append(hopper)
        self.chosen.append(chosen)
        self.barriers.append(barriers)
        self.is_escape.append(micro_path is not None)

    def snapshot(self, state):
        self.snapshot_times.append(self.total_time)
        self.snapshot_species.append(state.species.copy())
        self.snapshot_vacancy.append(int(state.vacancy))

    @classmethod
    def from_arrays(cls, a, extra_capacity=0):
        """Continue a stored trajectory (dict of checkpoint arrays)."""
        n = len(a["delta_t_s"])
        cap = n + 1 + int(extra_capacity)
        t = cls.__new__(cls)
        t.initial_species = np.asarray(a["initial_species"], np.int8).copy()
        t.initial_vacancy = int(a["initial_vacancy_index"])
        t.n_candidates = int(a["barriers_eV"].shape[1])
        t.total_time = float(a["total_time_s"])
        t.times = _Growable(np.float64, data=a["times_s"], capacity=cap)
        t.delta_t = _Growable(np.float64, data=a["delta_t_s"], capacity=cap)
        t.vacancy = _Growable(np.int32, data=a["vacancy_indices"], capacity=cap)
        t.unfolded = _Growable(np.float64, (3,),
                               data=a["vacancy_positions_unfolded"],
                               capacity=cap)
        t.hopper = _Growable(np.int8, data=a["hopper_species"], capacity=cap)
        t.chosen = _Growable(np.int8, data=a["chosen_jump_local"], capacity=cap)
        t.barriers = _Growable(np.float32, (t.n_candidates,),
                               data=a["barriers_eV"], capacity=cap)
        t.is_escape = _Growable(bool, data=a["is_escape"], capacity=cap)
        t.micro_paths = dict(a["escape_micro_paths"])
        t.snapshot_times = list(np.asarray(a["snapshot_times_s"]).tolist())
        t.snapshot_species = list(np.asarray(a["snapshot_species"]))
        t.snapshot_vacancy = [int(v) for v in a["snapshot_vacancy_indices"]]
        return t


@dataclass
class KMCRun:
    """A run read back from its checkpoint."""
    system: str
    T_K: float
    attempt_frequency_Hz: float
    lattice_parameter_A: float
    jump_distance_A: float
    final_state: object            # LatticeState after the last step
    n_steps: int
    total_time_s: float
    initial_species: np.ndarray
    initial_vacancy_index: int
    times_s: np.ndarray
    delta_t_s: np.ndarray
    vacancy_indices: np.ndarray
    hopper_species: np.ndarray
    chosen_jump_local: np.ndarray
    barriers_eV: np.ndarray
    vacancy_positions_unfolded: np.ndarray
    escape_layer: bool             # the run had the escape layer switched on
    is_escape: np.ndarray
    escape_micro_paths: dict
    snapshot_times_s: np.ndarray = None
    snapshot_species: np.ndarray = None
    snapshot_vacancy_indices: np.ndarray = None

    @property
    def cell(self):
        return self.final_state.cell

    @property
    def elements(self):
        return self.final_state.elements

    @property
    def n_sites(self):
        return self.cell.n_sites

    @property
    def times_with_zero_s(self):
        return np.concatenate(([0.0], self.times_s))

    @property
    def n_snapshots(self):
        return 0 if self.snapshot_times_s is None else len(self.snapshot_times_s)

    @property
    def n_escapes(self):
        return int(np.count_nonzero(self.is_escape))

    def hops(self):
        """(step, from_site, to_site) of every vacancy exchange in order; an
        escape contributes the exchanges along its micro-path."""
        n = self.n_steps
        vac = np.asarray(self.vacancy_indices, np.int64)
        esc = np.asarray(self.is_escape[:n], bool)
        plain = np.flatnonzero(~esc)
        if not esc.any():
            return plain, vac[plain], vac[plain + 1]
        steps, src, dst = [], [], []
        for k in range(n):
            if esc[k]:
                v = vac[k]
                for t in self.escape_micro_paths[k]:
                    steps.append(k), src.append(v), dst.append(int(t))
                    v = int(t)
            else:
                steps.append(k), src.append(vac[k]), dst.append(vac[k + 1])
        return np.asarray(steps), np.asarray(src), np.asarray(dst)

    def head(self, n):
        """The first n steps as a run of their own (self if n >= n_steps)."""
        n = int(n)
        if n >= self.n_steps:
            return self
        steps, src, dst = self.hops()
        keep = steps < n
        sp = np.asarray(self.initial_species).tolist()
        for v, t in zip(src[keep].tolist(), dst[keep].tolist()):
            sp[v], sp[t] = sp[t], sp[v]
        vac = int(self.vacancy_indices[n])
        state = LatticeState(self.cell, list(self.elements),
                             np.asarray(sp, np.int8), vac)
        snap = {}
        if self.snapshot_times_s is not None:
            at = np.searchsorted(self.times_s, self.snapshot_times_s, side="right")
            m = at <= n
            snap = dict(snapshot_times_s=self.snapshot_times_s[m],
                        snapshot_species=self.snapshot_species[m],
                        snapshot_vacancy_indices=self.snapshot_vacancy_indices[m])
        return dataclasses.replace(
            self, final_state=state, n_steps=n,
            total_time_s=float(self.times_s[n - 1]) if n else 0.0,
            times_s=self.times_s[:n], delta_t_s=self.delta_t_s[:n],
            vacancy_indices=self.vacancy_indices[:n + 1],
            hopper_species=self.hopper_species[:n],
            chosen_jump_local=self.chosen_jump_local[:n],
            barriers_eV=self.barriers_eV[:n],
            vacancy_positions_unfolded=self.vacancy_positions_unfolded[:n + 1],
            is_escape=self.is_escape[:n],
            escape_micro_paths={k: p for k, p in self.escape_micro_paths.items()
                                if k < n},
            **snap)
