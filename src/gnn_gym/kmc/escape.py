"""Analytic escape from low-temperature traps of the BKL loop.

At low temperature the rejection-free loop can oscillate inside a small group
of configurations joined by low barriers, while every jump out of the group
is many k_B T higher. Each oscillation costs a step and advances the clock by
a short residence time, and the number of steps to leave grows as
exp[(E_exit - E_int) / k_B T]. This costs efficiency, not accuracy, and is
removed by solving the escape from such a group exactly as an absorbing
Markov chain:

  1. Detection: a configuration recurring more than `revisit_threshold`
     times within the last `window` steps flags a basin.
  2. The basin is flooded from that configuration over every jump whose
     barrier lies within `internal_n_kT` k_B T of the lowest barrier out of
     it; the configurations reached are the transient states, every other
     jump out of them is an exit. A state is a full configuration, so the
     same vacancy site with rearranged atoms is a different state.
  3. With the transient block A of the rate matrix (A_ij = r_ij, A_ii = -k_i)
     the mean first-passage time from the entry state e0 is
         tau_esc = e0^T (-A)^-1 1,
     the mean time spent in state i before leaving is w_i with
     w^T = e0^T (-A)^-1, and an exit of rate r from state i is taken with
     probability r w_i.
  4. The clock advances by tau_esc, an exit is drawn from these
     probabilities, and the cell is moved to the exit configuration along
     the shortest chain of internal jumps from the entry state followed by
     the exit jump. The back-and-forth inside the basin adds no net
     displacement. The escape counts as one step.

A basin that is still open after `max_basin_size` states (low barriers lead
out of the enumerated set) is a region of ordinary diffusion: it is left to
the BKL loop, and detection pauses for `cooldown_steps` observed steps.

The detector state (the window of recent configurations, the flag and the
cooldown) is part of the checkpoint, so a resumed run continues exactly as
an uninterrupted one.
"""
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from ..model.predictor import VACANCY

K_B = 8.617333262e-5        # eV/K


def sanitise(barriers, fallback_eV):
    """Replace non-finite or negative barriers by a prohibitive value."""
    b = np.asarray(barriers, dtype=np.float64)
    bad = ~np.isfinite(b) | (b < 0.0)
    if bad.any():
        print(f"  [kmc] {int(bad.sum())} non-finite or negative barriers "
              f"{b.tolist()} replaced by {fallback_eV} eV", flush=True)
        b = np.where(bad, fallback_eV, b)
    return b


# ---------------------------------------------------------------------------
# Basin and its absorbing Markov chain
# ---------------------------------------------------------------------------

@dataclass
class Basin:
    """Transient states of a trap and its rate structure.

    A          (m, m) transient block of the generator
    adjacency  adjacency[i] = [(j, site), ...]: internal jumps i -> j and the
               site the vacancy exchanges with
    exits      [(i, site, rate), ...]: absorbing jumps out of state i
    leaks      a jump below the cutoff leads out of the enumerated set
    """
    species: list
    vacancies: list
    A: np.ndarray
    adjacency: list
    exits: list
    entry: int
    leaks: bool

    @property
    def size(self):
        return len(self.species)

    def path_to(self, target):
        """Sites of the shortest chain of internal jumps entry -> target."""
        if target == self.entry:
            return []
        prev = {}
        seen = {self.entry}
        queue = deque([self.entry])
        while queue and target not in prev:
            i = queue.popleft()
            for j, site in self.adjacency[i]:
                if j not in seen:
                    seen.add(j)
                    prev[j] = (i, site)
                    if j == target:
                        break
                    queue.append(j)
        sites = []
        node = target
        while node != self.entry:
            node, site = prev[node]
            sites.append(site)
        return sites[::-1]


def _swapped(species, vacancy, target):
    sp = species.copy()
    sp[vacancy] = species[target]
    sp[target] = VACANCY
    return sp


def build_basin(state, predictor, T_K, attempt_frequency_Hz, internal_n_kT,
                max_basin_size, fallback_eV):
    """Flood the basin around `state` and assemble its generator block."""
    kT = K_B * float(T_K)
    nu = float(attempt_frequency_Hz)
    neighbours = state.neighbours
    index = {}
    species, vacancies, barriers, rates = [], [], [], []

    def add(sp, v):
        index[(v, sp.tobytes())] = len(species)
        b = sanitise(predictor.barriers(sp, v, neighbours[v]), fallback_eV)
        species.append(sp)
        vacancies.append(v)
        barriers.append(b)
        rates.append(nu * np.exp(-b / kT))

    add(state.species.copy(), int(state.vacancy))
    cutoff = float(np.min(barriers[0])) + float(internal_n_kT) * kT

    queue = deque([0])
    while queue:
        i = queue.popleft()
        v = vacancies[i]
        for k, t in enumerate(neighbours[v]):
            if barriers[i][k] > cutoff:
                continue
            sp = _swapped(species[i], v, t)
            if (int(t), sp.tobytes()) in index:
                continue
            if len(species) >= max_basin_size:
                continue
            add(sp, int(t))
            queue.append(len(species) - 1)

    m = len(species)
    A = np.zeros((m, m))
    adjacency = [[] for _ in range(m)]
    exits = []
    leaks = False
    for i in range(m):
        A[i, i] = -float(rates[i].sum())
        v = vacancies[i]
        for k, t in enumerate(neighbours[v]):
            r = float(rates[i][k])
            if barriers[i][k] > cutoff:
                exits.append((i, int(t), r))
                continue
            j = index.get((int(t), _swapped(species[i], v, t).tobytes()))
            if j is None:
                leaks = True
            else:
                A[i, j] += r
                adjacency[i].append((j, int(t)))
    return Basin(species, vacancies, A, adjacency, exits, 0, leaks)


def first_passage(basin):
    """(tau_esc, exit probabilities) from the entry state."""
    negA = -basin.A
    tau = float(np.linalg.solve(negA, np.ones(len(negA)))[basin.entry])
    e0 = np.zeros(len(negA))
    e0[basin.entry] = 1.0
    w = np.linalg.solve(negA.T, e0)
    p = np.array([r * w[i] for i, _, r in basin.exits])
    total = p.sum()
    if total <= 0 or not np.isfinite(total):
        p = np.full(len(basin.exits), 1.0 / len(basin.exits))
    else:
        p = p / total
    return tau, p


# ---------------------------------------------------------------------------
# Detector and escape
# ---------------------------------------------------------------------------

class BasinDetector:
    """Recurrence count of configurations over a rolling window of steps."""

    def __init__(self, window, revisit_threshold):
        self.window = int(window)
        self.threshold = int(revisit_threshold)
        self.keys = deque()
        self.counts = Counter()
        self.flagged = False

    def observe(self, key):
        self.keys.append(key)
        self.counts[key] += 1
        if len(self.keys) > self.window:
            old = self.keys.popleft()
            self.counts[old] -= 1
            if self.counts[old] <= 0:
                del self.counts[old]
        if self.counts[key] > self.threshold:
            self.flagged = True
        return self.flagged

    def reset(self):
        self.keys.clear()
        self.counts.clear()
        self.flagged = False


@dataclass
class EscapeOutcome:
    tau_s: float
    micro_path: list            # sites the vacancy was exchanged with
    displacement: np.ndarray    # unfolded vacancy displacement (A)
    basin_size: int


class Escaper:
    """Basin detection and analytic escape; one per trajectory."""

    def __init__(self, params, fallback_eV):
        self.params = params
        self.fallback_eV = float(fallback_eV)
        self.detector = BasinDetector(params.window, params.revisit_threshold)
        self.cooldown = 0

    @property
    def flagged(self):
        return self.detector.flagged

    def observe(self, state):
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        self.detector.observe(state.key())

    def escape(self, state, predictor, T_K, attempt_frequency_Hz, rng,
               step_length):
        """Move `state` to a sampled exit configuration.

        Returns None (and the BKL loop takes the step) if the basin did not
        close, has no exit, or its linear system cannot be solved.
        `step_length` converts integer lattice displacements to Angstrom.
        """
        p = self.params
        basin = build_basin(state, predictor, T_K, attempt_frequency_Hz,
                            p.internal_n_kT, p.max_basin_size,
                            self.fallback_eV)
        self.detector.reset()
        if basin.leaks:
            self.cooldown = p.cooldown_steps
            return None
        if not basin.exits:
            return None
        try:
            tau, prob = first_passage(basin)
            if not np.isfinite(tau) or tau <= 0:
                return None
            e = int(rng.choice(len(basin.exits), p=prob))
        except (np.linalg.LinAlgError, ValueError):
            return None
        i, exit_site, _ = basin.exits[e]
        path = basin.path_to(i) + [exit_site]

        cell = state.cell
        delta = np.zeros(3, dtype=np.int64)
        for t in path:
            delta += cell.min_image(cell.coords[t] - cell.coords[state.vacancy])
            state.swap(t)
        return EscapeOutcome(tau_s=tau, micro_path=path,
                             displacement=delta * step_length,
                             basin_size=basin.size)

    # -- checkpoint -----------------------------------------------------------
    def state_arrays(self, n_sites):
        keys = list(self.detector.keys)
        species = (np.stack([np.frombuffer(k[1], np.int8) for k in keys])
                   if keys else np.zeros((0, n_sites), np.int8))
        return {"escape_window_vacancy": np.array([k[0] for k in keys],
                                                  dtype=np.int64),
                "escape_window_species": species,
                "escape_flagged": np.asarray(self.detector.flagged),
                "escape_cooldown": np.asarray(self.cooldown, np.int64)}

    def restore(self, arrays):
        """Restore the detector from checkpoint arrays (if present)."""
        self.detector.reset()
        self.cooldown = 0
        if "escape_window_vacancy" not in arrays:
            return False
        for v, sp in zip(arrays["escape_window_vacancy"],
                         arrays["escape_window_species"]):
            key = (int(v), np.asarray(sp, np.int8).tobytes())
            self.detector.keys.append(key)
            self.detector.counts[key] += 1
        self.detector.flagged = bool(arrays["escape_flagged"])
        self.cooldown = int(arrays["escape_cooldown"])
        return True
