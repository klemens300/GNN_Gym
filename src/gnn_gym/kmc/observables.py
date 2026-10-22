"""Observables computed from a stored run.

vacancy_diffusion
    Time-averaged mean squared displacement of the vacancy over a time
    window. The KMC clock is irregular, so the unfolded trajectory is first
    resampled on a uniform time grid (previous-value hold, which is what the
    vacancy does between jumps); averaging over all time origins is what
    makes a single trajectory usable. D is the slope of MSD = 6 D t through
    the origin up to lags of `max_lag_frac` of the window, alpha the log-log
    slope of MSD against lag over the same lags (1 is diffusive).

tracer_msd
    Mean squared displacement of the atoms of each element after every
    step, from the replayed hop history (each jump moves one atom by minus
    the vacancy displacement; an escape replays its stored chain of jumps).
    Displacements are accumulated in integer lattice units, so the result
    carries no rounding error from the replay.

warren_cowley
    First-neighbour Warren-Cowley parameters alpha_ij = 1 - P(j|i) / c_j of
    one configuration, P(j|i) the fraction of j among the jump-shell
    neighbours of i-atoms and c_j the fraction of j among all atoms; bonds to
    the vacancy are not counted.
"""
import numpy as np

from ..model.predictor import VACANCY


def vacancy_diffusion(positions, t_pos, t0, t1, n_grid=4000, max_lag_frac=0.10):
    """(D, squared end-to-end distance of the window, MSD exponent).

    positions  (n+1, 3) unfolded vacancy positions before each step and after
               the last one; t_pos the matching times (t_pos[0] = 0)
    """
    if not np.isfinite(t1 - t0) or (t1 - t0) <= 0:
        return np.nan, 0.0, np.nan
    grid = np.linspace(t0, t1, n_grid)
    idx = np.clip(np.searchsorted(t_pos, grid, side="right") - 1,
                  0, len(positions) - 1)
    r = positions[idx]
    lags = np.arange(1, max(2, int(n_grid * max_lag_frac)))
    msd = np.empty(len(lags))
    for j, k in enumerate(lags):
        d = r[k:] - r[:-k]
        msd[j] = np.mean(np.sum(d * d, axis=1))
    lag_t = lags * (t1 - t0) / (n_grid - 1)
    slope = float(np.sum(lag_t * msd) / np.sum(lag_t * lag_t))
    ok = msd > 0
    alpha = (float(np.polyfit(np.log(lag_t[ok]), np.log(msd[ok]), 1)[0])
             if ok.sum() >= 3 else np.nan)
    return slope / 6.0, float(r[-1].dot(r[-1])), alpha


def linear_diffusion(t, msd, lo=0.5, hi=1.0):
    """D = slope/6 of a single-origin MSD(t) over [lo, hi] of the time span."""
    m = (t >= lo * t[-1]) & (t <= hi * t[-1]) & np.isfinite(msd)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(t[m], msd[m], 1)[0]) / 6.0


def tracer_msd(run):
    """{element: (n_steps+1,) MSD in A^2 of its atoms after each step}."""
    cell = run.cell
    init = np.asarray(run.initial_species)
    sites = np.flatnonzero(init != VACANCY)
    n_el = len(run.elements)
    steps, src, dst = run.hops()

    # which atom makes each hop: the one sitting on the target site
    atom_at = np.full(cell.n_sites, -1, np.int64)
    atom_at[sites] = np.arange(len(sites))
    at = atom_at.tolist()
    mover = np.empty(len(steps), np.int64)
    for h, (v, t) in enumerate(zip(src.tolist(), dst.tolist())):
        a = at[t]
        mover[h] = a
        at[v], at[t] = a, -1

    # integer displacement of every hop; cumulative per atom
    d = cell.min_image(cell.coords[src] - cell.coords[dst])
    order = np.argsort(mover, kind="stable")
    cum = np.cumsum(d[order], axis=0)
    first = np.r_[0, np.flatnonzero(np.diff(mover[order])) + 1]
    seg_start = np.repeat(first, np.diff(np.r_[first, len(order)]))
    offset = np.where(seg_start[:, None] > 0, cum[seg_start - 1], 0)
    new = np.empty_like(d)
    new[order] = cum - offset
    old = new - d
    dsq = (new ** 2).sum(1) - (old ** 2).sum(1)

    species = init[sites][mover].astype(np.int64)
    per_step = np.zeros((run.n_steps + 1, n_el), np.int64)
    np.add.at(per_step, (steps + 1, species), dsq)
    total = np.cumsum(per_step, axis=0)
    n_atoms = np.bincount(init[sites].astype(np.int64), minlength=n_el)
    unit2 = (run.lattice_parameter_A / cell.lattice.units_per_a) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        msd = np.where(n_atoms > 0, total * unit2 / np.maximum(n_atoms, 1), 0.0)
    return {e: msd[:, i] for i, e in enumerate(run.elements)}


def warren_cowley(species, neighbours, n_elements):
    """(n_elements, n_elements) first-shell alpha_ij; NaN where undefined."""
    sp = np.asarray(species, np.int64)
    src = np.repeat(np.arange(sp.size), neighbours.shape[1])
    dst = np.asarray(neighbours, np.int64).ravel()
    si, sj = sp[src], sp[dst]
    ok = (si != VACANCY) & (sj != VACANCY)
    pairs = np.zeros((n_elements, n_elements), np.int64)
    np.add.at(pairs, (si[ok], sj[ok]), 1)
    atoms = sp[sp != VACANCY]
    comp = np.bincount(atoms, minlength=n_elements) / atoms.size
    bonds = pairs.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = 1.0 - pairs / bonds[:, None] / comp[None, :]
    alpha[~np.isfinite(alpha)] = np.nan
    return alpha
