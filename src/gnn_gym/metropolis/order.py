"""Long-range order of stored configurations, read in reciprocal space, and
local B2 order in real space.

CONCENTRATION WAVES

    For element e the occupation n_e(r) minus its mean is placed on the
    integer grid of the supercell (box^3 points, spacing a / units_per_a) and
    Fourier transformed,

        F_e(k) = sum_r (n_e(r) - c_e) exp(-2 pi i k.r),

    on every wave vector the cell can carry, k = m units_per_a / box in units
    of 2 pi / a. Wave vectors of the reciprocal lattice of the crystal
    (h + k + l even for bcc) carry the mean, which is subtracted, so only
    superstructure points remain. Per element:

        psi_e   max over superstructure k of |F_e(k)| / N_e: the amplitude of
                the strongest concentration wave. For B2 order it sits at
                (1 0 0) and equals eta_e (1 for complete order). Antiphase
                domains move intensity to satellites but do not cancel it,
                whereas eta, which counts atoms on two fixed sublattices, is
                blind to them. In a random alloy psi_e is a finite-size floor
                of order 1 / sqrt(N_e).
        k_e     where that maximum lies, folded to |h| >= |k| >= |l|.
        I/N     |F_e(k_e)|^2 / N, the diffraction-like intensity (random
                alloy: c (1 - c); complete order: N c (1 - c)).
        p_e     projection of every element onto the leading wave of the
                configuration, Re(F_e conj(F_lead)) / (|F_lead| N_e): which
                elements share its crest and which its trough.

    S(k) = |F_e(k)|^2 / N on the whole grid is the occupation structure
    factor; `top_peaks` lists its strongest superstructure points.

LOCAL B2 ORDER

    For a pair (A, B), the B2 phase of a site is +1 for A on sublattice 0 or
    B on sublattice 1, -1 for the reverse, 0 for any other element. The local
    order of a site is the mean phase over the site and its first-shell
    neighbours, divided by the number of pair atoms among them: +-1 inside a
    perfect B2 domain of either variant, ~0 in a random alloy. Two variants
    meeting mark an antiphase boundary.
"""
import numpy as np


# ---------------------------------------------------------------------------
# reciprocal space
# ---------------------------------------------------------------------------

def translation_generators(lattice):
    """Integer translation vectors that generate the lattice: the cubic cell
    edges and the basis offsets."""
    edge = lattice.cell_edge * np.eye(3, dtype=np.int64)
    basis = np.asarray(lattice.basis, dtype=np.int64)
    return np.vstack([edge, basis[1:] - basis[0]])


def k_grid(cell):
    """(box, box, box, 3) wave vectors of the FFT grid in units of 2 pi / a,
    folded to [-1/2, 1/2) of the grid period; and the integer grid index m."""
    g = cell.box
    m = np.stack(np.meshgrid(*(np.arange(g),) * 3, indexing="ij"), -1)
    m = (m + g // 2) % g - g // 2
    return m * cell.lattice.units_per_a / g, m


def superstructure_mask(cell):
    """(mask, k): mask is False on the reciprocal-lattice points of the
    crystal (they carry only the subtracted mean), True elsewhere."""
    k, m = k_grid(cell)
    gen = translation_generators(cell.lattice)
    recip = np.all((m @ gen.T) % cell.box == 0, axis=-1)
    return ~recip, k


def _reciprocal_candidates(lattice):
    """Reciprocal-lattice vectors (units 2 pi / a) with components in
    {0, 1, 2}: enough to fold any k with components in [0, 1]."""
    gen = translation_generators(lattice)
    r = np.arange(3)
    G = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    G = G[np.any(G != 0, axis=1)]
    ok = np.all(np.isclose(np.mod(G @ gen.T / lattice.units_per_a + 0.5, 1.0),
                           0.5), axis=1)
    return G[ok].astype(float)


def fold(k, lattice):
    """k folded by the cubic symmetry to |h| >= |k| >= |l| and shifted by
    the reciprocal-lattice vector that minimises its length; rounded."""
    a = np.sort(np.abs(np.asarray(k, dtype=float)))[::-1]
    best = a
    for G in _reciprocal_candidates(lattice):
        c = np.sort(np.abs(a - G))[::-1]
        if np.linalg.norm(c) < np.linalg.norm(best) - 1e-9:
            best = c
    return tuple(float(x) for x in np.round(best, 4))


def concentration_wave(z, cell, e):
    """F_e(k) on the (box)^3 grid (numpy's FFT sign; |F| and relative phases
    are what is used)."""
    g = cell.box
    occ = np.zeros((g, g, g))
    sig = (z == e).astype(float)
    occ[tuple(cell.coords.T)] = sig - sig.mean()
    return np.fft.fftn(occ)


def structure_factor(z, cell, e):
    """S_e(k) = |F_e(k)|^2 / N on the grid."""
    return np.abs(concentration_wave(z, cell, e)) ** 2 / z.size


def concentration_waves(z, cell, n_species, mask_k=None):
    """{species: psi, idx, k, I_over_N, eta, p, lead} for one configuration
    (species absent from z are left out; see the module docstring)."""
    mask, kgrid = mask_k if mask_k is not None else superstructure_mask(cell)
    n = z.size
    sub = cell.sublattice()
    present = [e for e in range(n_species) if (z == e).any()]
    F = {e: concentration_wave(z, cell, e) for e in present}
    out = {}
    for e in present:
        n_e = int((z == e).sum())
        amp = np.where(mask, np.abs(F[e]), 0.0)
        idx = np.unravel_index(int(np.argmax(amp)), amp.shape)
        out[e] = dict(N_e=n_e, psi=float(amp[idx]) / n_e, idx=idx,
                      k=fold(kgrid[idx], cell.lattice),
                      I_over_N=float(amp[idx] ** 2) / n,
                      eta=2.0 * abs(((z == e) & (sub == 0)).sum() / n_e - 0.5))
    lead = max(out, key=lambda e: out[e]["psi"])
    ref = F[lead][out[lead]["idx"]]
    for e in present:
        val = F[e][out[lead]["idx"]]
        out[e]["p"] = float(np.real(val * np.conj(ref))
                            / (abs(ref) * out[e]["N_e"]))
        out[e]["lead"] = lead
    return out


def top_peaks(S, cell, n_top=6):
    """Strongest superstructure points of S(k), k in units of 2 pi / a in
    [-1, 1), one representative per set of |components|: [(k, S), ...]."""
    mask, kgrid = superstructure_mask(cell)
    order = np.argsort(S.ravel())[::-1]
    seen, out = set(), []
    for flat in order:
        idx = np.unravel_index(flat, S.shape)
        if not mask[idx]:
            continue
        k = kgrid[idx]
        key = tuple(sorted(np.abs(np.round(k, 6))))
        if key in seen:
            continue
        seen.add(key)
        out.append((k, float(S[idx])))
        if len(out) >= n_top:
            break
    return out


def structure_factor_map(z, cell, e):
    """S_e(h, k) = max over l of S_e(h, k, l), centred on k = 0 and closed
    periodically (one extra row and column), with its axis in 2 pi / a."""
    g = cell.box
    S = np.fft.fftshift(structure_factor(z, cell, e))
    proj = np.pad(S.max(axis=2), ((0, 1), (0, 1)), mode="wrap")
    axis = (np.arange(g + 1) - g // 2) * cell.lattice.units_per_a / g
    return proj, axis


# ---------------------------------------------------------------------------
# real space
# ---------------------------------------------------------------------------

def b2_phase(z, sublattice, pair):
    """+1: A on sublattice 0 or B on sublattice 1; -1: the reverse; 0: other."""
    A, B = pair
    on0 = sublattice == 0
    ph = np.zeros(len(z))
    ph[((z == A) & on0) | ((z == B) & ~on0)] = 1
    ph[((z == A) & ~on0) | ((z == B) & on0)] = -1
    return ph


def local_b2_order(z, cell, pair):
    """Signed local B2 order s in [-1, 1] per site and n, the number of pair
    atoms among the site and its first-shell neighbours."""
    nb = cell.neighbours(cell.lattice.jump_shell)
    ph = b2_phase(z, cell.sublattice(), pair)
    tot = ph + ph[nb].sum(axis=1)
    n = (ph != 0).astype(int) + (ph[nb] != 0).sum(axis=1)
    s = np.where(n > 0, tot / np.maximum(n, 1), 0.0)
    return s, n


def b2_variant(s, n, thresh, nmin):
    """+1 / -1: site inside a B2 domain of that variant (|s| > thresh with at
    least nmin pair atoms around it); 0: no local B2 order."""
    return np.where((np.abs(s) > thresh) & (n >= nmin), np.sign(s), 0).astype(int)


def layer_sites(cell, z0):
    """Sites of the double layer z0, z0 + 1 (integer units) perpendicular to z:
    one corner plane and one body-centre plane of a bcc cell."""
    box = cell.box
    return np.flatnonzero(np.isin(cell.coords[:, 2], [z0 % box, (z0 + 1) % box]))


def local_pair_order(z, cell, pair):
    """Unsigned local order of a pair per site: (unlike - like pair bonds) / z
    over the first shell, averaged over the site and its neighbours. Random
    alloy: 0; perfect B2 of the pair: +1."""
    nb = cell.neighbours(cell.lattice.jump_shell)
    zc = nb.shape[1]
    a, b = pair
    isa, isb = z == a, z == b
    s = np.zeros(len(z))
    for j in range(zc):
        na, nbb = isa[nb[:, j]], isb[nb[:, j]]
        s += (isa & nbb) | (isb & na)
        s -= (isa & na) | (isb & nbb)
    s /= float(zc)
    return (s + s[nb].sum(axis=1)) / (zc + 1.0)
