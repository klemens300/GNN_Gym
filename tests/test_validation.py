"""Ordered candidates and the window convention of the ordered-NEB check."""
import numpy as np

from gnn_gym.lattice import BCC, Supercell
from gnn_gym.validation.ground_state import candidates, partitions
from gnn_gym.validation.ordered_neb import cut_window, reflections, warren_cowley

LAYERS = ["Nb", "Mo", "Ta", "W", "W", "Ta", "Mo", "Nb"]


def test_partitions():
    assert partitions(["Ta", "W"]) == [(("Ta",), ("W",))]
    p = partitions(["Mo", "Nb", "Ta", "W"])
    assert len(p) == 3 and all("Mo" in a for a, _ in p)


def test_candidates():
    cell = Supercell(BCC, 8)
    els = ["Mo", "Nb", "Ta", "W"]
    cand = candidates(cell, els, 2, LAYERS)
    assert set(cand) == {"random", "layered", "B2 {Mo,Nb}|{Ta,W}",
                         "B2 {Mo,Ta}|{Nb,W}", "B2 {Mo,W}|{Nb,Ta}"}
    sub = cell.sublattice()
    for name, (cells, dS) in cand.items():
        for syms in cells:
            assert all(syms.count(e) == cell.n_sites // 4 for e in els)
        if name.startswith("B2"):
            a, b = name[4:].split("|")
            ga = a.strip("{}").split(",")
            assert all((s in ga) == (k == 0) for s, k in zip(cells[0], sub))
            assert dS == np.log(2.0)
    lay = cand["layered"][0][0]
    z = cell.coords[:, 2]
    assert all(lay[i] == LAYERS[z[i] % 8] for i in range(cell.n_sites))
    # deterministic in the seeds
    again = candidates(cell, els, 2, LAYERS)
    assert again["random"][0] == cand["random"][0]


def test_reflected_windows():
    parent = Supercell(BCC, 12)
    win = Supercell(BCC, 7)
    rng = np.random.default_rng(0)
    sp = rng.integers(0, 4, parent.n_sites)
    vac = 1000
    sp[vac] = -1
    state = dict(species=sp, vac=vac)
    v, j = win.reference_jump()
    targets = set()
    for d in reflections(BCC):
        w, pj = cut_window(parent, state, win, d)
        assert w[v] == -1 and (w >= 0).sum() == win.n_sites - 1
        delta = parent.min_image(parent.coords[pj] - parent.coords[vac])
        assert np.array_equal(delta, np.asarray(BCC.reference_jump) * d)
        targets.add(pj)
    # the eight windows cover the eight first neighbours of the vacancy
    assert targets == set(parent.neighbours(1)[vac].tolist())


def test_warren_cowley_random_and_ordered():
    cell = Supercell(BCC, 6)
    nn = cell.neighbours(1)
    b2 = cell.sublattice()                    # perfect B2 of species 0 and 1
    assert warren_cowley(b2, nn, (0, 1)) == -1.0
    rnd = np.random.default_rng(0).permutation(b2)
    assert abs(warren_cowley(rnd, nn, (0, 1))) < 0.1
