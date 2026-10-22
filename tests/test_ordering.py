"""Ordered candidates of the ordering validation."""
import numpy as np

from gnn_gym.lattice import BCC, Supercell
from gnn_gym.validation.ordering import candidates, t_ord

LAYERS = ["Nb", "Mo", "Ta", "W", "W", "Ta", "Mo", "Nb"]


def test_candidates_quaternary():
    cell = Supercell(BCC, 4)
    cand = candidates(cell, ["Mo", "Nb", "Ta", "W"], n_random=2,
                      random_seed=1000, b2_seed=100, layer_sequence=LAYERS)
    assert sorted(cand) == sorted(["random", "layered", "B2 {Mo,Nb}|{Ta,W}",
                                   "B2 {Mo,Ta}|{Nb,W}", "B2 {Mo,W}|{Nb,Ta}"])
    sub = cell.sublattice()
    for name, (decs, dS) in cand.items():
        for d in decs:
            d = np.array(d)
            assert all((d == e).sum() == cell.n_sites // 4 for e in ("Mo", "Nb", "Ta", "W"))
        if name.startswith("B2"):
            a, b = name[3:].split("|")
            assert set(d[sub == 0]) == set(a.strip("{}").split(","))
            assert set(d[sub == 1]) == set(b.strip("{}").split(","))
            assert np.isclose(dS, np.log(2.0))
    assert np.isclose(cand["layered"][1], np.log(4.0))
    assert cand["random"][1] == 0.0


def test_candidates_binary_and_t_ord():
    cell = Supercell(BCC, 4)
    cand = candidates(cell, ["Ta", "W"], 1, 1000, 100, layer_sequence=LAYERS)
    assert sorted(cand) == ["B2 {Ta}|{W}", "random"]
    # ideal-entropy estimate: |dE| / (k_B ln 2)
    assert abs(t_ord(-23.26, np.log(2.0)) - 389.4) < 0.1
    assert t_ord(+0.23, np.log(2.0)) is None
