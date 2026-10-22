import numpy as np
import pytest

from gnn_gym.lattice import (BCC, Supercell, counts_from_fractions, decorate,
                             hop_ball, locality_report)


def test_bcc_shells():
    assert [BCC.coordination(s) for s in (1, 2, 3)] == [8, 6, 12]
    for s, d2 in zip((1, 2, 3), (3, 4, 8)):
        assert np.all((BCC.shell_vectors(s) ** 2).sum(1) == d2)
    assert BCC.angle_dots() == (-3, -1, 1, 3)
    assert BCC.shell_distance(2, 3.2) == pytest.approx(3.2)
    assert BCC.shell_distance(1, 3.2) == pytest.approx(3.2 * np.sqrt(3) / 2)


def test_site_order_corner_then_body_centre():
    c = Supercell(BCC, 3)
    assert c.n_sites == 54
    assert c.coords[:3].tolist() == [[0, 0, 0], [1, 1, 1], [0, 0, 2]]
    assert np.array_equal(c.sublattice()[:4], [0, 1, 0, 1])
    assert c.index(c.coords[17]) == 17
    assert c.index(c.coords[17] + c.box) == 17


def test_neighbour_tables():
    c = Supercell(BCC, 4)
    for shell in (1, 2):
        tab = c.neighbours(shell)
        vec = BCC.shell_vectors(shell)
        d = c.min_image(c.coords[tab] - c.coords[:, None, :])
        assert np.array_equal(d, np.broadcast_to(vec, d.shape))
        # symmetric relation
        for i in (0, 5, 77):
            for j in tab[i]:
                assert i in tab[j]
    srt = c.neighbours(1, sort=True)
    assert np.array_equal(np.sort(c.neighbours(1), axis=1), srt)


def test_cell_too_small_is_refused():
    with pytest.raises(ValueError):
        Supercell(BCC, 1).neighbours(1)


def test_reference_jump_is_first_shell():
    for size in (4, 7, 12):
        c = Supercell(BCC, size)
        v, j = c.reference_jump()
        assert j in c.neighbours(1)[v]
        assert tuple(c.coords[v]) == (2 * (size // 2),) * 3


def test_replication_preserves_environments():
    small = Supercell(BCC, 4)
    big = Supercell(BCC, 8)
    rep = small.replication_map(2)
    z = np.random.default_rng(0).integers(0, 4, small.n_sites)
    zb = z[rep]
    for i in (0, 50, 127):
        k = big.index(small.coords[i])
        assert np.array_equal(zb[big.neighbours(2)[k]], z[small.neighbours(2)[i]])


def test_locality():
    assert locality_report(BCC, 7, 2, (1, 2))["local"]
    assert locality_report(BCC, 7, 3, (1, 2))["local"]
    rep = locality_report(BCC, 7, 4, (1, 2))
    assert not rep["local"]
    c = Supercell(BCC, 7)
    v, j = c.reference_jump()
    # the read-out ball of the paper: all sites within 2 hops of V or J
    assert hop_ball(c.adjacency_lists((1, 2)), [v, j], 2).sum() == 84


def test_decoration_is_deterministic():
    counts = {"W": 3, "Mo": 5}
    a = decorate(counts, 11, 8)
    assert a == decorate(counts, 11, 8)
    assert sorted(a) == ["Mo"] * 5 + ["W"] * 3
    with pytest.raises(ValueError):
        decorate(counts, 11, 9)
    assert counts_from_fractions({"Mo": 5 / 8, "W": 3 / 8}, 8) == counts
