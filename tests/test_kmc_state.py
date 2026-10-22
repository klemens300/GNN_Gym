"""Start configuration and vacancy exchange of the KMC state."""
import numpy as np
import pytest

from gnn_gym.kmc.runner import seed_streams
from gnn_gym.kmc.state import VACANCY, centre_site, random_state
from kmc_mocks import ELEMENTS, EQUIATOMIC, cell


@pytest.mark.parametrize("size", [4, 5, 12])
def test_centre_site_is_box_centre(size):
    c = cell(size)
    assert tuple(c.coords[centre_site(c)]) == (size, size, size)


def test_random_state_composition_and_vacancy():
    c = cell(4)
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(42))
    assert st.species.dtype == np.int8
    assert int((st.species == VACANCY).sum()) == 1
    assert st.vacancy == centre_site(c)
    counts = np.bincount(st.species[st.species >= 0], minlength=4)
    assert counts.sum() == c.n_sites - 1
    assert all(abs(n - c.n_sites // 4) <= 1 for n in counts)


def test_random_state_equals_shuffled_symbol_list():
    """The decoration is the composition dealt in element order and shuffled
    once; shuffling the species codes consumes the generator exactly like
    shuffling the list of element symbols."""
    c = cell(4)
    seed = seed_streams(1142)[0]
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(seed))
    symbols = [e for e in ELEMENTS for _ in range(c.n_sites // 4)]
    np.random.default_rng(seed).shuffle(symbols)
    ref = np.array([ELEMENTS.index(s) for s in symbols], np.int8)
    ref[centre_site(c)] = VACANCY
    np.testing.assert_array_equal(st.species, ref)


def test_targets_are_first_neighbours():
    c = cell(4)
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(0))
    for v in (0, 17, 100, c.n_sites - 1):
        nb = st.neighbours[v]
        assert len(set(nb.tolist())) == 8 and v not in nb
        assert np.all(np.diff(nb) > 0)                   # candidate order
        d = c.min_image(c.coords[nb] - c.coords[v])
        assert np.all((d ** 2).sum(1) == 3)              # <111>, units a/2


def test_swap_moves_vacancy_and_conserves_species():
    c = cell(4)
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(0))
    before = np.bincount(st.species[st.species >= 0], minlength=4)
    key = st.key()
    v0, t = st.vacancy, int(st.targets()[3])
    moved = int(st.species[t])
    st.swap(t)
    assert st.vacancy == t and st.species[t] == VACANCY
    assert st.species[v0] == moved
    np.testing.assert_array_equal(
        np.bincount(st.species[st.species >= 0], minlength=4), before)
    assert st.key() != key
    with pytest.raises(ValueError):
        st.swap(st.vacancy)


def test_copy_is_independent():
    c = cell(4)
    st = random_state(c, ELEMENTS, EQUIATOMIC, np.random.default_rng(0))
    other = st.copy()
    st.swap(int(st.targets()[0]))
    assert other.vacancy != st.vacancy
    assert int((other.species == VACANCY).sum()) == 1
