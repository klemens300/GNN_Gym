"""Reference-data conventions that need no uMLIP: composition draw,
decoration, dataset row reconstruction, formation energies, windows."""
import csv
from itertools import combinations

import numpy as np
import pytest

from gnn_gym.lattice import BCC, Supercell, counts_from_fractions
from gnn_gym.oracle.elastic import composition_grid, decorate_cell, planned_cells
from gnn_gym.oracle.formation import (fit_regular_solution, kernel_resample,
                                      partial_energies)
from gnn_gym.oracle.jumps import jump_cell, sample_composition
from gnn_gym.oracle.structures import (composition_counts, realised_fractions,
                                       vegard_lattice_parameter)
from gnn_gym.oracle.window_study import cut_window, parent_cell
from gnn_gym.paths import load_yaml
from gnn_gym.systems import get_system

ELS = ["Mo", "Nb", "Ta", "W"]


def first_rows(path, n):
    with open(path, newline="") as fh:
        return [r for _, r in zip(range(n), csv.DictReader(fh))]


def test_largest_remainder_counts():
    assert composition_counts({"A": 0.5, "B": 0.5}, 7) == {"A": 4, "B": 3}
    c = composition_counts({"Mo": 0.31, "Nb": 0.34, "W": 0.35}, 686)
    assert sum(c.values()) == 686
    assert c == {"Mo": 213, "Nb": 233, "W": 240}
    rng = np.random.default_rng(1)
    for _ in range(50):
        comp = dict(zip(ELS, rng.dirichlet(np.ones(4))))
        c = composition_counts(comp, 686)
        assert sum(c.values()) == 686
        assert all(abs(c[e] - comp[e] * 686) < 1 for e in ELS)
        # a stored row's realised fractions give back the same counts
        assert counts_from_fractions({e: c[e] / 686 for e in ELS}, 686) == \
            {e: n for e, n in c.items() if n > 0}


def test_composition_subsets_uniform():
    rng = np.random.default_rng(0)
    sizes = [len(sample_composition(ELS, rng)) for _ in range(3000)]
    # 15 subsets: 4 unary, 6 binary, 4 ternary, 1 quaternary
    frac = np.bincount(sizes, minlength=5)[1:] / len(sizes)
    assert frac == pytest.approx([4 / 15, 6 / 15, 4 / 15, 1 / 15], abs=0.03)


@pytest.mark.parametrize("stream,seed_base", [("train", 900000), ("referee", 950000)])
def test_dataset_rows_rebuild(stream, seed_base):
    recipe = load_yaml("configs/oracle/recipe.yaml")
    cell = Supercell(BCC, 7)
    for r in first_rows(f"data/jumps/MoNbTaW_7cube_{stream}.csv", 25):
        assert int(r["seed"]) == seed_base + int(r["idx"])
        comp, syms, v, j = jump_cell(cell, ELS, int(r["seed"]))
        rf = realised_fractions(syms, ELS)
        assert "".join(sorted(comp)) == r["subsystem"]
        assert all(f"{rf[e]:.6f}" == r[f"frac_{e}"] for e in ELS)
        assert (syms[v], syms[j]) == (r["vac_species"], r["jump_species"])
        a0 = vegard_lattice_parameter(rf, recipe["vegard_A"])
        assert f"{a0:.9f}" == r["a_start_A"]


def test_partial_energies_euler():
    # sum_X x_X mu_X = e(x) and the fit recovers a regular solution exactly
    rng = np.random.default_rng(3)
    mu0 = np.array([-10.9, -10.1, -11.8, -13.0])
    W = np.zeros((4, 4))
    for k, (i, j) in enumerate(combinations(range(4), 2)):
        W[i, j] = W[j, i] = 0.1 * (k - 3)
    rows = []
    for _ in range(200):
        x = rng.dirichlet(np.ones(4))
        e = mu0 @ x + 0.5 * x @ W @ x
        rows.append({**{f"frac_{el}": str(v) for el, v in zip(ELS, x)},
                     "E_perfect_eV": str(e * 686), "n_sites": "686"})
    m, Wf, resid = fit_regular_solution(rows, ELS)
    assert np.allclose(m, mu0) and np.allclose(Wf, W, atol=1e-9) and resid < 1e-9
    x = np.full(4, 0.25)
    mu = partial_energies(m, Wf, x)
    assert x @ mu == pytest.approx(m @ x + 0.5 * x @ Wf @ x)


def test_kernel_resample_deterministic():
    sub = [{f"frac_{e}": str(v) for e, v in zip(ELS, x)}
           for x in np.random.default_rng(5).dirichlet(np.ones(4), 50)]
    a, n1 = kernel_resample(sub, ELS, np.full(4, .25), 0.1, 100, 0)
    b, n2 = kernel_resample(sub, ELS, np.full(4, .25), 0.1, 100, 0)
    assert [id(r) for r in a] == [id(r) for r in b] and n1 == n2 and 1 <= n1 <= 50


def test_elastic_sweep_cells():
    grid = composition_grid(ELS, 0.1)
    assert len(grid) == 286 and all(abs(sum(c.values()) - 1) < 1e-12 for c in grid)
    cell = Supercell(BCC, 4)
    comp = {"Mo": 0.3, "Nb": 0.6, "Ta": 0.1, "W": 0.0}
    syms = decorate_cell(cell, comp, "W", 7)
    assert len(syms) == 128 and syms[cell.index([4, 4, 4])] == "W"
    assert decorate_cell(cell, comp, "W", 7) == syms
    # the stored sweep: rows of a composition are (centre, seed) pairs
    with open("data/nu0/elastic_MoNbTaW.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2900
    cfg = load_yaml("configs/oracle/elastic_MoNbTaW.yaml")
    assert len(planned_cells(cfg)) == 286 * sum(n for _, n in cfg["centres"])


def test_windows_nested_and_complete():
    system = get_system("MoNbTaW")
    parent, syms, v, j = parent_cell(system, 12, 800000)
    assert sorted(set(syms)) == ELS and all(syms.count(e) == 864 for e in ELS)
    prev = None
    for n in (3, 4, 7, 12):
        w, ws, vw, jw = cut_window(parent, syms, n, v, j)
        small = Supercell(BCC, n)
        assert sorted(map(tuple, w)) == sorted(map(tuple, small.coords))
        assert (ws[vw], ws[jw]) == (syms[v], syms[j])
        assert np.array_equal(w[jw] - w[vw], BCC.reference_jump)
        if prev is not None:
            assert len(ws) > len(prev)
        prev = ws
