"""The KMC barrier interface: halo evaluation equals full-cell evaluation."""
import numpy as np
import pytest

from gnn_gym.lattice import BCC, Supercell
from gnn_gym.model.predictor import VACANCY, JumpPredictor
from gnn_gym.paths import resolve

DE = resolve("models/MoNbTaW/de_L2_s0.pt")
BSYM = resolve("models/MoNbTaW/bsym_site_s0.pt")
pytestmark = pytest.mark.skipif(not (DE.exists() and BSYM.exists()),
                                reason="model checkpoints not present")


@pytest.mark.parametrize("size", [6, 8, 10])
def test_halo_equals_full_cell(size):
    cell = Supercell(BCC, size)
    elements = ["Ta", "W", "Mo", "Nb"]          # order differs from the model's
    halo = JumpPredictor(DE, BSYM, cell, elements, device="cpu", halo=True,
                         verbose=False)
    full = JumpPredictor(DE, BSYM, cell, elements, device="cpu", halo=False,
                         verbose=False)
    # a halo that would cover the whole cell is replaced by the full cell;
    # at 8^3 the halo wraps around the periodic boundary
    assert halo.use_halo == (size >= 8)
    rng = np.random.default_rng(size)
    for _ in range(3):
        species = rng.integers(0, len(elements), cell.n_sites).astype(np.int8)
        vac = int(rng.integers(cell.n_sites))
        species[vac] = VACANCY
        targets = cell.neighbours(1, sort=True)[vac]
        a = halo.barriers(species, vac, targets)
        b = full.barriers(species, vac, targets)
        assert np.allclose(a, b, atol=2e-5), np.abs(a - b).max()


def test_element_subset_mapping():
    cell = Supercell(BCC, 10)
    p = JumpPredictor(DE, BSYM, cell, ["Ta", "W"], device="cpu", verbose=False)
    assert p.species_map.tolist() == [2, 3]
    with pytest.raises(ValueError):
        JumpPredictor(DE, BSYM, cell, ["Fe"], device="cpu", verbose=False)
