"""Exactness properties of the two jump models (CPU, deployed checkpoints)."""
import numpy as np
import pytest
import torch

from gnn_gym.lattice import Supercell
from gnn_gym.model import (LatticeGraph, ModelConfig, build_network, load_model,
                           readout_masks, save_checkpoint)
from gnn_gym.model.locality import jump_endpoints, training_cell
from gnn_gym.paths import resolve

DE = resolve("models/MoNbTaW/de_L2_s0.pt")
BSYM = resolve("models/MoNbTaW/bsym_site_s0.pt")
needs_models = pytest.mark.skipif(not (DE.exists() and BSYM.exists()),
                                  reason="model checkpoints not present")


def random_jump(cfg, cell, seed):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, len(cfg.elements), cell.n_sites)
    v, j = cell.reference_jump()
    zi, zf = jump_endpoints(z, v, j, cfg.vac_index)
    return (torch.from_numpy(zi).long()[None], torch.from_numpy(zf).long()[None])


@needs_models
def test_site_energy_mask_is_exact():
    """dE summed over the read-out mask equals the change of the full sum."""
    net, cfg, ck = load_model(DE, "cpu")
    net.double()
    net.encoder.graph.to(dtype=torch.float64)
    cell = training_cell(cfg)
    mask = torch.from_numpy(readout_masks(cfg, ck["reach"], cell)[0])
    with torch.no_grad():
        for seed in range(3):
            zi, zf = random_jump(cfg, cell, seed)
            masked = float(net(zi, zf, mask))
            full = float((net.site_energies(zf) - net.site_energies(zi)).sum())
            assert masked == pytest.approx(full, abs=1e-9)
            # antisymmetry: the reverse jump has the opposite energy change
            assert float(net(zf, zi, mask)) == pytest.approx(-masked, abs=1e-12)


@needs_models
def test_site_energy_closed_cycle_sums_to_zero():
    """A cycle of configurations A -> B -> C -> A: the dE add up to zero."""
    net, cfg, ck = load_model(DE, "cpu")
    net.double()
    net.encoder.graph.to(dtype=torch.float64)
    cell = training_cell(cfg)
    ones = torch.ones(cell.n_sites, dtype=torch.bool)
    rng = np.random.default_rng(5)
    states = [torch.from_numpy(rng.integers(0, cfg.n_species, cell.n_sites)).long()[None]
              for _ in range(3)]
    with torch.no_grad():
        total = sum(float(net(states[k], states[(k + 1) % 3], ones)) for k in range(3))
    assert abs(total) < 1e-9


@needs_models
def test_barrier_is_symmetric():
    net, cfg, ck = load_model(BSYM, "cpu")
    cell = training_cell(cfg)
    mask = torch.from_numpy(readout_masks(cfg, 2, cell)[1])
    with torch.no_grad():
        for seed in range(3):
            zi, zf = random_jump(cfg, cell, seed)
            assert float(net(zi, zf, mask)) == float(net(zf, zi, mask))


def test_checkpoint_roundtrip(tmp_path):
    cfg = ModelConfig(target="de", de_hidden=8, de_site_hidden=[8], size=4)
    torch.manual_seed(0)
    net = build_network(cfg)
    cell = Supercell(training_cell(cfg).lattice, 4)
    net.set_graph(LatticeGraph.for_supercell(cell, cfg.shells, False))
    path = tmp_path / "m.pt"
    save_checkpoint(path, net, cfg, reach=2, epoch=3)
    net2, cfg2, ck = load_model(path)
    assert cfg2 == cfg and ck["reach"] == 2 and ck["epoch"] == 3
    z = torch.randint(0, cfg.n_species, (2, cell.n_sites))
    with torch.no_grad():
        assert torch.equal(net.site_energies(z), net2.site_energies(z))
