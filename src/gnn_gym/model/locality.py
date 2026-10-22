"""Read-out masks and measured locality of the jump models.

The receptive field of an encoder is measured, not inferred from its layer
count: a line-graph step can carry information further than a site step, so
the only reliable number is how far a single species change actually travels
through the network (`measure_reach`). The site-energy mask is built from that
measured reach, so it cannot fall short of the sites whose energy changes.

`size_transfer_error` evaluates the same jump in the training cell and in a
2x2x2 replica of it, where every local environment is identical by
construction. Any difference is accuracy taken from the periodicity of the
training box, which a large KMC cell does not have.
"""
import numpy as np
import torch

from ..lattice import Supercell, get_lattice, hop_ball, hop_distances
from .graph import LatticeGraph


def training_cell(cfg):
    return Supercell(get_lattice(cfg.lattice), cfg.size)


def readout_masks(cfg, de_reach, cell=None, sites=None):
    """(site-energy mask, barrier read-out mask) as boolean site arrays.

    de_reach   measured reach of the site-energy encoder in hops; the sites
               whose energy can change are those within that many hops of
               the vacancy or the jumping atom
    """
    cell = cell if cell is not None else training_cell(cfg)
    v, j = sites if sites is not None else cell.reference_jump()
    adj = cell.adjacency_lists(cfg.shells)
    changed = hop_ball(adj, [v, j], de_reach)
    pooled = hop_ball(adj, [v, j], cfg.bsym_pool_hops)
    return changed, pooled


@torch.no_grad()
def measure_reach(net, cfg, cell=None, seed=0, tol=1e-9):
    """Largest graph distance at which a single species change moves a site
    feature. `net` must be bound to the graph of `cell` (default: the
    training cell)."""
    cell = cell if cell is not None else training_cell(cfg)
    net.eval()
    enc = net.encoder
    n = cell.n_sites
    rng = np.random.default_rng(seed)
    z = torch.from_numpy(rng.integers(0, len(cfg.elements), (1, n))).long()
    p = int(rng.integers(0, n))
    z2 = z.clone()
    z2[0, p] = (z[0, p] + 1) % len(cfg.elements)

    dev = next(net.parameters()).device
    x0 = enc(z.to(dev))
    x1 = enc(z2.to(dev))
    moved = (x1 - x0).abs().amax(-1)[0].cpu().numpy()
    thresh = tol * max(float(np.abs(x0.cpu().numpy()).max()), 1.0)
    affected = moved > thresh
    dist = hop_distances(cell.adjacency_lists(cfg.shells), p)
    return int(dist[affected].max()) if affected.any() else 0


def jump_endpoints(z, vacancy, jumper, vac_index):
    """Initial and final species vectors of one vacancy jump."""
    zi = z.copy()
    zi[vacancy] = vac_index
    zf = z.copy()
    zf[vacancy] = z[jumper]
    zf[jumper] = vac_index
    return zi, zf


@torch.no_grad()
def size_transfer_error(net, cfg, de_reach, factor=2, n_samples=8, seed=0,
                        device="cpu"):
    """Same jump, same neighbourhood, larger box: |difference| in meV."""
    net.eval()
    small = training_cell(cfg)
    big = Supercell(small.lattice, cfg.size * factor)
    small_graph = net.encoder.graph
    big_graph = LatticeGraph.for_supercell(
        big, cfg.shells, net.encoder.use_line,
        dtype=next(net.parameters()).dtype, device=device)

    v, j = small.reference_jump()
    v_big, j_big = big.index(small.coords[v]), big.index(small.coords[j])
    rep = small.replication_map(factor)
    which = 0 if cfg.target == "de" else 1
    m_small = torch.from_numpy(readout_masks(cfg, de_reach, small)[which]).to(device)
    m_big = torch.from_numpy(
        readout_masks(cfg, de_reach, big, (v_big, j_big))[which]).to(device)

    def as_batch(a):
        return torch.from_numpy(a).long().unsqueeze(0).to(device)

    rng = np.random.default_rng(seed)
    diffs = []
    try:
        for _ in range(n_samples):
            z = rng.integers(0, len(cfg.elements), small.n_sites)
            zi, zf = jump_endpoints(z, v, j, cfg.vac_index)
            zi_b, zf_b = jump_endpoints(z[rep], v_big, j_big, cfg.vac_index)
            net.set_graph(small_graph)
            e_small = float(net(as_batch(zi), as_batch(zf), m_small))
            net.set_graph(big_graph)
            e_big = float(net(as_batch(zi_b), as_batch(zf_b), m_big))
            diffs.append(abs(e_small - e_big))
    finally:
        net.set_graph(small_graph)
    d = np.asarray(diffs) * 1e3
    return {"factor": factor, "n": n_samples, "mean_meV": float(d.mean()),
            "max_meV": float(d.max())}
