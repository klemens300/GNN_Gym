"""Saving and loading jump-model checkpoints.

A checkpoint holds the architecture (`ModelConfig`), the weights and the
measured reach of the encoder; the reach fixes the read-out masks and the
size of the KMC halo. Everything else stored with it is plain Python data,
so checkpoints load with ``torch.load(weights_only=True)``.
"""
import torch

from ..lattice import Supercell, get_lattice
from .config import ModelConfig
from .graph import LatticeGraph
from .networks import build_network

FORMAT = "gnn_gym.jump_model.v1"


def save_checkpoint(path, net, cfg, reach, **meta):
    torch.save({"format": FORMAT, "model_config": cfg.to_dict(),
                "state_dict": net.state_dict(), "reach": int(reach), **meta},
               path)


def read_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} checkpoint "
                         f"(format {ck.get('format')!r})")
    return ck


def load_model(path, device="cpu", bind=True):
    """(network, ModelConfig, checkpoint dict).

    bind=True attaches the graph of the training cell, which is what training
    and evaluation on the jump datasets use. Simulations bind their own graph.
    """
    ck = read_checkpoint(path, device)
    cfg = ModelConfig.from_dict(ck["model_config"])
    lat = get_lattice(cfg.lattice)
    net = build_network(cfg, n_angle_classes=len(lat.angle_dots()))
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    if bind:
        net.set_graph(LatticeGraph.for_supercell(
            Supercell(lat, cfg.size), cfg.shells, cfg.line_graph,
            device=device))
    return net, cfg, ck
