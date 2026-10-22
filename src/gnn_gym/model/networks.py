"""The two jump models.

    E_a(A -> B) = B_sym(A, B) + 1/2 dE(A -> B)

Site-energy model (dE)
    A site energy eps_i is read off every lattice site and the change of their
    sum is the prediction, dE = sum_i [eps_i(B) - eps_i(A)]. A sum of site
    terms is a state function: around any closed sequence of vacancy jumps the
    predicted dE add up to exactly zero, so the rates obey detailed balance
    however accurate the individual site terms are. This holds as long as each
    eps_i depends only on a genuine local environment, i.e. the encoder's
    receptive field stays inside the minimum image of the training cell
    (checked by `locality.measure_reach` before training).

    Only sites within the receptive field of the jumping pair change their
    environment; every other site contributes the same eps_i to both
    endpoints. The sum is therefore restricted to that ball. This is exact,
    and it avoids taking a difference of two large float32 sums.

Barrier model (B_sym)
    B_sym is symmetric under exchanging the endpoints, so it cancels around
    closed loops and detailed balance does not constrain it. The symmetry is
    built in: the head only sees [x_A + x_B, |x_B - x_A|] of the endpoint
    features. With the per-site read-out the head runs on every site of a
    ball around the jumping pair and the outputs are averaged; each term is
    symmetric by itself, and the arrangement of local environments reaches
    the head before any averaging.
"""
import torch
import torch.nn as nn

from .encoder import LatticeEncoder, mlp


class SiteEnergyModel(nn.Module):
    def __init__(self, cfg, n_angle_classes=4):
        super().__init__()
        self.encoder = LatticeEncoder(
            cfg.n_species, cfg.de_hidden, cfg.de_layers, len(cfg.shells),
            use_line=cfg.de_line_graph, n_angle_classes=n_angle_classes)
        self.site = mlp([cfg.de_hidden] + list(cfg.de_site_hidden), 1)

    def set_graph(self, graph):
        self.encoder.set_graph(graph)

    def site_energies(self, z):
        """(B, N) site energies. Only differences are used, so their absolute
        level (the gauge) is arbitrary."""
        return self.site(self.encoder(z)).squeeze(-1)

    def forward(self, z_init, z_final, mask):
        """mask: (N,) or (B, N), the sites whose environment changes."""
        eps = self.site_energies(z_final) - self.site_energies(z_init)
        return (eps * mask.to(eps.dtype)).sum(-1)


class BarrierModel(nn.Module):
    def __init__(self, cfg, n_angle_classes=4):
        super().__init__()
        self.encoder = LatticeEncoder(
            cfg.n_species, cfg.bsym_hidden, cfg.bsym_layers, len(cfg.shells),
            use_line=cfg.bsym_line_graph, n_angle_classes=n_angle_classes)
        self.head = mlp([2 * cfg.bsym_hidden] + list(cfg.bsym_head_hidden), 1)
        self.site_sum = cfg.bsym_site_sum

    def set_graph(self, graph):
        self.encoder.set_graph(graph)

    def readout(self, xi, xf, mask):
        """Per-site read-out on already-encoded endpoint features.

        xi, xf: (B_i, N, F) and (B, N, F) with B_i either B or 1. A KMC step
        evaluates several candidate jumps that leave the same initial
        configuration; the caller encodes it once and passes it with batch
        size 1, which broadcasts. Since the encoder couples no samples, the
        result is the same as encoding the initial state once per candidate.
        """
        feat = torch.cat([xi + xf, (xf - xi).abs()], dim=-1)
        per_site = self.head(feat).squeeze(-1)
        if mask is None:
            return per_site.mean(-1)
        m = mask.to(per_site.dtype)
        return (per_site * m).sum(-1) / m.sum(-1).clamp(min=1.0)

    def _pooled(self, z, mask):
        x = self.encoder(z)
        m = mask.to(x.dtype)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        m = m.unsqueeze(-1)
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, z_init, z_final, mask):
        if not self.site_sum:
            zi = self._pooled(z_init, mask)
            zf = self._pooled(z_final, mask)
            feat = torch.cat([zi + zf, (zf - zi).abs()], dim=-1)
            return self.head(feat).squeeze(-1)

        xi, xf = self.encoder(z_init), self.encoder(z_final)
        if mask.dim() == 1:
            # Fixed read-out ball (training cell): gather once, plain mean.
            idx = mask.nonzero(as_tuple=True)[0]
            return self.readout(xi[:, idx], xf[:, idx], None)
        return self.readout(xi, xf, mask)


def build_network(cfg, n_angle_classes=4):
    cls = SiteEnergyModel if cfg.target == "de" else BarrierModel
    return cls(cfg, n_angle_classes=n_angle_classes)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
