"""Message passing on a fixed lattice graph.

One layer aggregates every neighbour shell with its own weights and updates
the site. With the line graph on, the layer also passes messages between
bonds that share a site, one learned channel gate per exact angle class, and
folds the bonds back into their source sites. An atom-centred convolution
treats the neighbours of a site as a set and cannot tell two heavy neighbours
on opposite sides from two side by side; the line graph restores that
arrangement, which decides the saddle point of a jump.

There is no BatchNorm: it would couple samples in a batch and sites across
the cell. LayerNorm acts per site over the feature dimension only, so a site
feature depends on nothing but the species within the encoder's reach. That
locality is what makes the site-energy sum a state function, and it also
makes the features of one sample independent of the rest of the batch.

The weights do not depend on the cell. `set_graph` binds the encoder to the
graph of any cell, e.g. the halo around the vacancy in a KMC supercell.
"""
import torch
import torch.nn as nn


def mlp(dims, out_dim, act=nn.SiLU):
    layers, prev = [], dims[0]
    for h in dims[1:]:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def spmm(a, x):
    """Sparse operator applied to batched node features (B, N, F)."""
    b, n, f = x.shape
    flat = x.permute(1, 0, 2).reshape(n, b * f)
    return torch.sparse.mm(a, flat).view(n, b, f).permute(1, 0, 2)


class LatticeEncoder(nn.Module):
    def __init__(self, n_species, hidden, n_layers, n_shells, use_line=False,
                 n_angle_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(n_species, hidden)
        self.n_layers = n_layers
        self.n_shells = n_shells
        self.use_line = use_line
        self.hidden = hidden

        self.atom_update = nn.ModuleList([
            mlp([(1 + n_shells) * hidden, 2 * hidden], hidden)
            for _ in range(n_layers)])
        self.atom_norm = nn.ModuleList([nn.LayerNorm(hidden)
                                        for _ in range(n_layers)])
        if use_line:
            self.bond_init = mlp([2 * hidden, 2 * hidden], hidden)
            # One channel gate per angle class keeps the bond update input at
            # 2 * hidden instead of (1 + n_classes) * hidden.
            self.angle_gate = nn.Parameter(
                torch.ones(n_angle_classes, hidden) / n_angle_classes)
            self.bond_update = nn.ModuleList([
                mlp([2 * hidden, 2 * hidden], hidden) for _ in range(n_layers)])
            self.bond_norm = nn.ModuleList([nn.LayerNorm(hidden)
                                            for _ in range(n_layers)])
            self.bond_to_atom = nn.ModuleList([
                mlp([2 * hidden, 2 * hidden], hidden) for _ in range(n_layers)])
        self.graph = None

    def set_graph(self, graph):
        if len(graph.adj) != self.n_shells:
            raise ValueError(f"graph has {len(graph.adj)} shells, encoder "
                             f"expects {self.n_shells}")
        if self.use_line:
            if not graph.with_line:
                raise ValueError("encoder needs a line graph, the graph has none")
            if len(graph.line) != self.angle_gate.shape[0]:
                raise ValueError("number of angle classes does not match")
        self.graph = graph

    def forward(self, z):
        """z: (B, N) species indices -> (B, N, F) site features."""
        g = self.graph
        if g is None:
            raise RuntimeError("encoder has no graph; call set_graph() first")
        x = self.embedding(z)
        b = None
        if self.use_line:
            b = self.bond_init(
                torch.cat([x[:, g.bond_src], x[:, g.bond_dst]], dim=-1))

        for k in range(self.n_layers):
            parts = [x] + [spmm(a, x) for a in g.adj]
            x = self.atom_norm[k](x + self.atom_update[k](
                torch.cat(parts, dim=-1)))
            if self.use_line:
                agg = 0.0
                for c, a in enumerate(g.line):
                    agg = agg + self.angle_gate[c] * spmm(a, b)
                b = self.bond_norm[k](b + self.bond_update[k](
                    torch.cat([b, agg], dim=-1)))
                nb = b.view(b.shape[0], g.n_sites, g.z, -1).mean(2)
                x = x + self.bond_to_atom[k](torch.cat([x, nb], dim=-1))
        return x
