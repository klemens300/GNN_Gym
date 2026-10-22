"""Sparse graph operators of one cell.

Every sample on a given cell has the same geometry; only the species differ.
The neighbour operators are therefore constants of the cell: row-normalised
sparse matrices (a neighbour mean) per neighbour shell, and, for the line
graph, one operator per exact bond-angle class. Message passing becomes a
product with a fixed sparse matrix, with no per-sample graph construction.

Bonds are the directed jump-shell bonds, indexed bond(i, slot) = i * z + slot
where slot is the column of the neighbour table. The bond -> site aggregation
is then a reshape to (batch, n_sites, z, features) and a mean over axis 2.

`LatticeGraph.from_tables` builds the operators from neighbour tables alone.
The full-cell graph and the halo sub-graph of the KMC predictor (whose
boundary sites point at a padding site) are both built through it, so they
cannot differ in construction.
"""
import numpy as np
import torch


def normalised_operator(edge_index, n_rows, n_cols, dtype=torch.float32,
                        device="cpu"):
    """Row-normalised sparse operator: row i averages over its edges."""
    src = torch.as_tensor(np.asarray(edge_index[0]), dtype=torch.long)
    dst = torch.as_tensor(np.asarray(edge_index[1]), dtype=torch.long)
    deg = torch.bincount(src, minlength=n_rows).clamp(min=1).to(dtype)
    val = (1.0 / deg)[src]
    a = torch.sparse_coo_tensor(torch.stack([src, dst]), val,
                                (n_rows, n_cols), dtype=dtype).coalesce()
    return a.to(device)


class LatticeGraph:
    """Neighbour-shell operators and, optionally, line-graph operators."""

    def __init__(self, n_sites, adj, line=(), bond_src=None, bond_dst=None,
                 z=None):
        self.n_sites = int(n_sites)
        self.adj = list(adj)
        self.line = list(line)
        self.with_line = bool(self.line)
        self.z = z
        self.bond_src = bond_src
        self.bond_dst = bond_dst
        self.n_bonds = None if z is None else self.n_sites * z

    @classmethod
    def from_tables(cls, tables, line_vectors=None, angle_dots=None,
                    dtype=torch.float32, device="cpu"):
        """Build from neighbour tables.

        tables        list of (n_sites, z_s) neighbour index arrays, one per
                      shell in model order; the first is the jump shell
        line_vectors  (n_sites * z, 3) integer bond vectors of the jump-shell
                      bonds, or None for no line graph
        angle_dots    the allowed bond-pair dot products (angle classes)
        """
        n = len(tables[0])
        adj = []
        for tab in tables:
            tab = np.asarray(tab)
            src = np.repeat(np.arange(n), tab.shape[1])
            adj.append(normalised_operator(np.stack([src, tab.reshape(-1)]),
                                           n, n, dtype, device))
        if line_vectors is None:
            return cls(n, adj)

        tab = np.asarray(tables[0])
        z = tab.shape[1]
        bond_src = np.repeat(np.arange(n), z)
        bond_dst = np.array(tab.reshape(-1))
        line = line_operators(np.asarray(line_vectors), bond_dst, n, z,
                              angle_dots, dtype, device)
        return cls(n, adj, line,
                   torch.as_tensor(bond_src, device=device),
                   torch.as_tensor(bond_dst, device=device),
                   z)

    @classmethod
    def for_supercell(cls, cell, shells, with_line, dtype=torch.float32,
                      device="cpu"):
        """The graph of a whole periodic supercell."""
        lat = cell.lattice
        if list(shells)[0] != lat.jump_shell:
            raise ValueError("the first graph shell must be the jump shell")
        tables = [cell.neighbours(s) for s in shells]
        vec = None
        if with_line:
            # Column t of the neighbour table is the t-th shell vector, so
            # every bond vector is known without looking at positions.
            vec = np.tile(lat.shell_vectors(lat.jump_shell), (cell.n_sites, 1))
        return cls.from_tables(tables, vec, lat.angle_dots(), dtype, device)

    def to(self, device=None, dtype=None):
        conv = lambda a: a.to(device=device, dtype=dtype)       # noqa: E731
        self.adj = [conv(a) for a in self.adj]
        self.line = [conv(a) for a in self.line]
        if self.with_line and device is not None:
            self.bond_src = self.bond_src.to(device)
            self.bond_dst = self.bond_dst.to(device)
        return self


def line_operators(bond_vectors, bond_dst, n_sites, z, angle_dots,
                   dtype=torch.float32, device="cpu"):
    """One row-normalised line-graph operator per angle class.

    Bond (i->j) receives from every bond (j->k); the class of the pair is the
    integer dot product of the two bond vectors.
    """
    n_bonds = n_sites * z
    rows, cols, dots = [], [], []
    for t in range(z):
        second = bond_dst * z + t
        rows.append(np.arange(n_bonds))
        cols.append(second)
        dots.append((bond_vectors * bond_vectors[second]).sum(1))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    dots = np.concatenate(dots)
    classes = np.asarray(angle_dots)
    bad = sorted(set(dots.tolist()) - set(classes.tolist()))
    if bad:
        raise ValueError(f"unexpected bond-pair dot products {bad}")
    cls = np.searchsorted(classes, dots)
    out = []
    for c in range(len(classes)):
        m = cls == c
        out.append(normalised_operator(np.stack([rows[m], cols[m]]),
                                       n_bonds, n_bonds, dtype, device))
    return out
