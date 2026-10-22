"""Crystal lattices in integer coordinates.

A lattice is described by its conventional cubic cell: an edge length and a
basis, both given as integer vectors in a common length unit (a/2 for bcc).
Every site of a supercell then has integer coordinates, squared distances are
integers, and a neighbour shell is an exact set of displacement vectors rather
than a distance window. The lattice parameter enters only when positions are
reported in Angstrom.

Everything downstream -- the model graphs, receptive fields, the KMC neighbour
table, the Metropolis sublattices -- is derived from these few numbers. A
further cubic lattice is therefore added by defining one more `Lattice`
instance; bcc is the one used and tested so far.

    bcc, units of a/2       |d|^2   neighbours   distance
        shell 1               3          8       sqrt(3)/2 a
        shell 2               4          6       a
        shell 3               8         12       sqrt(2) a
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Lattice:
    """A cubic lattice with integer site coordinates.

    name            identifier used in configs and checkpoints
    cell_edge       edge of the conventional cell in integer units
    basis           sites of the conventional cell in integer units
    units_per_a     integer units per lattice parameter a
    shell_d2        squared length of the displacement vectors of shell
                    1, 2, 3, ... in integer units
    reference_jump  jump vector of the training-cell convention: the vacancy
                    sits on the first basis site of the central conventional
                    cell, the jumping atom at this offset from it
    """
    name: str
    cell_edge: int
    basis: tuple
    units_per_a: float
    shell_d2: tuple
    reference_jump: tuple

    # A vacancy exchanges with a first-shell neighbour.
    jump_shell: int = 1

    def shell_vectors(self, shell):
        """Displacement vectors of one shell, lexicographically ordered."""
        return _shell_vectors(self, shell)

    def coordination(self, shell):
        return len(self.shell_vectors(shell))

    def angle_dots(self):
        """Exact angle classes between two consecutive jump-shell bonds.

        Bond (i->j) and bond (j->k) enclose an angle whose cosine is fixed by
        the integer dot product of the two bond vectors, so the classes are
        the distinct dot products between jump-shell vectors -- for bcc
        {-3, -1, 1, 3}, i.e. 180, 109.5, 70.5 and 0 degrees. No radial basis
        and no floating-point binning is needed.
        """
        v = self.shell_vectors(self.jump_shell)
        return tuple(int(d) for d in np.unique(v @ v.T))

    def shell_distance(self, shell, a):
        """Distance of a shell in Angstrom for lattice parameter a."""
        return float(np.sqrt(self.shell_d2[shell - 1])) / self.units_per_a * a


@lru_cache(maxsize=None)
def _shell_vectors(lattice, shell):
    d2 = lattice.shell_d2[shell - 1]
    r = int(np.ceil(np.sqrt(d2)))
    grid = np.arange(-r, r + 1)
    d = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    d = d[(d ** 2).sum(1) == d2]
    # Keep only lattice translations: from every basis site the vector has to
    # land on a site again. For bcc every integer vector of these lengths
    # qualifies; the filter matters for lattices whose integer grid is finer
    # than their translation group.
    basis = np.asarray(lattice.basis)
    edge = lattice.cell_edge
    allowed = {tuple(b) for b in basis % edge}
    ok = [all(tuple((b + v) % edge) in allowed for b in basis) for v in d]
    v = d[np.asarray(ok, dtype=bool)]
    if len(v) == 0:
        raise ValueError(f"{lattice.name}: shell {shell} has no lattice vectors")
    v.setflags(write=False)
    return v


BCC = Lattice(
    name="bcc",
    cell_edge=2,
    basis=((0, 0, 0), (1, 1, 1)),
    units_per_a=2.0,
    shell_d2=(3, 4, 8),
    reference_jump=(1, 1, 1),
)

LATTICES = {BCC.name: BCC}


def get_lattice(name):
    try:
        return LATTICES[name]
    except KeyError:
        raise ValueError(f"unknown lattice {name!r}; available: "
                         f"{sorted(LATTICES)}") from None
