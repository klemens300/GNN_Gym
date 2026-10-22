"""A periodic size^3 block of conventional cells.

Site order: conventional cells in C order of their (i, j, k) index, and within
each cell the basis in the order of `Lattice.basis`. For bcc this puts the
corner before the body centre and is the order of ase's
``bulk(..., cubic=True).repeat(size)``, so site indices agree with structures
built by ase.

All neighbour relations are built by displacing every site by the known shell
vectors and looking the target up in a dense coordinate index. That costs
O(n_sites) instead of the O(n_sites^2) of a distance matrix, which matters for
the 24^3 cells of the size checks.
"""
import numpy as np


class Supercell:
    def __init__(self, lattice, size):
        self.lattice = lattice
        self.size = int(size)
        self.box = lattice.cell_edge * self.size

        i, j, k = np.meshgrid(*(np.arange(self.size),) * 3, indexing="ij")
        cells = lattice.cell_edge * np.stack([i, j, k], -1).reshape(-1, 3)
        basis = np.asarray(lattice.basis, dtype=np.int64)
        coords = (cells[:, None, :] + basis[None]).reshape(-1, 3)
        self.coords = coords.astype(np.int64)
        self.coords.setflags(write=False)
        self.n_sites = len(self.coords)

        index = -np.ones((self.box,) * 3, dtype=np.int64)
        index[tuple(self.coords.T)] = np.arange(self.n_sites)
        if (index >= 0).sum() != self.n_sites:
            raise RuntimeError("site coordinates are not unique")
        self._index = index
        self._tables = {}

    # -- coordinates --------------------------------------------------------
    def index(self, coords):
        """Site index of one coordinate or an (..., 3) array of them.

        Coordinates are wrapped into the box first; a coordinate that is not
        a lattice site raises.
        """
        c = np.asarray(coords, dtype=np.int64) % self.box
        idx = self._index[c[..., 0], c[..., 1], c[..., 2]]
        if np.any(idx < 0):
            raise ValueError("coordinate is not a lattice site")
        return int(idx) if idx.ndim == 0 else idx

    def min_image(self, delta):
        """Minimum-image convention on integer displacement vectors."""
        h = self.box // 2
        return (np.asarray(delta) + h) % self.box - h

    def positions(self, a):
        """Cartesian site positions in Angstrom."""
        return self.coords * (a / self.lattice.units_per_a)

    def cell(self, a):
        """Cell matrix (rows are the box vectors) in Angstrom."""
        return np.eye(3) * self.box * (a / self.lattice.units_per_a)

    def sublattice(self):
        """Index of the basis site each lattice site belongs to."""
        basis = {tuple(b): n for n, b in enumerate(self.lattice.basis)}
        rel = self.coords % self.lattice.cell_edge
        return np.array([basis[tuple(r)] for r in rel], dtype=np.int64)

    def reference_jump(self):
        """(vacancy, jumping atom) of the training-cell convention.

        The vacancy is the first basis site of the central conventional cell,
        the jumping atom sits at `Lattice.reference_jump` from it, so both are
        in the same cell for every cell size.
        """
        c = self.lattice.cell_edge * (self.size // 2)
        v = np.asarray(self.lattice.basis[0]) + c
        return self.index(v), self.index(v + np.asarray(self.lattice.reference_jump))

    # -- neighbour relations -----------------------------------------------
    def neighbours(self, shell, sort=False):
        """(n_sites, z) neighbour indices of one shell.

        Column t is the neighbour along the t-th shell vector, which gives
        every site the same slot -> direction assignment; the line graph and
        the KMC halo rely on that. sort=True returns each row in ascending
        site order instead (the candidate order of the KMC).
        """
        key = (shell, sort)
        if key not in self._tables:
            vec = self.lattice.shell_vectors(shell)
            if 2 * int(np.abs(vec).max()) > self.box:
                raise ValueError(f"shell {shell} does not fit into a "
                                 f"{self.size}^3 cell")
            tab = self.index(self.coords[:, None, :] + vec[None])
            # In a cell too small for the shell, two vectors reach the same
            # periodic image and the neighbour would be counted twice.
            rows = np.sort(tab, axis=1)
            if np.any(rows[:, 1:] == rows[:, :-1]):
                raise ValueError(f"shell {shell}: repeated neighbour, the "
                                 f"{self.size}^3 cell is too small")
            if sort:
                tab = rows
            tab.setflags(write=False)
            self._tables[key] = tab
        return self._tables[key]

    def shell_edges(self, shells):
        """{shell: (2, n_sites * z) edge index}, both directions present."""
        out = {}
        for s in shells:
            tab = self.neighbours(s)
            src = np.repeat(np.arange(self.n_sites), tab.shape[1])
            out[s] = np.stack([src, tab.reshape(-1)]).astype(np.int64)
        return out

    def adjacency_lists(self, shells):
        """Per-site neighbour lists over all given shells (for graph search)."""
        tabs = [self.neighbours(s) for s in shells]
        return [np.concatenate([t[i] for t in tabs]).tolist()
                for i in range(self.n_sites)]

    def replication_map(self, factor=2):
        """Site map of the (factor*size)^3 cell onto this cell.

        Applying it to a decoration gives a larger cell in which every local
        environment is identical to one in this cell; only the box is larger.
        """
        big = Supercell(self.lattice, self.size * factor)
        return self.index(big.coords % self.box)
