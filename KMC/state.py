"""
KMCState: lattice geometry + species + vacancy tracking.

Internal representation is a logical site model (no ASE in the hot loop):
- positions:  [N_sites, 3]    cartesian site coordinates       (read-only)
- cell:       [3, 3]          orthogonal cubic cell vectors    (read-only)
- nn_table:   [N_sites, 8]    precomputed nearest-neighbour ids (read-only)
- species:    [N_sites]       int8 element index per site       (mutable)
- vacancy_index: int          current vacancy site              (mutable)

Vacancy is encoded by the reserved species value VACANCY_SPECIES = -1.

ASE conversion is on-demand via to_atoms(): two modes are supported,
one for GNN inference (N-1 atoms, vacancy site omitted) and one for
visualization (vacancy as a pseudo-atom with symbol "X" so OVITO can render
it). The geometry (positions/cell/nn_table) is cached at module level so that
many KMCState instances share the same arrays without recomputation.
"""

from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.build import bulk


# Reserved species value marking a site as the vacancy
VACANCY_SPECIES = -1


# ---------------------------------------------------------------------------
# Module-level geometry cache
# ---------------------------------------------------------------------------

# Keyed by (supercell_size, lattice_parameter_A); shared (read-only) across
# all KMCState instances built for the same geometry.
_GEOMETRY_CACHE: dict = {}


def _build_bcc_geometry(
    supercell_size: int, lattice_parameter_A: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build BCC supercell positions, cell, and NN table (cached)."""
    key = (int(supercell_size), float(lattice_parameter_A))
    if key not in _GEOMETRY_CACHE:
        crystal = bulk(
            "Fe",
            crystalstructure="bcc",
            a=lattice_parameter_A,
            cubic=True,
        )
        supercell = crystal.repeat([supercell_size] * 3)
        positions = supercell.positions.copy()
        cell = supercell.cell.array.copy()
        nn_table = _compute_bcc_nn_table(positions, cell, lattice_parameter_A)
        # Make read-only to defend against accidental mutation
        positions.setflags(write=False)
        cell.setflags(write=False)
        nn_table.setflags(write=False)
        _GEOMETRY_CACHE[key] = (positions, cell, nn_table)
    return _GEOMETRY_CACHE[key]


def _compute_bcc_nn_table(
    positions: np.ndarray, cell: np.ndarray, lattice_parameter_A: float
) -> np.ndarray:
    """Precompute the [N_sites, 8] NN-index table with PBC.

    BCC nearest-neighbour distance is a * sqrt(3) / 2.
    """
    n_sites = positions.shape[0]
    d_nn = lattice_parameter_A * np.sqrt(3.0) / 2.0
    tol = 0.05  # Angstrom tolerance for "is a NN"

    L = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])  # cubic / orthogonal cell

    nn_table = np.full((n_sites, 8), -1, dtype=np.int64)
    for i in range(n_sites):
        deltas = positions - positions[i]
        for ax in range(3):
            deltas[:, ax] -= np.round(deltas[:, ax] / L[ax]) * L[ax]
        distances = np.linalg.norm(deltas, axis=1)
        mask = (distances > 1e-6) & (np.abs(distances - d_nn) < tol)
        nn_indices = np.where(mask)[0]
        if len(nn_indices) != 8:
            raise RuntimeError(
                f"Site {i}: expected 8 nearest neighbours at distance "
                f"{d_nn:.4f} A, found {len(nn_indices)}"
            )
        nn_table[i] = nn_indices
    return nn_table


def _find_center_site_index(positions: np.ndarray, cell: np.ndarray) -> int:
    """Return the index of the site closest to the cell centre (with PBC)."""
    center = np.diagonal(cell) / 2.0
    return _find_site_closest_to_xyz(positions, cell, center)


def _find_site_closest_to_xyz(
    positions: np.ndarray,
    cell: np.ndarray,
    target_xyz,
) -> int:
    """Return the index of the site closest to ``target_xyz`` (with PBC).

    Coordinates are in Angstrom; the cell is assumed orthogonal-cubic so
    the diagonal entries are the box lengths along each axis.
    """
    target = np.asarray(target_xyz, dtype=float)
    if target.shape != (3,):
        raise ValueError(
            f"target_xyz must have shape (3,), got {target.shape}"
        )
    L = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
    deltas = positions - target
    for ax in range(3):
        deltas[:, ax] -= np.round(deltas[:, ax] / L[ax]) * L[ax]
    distances = np.linalg.norm(deltas, axis=1)
    return int(np.argmin(distances))


# ---------------------------------------------------------------------------
# KMCState
# ---------------------------------------------------------------------------

class KMCState:
    """Logical lattice + species + vacancy state for BCC vacancy KMC."""

    def __init__(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        nn_table: np.ndarray,
        species: np.ndarray,
        vacancy_index: int,
        element_symbols: List[str],
        jump_distance_A: float,
    ):
        self.positions = positions
        self.cell = cell
        self.nn_table = nn_table
        self.species = species
        self.vacancy_index = int(vacancy_index)
        self.element_symbols = list(element_symbols)
        self.jump_distance_A = float(jump_distance_A)

    # ------- Properties -------

    @property
    def n_sites(self) -> int:
        return int(self.species.shape[0])

    # ------- Constructors -------

    @classmethod
    def from_random_composition(
        cls, config, rng: Optional[np.random.Generator] = None
    ) -> "KMCState":
        """Random composition assignment with vacancy at the chosen position."""
        positions, cell, nn_table = _build_bcc_geometry(
            config.supercell_size, config.lattice_parameter_A
        )
        n_sites = positions.shape[0]

        if rng is None:
            rng = np.random.default_rng(config.random_seed)

        # Build a symbol list matching the composition fractions, then shuffle
        symbols: List[str] = []
        for el, frac in config.composition.items():
            symbols.extend([el] * int(round(frac * n_sites)))
        # Pad / truncate to exactly n_sites
        while len(symbols) < n_sites:
            symbols.append(config.elements[0])
        symbols = symbols[:n_sites]
        rng.shuffle(symbols)

        element_to_idx = {el: i for i, el in enumerate(config.elements)}
        species = np.array(
            [element_to_idx[s] for s in symbols], dtype=np.int8
        )

        # Place vacancy
        if config.vacancy_initial_position == "center":
            vac_idx = _find_center_site_index(positions, cell)
        elif config.vacancy_initial_position == "random":
            vac_idx = int(rng.integers(0, n_sites))
        elif config.vacancy_initial_position == "explicit":
            if config.vacancy_initial_xyz is None:
                # Defensive: KMCConfig.__post_init__ should have caught this.
                raise ValueError(
                    "vacancy_initial_position='explicit' but "
                    "vacancy_initial_xyz is None"
                )
            vac_idx = _find_site_closest_to_xyz(
                positions, cell, config.vacancy_initial_xyz
            )
        else:
            raise ValueError(
                f"Unknown vacancy_initial_position: "
                f"{config.vacancy_initial_position!r}"
            )
        species[vac_idx] = VACANCY_SPECIES

        jump_dist = config.lattice_parameter_A * np.sqrt(3.0) / 2.0
        return cls(
            positions, cell, nn_table, species, vac_idx,
            config.elements, jump_dist,
        )

    @classmethod
    def from_bicrystal(
        cls, config, rng: Optional[np.random.Generator] = None
    ) -> "KMCState":
        """Bicrystal: half/half along bicrystal_axis, vacancy at the interface."""
        positions, cell, nn_table = _build_bcc_geometry(
            config.supercell_size, config.lattice_parameter_A
        )
        n_sites = positions.shape[0]

        axis_map = {"x": 0, "y": 1, "z": 2}
        ax = axis_map[config.bicrystal_axis]
        L = cell[ax, ax]
        midpoint = L / 2.0

        el_a, el_b = config.bicrystal_elements
        for el in (el_a, el_b):
            if el not in config.elements:
                raise ValueError(
                    f"Bicrystal element '{el}' not in elements list "
                    f"{config.elements}"
                )
        element_to_idx = {el: i for i, el in enumerate(config.elements)}

        species = np.empty(n_sites, dtype=np.int8)
        for i in range(n_sites):
            species[i] = element_to_idx[
                el_a if positions[i, ax] < midpoint else el_b
            ]

        # Place the vacancy on the site closest to the interface plane
        distances_to_interface = np.abs(positions[:, ax] - midpoint)
        vac_idx = int(np.argmin(distances_to_interface))
        species[vac_idx] = VACANCY_SPECIES

        jump_dist = config.lattice_parameter_A * np.sqrt(3.0) / 2.0
        return cls(
            positions, cell, nn_table, species, vac_idx,
            config.elements, jump_dist,
        )

    @classmethod
    def from_slabs(
        cls, config, rng: Optional[np.random.Generator] = None
    ) -> "KMCState":
        """Multi-slab initial state: split the box along slab_axis into
        len(slab_elements) equal-thickness layers and assign each element
        to one layer in listed order. The vacancy is placed on the most
        central slab boundary plane.

        For two slabs this reproduces the bicrystal geometry; the dedicated
        from_bicrystal classmethod is kept for backwards compatibility.
        """
        positions, cell, nn_table = _build_bcc_geometry(
            config.supercell_size, config.lattice_parameter_A
        )
        n_sites = positions.shape[0]

        if config.slab_elements is None:
            raise ValueError(
                "from_slabs requires config.slab_elements to be set"
            )

        axis_map = {"x": 0, "y": 1, "z": 2}
        ax = axis_map[config.slab_axis]
        L = cell[ax, ax]

        slab_elements = list(config.slab_elements)
        n_slabs = len(slab_elements)
        for el in slab_elements:
            if el not in config.elements:
                raise ValueError(
                    f"Slab element '{el}' not in elements list "
                    f"{config.elements}"
                )

        element_to_idx = {el: i for i, el in enumerate(config.elements)}

        # Assign each site to a slab via its fractional position along the
        # axis. Slabs span [k*L/N, (k+1)*L/N) for k = 0..N-1; the upper
        # endpoint is clipped down so the last slab includes its boundary.
        species = np.empty(n_sites, dtype=np.int8)
        for i in range(n_sites):
            frac = positions[i, ax] / L
            k = min(int(frac * n_slabs), n_slabs - 1)
            species[i] = element_to_idx[slab_elements[k]]

        # Vacancy on the most central slab boundary plane.
        # For N slabs the boundaries lie at L*k/N for k = 1..N-1; the most
        # central one is at k = N // 2 (= L/2 for even N).
        central_boundary = L * (n_slabs // 2) / n_slabs
        distances_to_boundary = np.abs(positions[:, ax] - central_boundary)
        vac_idx = int(np.argmin(distances_to_boundary))
        species[vac_idx] = VACANCY_SPECIES

        jump_dist = config.lattice_parameter_A * np.sqrt(3.0) / 2.0
        return cls(
            positions, cell, nn_table, species, vac_idx,
            config.elements, jump_dist,
        )

    @classmethod
    def from_symbol_array(
        cls,
        config,
        symbols: List[str],
        vacancy_index: int,
    ) -> "KMCState":
        """Custom initial state from explicit symbol list and vacancy site."""
        positions, cell, nn_table = _build_bcc_geometry(
            config.supercell_size, config.lattice_parameter_A
        )
        n_sites = positions.shape[0]
        if len(symbols) != n_sites:
            raise ValueError(
                f"symbols length {len(symbols)} does not match "
                f"n_sites {n_sites}"
            )
        if not (0 <= vacancy_index < n_sites):
            raise ValueError(
                f"vacancy_index {vacancy_index} out of range [0, {n_sites})"
            )

        element_to_idx = {el: i for i, el in enumerate(config.elements)}
        species = np.array(
            [element_to_idx[s] for s in symbols], dtype=np.int8
        )
        species[vacancy_index] = VACANCY_SPECIES

        jump_dist = config.lattice_parameter_A * np.sqrt(3.0) / 2.0
        return cls(
            positions, cell, nn_table, species, vacancy_index,
            config.elements, jump_dist,
        )

    # ------- Mutation -------

    def swap_vacancy(self, target_site_index: int) -> None:
        """In-place: swap the vacancy with the atom at target_site_index."""
        target_site_index = int(target_site_index)
        if self.species[target_site_index] == VACANCY_SPECIES:
            raise ValueError(
                f"Target site {target_site_index} is already the vacancy"
            )
        # Move the atom from the target site into the (currently empty) vacancy
        self.species[self.vacancy_index] = self.species[target_site_index]
        self.species[target_site_index] = VACANCY_SPECIES
        self.vacancy_index = target_site_index

    # ------- Queries -------

    def get_neighbor_atom_indices(self) -> np.ndarray:
        """Return the 8 site indices of atoms surrounding the current vacancy."""
        return self.nn_table[self.vacancy_index].copy()

    def copy(self) -> "KMCState":
        """Return a copy with shared geometry but independent species/vacancy."""
        return KMCState(
            self.positions, self.cell, self.nn_table,
            self.species.copy(), self.vacancy_index,
            self.element_symbols, self.jump_distance_A,
        )

    # ------- Conversion -------

    def to_atoms(
        self,
        include_vacancy: bool = False,
        vacancy_symbol: str = "X",
    ) -> Atoms:
        """Build an ASE Atoms object from the current state.

        Args:
            include_vacancy: If False (default), omit the vacancy site -> N-1
                atoms, matching the GNN training topology. If True, include
                the vacancy site as a pseudo-atom with symbol
                `vacancy_symbol` so that OVITO and other viewers can render
                it.
            vacancy_symbol: Pseudo-atom symbol for the vacancy when
                include_vacancy=True. Default "X".
        """
        if include_vacancy:
            symbols = []
            for sp in self.species:
                if sp == VACANCY_SPECIES:
                    symbols.append(vacancy_symbol)
                else:
                    symbols.append(self.element_symbols[int(sp)])
            atoms = Atoms(
                symbols=symbols,
                positions=self.positions,
                cell=self.cell,
                pbc=True,
            )
        else:
            mask = self.species != VACANCY_SPECIES
            symbols = [
                self.element_symbols[int(s)] for s in self.species[mask]
            ]
            atoms = Atoms(
                symbols=symbols,
                positions=self.positions[mask],
                cell=self.cell,
                pbc=True,
            )
        return atoms
