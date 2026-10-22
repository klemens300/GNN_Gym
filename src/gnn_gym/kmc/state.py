"""Occupation of a supercell with a single vacancy.

The state is an integer species array over the sites of a
gnn_gym.lattice.Supercell, with VACANCY on the vacancy site. Geometry never
changes during a run: the candidate jumps of a site are its jump-shell
neighbours in ascending site order (the order in which barriers, rates and
the chosen candidate are reported), and displacements are integer
minimum-image vectors of the lattice coordinates.
"""
import numpy as np

from ..model.predictor import VACANCY


class LatticeState:
    def __init__(self, cell, elements, species, vacancy):
        self.cell = cell
        self.elements = tuple(elements)
        self.species = np.asarray(species, dtype=np.int8)
        self.vacancy = int(vacancy)
        self.neighbours = cell.neighbours(cell.lattice.jump_shell, sort=True)
        if self.species[self.vacancy] != VACANCY:
            raise ValueError("the vacancy site does not hold the vacancy")

    @property
    def n_sites(self):
        return self.cell.n_sites

    def targets(self):
        """Sites the vacancy can exchange with, in candidate order."""
        return self.neighbours[self.vacancy]

    def swap(self, target):
        """Exchange the vacancy with the atom on `target`."""
        target = int(target)
        if self.species[target] == VACANCY:
            raise ValueError(f"site {target} is the vacancy")
        self.species[self.vacancy] = self.species[target]
        self.species[target] = VACANCY
        self.vacancy = target

    def key(self):
        """Hashable identity of the configuration."""
        return self.vacancy, self.species.tobytes()

    def copy(self):
        return LatticeState(self.cell, self.elements, self.species.copy(),
                            self.vacancy)


def centre_site(cell):
    """The site closest to the centre of the box (first one on a tie)."""
    # doubled coordinates keep the centre on the integer grid
    d = 2 * cell.coords - cell.box
    d = (d + cell.box) % (2 * cell.box) - cell.box
    return int(np.argmin((d ** 2).sum(axis=1)))


def random_state(cell, elements, composition, rng):
    """Random decoration at the given fractions, vacancy at the box centre.

    The site list is filled in the order of `elements` with round(x n) sites
    each (completed with the first element, cut to the number of sites) and
    shuffled with `rng`; the centre site is then emptied.
    """
    n = cell.n_sites
    counts = [int(round(composition[e] * n)) for e in elements]
    species = np.repeat(np.arange(len(elements), dtype=np.int8), counts)
    if len(species) < n:
        species = np.concatenate(
            [species, np.zeros(n - len(species), dtype=np.int8)])
    species = species[:n].copy()
    rng.shuffle(species)
    vacancy = centre_site(cell)
    species[vacancy] = VACANCY
    return LatticeState(cell, elements, species, vacancy)
