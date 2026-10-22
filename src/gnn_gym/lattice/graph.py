"""Graph searches on a supercell: hop distances, hop balls, locality.

A site feature produced by an n-layer encoder depends on the species inside
the n-hop ball around that site. The model is strictly local if that ball does
not wrap around the periodic box, i.e. if it holds as many sites as the same
ball in an unbounded lattice. `locality_report` checks exactly that by
counting, instead of comparing a cutoff radius with half the box.
"""
import numpy as np

from .supercell import Supercell


def hop_distances(adj, source):
    """Graph distance of every site from `source` (-1 if unreachable)."""
    dist = np.full(len(adj), -1, dtype=int)
    dist[source] = 0
    frontier, d = [source], 0
    while frontier:
        d += 1
        nxt = []
        for a in frontier:
            for b in adj[a]:
                if dist[b] < 0:
                    dist[b] = d
                    nxt.append(b)
        frontier = nxt
    return dist


def hop_ball(adj, sources, n_hops):
    """Boolean mask of all sites within n_hops graph steps of `sources`."""
    seen = np.zeros(len(adj), dtype=bool)
    frontier = [int(s) for s in sources]
    seen[frontier] = True
    for _ in range(n_hops):
        nxt = []
        for a in frontier:
            for b in adj[a]:
                if not seen[b]:
                    seen[b] = True
                    nxt.append(b)
        frontier = nxt
    return seen


def locality_report(lattice, size, n_hops, shells, a=None):
    """Does the n-hop receptive field around the vacancy fit into the cell?

    The ball is counted in the given cell and in one at least three times as
    large. Equal counts mean the ball does not see its own periodic images, so
    every site energy is a function of a genuine local environment and their
    sum is a state function.
    """
    cell = Supercell(lattice, size)
    v, _ = cell.reference_jump()
    ball = hop_ball(cell.adjacency_lists(shells), [v], n_hops)

    big = Supercell(lattice, max(3 * size, 2 * n_hops + 3))
    vb, _ = big.reference_jump()
    n_free = int(hop_ball(big.adjacency_lists(shells), [vb], n_hops).sum())

    d = cell.min_image(cell.coords[ball] - cell.coords[v])
    r_max = float(np.sqrt((d ** 2).sum(-1).max())) / lattice.units_per_a
    half_box = 0.5 * size * lattice.cell_edge / lattice.units_per_a
    return {
        "n_hops": int(n_hops), "shells": list(shells),
        "sites_in_ball": int(ball.sum()), "sites_unbounded": n_free,
        "local": int(ball.sum()) == n_free,
        "r_max_a": r_max,
        "r_max_A": None if a is None else r_max * a,
        "half_box_A": None if a is None else half_box * a,
    }
