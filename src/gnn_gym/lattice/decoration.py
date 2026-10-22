"""Random decorations of a supercell with a fixed composition.

The jump datasets store scalars only: the realised composition and a seed.
Every structure is rebuilt from them with `decorate`, so the reference
generator and the model data loader must use this one function. The symbol
list is filled in alphabetical element order and shuffled with numpy's
default generator; changing either breaks the reconstruction of the stored
datasets.
"""
import numpy as np


def decorate(counts, seed, n_sites=None):
    """Symbol list of a random decoration from integer per-element counts."""
    symbols = [e for e in sorted(counts) for _ in range(counts[e])]
    if n_sites is not None and len(symbols) != n_sites:
        raise ValueError(f"{len(symbols)} symbols for {n_sites} sites")
    np.random.default_rng(seed).shuffle(symbols)
    return symbols


def counts_from_fractions(fractions, n_sites):
    """Integer counts from realised fractions (counts / n_sites).

    The datasets print the realised fractions with six decimals, so rounding
    recovers the counts exactly. A rounding residual, possible only for
    inputs that were not realised fractions, goes to the largest element so
    the total stays n_sites.
    """
    counts = {e: int(round(f * n_sites)) for e, f in fractions.items()}
    counts = {e: c for e, c in counts.items() if c > 0}
    rest = n_sites - sum(counts.values())
    if rest:
        big = max(counts, key=counts.get)
        counts[big] += rest
    return counts
