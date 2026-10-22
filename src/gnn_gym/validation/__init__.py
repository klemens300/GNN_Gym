"""Checks of the oracle and the lattice model against independent references.

    ground_state  uMLIP energies of ordered candidates (ordering energetics)
    ordering      the same candidates with the site-energy model
    dft           uMLIP and lattice model against DFT-NEB barriers
    ordered_neb   uMLIP-NEB against the model in the trapped KMC states
    speed         cost per KMC step and per barrier

ase and fairchem are imported only where a calculation needs them.
"""
