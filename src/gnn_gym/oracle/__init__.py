"""The uMLIP reference (UMA through fairchem and ASE).

    uma           calculator wrapper: relaxations, climbing-image NEB
    structures    decorated supercells as ASE Atoms, jump endpoints
    calibrate     lattice parameter of a composition
    jumps         reference jump datasets (training and referee set)
    check_jumps   quality check of a jump dataset
    formation     vacancy formation energies from the jump datasets (c_v)
    elastic       elastic constants on the composition grid (attempt frequency)
    window_study  cell-size convergence of the jump labels
    neb_settings  image count and force tolerance of the NEB

ase and fairchem are imported only where a calculation needs them, so the
analysis parts run without them.
"""
