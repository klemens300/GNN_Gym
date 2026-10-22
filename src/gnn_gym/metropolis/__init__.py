"""Equilibrium order of the site-energy model by canonical Metropolis Monte Carlo.

    sampler      swap moves evaluated on halos around the swapped sites
    run          cooling and heating chains through a list of temperatures
    observables  Warren-Cowley parameters, sublattice order, heat capacity
    order        concentration waves, structure factor, local B2 order
    autocorr     autocorrelation times and equilibration tests of the traces
    exafs        neighbour fractions in the two-group observable of EXAFS
    analysis     every quantity the paper reads from the chains

The site-energy model is the one that supplies dE to the KMC, so Metropolis
and KMC target the same Boltzmann distribution (the Metropolis chain samples
it exactly with the until_accept walk of the sampler): a gap between the
order the KMC reaches and the equilibrium order here is kinetic.
"""
