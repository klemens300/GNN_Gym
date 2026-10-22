# GNN_Gym

A learned lattice model for vacancy-mediated kinetics in multicomponent
alloys, with kinetic Monte Carlo (KMC) and Metropolis Monte Carlo on top of
it. This repository contains the code, models and data of

> K. Lechner, J. Zhang, P. Wagatha, W. Knabl, H. Clemens, D. Holec,
> *Learned lattice models for kinetic and Metropolis Monte Carlo in
> multicomponent alloys: kinetic freezing of chemical order in Mo–Nb–Ta–W*
> (2026).

## What it does

The migration barrier of a vacancy jump A -> B is split into

    E_a(A -> B) = B_sym(A, B) + 1/2 dE(A -> B)

and each part is learned by its own graph network on the fixed bcc lattice:

- **Site-energy model (dE).** A site energy is read off every lattice site
  and dE is the change of their sum. A sum of site terms is a state function,
  so the predicted dE add up to zero around every closed sequence of jumps
  and the KMC rates obey detailed balance exactly. The receptive field is
  measured and kept inside the training cell.
- **Barrier model (B_sym).** Symmetric in the two endpoints by construction,
  read out per site around the jumping pair, with a line graph over the
  first-neighbour bonds.

Both models are trained on vacancy jumps computed with climbing-image NEB
and a universal machine-learned interatomic potential (UMA, fairchem) in
7x7x7 bcc cells. The trained models drive

- a rejection-free (BKL) KMC with an escape layer for trapped vacancies,
  converted to physical time with the vacancy concentration c_v(T) and a
  Debye attempt frequency, and
- a Metropolis swap Monte Carlo of the equilibrium order.

## Installation

```bash
conda env create -f environment.yml
conda activate gnn-gym
pip install -e ".[oracle,test]"
```

The KMC, Metropolis and the model need only numpy, scipy, torch and pyyaml
(`pip install -e .`). The `oracle` extra (ase, fairchem-core, matscipy) is
needed only to generate reference data with the uMLIP; the UMA checkpoint is
downloaded from Hugging Face on first use and requires accepting its licence.

## Quick start

```bash
# error of the deployed models on the 1500 referee jumps
python -m gnn_gym.training.evaluate \
    --de models/MoNbTaW/de_L2_s0.pt --bsym models/MoNbTaW/bsym_site_s0.pt

# retrain the site-energy model (about 5 min on one GPU)
python -m gnn_gym.training.train configs/training/de_L2.yaml --out-dir runs/models

# tests
pytest
```

## Repository layout

```
src/gnn_gym/
  lattice/      lattices in integer coordinates, supercells, graph searches
  model/        site-energy and barrier models, checkpoints, KMC predictor
  training/     jump datasets, training, evaluation, learning curve
  oracle/       uMLIP reference: lattice parameter, NEB jump datasets, ...
  realtime/     vacancy concentration, attempt frequency, physical time
  kmc/          BKL kinetic Monte Carlo with escape layer
  metropolis/   swap Metropolis Monte Carlo, equilibration, order parameters
  analysis/     diffusion coefficients, correlation factors, trapping
  validation/   comparisons with uMLIP and DFT, timing
configs/        one YAML file per paper run
models/         trained models used in the paper
data/           input data (jump datasets, c_v spectra, nu0 grid, literature)
tests/
```

## Reproducing the paper

`REPRODUCE.md` lists the commands for every step, from the uMLIP reference
data to the analysis of the simulations. They write their outputs to
`results/` and `runs/`. The simulation trajectories of the paper and the DFT
reference data are archived separately (see REPRODUCE.md) and can be placed
in `runs/` and `data/dft/` to rerun the analysis without rerunning the
simulations.

## Working with an AI assistant

`AGENTS.md` describes the code for coding agents and AI assistants: the
structure, the properties that must not break (with the tests that guard
them), how to add another element set or lattice, and the facts needed to
answer questions about models, methods and results. Agents that read
`AGENTS.md` or `CLAUDE.md` pick it up automatically when started in the
repository; for a chat assistant, attach the file to the conversation.

## Scope

The lattice, the element set and the cell sizes are inputs, not constants of
the code: lattices are defined in `gnn_gym/lattice/lattice.py` (bcc so far),
material systems in `configs/systems.yaml`. Other element sets on the bcc
lattice need a new jump dataset and retraining; other lattices need a further
`Lattice` definition.

## Citation and licence

See `CITATION.cff`. MIT licence, see `LICENSE`.
