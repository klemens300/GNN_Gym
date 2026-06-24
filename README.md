# GNN-accelerated kinetic Monte Carlo for vacancy diffusion in MoNbTaW

A graph neural network (GNN) surrogate for vacancy **migration barriers** in
body-centred-cubic (BCC) refractory high-entropy alloys, deployed inside a
kinetic Monte Carlo (KMC) engine to reach diffusion and chemical-ordering
time scales that are out of reach for direct first-principles simulation.

This repository is the code accompanying the paper and contains the full
workflow: training the surrogate, running KMC with it, and recovering
physical time scales.

> Loading this repo into an AI assistant? See [`AGENTS.md`](AGENTS.md) for a
> task-oriented map of the code, exact entry-point commands, and the critical
> conventions an assistant needs to give correct guidance.

---

## The idea in one paragraph

A KMC simulation of vacancy diffusion repeats one expensive step millions of
times: map the local atomic environment around the migrating vacancy to its
migration barrier. That mapping is a function, and a function can be learned.
We replace the expensive evaluator recursively — density functional theory
(DFT) for the barrier was already replaced by a universal machine-learned
interatomic potential (uMLIP) running a nudged elastic band (NEB); here we go
one level further and learn the **entire NEB result** with a single GNN
forward pass. Because the barrier depends, to good approximation, on the local
environment of the vacancy, a locality-respecting GNN learns it accurately
from a finite dataset.

The workflow has three stages:

1. **Build** the surrogate — `gnn/`. Generate NEB barriers with a uMLIP
   oracle (UMA, `uma-s-1p1`), grow the dataset by active learning, and train
   the GNN to predict the barrier from unrelaxed endpoint structures.
2. **Deploy** it in the KMC engine — `KMC/`. At every Bortz–Kalos–Lebowitz
   (BKL) step the GNN evaluates the eight nearest-neighbour barriers around
   the vacancy in one batched forward pass.
3. **Rescale** to real time — `diffusion/`. A small number of additional
   uMLIP supercell calculations supply the vacancy formation energy
   `E_f^V`, the Debye attempt frequency `nu_0`, and the jump distance `a`,
   which convert simulated KMC time into physical time.

---

## Repository layout

```
GNN_Gym/
├── gnn/                # Stage 1 — GNN surrogate (training + active learning)
│   ├── config.py             # central configuration (elements, paths, hyperparameters)
│   ├── oracle.py             # uMLIP (UMA) NEB oracle — generates ground-truth barriers
│   ├── graph_builder.py      # structures -> PyG graphs (RBF edges, line graph, embeddings)
│   ├── model.py              # ALIGNN-style GNN encoder + MLP barrier head
│   ├── dataset.py            # CSV/NPZ dataset with mixed-fidelity trajectory sampling
│   ├── trainer.py            # training loop (early stopping, schedulers, W&B logging)
│   ├── inference.py          # batched prediction + active-learning query strategy
│   ├── fixed_test_set.py     # fixed evaluation set construction
│   ├── active_learning.py    # the active-learning driver (entry point)
│   ├── atomic_properties.py  # per-element property table + Vegard lattice parameters
│   └── utils.py              # model I/O, seeding, prediction helpers
│
├── KMC/                # Stage 2 — kinetic Monte Carlo engine
│   ├── config.py             # KMCConfig (JSON-serialisable run specification)
│   ├── state.py              # lattice state, vacancy bookkeeping
│   ├── engine.py             # BKL event selection + time advance
│   ├── barrier_predictor.py  # <-- the GNN <-> KMC bridge (see below)
│   ├── observables.py        # MSD, Warren–Cowley short-range order
│   ├── analysis.py           # Arrhenius / tau_order fits, real-time rescaling
│   ├── runner.py             # single ensemble + temperature-sweep drivers
│   ├── main.py               # CLI entry point (python -m KMC.main config.json)
│   ├── demos/                # benchmarks and illustrative scripts
│   └── tests/                # phase-by-phase KMC test suite
│
├── diffusion/          # Stage 3 — real-time rescaling quantities
│   ├── diffusion_oracle.py   # uMLIP relaxation, elastic constants, E_f^V, NEB
│   ├── diffusion_physics.py  # D(T), Debye frequency, jump distance (pure formulae)
│   ├── diffusion_calc.py     # composition-space sweep driver
│   ├── results.py            # DiffusionResult dataclass
│   └── config.py             # DiffusionConfig
│
├── tests/              # cross-package import + bridge smoke tests
├── examples/           # ready-to-edit KMC run configurations
├── AGENTS.md           # orientation file for AI assistants (task cookbook + gotchas)
├── pyproject.toml      # installable package definition (gnn, KMC, diffusion)
├── environment.yml     # curated conda environment
└── environment.lock.yml# fully pinned export for bit-for-bit reproduction
```

### How KMC and the GNN are wired together

`KMC/barrier_predictor.py` is the single integration point. `GNNBarrierPredictor`
loads the trained checkpoint and reuses the **same** `GraphBuilder`,
`atomic_properties` table and graph hyperparameters from the `gnn` package
that were used at training time, so the features the model sees in the KMC hot
loop are identical to those it was trained on. The KMC engine itself only
depends on the `BarrierPredictor` protocol (eight site indices in, eight
barriers in eV out), so the mock and GNN backends are interchangeable. For
performance the GNN backend caches the static BCC lattice graph once per run
and derives the nine per-step subgraphs by edge masking instead of rebuilding
neighbour lists every step.

---

## The GNN model

The surrogate is an ALIGNN-style graph network mapping a (initial, final)
structure pair to the **backward** migration barrier (the training target for
vacancy diffusion):

- **Inputs per atom:** learned element embedding + physical properties
  (atomic number, mass, radius, electronegativity, ionization energy,
  electron affinity, melting point, density).
- **Edges:** distances expanded in a radial basis (64 Gaussians, cutoff
  3.5 Å); a line graph carries bond-angle information.
- **Encoder:** 3 message-passing layers, hidden dimension 128.
- **Head:** MLP `[1024, 512, 256, 128]` over `[z_initial, z_final,
  z_final − z_initial]`.

Training uses a **mixed-fidelity** scheme: structures are retained at several
relaxation stages (an unrelaxed → relaxed trajectory), all sharing the same
converged NEB barrier label, with a progress-weighted Huber loss. This lets
the model predict the relaxed barrier directly from **unrelaxed** inputs, so
no relaxation is needed inside the KMC loop. Architecture defaults live in
`gnn/config.py`.

---

## Installation

```bash
# 1) Create the environment (PyTorch / PyG build must match your CUDA)
conda env create -f environment.yml
conda activate gnn-kmc

# 2) Register the packages (gnn, KMC, diffusion) as importable
pip install -e .

# 3) (optional) UMA oracle + elastic constants, only needed to GENERATE data
pip install fairchem-core matscipy
```

`pip install -e .` is what makes `import gnn`, `import KMC` and
`import diffusion` resolve from anywhere — it is required before running the
tests or the entry points below.

---

## Usage

### Run KMC with a trained model

```bash
python -m KMC.main examples/kmc_single_temperature.json   # single temperature
python -m KMC.main examples/kmc_temperature_sweep.json    # Arrhenius sweep + ensemble
```

Edit the JSON to point `gnn_model_path` at your checkpoint and set the
composition, supercell size, temperature(s) and number of realisations. The
config schema is `KMCConfig` in `KMC/config.py`.

### Train the GNN (active learning)

```bash
python -m gnn.active_learning
```

Set the element system, dataset paths and hyperparameters in `gnn/config.py`
first. **Note:** `gnn/config.py` ships with the cluster paths used for the
paper (`base_database_dir`, `base_production_dir`); change these to your own
locations before running.

### Generate real-time rescaling data

```bash
python -m diffusion.diffusion_calc --step 0.10 --moving-atoms W Mo
```

Produces `E_f^V`, `nu_0` and `a` per composition for the KMC real-time
rescaling (`KMC/analysis.py`).

---

## Tests

```bash
pip install -e ".[dev]"
pytest                       # cross-package smoke tests + KMC suite
pytest tests/                # just the refactor import/bridge smoke tests
```

The `tests/` suite verifies that every package imports cleanly (guarding the
refactor) and that the KMC ↔ GNN bridge satisfies its protocol. `KMC/tests/`
contains the detailed phase-by-phase engine tests.

---

## Reproducibility note

`environment.yml` is a curated, human-readable specification. The exact,
fully pinned environment used to produce the paper results (including the
specific PyTorch / CUDA build) is preserved in `environment.lock.yml`.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the accompanying paper (citation to be added
on publication).
