# AGENTS.md — orientation for AI assistants

This file helps an AI assistant (or any automated agent) understand this
repository quickly and give a user correct, actionable guidance. It is a
task-oriented companion to `README.md`. If you are an assistant reading this:
start here, then consult the named source files for specifics.

---

## What this repository is

A two-part scientific code for studying vacancy diffusion and chemical
ordering in the **MoNbTaW** body-centred-cubic (BCC) refractory high-entropy
alloy:

1. A **graph neural network (GNN)** that predicts a vacancy's migration
   barrier from its local atomic environment (replacing an expensive
   uMLIP-NEB calculation with a single forward pass).
2. A **kinetic Monte Carlo (KMC)** engine that calls the GNN at every step to
   evolve the vacancy and measure diffusion / short-range order.

Mental model — three stages:

- **Build** (`gnn/`): a uMLIP oracle (UMA, `uma-s-1p1`) computes NEB barriers;
  active learning grows the dataset; the GNN is trained on it.
- **Deploy** (`KMC/`): the trained GNN evaluates the eight nearest-neighbour
  barriers around the vacancy at every BKL step.
- **Rescale** (`diffusion/`): extra uMLIP supercell runs give the vacancy
  formation energy `E_f^V`, attempt frequency `nu_0` and jump distance `a`,
  which convert simulated KMC time into physical time.

---

## Repo map (where to look)

| Path | Role | Key files |
|------|------|-----------|
| `gnn/` | GNN surrogate training + active learning | `config.py`, `model.py`, `graph_builder.py`, `oracle.py`, `trainer.py`, `active_learning.py` |
| `KMC/` | KMC engine + analysis | `config.py`, `engine.py`, `state.py`, `barrier_predictor.py`, `runner.py`, `main.py`, `analysis.py` |
| `diffusion/` | Real-time rescaling quantities | `diffusion_oracle.py`, `diffusion_physics.py`, `diffusion_calc.py` |
| `tests/` | Cross-package import + bridge smoke tests | `test_imports.py`, `test_kmc_gnn_bridge.py` |
| `KMC/tests/` | Detailed KMC engine test suite | `test_phase*.py` |
| `examples/` | Ready-to-edit KMC run configs | `kmc_single_temperature.json`, `kmc_temperature_sweep.json` |

The single integration point between the two halves is
`KMC/barrier_predictor.py::GNNBarrierPredictor`.

---

## Setup (do this first)

```bash
conda env create -f environment.yml
conda activate gnn-kmc
pip install -e .            # registers gnn, KMC, diffusion as importable packages
# optional, only to GENERATE training/rescaling data:
pip install fairchem-core matscipy
```

`pip install -e .` is mandatory before running anything — it is what makes
`import gnn`, `import KMC`, `import diffusion` resolve.

---

## Task cookbook (map a user request to an action)

**"Run a KMC simulation."**
Edit a copy of `examples/kmc_single_temperature.json` (set `gnn_model_path`,
`composition`, `supercell_size`, `temperature_K`, `n_steps`, `output_dir`),
then:
```bash
python -m KMC.main path/to/your_config.json
```
For an Arrhenius temperature sweep with error bars use
`examples/kmc_temperature_sweep.json` (`temperatures_K_sweep` +
`n_realizations_per_T`). The config schema is `KMCConfig` in `KMC/config.py`.

**"Train the GNN / run active learning."**
Set elements, dataset paths and hyperparameters in `gnn/config.py`, then:
```bash
python -m gnn.active_learning
```
Single from-scratch training (no AL loop) is controlled by `train_only_mode`
in `gnn/config.py`.

**"Generate the real-time rescaling data (E_f^V, nu_0, a)."**
```bash
python -m diffusion.diffusion_calc --step 0.10 --moving-atoms W Mo
```
Requires `fairchem-core` and `matscipy`. Output feeds `KMC/analysis.py`
(`DiffusionLookup`, cached in a `diffusion_cache.json`).

**"Use a different alloy system (different elements)."**
Change `elements` in `gnn/config.py`; paths auto-generate from the sorted
element list. Add any missing elements to `gnn/atomic_properties.py` (property
row + BCC lattice parameter for Vegard's law). Retrain. Then set the matching
`elements` / `composition` in the KMC config. See the ordering gotcha below.

**"Run the tests."**
```bash
pytest tests/ -v        # fast: import + bridge smoke tests (guards refactors)
pytest KMC/tests/ -v    # full KMC engine suite
```

---

## Critical gotchas (get these wrong and results are silently incorrect)

1. **Element-index ordering must match between training and KMC.**
   The GNN maps species to integer indices by the position in
   `config.elements` (default `['Mo', 'Nb', 'Ta', 'W']` → Mo=0, Nb=1, Ta=2,
   W=3). The KMC state assigns species the same way from its own
   `config.elements`. If the two lists differ in order, the model reads the
   wrong element at each site and predictions become meaningless. When running
   a sub-system (e.g. only Ta–W), keep the full `['Mo','Nb','Ta','W']` order
   and set the absent elements to zero in `composition` rather than reordering.

2. **The training target is the BACKWARD barrier.**
   The GNN predicts `E_TS − E_post_jump` (see `gnn/dataset.py`). Do not
   reinterpret outputs as forward barriers.

3. **The KMC call order is deliberately swapped.**
   To get the forward barrier of the current state A hopping into candidate B,
   `barrier_predictor.py` calls `model(initial=B, final=A)`. This is intended;
   see the `GNNBarrierPredictor` docstring.

4. **Feature construction must stay identical train ↔ inference.**
   `GNNBarrierPredictor` reuses the `gnn` package's `GraphBuilder`,
   `atomic_properties` table and graph hyperparameters (cutoff 3.5 Å, 64 RBF
   Gaussians, line graph). Changing graph settings only on one side breaks the
   model. Change them in `gnn/config.py` and retrain.

5. **Default paths are placeholders.**
   `gnn/config.py` (`base_database_dir`, `base_production_dir`) and
   `KMC/config.py` (`gnn_model_path`, `diffusion_cache_path`) ship with
   placeholder/relative paths. Point them at real locations before running.

6. **Real-time rescaling is optional and separate.**
   KMC dynamics do not need `E_f^V`/`nu_0`/`a`; those only convert simulated
   time to physical time in post-processing (`KMC/analysis.py`). Without a
   populated `diffusion_cache.json`, runs proceed with "no rescaling".

---

## Model architecture (for answering "how does the GNN work?")

ALIGNN-style network mapping an (initial, final) structure pair to the
backward barrier: learned element embeddings + 8 atomic properties per node;
RBF-expanded edges (64 Gaussians, 3.5 Å cutoff) plus a line graph for bond
angles; 3 message-passing layers, hidden dim 128; MLP head
`[1024, 512, 256, 128]` over `[z_init, z_final, z_final − z_init]`. Trained
with a mixed-fidelity, progress-weighted Huber loss so the relaxed barrier is
predicted from unrelaxed inputs (no relaxation needed inside the KMC loop).
Defaults: `gnn/config.py`. Oracle: UMA `uma-s-1p1` (not CHGNet).

---

## Conventions

- Keep code comments in English.
- The three packages are independent installs sharing one repo; only
  `KMC/barrier_predictor.py` (GNN) and `KMC/analysis.py` (diffusion) cross
  package boundaries — both via plain `from gnn...` / `from diffusion...`
  imports.
- After any refactor, run `pytest tests/` — its import sweep catches a broken
  import immediately.
