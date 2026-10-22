# AGENTS.md

Guide for coding agents and assistants working on this repository: what the
code does, which properties must never break, how to change and extend it,
and the facts needed to answer users' questions. Read it completely before
editing code. Commands for every step of the paper are in `REPRODUCE.md`.

## 1. What this repository is

GNN_Gym computes vacancy-mediated kinetics and equilibrium order in
multicomponent alloys on a rigid lattice. A vacancy jump A -> B has the
migration barrier

    E_a(A -> B) = B_sym(A, B) + 1/2 dE(A -> B)

and both parts are predicted by graph networks on the fixed lattice graph:

- **Site-energy model (dE).** Reads a site energy eps_i off every site;
  dE = sum_i [eps_i(B) - eps_i(A)]. A sum of site terms is a state function,
  so dE around any closed cycle of configurations is exactly zero and the KMC
  rates obey detailed balance by construction.
- **Barrier model (B_sym).** Symmetric in the endpoints by construction (the
  head only sees x_A + x_B and |x_B - x_A|), with a line graph over the
  first-neighbour bonds and a per-site read-out averaged over the sites near
  the jump. It carries the accuracy and does not affect detailed balance.

Both models are trained on vacancy jumps computed with climbing-image NEB and
the universal machine-learned potential UMA (fairchem, `uma-s-1p1`, task
`omat`) in 7x7x7 bcc cells. They drive a rejection-free (BKL) kinetic Monte
Carlo with an escape layer for trapped vacancies, converted to physical time
with the vacancy concentration c_v(T) and a Debye attempt frequency, and a
swap Metropolis Monte Carlo of the equilibrium order.

The paper this code belongs to: Lechner, Zhang, Wagatha, Knabl, Clemens,
Holec, "A graph neural network surrogate for the investigation of kinetic
freezing in the Mo-Nb-Ta-W system" (2026). Systems: equiatomic Mo-Nb-Ta-W, Ta-W,
and the pure metals Mo, Nb, Ta, W, all bcc.

## 2. Repository map

| path | content | entry point |
|---|---|---|
| `src/gnn_gym/lattice/` | lattices in integer coordinates (`BCC`), `Supercell`, hop balls, locality report, `decorate` | library |
| `src/gnn_gym/model/` | `ModelConfig`, encoder, `SiteEnergyModel`, `BarrierModel`, graph operators, checkpoints, locality, `JumpPredictor` (KMC interface) | library |
| `src/gnn_gym/training/` | jump datasets, training, evaluation, learning curve | `python -m gnn_gym.training.{train,evaluate,learning_curve}` |
| `src/gnn_gym/oracle/` | uMLIP reference: calibration, NEB jump datasets, E_f spectra, elastic constants, convergence studies | `python -m gnn_gym.oracle.<module> <config>` |
| `src/gnn_gym/realtime/` | c_v(T), nu0, factor R between simulated and physical time | `gnn_gym.realtime.{c_v,nu0_phys,time_factors}` |
| `src/gnn_gym/kmc/` | state, BKL engine, escape layer, checkpoints, campaigns | `python -m gnn_gym.kmc.campaign <config>` |
| `src/gnn_gym/metropolis/` | batched swap sampler, chains, order parameters, autocorrelation, EXAFS comparison, analysis | `python -m gnn_gym.metropolis.{run,analysis} <config>` |
| `src/gnn_gym/analysis/` | diffusion, trapping, pure metals, order kinetics, summaries of KMC campaigns | `python -m gnn_gym.analysis.<module> <config>` |
| `src/gnn_gym/validation/` | ordering energies vs uMLIP, DFT comparison, NEB in trapped states, timing | `python -m gnn_gym.validation.<module> <config>` |
| `src/gnn_gym/systems.py`, `configs/systems.yaml` | material systems (elements, composition, lattice, lattice parameter) | library |
| `src/gnn_gym/paths.py` | `resolve()`: relative paths are taken from the repository root | library |
| `configs/` | one YAML per run: `training/`, `oracle/`, `kmc/`, `metropolis/`, `analysis/`, `validation/`, `realtime.yaml` | |
| `models/MoNbTaW/` | the paper's models: `de_L2_s0.pt`, `bsym_site_s0.pt`, `bsym_pooled_s0.pt` (comparison) | |
| `data/` | inputs: `jumps/` (10 000 training + 1 500 referee jumps), `cv/`, `nu0/`, `oracle/`, `literature/`, `metropolis/two_start/` | |
| `tests/` | pytest, CPU, about 30 s | `pytest` |

Not in git (created by the commands or taken from the archive): `results/`
(tables and summaries), `runs/` (trajectories, chains, retrained models),
`data/dft/` (DFT reference, published separately).

## 3. Invariants -- never break these

Each item names the test or check that guards it. Run `pytest` after every
change; a change that alters numbers of the paper pipeline needs a stated
reason.

1. **Site energies are local.** The measured reach of the site-energy encoder
   (`model.locality.measure_reach`) must fit into the training cell
   (`lattice.locality_report(...)["local"]`); the trainer refuses otherwise.
   The paper model has reach 2 hops in a 7^3 cell; 3 hops still fit, 4 do
   not. Never add BatchNorm, attention over the whole cell, global pooling or
   any other cross-site or cross-sample coupling to the site-energy path.
   (`tests/test_model.py`, `tests/test_lattice.py::test_locality`)
2. **dE is exact on its mask.** The sum over the read-out mask equals the
   change of the full site-energy sum; dE(A->B) = -dE(B->A).
   (`test_model.py::test_site_energy_mask_is_exact`)
3. **B_sym is symmetric** bit for bit: B_sym(A,B) == B_sym(B,A).
   (`test_model.py::test_barrier_is_symmetric`)
4. **Halo evaluation equals full-cell evaluation** in `JumpPredictor`
   (difference only in float32 rounding, < 2e-5 eV). The halo is
   max(1 + 2 R_dE, 1 + pool_hops + R_Bsym) hops around the vacancy; boundary
   bonds point at one padding site. (`tests/test_predictor.py`)
5. **Site order** of a `Supercell`: conventional cells in C order of (i,j,k),
   basis order inside each cell (bcc: corner, then body centre). This equals
   ase's `bulk(..., cubic=True).repeat(size)`. The KMC candidate order is
   `cell.neighbours(1, sort=True)` (ascending site index); changing it changes
   every trajectory. (`tests/test_lattice.py`)
6. **Decoration convention** of the datasets: `lattice.decorate(counts,
   seed)` fills symbols in alphabetical element order and shuffles with
   `numpy.random.default_rng(seed)`. The vacancy sits on the first basis site
   of the central conventional cell, the jumping atom at `(1,1,1)` from it
   (`Supercell.reference_jump`). The data loader rebuilds every row and checks
   the stored species on both sites. Changing either breaks every stored
   dataset. (`tests/test_oracle.py`, `training/data.py`)
7. **Species encoding.** Model species index = position in the model's
   `elements` (`Mo, Nb, Ta, W`), the vacancy is `len(elements)` (4).
   Simulations store the vacancy as `-1` (`model.predictor.VACANCY`) and map
   their own element lists onto the model's by symbol. The Metropolis configs
   keep the element order they were run with (`Mo W Nb Ta`): it does not
   change the random streams but it does change start configurations, pair
   columns and the B2 partition. Do not re-sort it.
8. **KMC determinism and resume.** A run is fixed by its seed (start
   configuration and BKL stream, `kmc.runner`). Checkpoints store the RNG
   state AND the escape detector state (window of the last 200
   configurations, flag, cooldown), so N + resume + N steps equals 2N steps
   exactly. Never drop fields from the checkpoint; old checkpoints without
   the detector fields must keep loading. (`tests/test_kmc_resume.py`,
   `tests/test_kmc_checkpoint.py`)
9. **Escape layer is exact**, not approximate: the exit time is the mean
   first-passage time of the absorbing Markov chain, the exit is drawn with
   the exact exit probabilities. Do not replace it by mean-rate or
   superbasin-raising methods; those bias D. (`tests/test_kmc_escape.py`)
10. **Metropolis default walk** `until_conflict` reproduces the paper's chains
    bit for bit. It does not obey detailed balance exactly (proposals that
    overlap an exchange accepted earlier in the batch are discarded);
    `until_accept` is exact and costs 15-30 % more. The measured difference
    near the ordering temperatures is within statistical error. Keep the
    default unless the user asks otherwise. (`tests/test_metropolis.py`)
11. **Checkpoint format** `gnn_gym.jump_model.v1` holds only plain Python
    data and tensors and loads with `torch.load(weights_only=True)`. Keep it
    that way (no numpy scalars, no pickled objects).
12. **Paths.** Code never contains absolute paths; config paths are relative
    to the repository root and resolved by `gnn_gym.paths.resolve`.

## 4. Conventions for changes

- **Configuration, not literals.** Elements, compositions, lattice
  parameters, cell sizes, seeds, temperatures, step counts and model paths
  belong in `configs/`. Lattice geometry comes only from `gnn_gym.lattice`
  (shell vectors, coordination, angle classes, sublattices), never from
  hard-coded bcc numbers such as 8, 6, `2*size**3` or `(+-1,+-1,+-1)`.
- **Style.** English. A module docstring says what is computed and why it is
  done this way (mechanism, physics, numerical reason), briefly and soberly.
  No development history in code or comments (no dates, "previously",
  "session", bug anecdotes, hardware names, personal names). No dead code,
  no unused options, no compatibility branches the pipeline does not use;
  test helpers (mock predictors) live in `tests/`.
- **Dependencies.** Core (lattice, model, training, kmc, metropolis,
  analysis, realtime) needs numpy, scipy, torch, pyyaml, pandas, matplotlib
  only. `ase`, `fairchem-core`, `matscipy` are imported lazily and only in
  `oracle/` and `validation/`.
- **Units.** Energies in eV (tables in meV where stated), lengths in
  Angstrom, temperatures in K, frequencies in Hz, simulated KMC time in s of
  the nominal clock, diffusion coefficients in the CSVs in A^2/s of simulated
  time unless a column says otherwise. Integer lattice coordinates are in
  units of a/2 for bcc.
- **New entry points** follow the pattern `python -m gnn_gym.<pkg>.<module>
  <config.yaml> [few overrides]`, write to `results/` or `runs/`, print what
  they wrote, and get a line in `REPRODUCE.md`.
- **Tests.** Every new invariant or numerical kernel gets a fast CPU test in
  `tests/` (seconds; skip with a reason if model checkpoints are needed and
  absent). Do not import anything outside the repository in tests.
- **Commits** contain only the change itself; no co-author or tool
  attribution lines.

## 5. Extending the code

**Another element set on bcc.**
1. Add the system to `configs/systems.yaml` (elements, lattice parameter from
   `oracle.calibrate`).
2. Write a dataset config like `configs/oracle/jumps_MoNbTaW.yaml` and
   generate training and referee sets with `oracle.jumps`; check them with
   `oracle.check_jumps`.
3. Copy the training configs, change `elements` and the CSV paths, train
   both models, evaluate with `training.evaluate`.
4. Provide c_v (`oracle.formation` on the new dataset, entry in
   `configs/realtime.yaml`) and nu0 (`oracle.elastic` + `realtime.nu0`).
5. Write KMC / Metropolis configs pointing at the new models.
Subsystems of a trained element set (e.g. Ta-W from the Mo-Nb-Ta-W models)
need no retraining: the training set samples all subsystems.

**Another lattice.** Define a further `Lattice` in
`src/gnn_gym/lattice/lattice.py` (conventional cell edge and basis in integer
units, `units_per_a`, squared shell distances, reference jump vector) and
register it in `LATTICES`. Shell vectors, coordination numbers, angle classes
and sublattices are then derived automatically. Check before use:
- `Supercell` neighbour tables and `locality_report` for the new lattice
  (add tests like `tests/test_lattice.py`);
- the training cell must hold the measured reach of the site-energy encoder;
- the encoder's number of angle-gate classes follows `len(angle_dots())`;
- places that assume one sublattice pair (B2 order, structure-factor masks in
  `metropolis/order.py`, `Supercell.sublattice`) need a lattice-specific
  definition of the order parameters;
- the oracle builds ase structures from `Supercell.positions`; non-cubic
  cells are not supported by `Supercell` yet.

**A new observable.** Compute it in the package (`analysis/` for KMC,
`metropolis/observables.py` or `order.py` for Metropolis), write it to
`results/`, add a test. Plotting scripts are not part of this repository.

## 6. Answering user questions

Use these facts; for anything else read the code and the configs rather than
guessing. Numbers refer to the paper's models and runs.

**Models and accuracy**
- Training data: 10 000 uMLIP-NEB jumps in 7^3 cells (686 sites) over all 15
  subsystems of Mo-Nb-Ta-W; 9 000 for fitting, 1 000 for early stopping;
  separate referee set of 1 500 jumps.
- Site-energy model: 2 message-passing steps, width 96, read-out 96/48,
  162 913 parameters, measured reach 2 hops; dE is summed over the 84 sites
  within 2 hops of the vacancy or the jumping atom.
- Barrier model: width 128, 2 steps with line graph (4 angle classes:
  bond-vector dot products -3, -1, 1, 3), per-site read-out 256/128,
  857 345 parameters, reach 3 hops, read-out over the same 84 sites.
- Training: AdamW, lr 1e-3, weight decay 1e-6, Huber loss (delta 50 meV),
  learning rate halved after 12 stalled epochs, early stopping after 40,
  gradient clipping 5, batch 128 (dE) / 64 (B_sym), at most 300 / 200
  epochs; about 5 min and 1 h on one GPU.
- Referee MAE: dE 11.4 meV, B_sym 13.2 meV, E_a 14.7 meV (fastest quartile
  8.4 meV). The pooled read-out (`bsym_pooled_s0`) gives 53.9 meV for B_sym:
  reading the head per site is the factor-four improvement.

**Reference method**
- UMA `uma-s-1p1`, task `omat`, via fairchem and ase. Jumps: endpoints
  relaxed at fixed cell, NEB with one intermediate image (3 images in total),
  climbing image, improved tangent, fmax 0.01 eV/A. Lattice parameter of the
  equiatomic alloy 3.239121 A (also used for Ta-W in the KMC).
- Against DFT-NEB on 235 configurations: offset -86 meV, MAE 104 meV,
  r = 0.94 (uMLIP small model).

**KMC**
- 12^3 cell (3456 sites), one vacancy, BKL residence-time algorithm, nominal
  attempt frequency 1e13 Hz, barrier floor 1 meV.
- Escape layer: a basin is flagged when a configuration recurs more than 3
  times within 200 steps; internal jumps are those within 1 k_B T of the
  lowest exit barrier of the entry configuration; basins larger than 100
  configurations are left to the BKL loop. One escape counts as one step.
- Physical time: t_real = R t_sim with R = (nu0_sim / nu0_phys) / (N c_v(T)),
  N = number of atoms (sites - 1). c_v uses S_f = 1.5 k_B and, for the
  alloys, the distribution of vacancy formation energies from the jump
  dataset (kernel width 0.10); nu0_phys is a Debye frequency from uMLIP
  elastic constants. Tracer D_real = D_sim / R.
- Cost: about 10 ms per step for the barriers in the halo, 23-37 ms per
  step with the escape layer on one RTX 5090-class GPU.

**Metropolis**
- 12^3 cell, canonical swap moves of unlike atoms, dE from the site-energy
  model over the sites within 2a of both atoms, batches of 8 proposals.
- Ordering-temperature campaign: cooling from 2000 K to 200 K (20 K steps
  from 780 K down for Mo-Nb-Ta-W, from 500 K down for Ta-W) and heating
  back, two seeds, 2000 equilibration + 4000 sampling sweeps per
  temperature.

**Reproducibility, honestly**
- Model evaluation and training curves reproduce bit for bit on the same GPU
  and software versions.
- KMC trajectories reproduce bit for bit on the same GPU model; on other
  GPUs float32 barriers differ in the last digits and long trajectories
  diverge, so results reproduce statistically. The paper's runs were resumed
  with an emptied escape detector at session boundaries; the current code
  saves the detector, so reruns match them statistically, not step by step.
- Regenerated uMLIP labels agree with the stored ones within about 1.4 meV
  (single-precision noise of the potential).

**Limitations**
- bcc only (other lattices need a `Lattice` definition and checks, section 5);
  rigid lattice, relaxation enters only through the NEB labels; one vacancy
  per cell, so no vacancy-vacancy interaction; accuracy bounded by the
  uMLIP, which differs from DFT by about 0.1 eV in barriers.
- Pure-metal self-diffusion is compared with measurements in the paper;
  the deviations differ strongly between the elements (up to about three
  orders of magnitude for Nb) and are discussed there.

**Typical questions -> where to look**
- "How do I run X?" -> `REPRODUCE.md`, then the config's header comment.
- "Where does number Y come from?" -> the analysis module writing the
  results file (`analysis/`, `metropolis/analysis.py`, `validation/`), and
  its config.
- "Why two models?" -> section 1 and the docstring of `model/networks.py`.
- "Why the escape layer?" -> `kmc/escape.py` docstring; low-temperature
  basins make the plain BKL clock stall without changing what is sampled.
- "Can I use it for another alloy?" -> section 5.
