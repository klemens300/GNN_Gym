# Reproducing the paper

All commands run from the repository root. Outputs go to `results/` (small
tables) and `runs/` (simulation trajectories); both are created by the
commands and are not part of the repository.

## External data

| what | where | needed for |
|---|---|---|
| KMC trajectories, Metropolis traces | archive, DOI to be added; unpack into `runs/` | analysis without rerunning the simulations |
| DFT reference (235 jumps: structures, NEB barriers) and the uMLIP barriers on them | archive, DOI to be added; unpack into `data/dft/` | uMLIP/DFT comparison |
| uMLIP checkpoint `uma-s-1p1` | Hugging Face, downloaded by fairchem | everything in `gnn_gym.oracle` and `validation.{ground_state,dft,ordered_neb}` |

## 1. Reference data (uMLIP)

```bash
python -m gnn_gym.oracle.calibrate  configs/oracle/calibrate.yaml        # lattice parameters
python -m gnn_gym.oracle.jumps      configs/oracle/jumps_MoNbTaW.yaml --stream train
python -m gnn_gym.oracle.jumps      configs/oracle/jumps_MoNbTaW.yaml --stream referee
python -m gnn_gym.oracle.check_jumps data/jumps/*.csv
python -m gnn_gym.oracle.window_study configs/oracle/window_study.yaml run   # then: analyse
python -m gnn_gym.oracle.neb_settings configs/oracle/neb_settings.yaml run   # then: analyse
python -m gnn_gym.oracle.formation  configs/oracle/formation.yaml        # E_f spectra -> c_v
python -m gnn_gym.oracle.elastic    configs/oracle/elastic_MoNbTaW.yaml --replay data/nu0/elastic_MoNbTaW.csv
python -m gnn_gym.realtime.nu0      configs/realtime.yaml --build        # Debye nu0 grid and fit
```

The jump datasets used in the paper are included (`data/jumps/`); regenerating
them takes about 30 s per jump on one GPU.

## 2. Models

```bash
python -m gnn_gym.training.train configs/training/de_L2.yaml        # site-energy model
python -m gnn_gym.training.train configs/training/bsym_site.yaml    # barrier model
python -m gnn_gym.training.train configs/training/bsym_pooled.yaml  # pooled read-out (comparison)
python -m gnn_gym.training.learning_curve configs/training/de_L3.yaml \
    --limits 1000 2500 5000 10000 --out results/model/learning_curve_de.csv
python -m gnn_gym.training.evaluate --de models/MoNbTaW/de_L2_s0.pt \
    --bsym models/MoNbTaW/bsym_site_s0.pt --predictions results/model/referee_predictions.csv
```

The trained models of the paper are in `models/MoNbTaW/`. Retraining writes to
`runs/models/`; on the same GPU and software versions it reproduces the
training curves, elsewhere it reproduces them statistically.

## 3. Metropolis Monte Carlo

```bash
python -m gnn_gym.metropolis.run configs/metropolis/tord_MoNbTaW.yaml   # cooling, then heating chains
python -m gnn_gym.metropolis.run configs/metropolis/tord_TaW.yaml
python -m gnn_gym.metropolis.run configs/metropolis/two_start.yaml
python -m gnn_gym.metropolis.analysis configs/metropolis/analysis.yaml  # order, C_v, T_ord, alpha_eq, low-T maps, EXAFS
python -m gnn_gym.validation.ground_state configs/validation/ground_state_MoNbTaW.yaml   # uMLIP
python -m gnn_gym.validation.ground_state configs/validation/ground_state_TaW.yaml
python -m gnn_gym.validation.ordering configs/validation/ordering.yaml  # ordering energies, model vs uMLIP
```

The paper's chains use the batched walk (`walk: until_conflict`). The exact
single-exchange walk (`walk: until_accept`) agrees within statistical error
near the ordering temperatures and costs 15-30 % more time.

## 4. Kinetic Monte Carlo

```bash
python -m gnn_gym.kmc.campaign configs/kmc/escape_local.yaml   # Mo-Nb-Ta-W, 700-1500 K
python -m gnn_gym.kmc.campaign configs/kmc/escape.yaml         # Ta-W, 500-1500 K
python -m gnn_gym.kmc.campaign configs/kmc/escape_s1.yaml      # second and third realisation
python -m gnn_gym.kmc.campaign configs/kmc/escape_s2.yaml
python -m gnn_gym.kmc.campaign configs/kmc/pure_s0.yaml        # Mo, Nb, Ta, W (also pure_s1, pure_s2)
python -m gnn_gym.kmc.campaign configs/kmc/no_escape.yaml      # comparison without escape layer
```

Every point carries the seed it was run with. Rerunning the same command
continues unfinished points. A trajectory is reproduced exactly only on the
same GPU model (single-precision barriers differ in the last digits between
GPUs); the paper's runs were also interrupted and resumed with a detector
state that the code now saves, so reruns reproduce them statistically.

## 5. Analysis

```bash
python -m gnn_gym.analysis.analyze_campaign configs/analysis/campaign_alloys.yaml
python -m gnn_gym.analysis.analyze_campaign configs/analysis/campaign_pure.yaml
python -m gnn_gym.analysis.analyze_campaign configs/analysis/campaign_no_escape.yaml
python -m gnn_gym.analysis.pure_elements    configs/analysis/pure_elements.yaml
python -m gnn_gym.analysis.escape_vs_plain  configs/analysis/escape_vs_plain.yaml
python -m gnn_gym.analysis.order_kinetics   configs/analysis/order_kinetics.yaml
python -m gnn_gym.analysis.diffusion_summary configs/analysis/diffusion_summary.yaml
python -m gnn_gym.analysis.run_details      configs/analysis/run_details.yaml
python -m gnn_gym.analysis.check_trapping   runs/no_escape/kmc_* runs/escape*/kmc_* --csv results/kmc/trapping.csv
python -m gnn_gym.validation.dft            configs/validation/dft.yaml umlip   # then: model, stats
python -m gnn_gym.validation.ordered_neb    configs/validation/ordered_neb.yaml
python -m gnn_gym.validation.speed          configs/validation/speed.yaml
```

`analysis/order_kinetics` needs `results/metropolis/alpha_eq_kmc.csv` from
step 3.
