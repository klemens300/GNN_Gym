# Active Learning for Diffusion Barrier Prediction

Fast and accurate prediction of vacancy diffusion barriers in high-entropy alloys using Graph Neural Networks (GNNs) and Active Learning.

> **Purpose:** This repository provides a complete Active Learning workflow to accelerate kinetic Monte Carlo (KMC) simulations by efficiently predicting diffusion barriers using a GNN surrogate model trained on CHGNet NEB calculations.

---

## What Does This Do?

**Problem:** Kinetic Monte Carlo simulations require thousands of diffusion barrier calculations. Traditional DFT or NEB calculations are too slow for high-throughput screening.

**Solution:** Train a Graph Neural Network to predict diffusion barriers orders of magnitude faster than CHGNet, while maintaining high accuracy through Active Learning.

### Key Concept: Active Learning Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Active Learning Loop                      │
└─────────────────────────────────────────────────────────────┘

1. Oracle (CHGNet NEB)
   └─> Calculate diffusion barriers for compositions
        (Slow but accurate)

2. Train GNN
   └─> Learn from Oracle data
        (Fast surrogate model)

3. Inference
   └─> GNN predicts barriers on test set
        (Identify prediction uncertainty)

4. Query Strategy
   └─> Select most uncertain/informative samples
        (Error-weighted sampling)

5. Back to Oracle
   └─> Calculate selected samples, add to training data
        (Iteratively improve model)

6. Convergence Check
   └─> Stop when MAE < threshold or max cycles reached
        (Automatically finds optimal dataset size)
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/active-learning-diffusion.git
cd active-learning-diffusion

# Install dependencies
pip install torch torch-geometric
pip install ase chgnet pymatgen
pip install pandas numpy matplotlib
pip install wandb  # Optional: for experiment tracking
```

### Basic Usage

1. **Configure your system** in `config.py`:
```python
# Material system
elements = ['Mo', 'Nb', 'Ta', 'W']  # Your alloy elements

# Active Learning parameters
al_initial_samples = 1000      # Initial random samples
al_n_query = 250               # Samples to add per cycle
al_max_cycles = 20             # Maximum AL cycles

# Convergence criteria
al_convergence_threshold_mae = 0.01  # Stop when MAE < 0.01 eV
```

2. **Run Active Learning**:
```bash
jupyter notebook UT9_active_learning.ipynb
# or
python UT9_active_learning.ipynb  # If using jupytext
```

3. **Monitor progress**:
- Logs: `logs/active_learning.log`
- Checkpoints: `checkpoints/cycle_*/best_model.pt`
- Results: `active_learning_results/`
- Weights & Biases: Dashboard (if enabled)

---

## Project Structure

```
.
├── config.py                    # Central configuration
├── UT9_active_learning.ipynb   # Main Active Learning script
│
├── oracle.py                    # CHGNet NEB calculator (Oracle)
├── model.py                     # GNN architecture
├── trainer.py                   # Training loop with early stopping
├── inference.py                 # Prediction and query strategy
├── dataset.py                   # Data loading and preprocessing
│
├── template_graph_builder.py   # Fast graph construction
├── atomic_properties.py         # Element feature database
├── utils.py                     # Helper functions
│
├── checkpoints/                 # Saved models per cycle
├── logs/                        # Training and AL logs
├── MoNbTaW/                     # Structure database (auto-generated)
└── MoNbTaW.csv                  # Diffusion barrier database
```

---

## How It Works

### 1. Oracle: CHGNet NEB Calculations

**Purpose:** Generate ground truth diffusion barriers.

**Process:**
- Create BCC supercell with composition (e.g., Mo₂₅Nb₂₅Ta₂₅W₂₅)
- Create vacancy at center
- Relax initial structure
- Select random neighbor atom
- Move atom to vacancy → final structure
- Run NEB to find minimum energy path
- Extract forward/backward barriers

**Output:** Structure files (.cif) and barriers stored in database.

**File:** `oracle.py`

### 2. GNN Architecture

**Purpose:** Learn structure-property relationship for fast prediction.

**Architecture:**
```
Graph Neural Network (GNN)
│
├─ Node Features:
│  ├─ Position (3D coordinates)
│  ├─ Element one-hot encoding
│  └─ Atomic properties (radius, mass, electronegativity, valence)
│
├─ Message Passing (5 layers):
│  └─ GraphConvLayer with edge features (distances)
│
├─ Graph Pooling:
│  └─ Global mean pooling
│
└─ Predictor (MLP):
   ├─ Input: [emb_initial, emb_final, delta_emb]
   └─ Output: Energy barrier (eV)
```

**Key Features:**
- Template-based graph construction (100x faster)
- Shared encoder for initial/final structures
- Difference embedding captures transition

**Files:** `model.py`, `template_graph_builder.py`

### 3. Active Learning Strategy

**Purpose:** Efficiently sample the most informative compositions.

**Query Strategy:** Error-weighted sampling
- Calculate prediction errors on test set
- Sample proportional to relative error
- High-error samples = high information gain

**Convergence Criteria:**
- MAE threshold (e.g., < 0.01 eV)
- Patience: No improvement for N cycles
- Maximum cycles reached

**Files:** `inference.py`, `UT9_active_learning.ipynb`

### 4. Training

**Features:**
- Early stopping with patience
- Learning rate scheduling (Cosine Warm Restarts)
- Gradient clipping
- Checkpoint management
- Weights & Biases integration

**Metrics:**
- Mean Absolute Error (MAE)
- Relative MAE
- Training/Validation loss

**File:** `trainer.py`

---

## Configuration Guide

### Key Parameters in `config.py`

#### Active Learning
```python
al_initial_samples = 1000        # Initial dataset size
al_n_test = 500                  # Test samples per cycle
al_n_query = 250                 # Query samples per cycle
al_max_cycles = 20               # Maximum AL cycles

# Convergence
al_convergence_check = True
al_convergence_metric = "mae"    # or "rel_mae"
al_convergence_threshold_mae = 0.01  # eV
al_convergence_patience = 5      # cycles
```

#### Model Architecture
```python
gnn_hidden_dim = 64              # GNN layer width
gnn_num_layers = 5               # Message passing depth
gnn_embedding_dim = 64           # Graph embedding size
mlp_hidden_dims = [1024, 512, 256]  # Predictor MLP
```

#### Training
```python
learning_rate = 5e-4
batch_size = 32
epochs = 10000
patience = 100                   # Early stopping

# Final model (after convergence)
final_model_patience = 666       # Higher patience for final training
```

#### NEB Calculation
```python
neb_images = 3                   # NEB path resolution
neb_fmax = 0.1                   # Force convergence (eV/Å)
neb_max_steps = 500              # Maximum steps
```

---

## Output Files

### Database
- **`MoNbTaW.csv`**: Central database with all calculated barriers
  - Columns: composition, barriers, energies, timings, paths

### Structures
- **`MoNbTaW/`**: Directory tree organized by composition and run
  ```
  MoNbTaW/
  ├── Mo25Nb25Ta25W25/
  │   ├── run_1/
  │   │   ├── initial_relaxed.cif
  │   │   ├── final_relaxed.cif
  │   │   ├── neb_image_*.cif
  │   │   └── results.json
  │   └── run_2/
  └── Mo30Nb20Ta20W30/
  ```

### Models
- **`checkpoints/cycle_*/best_model.pt`**: Model for each AL cycle
- **`checkpoints/final_model/best_model.pt`**: Final optimized model

### Results
- **`active_learning_results/`**:
  - `cycle_*_predictions.csv`: Predictions for each cycle
  - `convergence_history.json`: AL convergence metrics

### Logs
- **`logs/active_learning.log`**: Main AL loop
- **`logs/oracle.log`**: NEB calculations
- **`logs/inference_cycle_*.log`**: Per-cycle inference
- **`checkpoints/cycle_*/training.log`**: Training logs

---

## Example Workflow

### Scenario: Screen Mo-Nb-Ta-W system for diffusion barriers

```python
# 1. Configure in config.py
elements = ['Mo', 'Nb', 'Ta', 'W']
al_initial_samples = 1000
al_n_query = 250
al_convergence_threshold_mae = 0.01  # eV

# 2. Run Active Learning
# Execute: UT9_active_learning.ipynb

# 3. Monitor
# - Watch logs/active_learning.log for progress
# - Check convergence in active_learning_results/

# 4. Results
# - Cycle 0: 1000 samples, MAE = 0.15 eV
# - Cycle 1: 1250 samples, MAE = 0.08 eV
# - Cycle 2: 1500 samples, MAE = 0.04 eV
# - Cycle 3: 1750 samples, MAE = 0.02 eV
# - Cycle 4: 2000 samples, MAE = 0.009 eV ✓ CONVERGED

# 5. Use final model
model = load_model_for_inference('checkpoints/final_model/best_model.pt')
barrier = predict_single(model, initial_graph, final_graph)
```

---

## Advanced Usage

### Custom Element Systems

```python
# In config.py
elements = ['Ti', 'Zr', 'Hf', 'V', 'Nb']  # Your elements

# Add atomic properties in atomic_properties.py
ATOMIC_PROPERTIES = {
    'Ti': {
        'atomic_radius': 1.76,
        'atomic_mass': 47.867,
        'electronegativity': 1.54,
        'valence': 4
    },
    # ... add your elements
}
```

### Using Trained Models for Prediction

```python
from utils import load_model_for_inference, predict_single
from template_graph_builder import TemplateGraphBuilder
from config import Config

# Load model
config = Config()
model, checkpoint = load_model_for_inference(
    'checkpoints/final_model/best_model.pt',
    config
)

# Build graphs for prediction
builder = TemplateGraphBuilder(config)
initial_graph, final_graph = builder.build_pair_graph(
    'path/to/initial.cif',
    'path/to/final.cif',
    backward_barrier=0.0  # Dummy value for prediction
)

# Predict barrier
barrier = predict_single(model, initial_graph, final_graph, device='cuda')
print(f"Predicted barrier: {barrier:.3f} eV")
```

### Batch Predictions

```python
from utils import predict_batch

# Predict multiple barriers
barriers = predict_batch(
    model,
    initial_graphs,  # List of graphs
    final_graphs,    # List of graphs
    device='cuda',
    batch_size=64
)
```

---

## Monitoring with Weights & Biases

Enable experiment tracking:

```python
# In config.py
use_wandb = True
wandb_project = "diffusion-barriers"
wandb_entity = "your-username"  # Optional
```

**Tracked Metrics:**
- Training/Validation loss and MAE
- Learning rate schedule
- Patience counter
- Best metrics per cycle
- Dataset size growth

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# Reduce batch size in config.py
batch_size = 16  # or smaller
```

**2. Convergence too slow**
```python
# Increase query samples per cycle
al_n_query = 500  # More samples per cycle

# Or relax convergence threshold
al_convergence_threshold_mae = 0.02  # Less strict
```

**3. NEB calculations failing**
```python
# Increase NEB steps
neb_max_steps = 1000

# Or relax force convergence
neb_fmax = 0.2
```

**4. Model not improving**
```python
# Increase model capacity
gnn_num_layers = 7
mlp_hidden_dims = [2048, 1024, 512, 256]

# Or use different scheduler
scheduler_type = "cosine_warm_restarts"
scheduler_t_0 = 100
```

---

## File Format Reference

### Database CSV (`MoNbTaW.csv`)

```csv
composition_string,Mo,Nb,Ta,W,run_number,calculator,model,diffusing_element,forward_barrier_eV,backward_barrier_eV,E_initial_eV,E_final_eV,initial_relax_time_s,final_relax_time_s,neb_time_s,total_time_s,structure_folder,timestamp
Mo25Nb25Ta25W25,0.25,0.25,0.25,0.25,1,CHGNet,CHGNet-pretrained,W,1.234,0.987,-1234.56,-1233.57,12.3,11.8,45.2,69.3,MoNbTaW/Mo25Nb25Ta25W25/run_1,2024-01-01 12:00:00
```

### Model Checkpoint

```python
checkpoint = {
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'epoch': 150,
    'best_val_mae': 0.0234,
    'best_val_rel_mae': 0.0456,
    'history': {...},
    'config': {...},
    'cycle': 3
}
```

---

## Dependencies

### Required
- **PyTorch** (≥ 2.0): Neural network framework
- **PyTorch Geometric**: Graph neural network library
- **ASE**: Atomic structure manipulation
- **CHGNet**: Machine learning interatomic potential
- **pymatgen**: Materials analysis
- **pandas**: Data management
- **numpy**: Numerical computing

### Optional
- **wandb**: Experiment tracking
- **matplotlib**: Visualization
- **jupyter**: Interactive notebook

---

## Tips for Best Results

1. **Start small**: Begin with `al_initial_samples = 500-1000` to verify setup

2. **Monitor convergence**: Check `logs/active_learning.log` regularly

3. **Adjust query size**: Balance between speed and data efficiency
   - Small query (50-100): More cycles, finer control
   - Large query (500-1000): Fewer cycles, faster completion

4. **Use GPU**: Essential for reasonable training times
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

5. **Checkpoint often**: Models are saved automatically per cycle

6. **Validate final model**: Test on held-out compositions before KMC

---

## Code Overview

### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Configuration management | `Config` class |
| `oracle.py` | CHGNet NEB calculations | `Oracle.calculate()` |
| `model.py` | GNN architecture | `DiffusionBarrierModel` |
| `trainer.py` | Training loop | `Trainer.train()` |
| `inference.py` | Prediction & query | `run_inference_cycle()` |
| `dataset.py` | Data loading | `create_dataloaders()` |
| `template_graph_builder.py` | Graph construction | `TemplateGraphBuilder` |
| `utils.py` | Helper functions | `load_model_for_inference()` |

### Data Flow

```
CSV Database (MoNbTaW.csv)
    ↓
Dataset (dataset.py)
    ↓
DataLoader (batched graphs)
    ↓
GNN Model (model.py)
    ↓
Predictions
    ↓
Query Strategy (inference.py)
    ↓
Oracle (oracle.py)
    ↓
Back to CSV Database
```

---

## Contributing

This repository serves as supplementary information for a research paper. For questions or issues:

1. Check the troubleshooting section
2. Review logs for error messages
3. Verify configuration in `config.py`

---

## License

[Add your license here]

---

## Acknowledgments

**Tools:**
- [CHGNet](https://github.com/CederGroupHub/chgnet): Machine learning interatomic potential
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/): Graph neural networks
- [ASE](https://wiki.fysik.dtu.dk/ase/): Atomic Simulation Environment

---

## Contact

[Your contact information or link to paper]

---

**Last Updated:** [Date]

**Repository:** Supplementary material for [Paper Title/DOI]