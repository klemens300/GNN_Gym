# GNN_Gym - Diffusion Barrier Prediction with Graph Neural Networks

Deep learning framework for predicting vacancy diffusion barriers in Body-Centered Cubic (BCC) high-entropy alloys using Graph Neural Networks.

## Overview

This project implements a complete machine learning pipeline for predicting energy barriers of vacancy diffusion in high-entropy alloys, specifically focusing on the Mo-Nb-Ta-W system. The framework combines:

- **Oracle System**: Automated NEB calculations using CHGNet
- **Template-based Graph Construction**: Ultra-fast graph generation for identical BCC geometries
- **Graph Neural Network**: Custom message-passing architecture for barrier prediction
- **Training Pipeline**: Comprehensive training loop with early stopping and Weights & Biases integration

## Key Features

- **Oracle for Data Generation**: Automated NEB (Nudged Elastic Band) calculations for generating training data
- **Template-Based Graph Builder**: 100x faster than traditional graph construction methods
- **Custom GNN Architecture**: Message-passing neural network designed for crystal structures
- **Complete Training Pipeline**: Built-in training loop with validation, checkpointing, and logging
- **Automatic Element Detection**: Adapts to any element system in the database
- **PBC-Aware**: Correctly handles periodic boundary conditions
- **Jupyter Notebooks**: Interactive notebooks for testing and training

## Project Structure

```
GNN_Gym/
├── README.md                          # This file
│
├── Core Components
│   ├── config.py                      # Central configuration
│   ├── atomic_properties.py           # Element property database
│   ├── template_graph_builder.py      # Fast graph construction
│   ├── model.py                       # GNN architecture
│   ├── dataset.py                     # PyTorch dataset & dataloaders
│   ├── trainer.py                     # Training loop
│   ├── oracle.py                      # NEB calculation oracle
│   └── utils.py                       # Utility functions
│
├── Jupyter Notebooks
│   ├── UT1_test_oracle.ipynb          # Oracle testing
│   ├── UT2_bulk_calculation.ipynb     # Bulk NEB calculations
│   ├── UT3_check_computational_time_database.ipynb  # Performance analysis
│   ├── UT4_Test_graph_creation.ipynb  # Graph builder testing
│   ├── UT5_Test_dataloader.ipynb      # Dataloader testing
│   └── train.ipynb                    # Model training
│
└── Data (generated)
    ├── database_navi.csv              # Database metadata
    ├── database/                      # NEB calculation results
    └── checkpoints/                   # Model checkpoints
```

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torch-geometric
pip install pymatgen ase pandas numpy

# For Oracle (NEB calculations)
pip install chgnet

# For training visualization
pip install wandb

# For Jupyter notebooks
pip install jupyter matplotlib
```

### Quick Setup

```bash
# Clone repository
git clone <your-repo-url>
cd GNN_Gym

# Install dependencies
pip install torch torch-geometric pymatgen ase pandas numpy chgnet wandb

# Ready to go!
```

## Quick Start

### 1. Configure Your Experiment

Edit `config.py` to set your parameters:

```python
from config import Config

config = Config()
config.batch_size = 32
config.learning_rate = 5e-4
config.epochs = 1000
config.use_wandb = True  # Enable Weights & Biases logging
```

### 2. Generate Training Data (Oracle)

Use the Oracle to generate NEB calculation data:

```python
from oracle import Oracle
from config import Config

config = Config()
oracle = Oracle(config)

# Generate data for specific composition
composition = {'Mo': 0.25, 'Nb': 0.25, 'Ta': 0.25, 'W': 0.25}
oracle.calculate(composition)
```

See `UT1_test_oracle.ipynb` and `UT2_bulk_calculation.ipynb` for examples.

### 3. Build Graphs from Data

The template-based graph builder creates graphs on-the-fly:

```python
from template_graph_builder import TemplateGraphBuilder
from config import Config

config = Config()
builder = TemplateGraphBuilder(config)

# Build graph pair with label
initial_graph, final_graph = builder.build_pair_graph(
    initial_cif="path/to/initial.cif",
    final_cif="path/to/final.cif",
    backward_barrier=1.23  # eV
)
```

### 4. Train the Model

Use the provided training pipeline:

```python
from config import Config
from dataset import create_dataloaders
from model import DiffusionBarrierModel, create_model_from_config
from trainer import Trainer

# Setup
config = Config()
train_loader, val_loader = create_dataloaders(config, val_split=0.1)

# Get dynamic node dimension
node_input_dim = 3 + len(builder.elements) + 4

# Create model
model = create_model_from_config(config, node_input_dim)

# Train
trainer = Trainer(model, config, save_dir="checkpoints")
history = trainer.train(train_loader, val_loader)
```

Or use the interactive notebook: `train.ipynb`

## Core Components

### Oracle (`oracle.py`)

Automated system for generating NEB calculation data:

- Creates BCC supercells with arbitrary compositions
- Places vacancies and selects diffusion paths
- Performs structure relaxation using CHGNet
- Runs NEB calculations to find energy barriers
- Saves all results (structures, energies, barriers)

**Key features:**
- Silent mode (no verbose output)
- Automatic run numbering per composition
- Complete data tracking in CSV
- Memory cleanup after calculations

### Graph Builder (`template_graph_builder.py`)

Template-based graph construction for ultra-fast performance:

- **Template approach**: Build connectivity once, reuse for all structures
- **Speed**: ~20ms per graph pair (100 pairs in 2 seconds)
- **Memory efficient**: No pre-computed files needed
- **Automatic element detection**: Adapts to any element system

**Node Features** (7 + N):
- Cartesian positions (3)
- Element one-hot encoding (N elements)
- Atomic properties (4): radius, mass, electronegativity, valence

**Edge Features** (1):
- Interatomic distances

### Model Architecture (`model.py`)

Custom Graph Neural Network for diffusion barrier prediction:

**Components:**
1. **GraphConvLayer**: Message passing layer with edge features
2. **GNNEncoder**: Stack of graph convolution layers
3. **BarrierPredictor**: MLP for barrier prediction
4. **DiffusionBarrierModel**: Complete model combining encoder + predictor

**Architecture:**
- Shared encoder for initial and final structures
- Difference embeddings capture structural changes
- Deep MLP predictor (default: [1024, 512, 256])
- BatchNorm and Dropout for regularization

### Dataset (`dataset.py`)

PyTorch dataset for on-the-fly graph loading:

- Loads data from CSV database
- Builds graphs on-the-fly (fast with template!)
- Automatic barrier filtering for data quality
- Train/validation splitting
- Custom collate function for batching

### Trainer (`trainer.py`)

Complete training pipeline with:

- Training and validation loops
- Early stopping with patience
- Learning rate scheduling (Plateau, Cosine, Step)
- Gradient clipping
- Checkpoint saving/loading
- Weights & Biases integration
- Training history tracking

## Usage Examples

### Training a Model

```python
from config import Config
from dataset import create_dataloaders
from template_graph_builder import TemplateGraphBuilder
from model import create_model_from_config
from trainer import Trainer

# Configuration
config = Config()
config.batch_size = 32
config.learning_rate = 5e-4
config.epochs = 1000
config.patience = 50
config.use_wandb = True

# Create dataloaders
train_loader, val_loader = create_dataloaders(config, val_split=0.1)

# Get node input dimension dynamically
builder = TemplateGraphBuilder(config)
node_input_dim = 3 + len(builder.elements) + 4

# Create model
model = create_model_from_config(config, node_input_dim)

# Train
trainer = Trainer(model, config, save_dir="checkpoints")
history = trainer.train(train_loader, val_loader)
```

### Making Predictions

```python
import torch
from model import DiffusionBarrierModel

# Load trained model
model = DiffusionBarrierModel(node_input_dim=12)
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict barrier
with torch.no_grad():
    prediction = model(initial_graph, final_graph)
    barrier_eV = prediction.item()
    print(f"Predicted barrier: {barrier_eV:.3f} eV")
```

### Generating New Data

```python
from oracle import Oracle
from config import Config
import numpy as np

config = Config()

# Use context manager for automatic cleanup
with Oracle(config) as oracle:
    # Generate data for multiple compositions
    for _ in range(10):
        # Random quaternary composition
        fractions = np.random.dirichlet([1, 1, 1, 1])
        composition = {
            'Mo': fractions[0],
            'Nb': fractions[1],
            'Ta': fractions[2],
            'W': fractions[3]
        }
        oracle.calculate(composition)
```

## Jupyter Notebooks

Interactive notebooks for testing and development:

- **UT1_test_oracle.ipynb**: Test Oracle functionality
- **UT2_bulk_calculation.ipynb**: Generate multiple NEB calculations in bulk
- **UT3_check_computational_time_database.ipynb**: Analyze Oracle performance
- **UT4_Test_graph_creation.ipynb**: Test and validate graph construction
- **UT5_Test_dataloader.ipynb**: Test dataset and dataloader functionality
- **train.ipynb**: Complete training workflow

## Configuration

All settings are centralized in `config.py`:

### Key Parameters

```python
# Crystal structure
supercell_size: int = 4              # 4x4x4 BCC supercell
lattice_parameter: float = 3.2       # Angstrom

# Graph construction
cutoff_radius: float = 3.5           # Neighbor cutoff (Å)
max_neighbors: int = 50              # Max neighbors per atom

# Data
batch_size: int = 32
min_barrier: float = 0.1             # Filter out noise
max_barrier: float = 15.0            # Filter out unrealistic barriers
val_split: float = 0.1               # 10% validation

# Model architecture
gnn_hidden_dim: int = 64
gnn_num_layers: int = 5
gnn_embedding_dim: int = 64
mlp_hidden_dims: List[int] = [1024, 512, 256]
dropout: float = 0.15

# Training
learning_rate: float = 5e-4
weight_decay: float = 0.01
epochs: int = 1000
patience: int = 50

# Scheduler
use_scheduler: bool = True
scheduler_type: str = "plateau"      # "plateau", "cosine", "step"
scheduler_factor: float = 0.5
scheduler_patience: int = 10

# Logging
use_wandb: bool = True
wandb_project: str = "diffusion-barrier"
```

## Performance

- **Graph Construction**: ~20ms per graph pair
- **Training Speed**: Depends on dataset size and hardware
- **Oracle NEB Calculation**: ~60-120 seconds per structure pair (with CHGNet)
- **Memory**: Efficient - no pre-computed graph files needed

## Database Format

The Oracle generates a CSV database with the following structure:

| Column | Description |
|--------|-------------|
| composition_string | e.g., "Mo25Nb25Ta25W25" |
| Mo, Nb, Ta, W | Element fractions |
| run_number | Incremental run number per composition |
| calculator | "CHGNet" |
| diffusing_element | Element that diffuses |
| forward_barrier_eV | Forward barrier |
| backward_barrier_eV | Backward barrier (used as label) |
| E_initial_eV | Initial structure energy |
| E_final_eV | Final structure energy |
| structure_folder | Path to CIF files |
| timestamp | Calculation timestamp |

## Key Design Decisions

### Why Template-Based Graphs?

All BCC structures with the same supercell size share identical geometry:
- **Fixed connectivity** → Build graph topology once, reuse forever
- **Only positions & elements change** → Fast node feature updates
- **Result**: 100x speedup compared to rebuilding graphs from scratch

### Why Both Initial and Final Graphs?

The GNN learns from the **structural transition**:
1. Encode initial structure → embedding
2. Encode final structure → embedding
3. Compute difference embedding (Δemb = final - initial)
4. Predict barrier from [initial_emb, final_emb, Δemb]

This captures the structural changes that determine the barrier.

### Why Backward Barrier?

The model is trained on backward barriers (final → initial) as the primary prediction target, though the framework can be easily adapted for forward barriers.

## Extending the Project

### Add New Elements

1. Add atomic properties to `atomic_properties.py`:
```python
ATOMIC_PROPERTIES = {
    'Mo': {'atomic_radius': 1.47, 'atomic_mass': 95.95, ...},
    'YourElement': {'atomic_radius': X.XX, 'atomic_mass': XX.XX, ...}
}
```

2. Update `config.py` with new elements list

3. Generate data with Oracle - automatic element detection!

### Modify Model Architecture

Edit `config.py` to change model hyperparameters:

```python
config.gnn_hidden_dim = 128           # Larger GNN
config.gnn_num_layers = 10            # Deeper GNN
config.mlp_hidden_dims = [2048, 1024, 512, 256]  # Larger MLP
```

### Use Different Loss Functions

Modify `trainer.py`:

```python
# L1 Loss instead of MSE
self.criterion = nn.L1Loss()

# Or Huber Loss
self.criterion = nn.SmoothL1Loss()
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in config
- Reduce model size (`gnn_hidden_dim`, `mlp_hidden_dims`)
- Use gradient accumulation

### Oracle Calculations Too Slow

- Reduce `neb_images` in oracle config
- Use faster relaxation settings (higher `fmax`)
- Run calculations in parallel (multiple Oracle instances)

### Poor Model Performance

- Check data quality (barrier distribution)
- Adjust barrier filtering (`min_barrier`, `max_barrier`)
- Increase model capacity
- Add more training data
- Tune hyperparameters

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gnn_gym_2024,
  title={GNN_Gym: Graph Neural Networks for Diffusion Barrier Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/GNN_Gym}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **CHGNet**: Used for NEB calculations and structure relaxation
- **PyTorch Geometric**: Graph neural network framework
- **ASE**: Atomic Simulation Environment
- **Weights & Biases**: Experiment tracking and visualization

## Contact

For questions, issues, or contributions, please open a GitHub issue or contact the maintainers.
