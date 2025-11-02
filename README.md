# Diffusion Barrier Prediction with GNN

Graph Neural Network for predicting vacancy diffusion barriers in BCC alloys using template-based graph construction.

## Overview

This project implements an active learning pipeline for predicting energy barriers of vacancy diffusion in high-entropy alloys (Mo-Nb-Ta-W system). The key innovation is a **template-based graph builder** that leverages the identical geometry of BCC structures for fast, on-the-fly graph generation.

## Features

- **Template-Based Graph Construction**: 100x faster than traditional methods, no pre-computed files needed
- **Automatic Element Detection**: Adapts to any element system in the database
- **PBC-Aware**: Correctly handles periodic boundary conditions for position differences
- **Active Learning Ready**: Built for iterative model improvement with oracle feedback
- **Flexible Architecture**: Works with any number of elements (ternary, quaternary, quinary, etc.)

## Script Overview

### Core Scripts

**model.py** - GNN Architecture
- `GraphConvLayer`: Custom message passing layer with edge features
- `GNNEncoder`: Stacks multiple graph conv layers
- `BarrierPredictor`: MLP that predicts barriers from embeddings
- `DiffusionBarrierModel`: Full model (encoder + predictor)

**dataset.py** - Data Loading
- `DiffusionBarrierDataset`: PyTorch dataset with on-the-fly graph building
- `create_dataloaders()`: Creates train/val dataloaders with barrier filtering
- Template-based construction for fast graph generation

**trainer.py** - Training
- Training and validation loops
- Early stopping with patience
- Learning rate scheduling (Plateau, Cosine, Step)
- Checkpoint saving/loading
- Weights & Biases integration

**inference.py** - Active Learning
- `run_inference_cycle()`: Complete AL cycle (test generation → oracle → predictions → query selection)
- `generate_test_compositions()`: Uniform grid sampling over composition space
- `select_samples_by_error()`: Error-weighted query strategy
- GPU memory cleanup utilities

**oracle.py** - Ground Truth Generation
- Creates BCC supercells with arbitrary compositions
- Vacancy creation and neighbor selection
- Structure relaxation using CHGNet
- NEB calculations for diffusion barriers
- Automatic data storage with run numbering

**utils.py** - Utilities
- `load_model_for_inference()`: Load trained models
- `save_model_for_inference()`: Save with full metadata
- `predict_single()`: Single structure pair prediction
- `predict_batch()`: Batch predictions
- Model info and validation utilities

## Architecture

### Graph Representation

**Node Features** (7 + N):
- Cartesian positions (3) - from CIF structure
- Element one-hot encoding (N) - automatically detected
- Atomic properties (4) - radius, mass, electronegativity, valence

**Edge Features** (1):
- Interatomic distances (from template)

**Label**:
- Energy barrier in eV (required)

### Key Components

**Core Architecture:**
```
model.py                   # GNN model (GraphConv layers + barrier predictor)
template_graph_builder.py  # Fast graph construction with template
config.py                  # Configuration management
atomic_properties.py       # Element property database
```

**Training & Data:**
```
dataset.py                 # PyTorch dataset with on-the-fly graph building
trainer.py                 # Training loop with early stopping & wandb
oracle.py                  # NEB calculations using CHGNet
```

**Active Learning & Inference:**
```
inference.py               # Active learning cycle (test generation, predictions, query selection)
utils.py                   # Model I/O, prediction utilities
```

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd <repo-name>

# Install core dependencies
pip install torch torch-geometric pymatgen ase pandas numpy

# Install CHGNet for oracle calculations (optional)
pip install chgnet

# Install Weights & Biases for training logging (optional)
pip install wandb
```

## Quick Start

### 1. Initialize Template Builder
```python
from template_graph_builder import TemplateGraphBuilder
from config import Config

config = Config()
builder = TemplateGraphBuilder(config)
```

The builder automatically detects all elements from your database.

### 2. Build Graph Pairs
```python
# Build graph pair with required label
initial_graph, final_graph = builder.build_pair_graph(
    initial_cif="path/to/initial.cif",
    final_cif="path/to/final.cif",
    backward_barrier=1.23  # eV (REQUIRED)
)

# Both graphs have the same label
print(initial_graph.y)  # tensor([1.23])
print(final_graph.y)    # tensor([1.23])
```

### 3. Use in GNN Training
```python
# Your GNN model
class DiffusionBarrierGNN(nn.Module):
    def forward(self, initial_graph, final_graph):
        emb_initial = self.encoder(initial_graph)
        emb_final = self.encoder(final_graph)
        delta = emb_final - emb_initial
        barrier = self.predictor(delta)
        return barrier

# Training loop
for initial_graph, final_graph in dataloader:
    prediction = model(initial_graph, final_graph)
    loss = criterion(prediction, initial_graph.y)
    loss.backward()
```

## Training Pipeline

### 1. Train a Model
```python
from config import Config
from dataset import create_dataloaders
from model import create_model_from_config
from template_graph_builder import TemplateGraphBuilder
from trainer import Trainer

# Setup
config = Config()
builder = TemplateGraphBuilder(config)
node_input_dim = 3 + len(builder.elements) + 4

# Create dataloaders
train_loader, val_loader = create_dataloaders(config, val_split=0.1)

# Create model
model = create_model_from_config(config, node_input_dim)

# Train
trainer = Trainer(model, config, save_dir="checkpoints")
history = trainer.train(train_loader, val_loader)
```

### 2. Active Learning Cycle
```python
from inference import run_inference_cycle
from oracle import Oracle

# Initialize oracle
oracle = Oracle(config)

# Run inference cycle
selected_samples, predictions = run_inference_cycle(
    cycle=0,
    model_path='checkpoints/best_model.pt',
    oracle=oracle,
    config=config
)

# selected_samples contains highest-error compositions for next training
```

### 3. Generate Training Data with Oracle
```python
from oracle import Oracle

oracle = Oracle(config)

# Calculate NEB barrier for a composition
composition = {'Mo': 0.25, 'Nb': 0.25, 'Ta': 0.25, 'W': 0.25}
success = oracle.calculate(composition)

# Results saved to database/ and appended to database_navi.csv
```

### 4. Make Predictions
```python
from utils import load_model_for_inference, predict_single
from template_graph_builder import TemplateGraphBuilder

# Load model
model, checkpoint = load_model_for_inference(
    'checkpoints/best_model.pt', config
)

# Build graphs
builder = TemplateGraphBuilder(config)
initial_graph, final_graph = builder.build_pair_graph(
    "path/to/initial.cif",
    "path/to/final.cif",
    backward_barrier=0.0  # Dummy value for prediction
)

# Predict barrier
barrier = predict_single(model, initial_graph, final_graph, device='cuda')
print(f"Predicted barrier: {barrier:.3f} eV")
```

## Configuration

Edit `config.py` to customize:
```python
@dataclass
class Config:
    # Crystal structure
    supercell_size: int = 4              # 4x4x4 BCC supercell
    lattice_parameter: float = 3.2       # Angstrom
    
    # Graph construction
    cutoff_radius: float = 3.5           # Neighbor cutoff (Å)
    max_neighbors: int = 50              # Max neighbors per atom
    
    # Data
    csv_path: str = "database_navi.csv"  # Database path
```

## Performance

**Speed**: ~20ms per graph pair (100 pairs in 2 seconds)

**Memory**: No pre-computed .pt files needed

**Scalability**: Tested on 20+ samples, supports any element system

## Testing

Run comprehensive test suite:
```bash
python test_template_graph_builder.py
```

**Tests include:**
1. Template creation and validation
2. Automatic element detection
3. Graph pair construction with labels
4. PBC-corrected position differences
5. Multiple samples validation
6. Speed benchmarks
7. Label requirement enforcement
8. Template immutability

## Project Structure
```
.
├── model.py                       # GNN architecture
├── template_graph_builder.py     # Fast graph builder with template
├── dataset.py                     # PyTorch dataset & dataloaders
├── trainer.py                     # Training loop with early stopping
├── inference.py                   # Active learning inference cycle
├── oracle.py                      # NEB calculations with CHGNet
├── utils.py                       # Model I/O and prediction utilities
├── config.py                      # Configuration management
├── atomic_properties.py           # Element property database
├── database/                      # NEB calculation results
│   └── <composition>/
│       └── run_*/
│           ├── initial_relaxed.cif
│           ├── final_relaxed.cif
│           └── neb_image_*.cif
├── database_navi.csv              # Metadata (barriers, paths)
├── checkpoints/                   # Saved model checkpoints
└── README.md
```

## Database Format

CSV columns:
- `composition_string`: e.g., "Mo25Nb25Ta25W25"
- `structure_folder`: Path to CIF files
- `forward_barrier_eV`: Forward barrier
- `backward_barrier_eV`: Backward barrier (used as label)

## Key Design Decisions

### Why Template-Based?

All structures share identical geometry (4x4x4 BCC with vacancy):
- **Connectivity is fixed** → Build once, reuse forever
- **Only positions & elements change** → Fast updates
- **Result**: 100x speedup vs. rebuilding graphs

### Why Both Graphs?

The GNN learns from the **transition** (initial → final):
- Encodes both structures
- Computes difference embedding
- Predicts barrier from structural change

### Why Backward Barrier?

Training focuses on backward barriers (final → initial) as the primary prediction target.

## Extending to New Elements

1. Add atomic properties to `atomic_properties.py`:
```python
ATOMIC_PROPERTIES = {
    'Mo': {'atomic_radius': 1.47, 'atomic_mass': 95.95, ...},
    'YourElement': {'atomic_radius': X.XX, 'atomic_mass': XX.XX, ...},
}
```

2. Generate data with new element

3. Template builder automatically detects and adapts!

## Citation

If you use this code, please cite:
```bibtex
@software{diffusion_gnn_2024,
  title={Template-Based GNN for Diffusion Barrier Prediction},
  author={Your Name},
  year={2024},
  url={your-repo-url}
}
```

## License

MIT License - see LICENSE file

## Contact

For questions or issues, please open a GitHub issue.