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
```
template_graph_builder.py  # Fast graph construction with template
config.py                  # Configuration management
atomic_properties.py       # Element property database
test_template_graph_builder.py  # Comprehensive test suite
```

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install torch torch-geometric pymatgen ase pandas numpy
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
├── template_graph_builder.py     # Core graph builder
├── config.py                      # Configuration
├── atomic_properties.py           # Element database
├── test_template_graph_builder.py # Test suite
├── database/                      # NEB calculation results
│   └── <composition>/
│       └── run_*/
│           ├── initial_relaxed.cif
│           └── final_relaxed.cif
├── database_navi.csv              # Metadata (barriers, paths)
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