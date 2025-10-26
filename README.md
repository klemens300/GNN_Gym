# GNN_Gym

Active Learning for diffusion barrier prediction in multi-component alloys using Graph Neural Networks.

---

## Project Structure
```
├── config.py              # Central configuration
├── oracle.py              # NEB calculations with CHGNet
├── test_oracle.py         # Oracle unit tests
├── bulk_calculation.py    # Bulk data generation
├── neb_database/          # Calculated structures (not in git)
└── data.csv               # Results database (not in git)
```

---

## Installation
```bash
pip install numpy pandas torch ase pymatgen chgnet
```

**Requirements:**
- Python 3.9+
- CUDA-capable GPU (recommended)

---

## Usage

### Test Oracle

Run unit tests with 4 sample calculations:
```bash
python test_oracle.py
```

### Generate Training Data

Adjust parameters in `bulk_calculation.py`:
```python
N_SAMPLES = 100              # Number of compositions
RUNS_PER_COMPOSITION = 1     # Runs per composition
SEED = 42                    # Random seed
```

Then run:
```bash
python bulk_calculation.py
```

---

## Configuration

All parameters are in `config.py`:

**Structure:**
- `supercell_size`: BCC supercell dimensions (default: 4x4x4)
- `lattice_parameter`: Lattice constant in Å (default: 3.2)
- `elements`: List of elements (default: ['Mo', 'Nb', 'Ta', 'W'])

**NEB:**
- `neb_images`: Number of NEB images (default: 5)
- `neb_fmax`: Force convergence in eV/Å (default: 0.05)
- `neb_max_steps`: Maximum optimization steps (default: 200)

**Relaxation:**
- `relax_fmax`: Force convergence in eV/Å (default: 0.05)
- `relax_max_steps`: Maximum steps (default: 500)

---

## Output

**Directory structure:**
```
neb_database/
└── Mo25Nb25Ta25W25/
    └── run_1/
        ├── initial_unrelaxed.cif
        ├── initial_relaxed.cif
        ├── final_unrelaxed.cif
        ├── final_relaxed.cif
        ├── neb_image_0.cif
        ├── neb_image_1.cif
        ├── ...
        └── results.json
```

**CSV database (`data.csv`):**
- Composition, barriers, energies, timestamps
- One row per calculation

**JSON results (`results.json`):**
- Complete calculation metadata
- NEB energies for all images
- Computation time

---

## Workflow

1. **Oracle creates structure:** Random BCC with vacancy at center
2. **Relaxation:** Initial structure relaxed with CHGNet
3. **Neighbor selection:** Random neighbor jumps to vacancy
4. **Relaxation:** Final structure relaxed
5. **NEB calculation:** Barrier computed between initial and final
6. **Save results:** Structures, energies, and barriers stored

---

## Notes

- Calculator: CHGNet (pretrained)
- Run numbering: Starts at 1 per composition
- Memory cleanup: Automatic via context manager
- GPU recommended for reasonable runtime

---

## License

[Add your license here]

## Citation

[Add citation when published]
