"""
Central configuration for all parameters.
Everything adjustable in one place!
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """
    Central configuration for NEB calculations and data generation.
    All parameters can be adjusted here.
    """
    
    # ============================================================
    # BCC STRUCTURE
    # ============================================================
    supercell_size: int = 4              # Size of BCC supercell (e.g., 4 = 4x4x4)
    lattice_parameter: float = 3.2       # BCC lattice parameter in Angstrom
    
    # ============================================================
    # ELEMENTS
    # ============================================================
    elements: List[str] = field(default_factory=lambda: ['Mo', 'Nb', 'Ta', 'W'])
    # Can be changed to any elements, e.g., ['Fe', 'Cr', 'Ni']
    
    # ============================================================
    # NEB PARAMETERS
    # ============================================================
    neb_images: int = 3                  # Number of NEB images (min. 3)
    neb_fmax: float = 0.05               # Force convergence criterion (eV/Angstrom)
    neb_max_steps: int = 200             # Maximum NEB optimization steps
    neb_spring_constant: float = 0.5     # Spring constant for NEB
    neb_climb: bool = True               # Use climbing image NEB
    
    # ============================================================
    # CHGNET RELAXATION
    # ============================================================
    relax_fmax: float = 0.05             # Force convergence for structure relaxation
    relax_max_steps: int = 500           # Maximum relaxation steps
    relax_cell: bool = False             # If True, relax cell; if False, only atoms
    
    # ============================================================
    # DATA STORAGE
    # ============================================================
    database_dir: str = "neb_database"   # Main database directory
    csv_path: str = "data.csv"           # CSV file for metadata
    
    # ============================================================
    # RANDOM SEED
    # ============================================================
    random_seed: int = 42                # For reproducibility


if __name__ == "__main__":
    # Display default configuration
    config = Config()
    
    print("="*70)
    print("CONFIGURATION")
    print("="*70)
    
    print("\nBCC STRUCTURE:")
    print(f"  Supercell size: {config.supercell_size}x{config.supercell_size}x{config.supercell_size}")
    print(f"  Lattice parameter: {config.lattice_parameter} Å")
    print(f"  Elements: {config.elements}")
    
    print("\nNEB PARAMETERS:")
    print(f"  Number of images: {config.neb_images}")
    print(f"  Force convergence: {config.neb_fmax} eV/Å")
    print(f"  Max steps: {config.neb_max_steps}")
    print(f"  Spring constant: {config.neb_spring_constant}")
    print(f"  Climbing image: {config.neb_climb}")
    
    print("\nRELAXATION:")
    print(f"  Force convergence: {config.relax_fmax} eV/Å")
    print(f"  Max steps: {config.relax_max_steps}")
    print(f"  Relax cell: {config.relax_cell}")
    
    print("\nDATA STORAGE:")
    print(f"  Database directory: {config.database_dir}")
    print(f"  CSV path: {config.csv_path}")
    
    print("\n" + "="*70)