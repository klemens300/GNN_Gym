"""
Configuration for Diffusion Barrier Prediction

All settings in one place for easy management.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Complete configuration for the project"""
    
    # ============================================================
    # PATHS
    # ============================================================
    csv_path: str = "database_navi.csv"
    checkpoint_dir: str = "checkpoints"
    
    # ============================================================
    # CRYSTAL STRUCTURE
    # ============================================================
    supercell_size: int = 4              # BCC supercell size (4x4x4)
    lattice_parameter: float = 3.2       # BCC lattice parameter (Angstrom)
    
    # ============================================================
    # GRAPH CONSTRUCTION
    # ============================================================
    cutoff_radius: float = 3.5           # Neighbor cutoff radius (Angstrom)
    max_neighbors: int = 50              # Maximum neighbors per atom
    
    # ============================================================
    # DATA
    # ============================================================
    batch_size: int = 32                 # Batch size for training
    num_workers: int = 0                 # DataLoader workers (0 for debugging)
    
    # Data cleanup (barrier filtering)
    min_barrier: float = 0.1             # Minimum barrier (eV) - removes noise
    max_barrier: float = 15.0             # Maximum barrier (eV) - removes slow diffusion
    
    # Train/Val split
    val_split: float = 0.1               # Validation split ratio (10%)
    random_seed: int = 42                # Random seed for reproducibility
    
    # ============================================================
    # MODEL ARCHITECTURE
    # ============================================================
    # GNN Encoder
    gnn_hidden_dim: int = 64             # Hidden dimension for GNN layers
    gnn_num_layers: int = 5              # Number of message passing layers
    gnn_embedding_dim: int = 64          # Output dimension of GNN encoder
    
    # MLP Predictor
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    dropout: float = 0.15                # Dropout rate
    
    # ============================================================
    # TRAINING
    # ============================================================
    # Optimization
    learning_rate: float = 5e-4          # Initial learning rate
    weight_decay: float = 0.01           # L2 regularization (AdamW)
    gradient_clip_norm: float = 1.0      # Max gradient norm
    
    # Training loop
    epochs: int = 1000                   # Maximum number of epochs
    patience: int = 50                   # Early stopping patience
    save_interval: int = 50              # Save checkpoint every N epochs
    
    # Learning rate scheduling
    use_scheduler: bool = True           # Use learning rate scheduler
    scheduler_factor: float = 0.5        # Reduce LR by this factor
    scheduler_patience: int = 10         # Patience for LR reduction


# Alternative configurations
def get_fast_config() -> Config:
    """
    Fast config for testing/debugging.
    Small model, few epochs.
    """
    config = Config()
    config.gnn_hidden_dim = 32
    config.gnn_num_layers = 3
    config.gnn_embedding_dim = 32
    config.mlp_hidden_dims = [256, 128]
    config.batch_size = 8
    config.epochs = 100
    config.patience = 10
    return config


def get_production_config() -> Config:
    """
    Production config for best performance.
    Large model, many epochs.
    """
    config = Config()
    config.gnn_hidden_dim = 128
    config.gnn_num_layers = 8
    config.gnn_embedding_dim = 128
    config.mlp_hidden_dims = [2048, 1024, 512, 256]
    config.batch_size = 32
    config.epochs = 5000
    config.patience = 100
    config.learning_rate = 3e-4
    return config


# Display config
if __name__ == "__main__":
    print("="*70)
    print("DEFAULT CONFIGURATION")
    print("="*70)
    
    config = Config()
    
    print("\nPATHS:")
    print(f"  csv_path: {config.csv_path}")
    print(f"  checkpoint_dir: {config.checkpoint_dir}")
    
    print("\nCRYSTAL STRUCTURE:")
    print(f"  supercell_size: {config.supercell_size}")
    print(f"  lattice_parameter: {config.lattice_parameter}")
    
    print("\nGRAPH CONSTRUCTION:")
    print(f"  cutoff_radius: {config.cutoff_radius}")
    print(f"  max_neighbors: {config.max_neighbors}")
    
    print("\nDATA:")
    print(f"  batch_size: {config.batch_size}")
    print(f"  min_barrier: {config.min_barrier} eV")
    print(f"  max_barrier: {config.max_barrier} eV")
    print(f"  val_split: {config.val_split}")
    print(f"  random_seed: {config.random_seed}")
    
    print("\nMODEL ARCHITECTURE:")
    print(f"  gnn_hidden_dim: {config.gnn_hidden_dim}")
    print(f"  gnn_num_layers: {config.gnn_num_layers}")
    print(f"  gnn_embedding_dim: {config.gnn_embedding_dim}")
    print(f"  mlp_hidden_dims: {config.mlp_hidden_dims}")
    print(f"  dropout: {config.dropout}")
    
    print("\nTRAINING:")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  weight_decay: {config.weight_decay}")
    print(f"  gradient_clip_norm: {config.gradient_clip_norm}")
    print(f"  epochs: {config.epochs}")
    print(f"  patience: {config.patience}")
    print(f"  save_interval: {config.save_interval}")
    
    print("\nLEARNING RATE SCHEDULER:")
    print(f"  use_scheduler: {config.use_scheduler}")
    print(f"  scheduler_factor: {config.scheduler_factor}")
    print(f"  scheduler_patience: {config.scheduler_patience}")
    
    print("\n" + "="*70)
    
    # Alternative configs
    print("\nALTERNATIVE CONFIGURATIONS:")
    
    print("\n1. Fast Config (testing):")
    fast = get_fast_config()
    print(f"   Model: {fast.gnn_hidden_dim}D hidden, {fast.gnn_num_layers} layers")
    print(f"   Training: {fast.epochs} epochs, batch {fast.batch_size}")
    
    print("\n2. Production Config (best performance):")
    prod = get_production_config()
    print(f"   Model: {prod.gnn_hidden_dim}D hidden, {prod.gnn_num_layers} layers")
    print(f"   Training: {prod.epochs} epochs, batch {prod.batch_size}")
    
    print("\n" + "="*70)