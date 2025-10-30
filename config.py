"""
Configuration for Diffusion Barrier Prediction

All settings in one place for easy management.
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


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
    max_barrier: float = 15.0            # Maximum barrier (eV) - removes slow diffusion
    
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
    scheduler_type: str = "plateau"      # Scheduler type: "plateau", "cosine", "step", or "none"
    scheduler_factor: float = 0.5        # Reduce LR by this factor (plateau, step)
    scheduler_patience: int = 10         # Patience for LR reduction (plateau)
    scheduler_step_size: int = 100       # Step size for StepLR (step)
    scheduler_t_max: int = 500           # T_max for CosineAnnealingLR (cosine)
    scheduler_eta_min: float = 1e-6      # Minimum LR for CosineAnnealingLR (cosine)
    
    # ============================================================
    # LOGGING (Weights & Biases)
    # ============================================================
    use_wandb: bool = True                    # Enable/disable wandb
    wandb_project: str = "diffusion-barrier"  # Wandb project name
    wandb_entity: str = None                  # Wandb entity (username/team), None = default
    wandb_run_name: str = None                # Run name, None = auto-generated
    wandb_tags: List[str] = field(default_factory=list)  # Tags for the run
    wandb_notes: str = ""                     # Notes for the run
    wandb_log_interval: int = 1               # Log every N epochs
    wandb_watch_model: bool = True            # Watch model gradients
    wandb_watch_freq: int = 100               # Watch frequency (batches)
    
    def get_model_name(self) -> str:
        """
        Generate a descriptive model name based on configuration.
        
        Format: {timestamp}-GNN-{layers}x{hidden}-MLP-{mlp_structure}-{scheduler}
        
        Returns:
            model_name: Descriptive model name
        
        Example:
            "20241030-143522-GNN-5x64-MLP-1024-512-256-plateau"
        """
        # Timestamp first for easy sorting
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # MLP structure as string
        mlp_str = "-".join(map(str, self.mlp_hidden_dims))
        
        # Scheduler type
        scheduler = self.scheduler_type if self.use_scheduler else "none"
        
        # Build name (timestamp first!)
        model_name = (
            f"{timestamp}-"
            f"GNN-{self.gnn_num_layers}x{self.gnn_hidden_dim}-"
            f"MLP-{mlp_str}-"
            f"{scheduler}"
        )
        
        return model_name
    
    def get_experiment_name(self) -> str:
        """
        Generate experiment name for wandb with timestamp.
        
        Format: {date}-{time}-GNN
        
        Returns:
            experiment_name: Simple experiment name with timestamp
        
        Example:
            "20241030-143522-GNN"
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-GNN"


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
    print(f"  scheduler_type: {config.scheduler_type}")
    print(f"  scheduler_factor: {config.scheduler_factor}")
    print(f"  scheduler_patience: {config.scheduler_patience}")
    print(f"  scheduler_step_size: {config.scheduler_step_size}")
    print(f"  scheduler_t_max: {config.scheduler_t_max}")
    print(f"  scheduler_eta_min: {config.scheduler_eta_min}")
    
    print("\nLOGGING (Weights & Biases):")
    print(f"  use_wandb: {config.use_wandb}")
    print(f"  wandb_project: {config.wandb_project}")
    print(f"  wandb_entity: {config.wandb_entity}")
    print(f"  wandb_run_name: {config.wandb_run_name}")
    print(f"  wandb_tags: {config.wandb_tags}")
    print(f"  wandb_notes: {config.wandb_notes}")
    print(f"  wandb_log_interval: {config.wandb_log_interval}")
    print(f"  wandb_watch_model: {config.wandb_watch_model}")
    print(f"  wandb_watch_freq: {config.wandb_watch_freq}")
    
    print("\nGENERATED NAMES:")
    print(f"  Model name: {config.get_model_name()}")
    print(f"  Experiment name: {config.get_experiment_name()}")
    
    print("\n" + "="*70)