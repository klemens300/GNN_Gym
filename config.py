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
    csv_path: str = "MoNbTaW.csv"
    checkpoint_dir: str = "checkpoints"
    database_dir: str = "MoNbTaW"          # Directory for structure files
    
    # ============================================================
    # MATERIAL SYSTEM (Elements)
    # ============================================================
    elements: List[str] = field(default_factory=lambda: ['Mo', 'Nb', 'Ta', 'W'])
    
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
    num_workers: int = 0                 # DataLoader
    
    # Data cleanup (barrier filtering)
    min_barrier: float = 0.1             # Minimum barrier (eV)
    max_barrier: float = 5.0            # Maximum barrier (eV)
    
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
    dropout: float = 0.1                # Dropout rate
    
    # ============================================================
    # TRAINING
    # ============================================================
    # Optimization
    learning_rate: float = 5e-4          # Initial learning rate
    weight_decay: float = 0.01           # L2 regularization (AdamW)
    gradient_clip_norm: float = 1.0      # Max gradient norm
    
    # Training loop
    epochs: int = 5000                   # Maximum number of epochs
    patience: int = 50                   # Early stopping patience
    save_interval: int = 50              # Save checkpoint every N epochs
    
    # Learning rate scheduling
    use_scheduler: bool = True           # Use learning rate scheduler
    scheduler_type: str = "cosine"      # Scheduler type: "plateau", "cosine", "step", or "none" ONLY COSINE TESTED
    scheduler_factor: float = 0.5        # Reduce LR by this factor (plateau, step)
    scheduler_patience: int = 10         # Patience for LR reduction (plateau)
    scheduler_step_size: int = 100       # Step size for StepLR (step)
    scheduler_t_max: int = 100           # T_max for CosineAnnealingLR (cosine)
    scheduler_eta_min: float = 1e-6      # Minimum LR for CosineAnnealingLR (cosine)
    
    # ============================================================
    # NEB (Nudged Elastic Band) PARAMETERS
    # ============================================================
    neb_n_images: int = 3                  #ARTIFACT - WILL BE DELETED
    neb_images: int = 3                # Number of images in NEB path
    neb_spring_constant: float = 5.0     # Spring constant for NEB (eV/Angstrom^2)
    neb_fmax: float = 0.1               # Force convergence criterion (eV/Angstrom)
    neb_max_steps: int = 500             # Maximum optimization steps
    neb_climb: bool = True               # Use climbing image NEB
    neb_method: str = "aseneb"           # NEB method: "aseneb" or "dynneb"

    # STRUCTURE RELAXATION (CHGNet)
    relax_cell: bool = False              # Allow cell relaxation
    relax_fmax: float = 0.1              # Force convergence for relaxation (eV/Angstrom)
    relax_steps: int = 500               # Maximum relaxation steps
    relax_max_steps: int = 500           # Maximum relaxation steps (alias for compatibility)
    
    # ============================================================
    # ACTIVE LEARNING
    # ============================================================
    # Initial data generation (Cycle 0)
    al_initial_samples: int = 1000              # Initial random samples before AL starts

    # Test set generation
    al_n_test: int = 500                      # Number of test compositions per cycle
    al_test_strategy: str = 'uniform'         # Test generation strategy: 'uniform', ...

    # Query strategy
    al_n_query: int = 1000                      # Number of new training samples per cycle
    al_query_strategy: str = 'error_weighted' # Query strategy: 'error_weighted', ...

    # Active learning loop
    al_max_cycles: int = 10                   # Maximum number of AL cycles
    al_seed: int = 42                         # Random seed for AL (gets incremented per cycle)

    # Output
    al_results_dir: str = "active_learning_results"  # Directory for AL results
    
    # ============================================================
    # LOGGING (File Logging)
    # ============================================================
    log_dir: str = "logs"                        # Directory for log files
    log_level: str = "INFO"                      # Logging level: DEBUG, INFO, WARNING, ERROR
    log_to_console: bool = True                  # Also print to console
    
    # ============================================================
    # LOSS FUNCTION
    # ============================================================
    loss_function: str = "mse"                   # Loss function: "mse", "mae", "huber", "smooth_l1"
    
    # ============================================================
    # LOGGING (Weights & Biases)
    # ============================================================
    use_wandb: bool = True                    # Enable/disable wandb
    wandb_project: str = "GNN_Gym_final_test_debug_3"  # Wandb project name
    wandb_entity: str = None                  # Wandb entity (username/team), None = default
    wandb_run_name: str = None                # Run name, None = auto-generated
    wandb_tags: List[str] = field(default_factory=list)  # Tags for the run
    wandb_notes: str = ""                     # Notes for the run
    wandb_log_interval: int = 1               # Log every N epochs
    wandb_watch_model: bool = True            # Watch model gradients
    wandb_watch_freq: int = 10               # Watch frequency (batches)
    
    def get_model_name(self, n_samples: int = None, cycle: int = None) -> str:
        """
        Generate a descriptive model name based on configuration.
        
        Format: {timestamp}-samples{n}-cycle{c}-GNN-{layers}x{hidden}-MLP-{mlp_structure}-{scheduler}
        
        Args:
            n_samples: Number of training samples (optional)
            cycle: Active learning cycle number (optional)
        
        Returns:
            model_name: Descriptive model name
        
        Example:
            "20241030-143522-samples2500-cycle0-GNN-5x64-MLP-1024-512-256-cosine"
        """
        # Timestamp first for easy sorting
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # MLP structure as string
        mlp_str = "-".join(map(str, self.mlp_hidden_dims))
        
        # Scheduler type
        scheduler = self.scheduler_type if self.use_scheduler else "none"
        
        # Build name (timestamp first!)
        parts = [timestamp]
        
        # Add sample size if provided
        if n_samples is not None:
            parts.append(f"samples{n_samples}")
        
        # Add cycle if provided
        if cycle is not None:
            parts.append(f"cycle{cycle}")
        
        # Add architecture info
        parts.append(f"GNN-{self.gnn_num_layers}x{self.gnn_hidden_dim}")
        parts.append(f"MLP-{mlp_str}")
        parts.append(scheduler)
        
        model_name = "-".join(parts)
        
        return model_name
    
    def get_experiment_name(self, cycle: int = None) -> str:
        """
        Generate experiment name for wandb with timestamp and cycle.
        
        Format: {date}-{time}-cycle{c}-GNN
        
        Args:
            cycle: Active learning cycle number (optional)
        
        Returns:
            experiment_name: Simple experiment name with timestamp
        
        Example:
            "20241030-143522-cycle0-GNN"
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if cycle is not None:
            return f"{timestamp}-cycle{cycle}-GNN"
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
    print(f"  database_dir: {config.database_dir}")
    
    print("\nMATERIAL SYSTEM:")
    print(f"  elements: {config.elements}")
    
    print("\nCRYSTAL STRUCTURE:")
    print(f"  supercell_size: {config.supercell_size}")
    print(f"  lattice_parameter: {config.lattice_parameter}")
    
    print("\nSTRUCTURE RELAXATION:")
    print(f"  relax_cell: {config.relax_cell}")
    print(f"  relax_fmax: {config.relax_fmax} eV/Å")
    print(f"  relax_steps: {config.relax_steps}")
    
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
    
    print("\nNEB PARAMETERS:")
    print(f"  neb_n_images: {config.neb_n_images}")
    print(f"  neb_spring_constant: {config.neb_spring_constant} eV/Å²")
    print(f"  neb_fmax: {config.neb_fmax} eV/Å")
    print(f"  neb_max_steps: {config.neb_max_steps}")
    print(f"  neb_climb: {config.neb_climb}")
    print(f"  neb_method: {config.neb_method}")
    
    print("\nACTIVE LEARNING:")
    print(f"  al_n_test: {config.al_n_test}")
    print(f"  al_test_strategy: {config.al_test_strategy}")
    print(f"  al_n_query: {config.al_n_query}")
    print(f"  al_query_strategy: {config.al_query_strategy}")
    print(f"  al_max_cycles: {config.al_max_cycles}")
    print(f"  al_seed: {config.al_seed}")
    print(f"  al_results_dir: {config.al_results_dir}")
    
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