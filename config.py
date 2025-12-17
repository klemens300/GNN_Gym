"""
Configuration for Diffusion Barrier Prediction

All settings in one place for easy management.

UPDATED:
- Increased learning_rate from 5e-5 to 1e-3 (helps escape mean prediction)
- Changed loss_function from "mse" to "huber" (more robust for outliers)
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
    csv_path: str = "/home/klemens/databases/MoNbTaW.csv"
    checkpoint_dir: str = "checkpoints"
    database_dir: str = "/home/klemens/databases/MoNbTaW"          # Directory for structure files

    # ============================================================
    # TRAINING MODE
    # ============================================================
    train_only_mode: bool = True        # Skip AL loop, only train final model
    train_only_skip_cycles: bool = True  # Skip cycle training, only final model
    
    # ============================================================
    # CALCULATOR SETTINGS
    # ============================================================
    calculator: str = "fairchem"             # Calculator: "chgnet" or "fairchem"
    
    # FAIRChem Model Options:
    # - Universal Materials Accelerator (UMA) - Best for inorganic materials:
    #   * "uma-s-1p1" - Small, fast
    #   * "uma-m-1p1" - Medium, balanced (RECOMMENDED for Mo-Nb-Ta-W)
    #   * "uma-l-1p1" - Large, most accurate
    # - EquiformerV2 (OC20/OC22):
    #   * "EquiformerV2-31M-S2EF-OC20-All+MD"
    #   * "EquiformerV2-153M-S2EF-OC20-All+MD"
    # - GemNet:
    #   * "GemNet-OC-S2EFS-OC20+OC22"
    fairchem_model: str = "uma-m-1p1"        # Model name for FAIRChem
    
    # CHGNet settings (if calculator="chgnet")
    chgnet_model: str = "0.3.0"              # CHGNet version
    
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

    # Line Graph (ALIGNN-style) for bond angles
    use_line_graph: bool = True          # Enable line graph
    line_graph_cutoff: float = 3.5 

    # ============================================================
    # DATA
    # ============================================================
    batch_size: int = 64                 # Batch size for training
    num_workers: int = 0                 # DataLoader workers
    
    # Data cleanup (barrier filtering)
    min_barrier: float = 0.0             # Minimum barrier (eV)
    max_barrier: float = 50.0             # Maximum barrier (eV)
    
    # Train/Val split
    val_split: float = 0.1               # Validation split ratio (10%)
    random_seed: int = 42                # Random seed for reproducibility
    
    # ============================================================
    # MODEL ARCHITECTURE
    # ============================================================
    # Atom Graph (GNN Encoder)
    gnn_hidden_dim: int = 64             # Hidden dimension for GNN layers
    gnn_num_layers: int = 5              # Number of message passing layers
    gnn_embedding_dim: int = 64          # Output dimension of GNN encoder

    # Line Graph (for bond angles)
    use_line_graph: bool = True          # Use ALIGNN-style line graph
    line_graph_hidden_dim: int = 64      # Hidden dimension for line graph layers
    line_graph_num_layers: int = 3       # Number of line graph message passing layers
    line_graph_embedding_dim: int = 64   # Output dimension of line graph

    # MLP Predictor
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1                 # Dropout rate
    
    # ============================================================
    # TRAINING
    # ============================================================
    # Optimization
    learning_rate: float = 1e-3          # ✅ FIXED: Increased from 5e-5 to 1e-3
    weight_decay: float = 0.01           # L2 regularization (AdamW)
    gradient_clip_norm: float = 1.0      # Max gradient norm
    
    # Training loop
    epochs: int = 10000                  # Maximum number of epochs
    patience: int = 120                  # Early stopping patience (epochs)
    save_interval: int = 50              # Save checkpoint every N epochs
    
    # Final model training (after convergence or max cycles)
    final_model_patience: int = 666      # Higher patience for final model
    
    # ============================================================
    # LEARNING RATE SCHEDULER
    # ============================================================
    use_scheduler: bool = True
    scheduler_type: str = "cosine_warm_restarts"  # Type: "plateau", "step", "cosine", "cosine_warm_restarts"
    
    # --- ReduceLROnPlateau (scheduler_type="plateau") ---
    plateau_factor: float = 0.5              # Reduce LR by this factor
    plateau_patience: int = 10               # Epochs without improvement
    
    # --- StepLR (scheduler_type="step") ---
    step_size: int = 100                     # Step every N epochs
    step_gamma: float = 0.5                  # Multiply LR by gamma
    
    # --- CosineAnnealingLR (scheduler_type="cosine") ---
    cosine_t_max: int = 100                  # Period length
    cosine_eta_min: float = 1e-6             # Minimum LR
    
    # --- CosineAnnealingWarmRestarts (scheduler_type="cosine_warm_restarts") ---
    warm_restart_t_0: int = 1000              # First restart period (epochs)
    warm_restart_t_mult: float = 1.2         # Period multiplier after restart
    warm_restart_eta_min: float = 1e-4       # Minimum LR (increased from 5e-5)
    warm_restart_decay: float = 0.9          # LR decay factor after restart
    
    # ============================================================
    # NEB (Nudged Elastic Band) PARAMETERS
    # ============================================================
    neb_images: int = 3                  # Number of images in NEB path
    neb_spring_constant: float = 5.0     # Spring constant (eV/Angstrom^2)
    neb_fmax: float = 0.1                # Force convergence criterion (eV/Angstrom)
    neb_max_steps: int = 500             # Maximum optimization steps
    neb_climb: bool = True               # Use climbing image NEB
    neb_method: str = "aseneb"           # NEB method: "aseneb" or "dynneb"

    # ============================================================
    # STRUCTURE RELAXATION
    # ============================================================
    relax_cell: bool = False             # Allow cell relaxation
    relax_fmax: float = 0.1              # Force convergence (eV/Angstrom)
    relax_steps: int = 500               # Maximum relaxation steps
    relax_max_steps: int = 500           # Maximum relaxation steps (alias)
    
    # ============================================================
    # ACTIVE LEARNING
    # ============================================================
    # Initial data generation (Cycle 0)
    al_initial_samples: int = 20000       # Initial random samples before AL starts

    # Test set generation
    al_n_test: int = 4000                # Number of test compositions per cycle
    al_test_strategy: str = 'uniform'    # Test generation strategy

    # Query strategy
    al_n_query: int = 1000               # Number of new training samples per cycle
    al_query_strategy: str = 'error_weighted'  # Query strategy

    # Active learning loop
    al_max_cycles: int = 20              # Maximum number of AL cycles
    al_seed: int = 42                    # Random seed for AL

    # Convergence criteria
    al_convergence_check: bool = True                    # Enable convergence checking
    al_convergence_metric: str = "mae"                   # Metric: "mae" or "rel_mae"
    al_convergence_threshold_mae: float = 0.01           # MAE threshold (eV)
    al_convergence_threshold_rel_mae: float = 0.1        # Relative MAE threshold
    al_convergence_patience: int = 20                    # Cycles without improvement

    # Output
    al_results_dir: str = "active_learning_results"  # Directory for AL results
    
    # ============================================================
    # LOGGING (File Logging)
    # ============================================================
    log_dir: str = "logs"                # Directory for log files
    log_level: str = "INFO"              # Logging level
    log_to_console: bool = True          # Also print to console
    
    # ============================================================
    # LOSS FUNCTION
    # ============================================================
    loss_function: str = "huber"         # ✅ FIXED: Changed from "mse" to "huber"
                                         # Options: "mse", "mae", "huber", "smooth_l1"
                                         # Huber is more robust to outliers
    
    # ============================================================
    # LOGGING (Weights & Biases)
    # ============================================================
    use_wandb: bool = True                                    # Enable/disable wandb
    wandb_project: str = "GNN_Gym_MoNbTaW_fairchem_more_data"          # Wandb project name
    wandb_entity: str = None                                  # Wandb entity
    wandb_run_name: str = None                                # Run name
    wandb_tags: List[str] = field(default_factory=list)      # Tags
    wandb_notes: str = "Fixed: Line graph batching, residual connections, normalized aggregation, higher LR"
    wandb_log_interval: int = 1                               # Log every N epochs
    wandb_watch_model: bool = True                            # Watch model gradients
    wandb_watch_freq: int = 10                                # Watch frequency
    
    def get_model_name(self, n_samples: int = None, cycle: int = None) -> str:
        """Generate a descriptive model name based on configuration."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mlp_str = "-".join(map(str, self.mlp_hidden_dims))
        scheduler = self.scheduler_type if self.use_scheduler else "none"
        
        parts = [timestamp]
        
        if n_samples is not None:
            parts.append(f"samples{n_samples}")
        
        if cycle is not None:
            parts.append(f"cycle{cycle}")
        
        parts.append(f"GNN-{self.gnn_num_layers}x{self.gnn_hidden_dim}")
        parts.append(f"MLP-{mlp_str}")
        parts.append(scheduler)
        
        return "-".join(parts)
    
    def get_experiment_name(self, n_samples: int = None, cycle: int = None) -> str:
        """Generate experiment name for wandb with timestamp, dataset size, and cycle."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [timestamp]
        
        if n_samples is not None:
            parts.append(f"samples{n_samples}")
        
        if cycle is not None:
            parts.append(f"cycle{cycle}")
        
        parts.append("GNN-FIXED")  # Mark as fixed version
        
        return "-".join(parts)


# Display config
if __name__ == "__main__":
    print("="*70)
    print("DEFAULT CONFIGURATION (FIXED)")
    print("="*70)
    
    config = Config()
    
    print("\n✅ FIXES APPLIED:")
    print(f"  - Learning rate increased: 5e-5 → {config.learning_rate}")
    print(f"  - Loss function changed: mse → {config.loss_function}")
    print(f"  - Cosine warm restart eta_min: 5e-5 → {config.warm_restart_eta_min}")
    
    print("\nPATHS:")
    print(f"  csv_path: {config.csv_path}")
    print(f"  checkpoint_dir: {config.checkpoint_dir}")
    print(f"  database_dir: {config.database_dir}")
    
    print("\nCALCULATOR:")
    print(f"  calculator: {config.calculator}")
    if config.calculator == "fairchem":
        print(f"  fairchem_model: {config.fairchem_model}")
    elif config.calculator == "chgnet":
        print(f"  chgnet_model: {config.chgnet_model}")
    
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
    print(f"  use_line_graph: {config.use_line_graph}")
    print(f"  line_graph_cutoff: {config.line_graph_cutoff}")
    
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
    print(f"  use_line_graph: {config.use_line_graph}")
    print(f"  line_graph_hidden_dim: {config.line_graph_hidden_dim}")
    print(f"  line_graph_num_layers: {config.line_graph_num_layers}")
    print(f"  mlp_hidden_dims: {config.mlp_hidden_dims}")
    print(f"  dropout: {config.dropout}")
    
    print("\nTRAINING:")
    print(f"  learning_rate: {config.learning_rate} ✅ INCREASED")
    print(f"  weight_decay: {config.weight_decay}")
    print(f"  gradient_clip_norm: {config.gradient_clip_norm}")
    print(f"  epochs: {config.epochs}")
    print(f"  patience: {config.patience}")
    print(f"  final_model_patience: {config.final_model_patience}")
    print(f"  save_interval: {config.save_interval}")
    print(f"  loss_function: {config.loss_function} ✅ CHANGED")
    
    print("\nLEARNING RATE SCHEDULER:")
    print(f"  use_scheduler: {config.use_scheduler}")
    print(f"  scheduler_type: {config.scheduler_type}")
    if config.scheduler_type == "plateau":
        print(f"  plateau_factor: {config.plateau_factor}")
        print(f"  plateau_patience: {config.plateau_patience}")
    elif config.scheduler_type == "step":
        print(f"  step_size: {config.step_size}")
        print(f"  step_gamma: {config.step_gamma}")
    elif config.scheduler_type == "cosine":
        print(f"  cosine_t_max: {config.cosine_t_max}")
        print(f"  cosine_eta_min: {config.cosine_eta_min}")
    elif config.scheduler_type == "cosine_warm_restarts":
        print(f"  warm_restart_t_0: {config.warm_restart_t_0}")
        print(f"  warm_restart_t_mult: {config.warm_restart_t_mult}")
        print(f"  warm_restart_eta_min: {config.warm_restart_eta_min} ✅ INCREASED")
        print(f"  warm_restart_decay: {config.warm_restart_decay}")
    
    print("\n" + "="*70)