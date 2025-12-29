"""
Configuration for Diffusion Barrier Prediction

Auto-generates paths based on element list for easy system switching.
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime
from pathlib import Path


@dataclass
class Config:
    
    # ============================================================
    # MATERIAL SYSTEM (Elements)
    # Define elements here - all paths will be auto-generated
    # ============================================================
    elements: List[str] = field(default_factory=lambda: ['Mo', 'Nb', 'Ta', 'W'])
    
    # ============================================================
    # BASE DIRECTORIES
    # Only these need to be changed for different file systems
    # ============================================================
    base_database_dir: str = "home/klemens/databases/MoNbTaW"
    base_production_dir: str = "home/klemens/databases/MoNbTaW"
    
    # ============================================================
    # AUTO-GENERATED PATHS (do not modify directly)
    # These are computed from elements and base directories
    # ============================================================
    def __post_init__(self):
        """
        Auto-generate all paths based on elements.
        System name is created by alphabetically sorting elements.
        Example: ['Mo', 'Nb', 'Cr', 'V'] -> 'CrMoNbV'
        """
        # Generate system name from alphabetically sorted elements
        self.system_name = "".join(sorted(self.elements))
        
        # Training data paths (main CSV that grows during active learning)
        self.csv_path = f"{self.base_database_dir}/{self.system_name}.csv"
        self.database_dir = f"{self.base_database_dir}/{self.system_name}"
        
        # Test data paths (fixed test set, separate from training)
        self.al_test_set_csv = f"{self.base_database_dir}/{self.system_name}_test.csv"
        self.al_test_set_dir = f"{self.base_database_dir}/{self.system_name}_test"
        
        # Model checkpoint directory
        self.checkpoint_dir = f"{self.base_production_dir}/{self.system_name}/GNN_Gym/checkpoints"
        
        # Results directory
        self.al_results_dir = f"{self.base_production_dir}/{self.system_name}/GNN_Gym/active_learning_results"
        
        # Log directory
        self.log_dir = f"{self.base_production_dir}/{self.system_name}/GNN_Gym/logs"

    # ============================================================
    # TRAINING MODE
    # ============================================================

    # ============================================================
    # BARRIER VALIDATION (Oracle)
    # ============================================================
    barrier_min_cutoff: float = 0.0    # Minimum barrier to save (eV)
    barrier_max_cutoff: float = 5.0    # Maximum barrier to save (eV)


    train_only_mode: bool = True
    train_only_skip_cycles: bool = True
    
    # ============================================================
    # CALCULATOR SETTINGS
    # ============================================================
    calculator: str = "fairchem"
    fairchem_model: str = "uma-m-1p1"
    chgnet_model: str = "0.3.0"
    
    # ============================================================
    # CRYSTAL STRUCTURE
    # ============================================================
    supercell_size: int = 4
    lattice_parameter: float = 3.2
    
    # ============================================================
    # GRAPH CONSTRUCTION
    # ============================================================
    cutoff_radius: float = 3.5
    max_neighbors: int = 50
    use_line_graph: bool = True
    line_graph_cutoff: float = 3.5 

    # ============================================================
    # DATA 
    # ============================================================
    batch_size: int = 128            
    batch_size_val: int = 128         
    num_workers: int = 16              
    
    # Data cleanup (barrier filtering)
    min_barrier: float = 0.0
    max_barrier: float = 5.0
    
    # Train/Val split
    val_split: float = 0.1
    random_seed: int = 42
    
    # ============================================================
    # ATOM EMBEDDINGS
    # ============================================================
    # Learned embeddings for each element type
    atom_embedding_dim: int = 32           # Dimension of learned element embeddings
    use_atomic_properties: bool = True      # Concatenate physical properties to embeddings
    
    # ============================================================
    # MODEL ARCHITECTURE
    # ============================================================
    
    # Atom Graph (GNN Encoder)
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    gnn_embedding_dim: int = 64

    # Line Graph (for bond angles)
    use_line_graph: bool = True
    line_graph_hidden_dim: int = gnn_hidden_dim
    line_graph_num_layers: int = gnn_num_layers
    line_graph_embedding_dim: int = gnn_embedding_dim

    # MLP Predictor
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1
    
    # ============================================================
    # TRAINING
    # ============================================================
    # Optimization
    learning_rate: float = 1e-3    
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Training loop
    epochs: int = 10000
    patience: int = 120
    save_interval: int = 50
    final_model_patience: int = 200
    
    # ============================================================
    # OPTIMIZER
    # ============================================================
    use_fused_optimizer: bool = True 
    
    # ============================================================
    # MIXED PRECISION
    # ============================================================
    use_amp: bool = True     
    
    # ============================================================
    # MODEL COMPILATION
    # ============================================================
    compile_model: bool = False     
    compile_mode: str = 'reduce-overhead' 
    
    # ============================================================
    # CUDNN OPTIMIZATION
    # ============================================================
    cudnn_benchmark: bool = True 
    
    # ============================================================
    # DATALOADER OPTIMIZATION
    # ============================================================
    prefetch_factor: int = 4          
    drop_last: bool = True            
    
    # ============================================================
    # LEARNING RATE SCHEDULER
    # ============================================================
    use_scheduler: bool = True
    scheduler_type: str = "cosine_warm_restarts"
    
    # ReduceLROnPlateau parameters
    plateau_factor: float = 0.5
    plateau_patience: int = 10
    
    # StepLR parameters
    step_size: int = 100
    step_gamma: float = 0.5
    
    # CosineAnnealingLR parameters
    cosine_t_max: int = 100
    cosine_eta_min: float = 1e-6
    
    # CosineAnnealingWarmRestarts parameters
    warm_restart_t_0: int = 500
    warm_restart_t_mult: float = 1.2
    warm_restart_eta_min: float = 1e-4  
    warm_restart_decay: float = 0.9
    
    # ============================================================
    # NEB PARAMETERS
    # ============================================================
    neb_images: int = 3
    neb_spring_constant: float = 5.0
    neb_fmax: float = 0.1
    neb_max_steps: int = 500
    neb_climb: bool = True
    neb_method: str = "aseneb"

    # ============================================================
    # STRUCTURE RELAXATION
    # ============================================================
    relax_cell: bool = False
    relax_fmax: float = 0.1
    relax_steps: int = 500
    relax_max_steps: int = 500
    
    # ============================================================
    # ACTIVE LEARNING - FIXED TEST SET
    # ============================================================
    # Test set (fixed for all cycles, stored separately from training data)
    al_test_set_size: int = 1000
    al_test_set_strategy: str = "uniform"  # "uniform", "sobol", or "grid"
    
    # Training set (grows with each cycle via query samples)
    al_initial_training_samples: int = 5000
    al_n_query: int = 2500  # Number of query samples added to training set each cycle
    
    # Active Learning Loop
    al_query_strategy: str = 'error_weighted'
    al_max_cycles: int = 20
    al_seed: int = 42
    al_convergence_check: bool = True
    al_convergence_metric: str = "mae"
    al_convergence_threshold_mae: float = 0.01
    al_convergence_threshold_rel_mae: float = 0.1
    al_convergence_patience: int = 2000
    
    # ============================================================
    # LOGGING
    # ============================================================
    log_level: str = "INFO"
    log_to_console: bool = True
    
    # ============================================================
    # LOSS FUNCTION
    # ============================================================
    loss_function: str = "huber"
    
    # ============================================================
    # WEIGHTS & BIASES
    # ============================================================
    use_wandb: bool = True
    wandb_project: str = None  # Auto-set to system_name if None
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: str = "."
    wandb_log_interval: int = 1
    wandb_watch_model: bool = True
    wandb_watch_freq: int = 10
    
    def get_model_name(self, n_samples: int = None, cycle: int = None) -> str:
        """Generate a descriptive model name based on configuration."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mlp_str = "-".join(map(str, self.mlp_hidden_dims))
        scheduler = self.scheduler_type if self.use_scheduler else "none"
        
        # Use system name
        parts = [timestamp, self.system_name, "barrier"]
        
        if n_samples is not None:
            parts.append(f"N{n_samples}")
        
        if cycle is not None:
            parts.append(f"C{cycle}")
        
        parts.append(f"B{self.batch_size}")
        parts.append(f"GNN-{self.gnn_num_layers}x{self.gnn_hidden_dim}")
        parts.append(f"MLP-{mlp_str}")
        parts.append(scheduler)
        
        if self.use_amp:
            parts.append("AMP")
        if self.compile_model:
            parts.append("COMPILED")
        
        return "-".join(parts)
    
    def get_experiment_name(self, n_samples: int = None, cycle: int = None) -> str:
        """Generate experiment name for wandb."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        parts = [timestamp, self.system_name, "barrier"]
        
        if n_samples is not None:
            parts.append(f"samples{n_samples}")
        
        if cycle is not None:
            parts.append(f"cycle{cycle}")
        
        return "-".join(parts)
    
    def print_paths(self):
        """Print all auto-generated paths for verification."""
        print("="*70)
        print("AUTO-GENERATED PATHS")
        print("="*70)
        print(f"System: {self.system_name}")
        print(f"Elements: {self.elements}")
        print()
        print("Training Data:")
        print(f"  CSV: {self.csv_path}")
        print(f"  Dir: {self.database_dir}")
        print()
        print("Test Data:")
        print(f"  CSV: {self.al_test_set_csv}")
        print(f"  Dir: {self.al_test_set_dir}")
        print()
        print("Model & Results:")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Results: {self.al_results_dir}")
        print(f"  Logs: {self.log_dir}")
        print("="*70)


# Display config
if __name__ == "__main__":
    print("="*70)
    print("CONFIG TEST")
    print("="*70)
    
    # Test 1: MoNbCrV system
    print("\nTest 1: MoNbCrV")
    config1 = Config(elements=['Mo', 'Nb', 'Cr', 'V'])
    config1.print_paths()
    
    # Test 2: MoNbTaWTi system
    print("\nTest 2: MoNbTaWTi")
    config2 = Config(elements=['Mo', 'Nb', 'Ta', 'W', 'Ti'])
    config2.print_paths()
    
    # Test 3: Binary system
    print("\nTest 3: MoW")
    config3 = Config(elements=['Mo', 'W'])
    config3.print_paths()
    
    print("\nExample model names:")
    print(f"  Cycle 0: {config1.get_model_name(n_samples=5000, cycle=0)}")
    print(f"  Cycle 5: {config1.get_model_name(n_samples=10000, cycle=5)}")
    print(f"  Final:   {config1.get_model_name(n_samples=25000)}")
    
    print("\n" + "="*70)