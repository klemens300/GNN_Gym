"""
Configuration for Diffusion Barrier Prediction
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class Config:
    
    # ============================================================
    # PATHS
    # ============================================================
    csv_path: str = "/mulfs/home/p2467946/Diffusion_barrier_GNN/databases/MoNbCrV.csv"
    checkpoint_dir: str = "/mulfs/home/p2467946/Diffusion_barrier_GNN/production/MoNbCrV/GNN_Gym/checkpoints"
    database_dir: str = "/mulfs/home/p2467946/Diffusion_barrier_GNN/databases/MoNbCrV"

    # ============================================================
    # TRAINING MODE
    # ============================================================
    train_only_mode: bool = False
    train_only_skip_cycles: bool = False
    
    # ============================================================
    # CALCULATOR SETTINGS
    # ============================================================
    calculator: str = "fairchem"
    fairchem_model: str = "uma-m-1p1"
    chgnet_model: str = "0.3.0"
    
    # ============================================================
    # MATERIAL SYSTEM (Elements)
    # ============================================================
    elements: List[str] = field(default_factory=lambda: ['Mo', 'Nb', 'Cr', 'V'])
    
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
    max_barrier: float = 50.0
    
    # Train/Val split
    val_split: float = 0.1
    random_seed: int = 42
    
    # ============================================================
    # MODEL ARCHITECTURE - 
    # ============================================================
    
    # Atom Graph (GNN Encoder)
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_embedding_dim: int = 128

    # Line Graph (for bond angles)
    use_line_graph: bool = True
    line_graph_hidden_dim: int = 128
    line_graph_num_layers: int = 3
    line_graph_embedding_dim: int = 128

    # MLP Predictor
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    dropout: float = 0.1
    
    # ============================================================
    # TRAINING - 
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
    # LEARNING RATE SCHEDULER - 
    # ============================================================
    use_scheduler: bool = True
    scheduler_type: str = "cosine_warm_restarts"
    
    # --- ReduceLROnPlateau ---
    plateau_factor: float = 0.5
    plateau_patience: int = 10
    
    # --- StepLR ---
    step_size: int = 100
    step_gamma: float = 0.5
    
    # --- CosineAnnealingLR ---
    cosine_t_max: int = 100
    cosine_eta_min: float = 1e-6
    
    # --- CosineAnnealingWarmRestarts --- 
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
    # Test set (FIXED for all cycles, separate CSV)
    al_test_set_size: int = 1000
    al_test_set_strategy: str = "uniform"  # "uniform", "sobol", or "grid"
    al_test_set_csv: str = "/mulfs/home/p2467946/Diffusion_barrier_GNN/databases/MoNbCrV_test.csv"
    al_test_set_dir: str = "/mulfs/home/p2467946/Diffusion_barrier_GNN/databases/MoNbCrV_test"
    
    # Training set (grows with each cycle, MAIN CSV)
    al_initial_training_samples: int = 1000
    al_n_query: int = 250  # Added to TRAINING set each cycle
    
    # Active Learning Loop
    al_query_strategy: str = 'error_weighted'
    al_max_cycles: int = 20
    al_seed: int = 42
    al_convergence_check: bool = True
    al_convergence_metric: str = "mae"
    al_convergence_threshold_mae: float = 0.01
    al_convergence_threshold_rel_mae: float = 0.1
    al_convergence_patience: int = 2000
    al_results_dir: str = "active_learning_results"
    
    # ============================================================
    # LOGGING
    # ============================================================
    log_dir: str = "logs"
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
    wandb_project: str = "MoNbCrV"
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
        
        # Element string for identification
        elements_str = "".join(sorted(self.elements))
        
        # Start with timestamp, elements, and "barrier"
        parts = [timestamp, elements_str, "barrier"]
        
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
        
        # Element string
        elements_str = "".join(sorted(self.elements))
        
        parts = [timestamp, elements_str, "barrier"]
        
        if n_samples is not None:
            parts.append(f"samples{n_samples}")
        
        if cycle is not None:
            parts.append(f"cycle{cycle}")
        
        return "-".join(parts)


# Display config
if __name__ == "__main__":
    print("="*70)
    print("CONFIG")
    print("="*70)
    
    config = Config()
    
    print("\nExample model names:")
    print(f"  Cycle 0: {config.get_model_name(n_samples=5000, cycle=0)}")
    print(f"  Cycle 5: {config.get_model_name(n_samples=10000, cycle=5)}")
    print(f"  Final:   {config.get_model_name(n_samples=25000)}")
    
    print("\n" + "="*70)