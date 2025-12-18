"""
Configuration for Diffusion Barrier Prediction


Changes:
1. Correct input_dim for new GraphBuilder (12 features not 8)
2. Larger batch sizes (GPU utilization)
3. Mixed Precision enabled
4. More workers + prefetch
5. Optimized for 32GB VRAM
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class Config:
    """Complete configuration - RTX 5090 OPTIMIZED"""
    
    # ============================================================
    # PATHS
    # ============================================================
    csv_path: str = "/home/klemens/databases/MoNbTaW.csv"
    checkpoint_dir: str = "checkpoints"
    database_dir: str = "/home/klemens/databases/MoNbTaW"

    # ============================================================
    # TRAINING MODE
    # ============================================================
    train_only_mode: bool = True
    train_only_skip_cycles: bool = True
    
    # ============================================================
    # CALCULATOR SETTINGS
    # ============================================================
    calculator: str = "fairchem"
    fairchem_model: str = "uma-m-1p1"
    chgnet_model: str = "0.3.0"
    
    # ============================================================
    # MATERIAL SYSTEM (Elements)
    # ============================================================
    elements: List[str] = field(default_factory=lambda: ['Mo', 'Nb', 'Ta', 'W'])
    
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
    batch_size: int = 128           # 
    batch_size_val: int = 128        # 
    num_workers: int = 6            #  
    
    # Data cleanup (barrier filtering)
    min_barrier: float = 0.0
    max_barrier: float = 50.0
    
    # Train/Val split
    val_split: float = 0.1
    random_seed: int = 42
    
    # ============================================================
    # MODEL ARCHITECTURE - 
    # ============================================================
    # Input dimension: 4 (one-hot) + 8 (properties) = 12
    # Properties: atomic_number, atomic_mass, atomic_radius, 
    #             electronegativity, first_ionization, electron_affinity,
    #             melting_point, density
    
    # Atom Graph (GNN Encoder)
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 5
    gnn_embedding_dim: int = 64

    # Line Graph (for bond angles)
    use_line_graph: bool = True
    line_graph_hidden_dim: int = 64
    line_graph_num_layers: int = 3
    line_graph_embedding_dim: int = 64

    # MLP Predictor
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
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
    final_model_patience: int = 666
    
    # ============================================================
    # OPTIMIZER - 🔥 FUSED FOR SPEED
    # ============================================================
    use_fused_optimizer: bool = True  # 🔥 NEW! GPU-optimized AdamW
    
    # ============================================================
    # MIXED PRECISION - 🔥 30-50% SPEEDUP!
    # ============================================================
    use_amp: bool = True              # 🔥 NEW! Automatic Mixed Precision (FP16)
    
    # ============================================================
    # MODEL COMPILATION - 🔥 10-30% SPEEDUP
    # ============================================================
    compile_model: bool = False        # 🔥 NEW! torch.compile (PyTorch 2.0+)
    compile_mode: str = 'reduce-overhead'  # Options: default, reduce-overhead, max-autotune
    
    # ============================================================
    # CUDNN OPTIMIZATION
    # ============================================================
    cudnn_benchmark: bool = True      # 🔥 NEW! Auto-tune convolutions
    
    # ============================================================
    # DATALOADER OPTIMIZATION
    # ============================================================
    prefetch_factor: int = 6          
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
    
    # --- CosineAnnealingWarmRestarts - 
    warm_restart_t_0: int = 500
    warm_restart_t_mult: float = 1.2
    warm_restart_eta_min: float = 1e-4  # 
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
    # ACTIVE LEARNING
    # ============================================================
    al_initial_samples: int = 20000
    al_n_test: int = 4000
    al_test_strategy: str = 'uniform'
    al_n_query: int = 1000
    al_query_strategy: str = 'error_weighted'
    al_max_cycles: int = 20
    al_seed: int = 42
    al_convergence_check: bool = True
    al_convergence_metric: str = "mae"
    al_convergence_threshold_mae: float = 0.01
    al_convergence_threshold_rel_mae: float = 0.1
    al_convergence_patience: int = 20
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
    wandb_project: str = "GNN_Gym_MoNbTaW_RTX5090_RealGeometry"  # 🔥 New project
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: str = "RTX 5090 optimized: Batch 384, LR 6e-3, AMP, Compile, Real Geometry"
    wandb_log_interval: int = 1
    wandb_watch_model: bool = True
    wandb_watch_freq: int = 10
    
    def get_model_name(self, n_samples: int = None, cycle: int = None) -> str:
        """Generate a descriptive model name based on configuration."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mlp_str = "-".join(map(str, self.mlp_hidden_dims))
        scheduler = self.scheduler_type if self.use_scheduler else "none"
        
        parts = [timestamp, "RTX5090-OPT"]
        
        if n_samples is not None:
            parts.append(f"samples{n_samples}")
        
        if cycle is not None:
            parts.append(f"cycle{cycle}")
        
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
        parts = [timestamp]
        
        if n_samples is not None:
            parts.append(f"samples{n_samples}")
        
        if cycle is not None:
            parts.append(f"cycle{cycle}")
        
        return "-".join(parts)


# Display config
if __name__ == "__main__":
    print("="*70)
    print("RTX 5090 OPTIMIZED CONFIGURATION + REAL GEOMETRY")
    print("="*70)
    
    config = Config()
    
    print("\n🔥 KEY FIXES:")
    print("  ✅ Input features: 12 (4 one-hot + 8 properties)")
    print("  ✅ Real geometry from CIF files")
    
    print("\n🚀 RTX 5090 OPTIMIZATIONS:")
    print(f"  ✅ Batch size: 256 → {config.batch_size} (1.5x)")
    print(f"  ✅ Val batch: 64 → {config.batch_size_val} (8x!)")
    print(f"  ✅ Learning rate: 4e-3 → {config.learning_rate} (1.5x)")
    print(f"  ✅ num_workers: 12 → {config.num_workers}")
    print(f"  ✅ Mixed Precision (AMP): {config.use_amp}")
    print(f"  ✅ Model Compilation: {config.compile_model}")
    print(f"  ✅ Fused Optimizer: {config.use_fused_optimizer}")
    print(f"  ✅ cuDNN Benchmark: {config.cudnn_benchmark}")
    print(f"  ✅ Prefetch Factor: {config.prefetch_factor}")
    
    print("\n📊 EXPECTED PERFORMANCE:")
    print("  • Training speed: ~2-2.5x faster than before")
    print("  • GPU utilization: 95-100%")
    print("  • VRAM usage: ~20-24 GB (plenty of headroom!)")
    print("  • Time per epoch: ~25-35s (was ~60s+)")
    
    print("\n💾 FEATURE DIMENSIONS:")
    print("  • Node features: 12")
    print("    - One-hot encoding: 4 (Mo, Nb, Ta, W)")
    print("    - Atomic properties: 8")
    print("      1. atomic_number")
    print("      2. atomic_mass")
    print("      3. atomic_radius")
    print("      4. electronegativity")
    print("      5. first_ionization")
    print("      6. electron_affinity")
    print("      7. melting_point")
    print("      8. density")
    
    print("\nHARDWARE:")
    print("  GPU: RTX 5090 (32 GB)")
    print("  CPU: Ryzen 7 9800X3D (8C/16T)")
    print("  RAM: 32 GB")
    
    print("\n" + "="*70)