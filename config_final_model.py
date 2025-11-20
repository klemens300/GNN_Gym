"""
Configuration for Final Model Training

Simplified config without Active Learning parameters.
Focus on model architecture, training, and logging.
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class ConfigFinalModel:
    """Configuration for training a single model on existing data"""
    
    # ============================================================
    # PATHS
    # ============================================================
    csv_path: str = "MoNbTaW.csv"
    checkpoint_dir: str = "checkpoints_final_model"
    database_dir: str = "MoNbTaW"
    
    # ============================================================
    # MATERIAL SYSTEM
    # ============================================================
    elements: List[str] = field(default_factory=lambda: ['Mo', 'Nb', 'Ta', 'W'])
    
    # ============================================================
    # CRYSTAL STRUCTURE (needed for template graph building)
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
    batch_size: int = 32
    num_workers: int = 0
    
    # Data cleanup (barrier filtering)
    min_barrier: float = 0.1
    max_barrier: float = 5.0
    
    # Train/Val split
    val_split: float = 0.1
    random_seed: int = 42
    
    # ============================================================
    # MODEL ARCHITECTURE
    # ============================================================
    # GNN Encoder
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 5
    gnn_embedding_dim: int = 128
    
    # MLP Predictor
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [2048, 1024, 512])
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
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine_warm_restarts"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    scheduler_step_size: int = 100
    scheduler_t_max: int = 100
    scheduler_eta_min: float = 1e-6
    
    # Cosine Warm Restarts specific
    scheduler_t_0: int = 100
    scheduler_t_mult: int = 1.2
    scheduler_restart_decay: float = 0.9
    
    # ============================================================
    # LOSS FUNCTION
    # ============================================================
    loss_function: str = "mse"
    
    # ============================================================
    # LOGGING (File Logging)
    # ============================================================
    log_dir: str = "logs_final_model"
    log_level: str = "INFO"
    log_to_console: bool = True
    
    # ============================================================
    # LOGGING (Weights & Biases)
    # ============================================================
    use_wandb: bool = True
    wandb_project: str = "GNN_Final_Model_MoNbTaW"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: str = "Training final model on complete dataset"
    wandb_log_interval: int = 1
    wandb_watch_model: bool = True
    wandb_watch_freq: int = 10
    
    def get_model_name(self, n_samples: int = None) -> str:
        """
        Generate a descriptive model name based on configuration.
        
        Format: {timestamp}-samples{n}-GNN-{layers}x{hidden}-MLP-{mlp_structure}-{scheduler}
        
        Args:
            n_samples: Number of training samples (optional)
        
        Returns:
            model_name: Descriptive model name
        
        Example:
            "20241030-143522-samples2500-GNN-5x64-MLP-1024-512-256-cosine"
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        mlp_str = "-".join(map(str, self.mlp_hidden_dims))
        scheduler = self.scheduler_type if self.use_scheduler else "none"
        
        parts = [timestamp]
        
        if n_samples is not None:
            parts.append(f"samples{n_samples}")
        
        parts.append(f"GNN-{self.gnn_num_layers}x{self.gnn_hidden_dim}")
        parts.append(f"MLP-{mlp_str}")
        parts.append(scheduler)
        
        model_name = "-".join(parts)
        
        return model_name
    
    def get_experiment_name(self, n_samples: int = None, cycle: int = None) -> str:
        """
        Generate experiment name for wandb with timestamp and dataset size.
        
        Format: {date}-{time}-samples{n}-GNN
        
        Args:
            n_samples: Number of training samples (optional)
            cycle: Ignored (for compatibility with Trainer)
        
        Returns:
            experiment_name: Simple experiment name with timestamp
        
        Example:
            "20241030-143522-samples2500-GNN"
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [timestamp]
        
        if n_samples is not None:
            parts.append(f"samples{n_samples}")
        
        parts.append("GNN")
        
        return "-".join(parts)


# Display config
if __name__ == "__main__":
    print("="*70)
    print("FINAL MODEL CONFIGURATION")
    print("="*70)
    
    config = ConfigFinalModel()
    
    print("\nPATHS:")
    print(f"  csv_path: {config.csv_path}")
    print(f"  checkpoint_dir: {config.checkpoint_dir}")
    print(f"  database_dir: {config.database_dir}")
    
    print("\nMATERIAL SYSTEM:")
    print(f"  elements: {config.elements}")
    
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
    if config.scheduler_type == "cosine_warm_restarts":
        print(f"  scheduler_t_0: {config.scheduler_t_0}")
        print(f"  scheduler_t_mult: {config.scheduler_t_mult}")
        print(f"  scheduler_restart_decay: {config.scheduler_restart_decay}")
        print(f"  scheduler_eta_min: {config.scheduler_eta_min}")
    
    print("\nLOSS FUNCTION:")
    print(f"  loss_function: {config.loss_function}")
    
    print("\nLOGGING:")
    print(f"  log_dir: {config.log_dir}")
    print(f"  log_level: {config.log_level}")
    print(f"  log_to_console: {config.log_to_console}")
    
    print("\nWANDB:")
    print(f"  use_wandb: {config.use_wandb}")
    print(f"  wandb_project: {config.wandb_project}")
    print(f"  wandb_entity: {config.wandb_entity}")
    print(f"  wandb_notes: {config.wandb_notes}")
    
    print("\nGENERATED NAMES:")
    print(f"  Model name: {config.get_model_name(n_samples=2500)}")
    print(f"  Experiment name: {config.get_experiment_name(n_samples=2500)}")
    
    print("\n" + "="*70)