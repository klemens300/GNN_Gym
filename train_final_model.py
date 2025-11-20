"""
Train Final Model - Standalone Training Script

Train a single GNN model on existing dataset without active learning.
Perfect for testing different architectures and hyperparameters.

Usage:
    python train_final_model.py
"""

import sys
import traceback
import logging
import torch
from pathlib import Path
from datetime import datetime

# Project imports
from config_final_model import ConfigFinalModel
from dataset import create_dataloaders
from template_graph_builder import TemplateGraphBuilder
from model import create_model_from_config, count_parameters
from trainer import Trainer
from utils import get_node_input_dim

# Optional Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# Setup Logger
# ============================================================================

def setup_logger(config: ConfigFinalModel):
    """Setup main logger for training."""
    logger = logging.getLogger("final_model_training")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    logger.handlers = []
    
    # File handler
    log_file = Path(config.log_dir) / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, config.log_level.upper()))
    
    # Console handler
    if config.log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, config.log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    if config.log_to_console:
        ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    if config.log_to_console:
        logger.addHandler(ch)
    
    return logger


# ============================================================================
# Get Database Statistics
# ============================================================================

def get_database_stats(csv_path: str) -> dict:
    """Get statistics from CSV database."""
    import pandas as pd
    
    p = Path(csv_path)
    if not p.exists():
        return {"n_samples": 0, "n_compositions": 0}
    
    try:
        df = pd.read_csv(p)
    except Exception:
        return {"n_samples": 0, "n_compositions": 0}
    
    return {
        "n_samples": len(df),
        "n_compositions": df['composition_string'].nunique() if 'composition_string' in df.columns else 0
    }


# ============================================================================
# Main Training Function
# ============================================================================

def train_final_model(config: ConfigFinalModel, logger: logging.Logger):
    """
    Train a single model on the complete dataset.
    
    Workflow:
    1. Load data from CSV
    2. Create dataloaders (train/val split)
    3. Build model
    4. Train with early stopping
    5. Save best model
    
    Args:
        config: ConfigFinalModel instance
        logger: Logger instance
    """
    logger.info("="*70)
    logger.info("FINAL MODEL TRAINING")
    logger.info("="*70)
    
    # Get database statistics
    db_stats = get_database_stats(config.csv_path)
    logger.info(f"Database: {config.csv_path}")
    logger.info(f"  Samples: {db_stats['n_samples']}")
    logger.info(f"  Compositions: {db_stats['n_compositions']}")
    
    if db_stats['n_samples'] == 0:
        raise RuntimeError(f"No data found in {config.csv_path}")
    
    # Store dataset size for wandb naming
    config._current_dataset_size = db_stats['n_samples']
    
    # Create output directory
    outdir = Path(config.checkpoint_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {outdir}")
    
    try:
        # ====================================================================
        # 1. CREATE DATALOADERS
        # ====================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*70)
        
        result = create_dataloaders(config)
        
        if result is None:
            raise RuntimeError("create_dataloaders returned None")
        
        # Handle both tuple and single return value
        if isinstance(result, tuple) and len(result) == 2:
            train_loader, val_loader = result
        else:
            train_loader = result
            val_loader = None
        
        if train_loader is None:
            raise RuntimeError("train_loader is None - no training data available")
        
        logger.info(f"✓ Dataloaders created")
        logger.info(f"  Train batches: {len(train_loader)}")
        if val_loader is not None:
            logger.info(f"  Val batches: {len(val_loader)}")
        
        # ====================================================================
        # 2. BUILD MODEL
        # ====================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 2: BUILDING MODEL")
        logger.info("="*70)
        
        builder = TemplateGraphBuilder(config)
        node_input_dim = get_node_input_dim(builder)
        
        logger.info(f"Node input dimension: {node_input_dim}")
        logger.info(f"  Position features: 3")
        logger.info(f"  Element one-hot: {len(builder.elements)}")
        logger.info(f"  Atomic properties: 4")
        
        model = create_model_from_config(config, node_input_dim)
        
        # Count parameters
        params = count_parameters(model)
        logger.info(f"\nModel architecture:")
        logger.info(f"  GNN layers: {config.gnn_num_layers}")
        logger.info(f"  GNN hidden dim: {config.gnn_hidden_dim}")
        logger.info(f"  GNN embedding dim: {config.gnn_embedding_dim}")
        logger.info(f"  MLP hidden dims: {config.mlp_hidden_dims}")
        logger.info(f"  Dropout: {config.dropout}")
        logger.info(f"\nParameter counts:")
        logger.info(f"  Encoder: {params['encoder']:,}")
        logger.info(f"  Predictor: {params['predictor']:,}")
        logger.info(f"  Total: {params['total']:,}")
        
        # ====================================================================
        # 3. TRAIN MODEL
        # ====================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 3: TRAINING")
        logger.info("="*70)
        
        trainer = Trainer(
            model,
            config,
            save_dir=str(outdir),
            cycle=None,
            is_final_model=False  # Use regular patience from config
        )
        
        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {config.epochs}")
        logger.info(f"  Patience: {config.patience}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Weight decay: {config.weight_decay}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Scheduler: {config.scheduler_type if config.use_scheduler else 'None'}")
        logger.info(f"  Loss function: {config.loss_function}")
        
        # Train
        history = trainer.train(train_loader, val_loader, verbose=True)
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED")
        logger.info("="*70)
        logger.info(f"Best validation MAE: {trainer.best_val_mae:.4f} eV")
        logger.info(f"Best validation Rel MAE: {trainer.best_val_rel_mae:.4f}")
        logger.info(f"Best training MAE: {trainer.best_train_mae:.4f} eV")
        logger.info(f"Final epoch: {trainer.current_epoch}")
        logger.info(f"Best model saved: {outdir / 'best_model.pt'}")
        logger.info("="*70)
        
        return history
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        raise


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    # Load config
    config = ConfigFinalModel()
    
    # Setup logger
    logger = setup_logger(config)
    
    logger.info("="*70)
    logger.info("FINAL MODEL TRAINING SCRIPT")
    logger.info("="*70)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    # Print configuration
    logger.info("\nCONFIGURATION:")
    logger.info("-"*70)
    for key, val in config.__dict__.items():
        if not key.startswith('_'):
            logger.info(f"  {key:30s}: {val}")
    logger.info("="*70)
    
    try:
        # Train model
        history = train_final_model(config, logger)
        
        logger.info("\n" + "="*70)
        logger.info("SUCCESS!")
        logger.info("="*70)
        logger.info(f"Training completed successfully")
        logger.info(f"Checkpoints saved in: {config.checkpoint_dir}")
        logger.info(f"Logs saved in: {config.log_dir}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("TRAINING FAILED")
        logger.error("="*70)
        logger.error(f"Error: {e}")
        logger.error("="*70)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()