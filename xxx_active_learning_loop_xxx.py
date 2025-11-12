# === Active Learning Loop with Convergence Checking and Final Model Training ===
# Console: English; Comments: English

import sys
import traceback
import logging
from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
import torch

# --- Project imports ---
from config import Config
from oracle import Oracle
from inference import run_inference_cycle, cleanup_gpu, ConvergenceTracker, save_convergence_history
from trainer import Trainer
from dataset import create_dataloaders
from template_graph_builder import TemplateGraphBuilder
from model import create_model_from_config, count_parameters
from utils import get_node_input_dim, save_model_for_inference

# --- Optional Weights & Biases support ---
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ============================================================================
# Setup Main Logger
# ============================================================================

def setup_main_logger(config: Config):
    """Setup main logger for active learning loop."""
    logger = logging.getLogger("active_learning")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    log_file = Path(config.log_dir) / "active_learning.log"
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
# Helper functions
# ============================================================================

def is_csv_missing_or_empty(csv_path: str) -> bool:
    """Check if the CSV file is missing or empty."""
    p = Path(csv_path)
    if not p.exists():
        return True
    try:
        df = pd.read_csv(p)
        return len(df) == 0
    except Exception:
        return True


def sample_simplex_uniform(n: int, k: int) -> np.ndarray:
    """Sample n points uniformly on a (k-1)-simplex using Dirichlet distribution."""
    return np.random.dirichlet(alpha=np.ones(k), size=n)


def initial_data_creation_if_needed(config: Config, oracle: Oracle, logger: logging.Logger):
    """
    Create an initial dataset if CSV is missing or empty.
    Uses config.al_initial_samples to determine the number of samples.
    """
    csv_path = config.csv_path
    if not is_csv_missing_or_empty(csv_path):
        logger.info(f"Database found: {csv_path}")
        return

    elements = list(getattr(config, "elements", []))
    n_seed = int(getattr(config, "al_initial_samples", 0) or 0)
    if not elements or n_seed <= 0:
        raise RuntimeError(
            "Initial data creation requires 'elements' and 'al_initial_samples' > 0 in config."
        )

    logger.info("="*70)
    logger.info("INITIAL DATA CREATION (CSV empty or not found)")
    logger.info("="*70)
    logger.info(f"Elements: {elements}")
    logger.info(f"Samples to create: {n_seed}")
    logger.info(f"Target CSV: {csv_path}")

    weights = sample_simplex_uniform(n_seed, len(elements))
    compositions = []
    for row in weights:
        comp = {el: float(val) for el, val in zip(elements, row)}
        s = sum(comp.values())
        if abs(s - 1.0) > 1e-12:
            for el in comp:
                comp[el] /= s
        compositions.append(comp)

    successes = 0
    for i, comp in enumerate(compositions, 1):
        try:
            logger.info(f"  [{i}/{n_seed}] Calculating: {comp}")
            ok = oracle.calculate(comp)
            if ok is not False:
                successes += 1
        except Exception as e:
            logger.error(f"   Error at sample {i}: {e}")

    if successes == 0:
        raise RuntimeError("Initial data creation failed: no valid samples added.")
    logger.info(f"Initial data creation completed. {successes} samples added.")


def get_database_stats(csv_path: str) -> dict:
    """Return summary statistics for the current database CSV."""
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


def train_cycle_model(config: Config, cycle: int, logger: logging.Logger, is_final: bool = False) -> dict:
    """Train the model for the given active learning cycle."""
    logger.info("="*70)
    if is_final:
        logger.info("TRAINING FINAL MODEL")
    else:
        logger.info(f"TRAINING MODEL - CYCLE {cycle}")
    logger.info("="*70)

    outdir = Path(config.checkpoint_dir) / ("final_model" if is_final else f"cycle_{cycle}")
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        # Get current database size
        db_stats = get_database_stats(config.csv_path)
        n_samples = db_stats['n_samples']
        
        # Store dataset size in config for wandb naming
        config._current_dataset_size = n_samples
        
        # Create dataloaders - handle return value properly
        result = create_dataloaders(config)
        
        if result is None:
            raise RuntimeError("create_dataloaders returned None - check your CSV path and data")
        
        # Handle both tuple and single return value
        if isinstance(result, tuple) and len(result) == 2:
            train_loader, val_loader = result
        else:
            train_loader = result
            val_loader = None
        
        if train_loader is None:
            raise RuntimeError("train_loader is None - no training data available")
        
        builder = TemplateGraphBuilder(config)
        node_input_dim = get_node_input_dim(builder)
        model = create_model_from_config(config, node_input_dim)
        
        # Load best model from previous cycle if this is final training
        if is_final:
            # Find the cycle with best validation MAE
            best_cycle = None
            best_val_mae = float('inf')
            
            for c in range(cycle + 1):
                cycle_path = Path(config.checkpoint_dir) / f"cycle_{c}" / "best_model.pt"
                if cycle_path.exists():
                    checkpoint = torch.load(cycle_path, map_location='cpu')
                    val_mae = checkpoint.get('best_val_mae', float('inf'))
                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        best_cycle = c
            
            if best_cycle is not None:
                best_model_path = Path(config.checkpoint_dir) / f"cycle_{best_cycle}" / "best_model.pt"
                logger.info(f"Loading best model from cycle {best_cycle} (val MAE: {best_val_mae:.4f})")
                checkpoint = torch.load(best_model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer
        trainer = Trainer(model, config, save_dir=str(outdir), cycle=cycle, is_final_model=is_final)
        trainer.train(train_loader, val_loader, verbose=True)
        
        logger.info(f"Model training completed (Cycle {cycle})" if not is_final else "Final model training completed")
        
        return {
            'best_val_mae': trainer.best_val_mae,
            'best_val_rel_mae': trainer.best_val_rel_mae
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return None


def active_learning_loop(config: Config, logger: logging.Logger):
    """Main active learning loop with convergence checking and final model training."""
    logger.info("="*70)
    logger.info("ACTIVE LEARNING LOOP STARTING")
    logger.info("="*70)
    logger.info(f"Max cycles: {config.al_max_cycles}")
    logger.info(f"Test samples per cycle: {config.al_n_test}")
    logger.info(f"Query samples per cycle: {config.al_n_query}")
    logger.info(f"Elements: {config.elements}")
    logger.info(f"Convergence check: {config.al_convergence_check}")
    if config.al_convergence_check:
        logger.info(f"  Metric: {config.al_convergence_metric}")
        logger.info(f"  MAE threshold: {config.al_convergence_threshold_mae} eV")
        logger.info(f"  Rel MAE threshold: {config.al_convergence_threshold_rel_mae}")
        logger.info(f"  Patience: {config.al_convergence_patience} cycles")
    logger.info("="*70)

    oracle = Oracle(config)
    initial_data_creation_if_needed(config, oracle, logger)
    
    # Initialize convergence tracker
    convergence_tracker = ConvergenceTracker(config) if config.al_convergence_check else None
    converged = False
    last_cycle = 0

    for cycle in range(config.al_max_cycles):
        logger.info("="*70)
        logger.info(f"Cycle {cycle}/{config.al_max_cycles - 1}")
        logger.info("="*70)

        db_stats = get_database_stats(config.csv_path)
        logger.info(f"Current database: {db_stats['n_samples']} samples, {db_stats['n_compositions']} compositions")

        # Model path logic
        if cycle == 0:
            logger.info("Training initial model ...")
            train_cycle_model(config, cycle, logger)
        else:
            prev_model = Path(config.checkpoint_dir) / f"cycle_{cycle-1}" / "best_model.pt"
            if not prev_model.exists():
                logger.warning("Previous model not found, training new model ...")
                train_cycle_model(config, cycle, logger)
            else:
                logger.info(f"Using model: {prev_model}")

        # Determine current model path
        current_model = Path(config.checkpoint_dir) / f"cycle_{cycle}" / "best_model.pt"
        
        # Run inference with convergence tracking
        logger.info("Starting inference cycle ...")
        try:
            selected, predictions = run_inference_cycle(
                cycle, 
                str(current_model), 
                oracle, 
                config, 
                convergence_tracker=convergence_tracker,
                verbose=True
            )
            
            # Check convergence
            if convergence_tracker is not None:
                summary = convergence_tracker.get_summary()
                if summary['converged']:
                    logger.info("="*70)
                    logger.info("CONVERGENCE ACHIEVED!")
                    logger.info("="*70)
                    logger.info(f"Best {config.al_convergence_metric.upper()}: "
                              f"{summary['best_mae'] if config.al_convergence_metric == 'mae' else summary['best_rel_mae']:.4f}")
                    logger.info(f"Cycles without improvement: {summary['cycles_without_improvement']}")
                    converged = True
                    last_cycle = cycle
                    
                    # Save convergence history
                    save_convergence_history(
                        convergence_tracker,
                        Path(config.al_results_dir) / "convergence_history.json"
                    )
                    break
            
            # After inference, train next model
            if cycle < config.al_max_cycles - 1:
                logger.info("Training model with new data ...")
                train_cycle_model(config, cycle + 1, logger)
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            traceback.print_exc()
            continue

        logger.info(f"Cycle {cycle} completed.")
        last_cycle = cycle
    
    # Train final model with higher patience
    logger.info("="*70)
    if converged:
        logger.info("TRAINING FINAL MODEL (after convergence)")
    else:
        logger.info("TRAINING FINAL MODEL (max cycles reached)")
    logger.info("="*70)
    logger.info(f"Using higher patience: {config.final_model_patience} epochs")
    
    final_results = train_cycle_model(config, last_cycle, logger, is_final=True)
    
    if final_results:
        logger.info(f"Final model best val MAE: {final_results['best_val_mae']:.4f} eV")
        logger.info(f"Final model best val Rel MAE: {final_results['best_val_rel_mae']:.4f}")
    
    logger.info("="*70)
    logger.info("ACTIVE LEARNING COMPLETE")
    logger.info("="*70)


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    config = Config()
    logger = setup_main_logger(config)
    
    logger.info("="*70)
    logger.info("CURRENT CONFIG")
    logger.info("="*70)
    for key, val in config.__dict__.items():
        if not key.startswith('_'):
            logger.info(f"{key:30s}: {val}")
    logger.info("="*70)
    
    try:
        active_learning_loop(config, logger)
    except Exception as e:
        logger.error(f"Run aborted: {e}")
        traceback.print_exc()
        sys.exit(1)