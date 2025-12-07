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
from inference import (
    run_inference_cycle,
    cleanup_gpu,
    ConvergenceTracker,
    save_convergence_history,
    generate_and_calculate_query_data
)
from trainer import Trainer
from dataset import create_dataloaders
from template_graph_builder import TemplateGraphBuilder
from model import create_model_from_config, count_parameters
from utils import get_node_input_dim, save_model_for_inference, set_seed

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


def initial_data_creation_if_needed(config: Config, oracle: Oracle, logger: logging.Logger):
    """
    Create initial dataset with intelligent resume capability.
    
    Logic:
    - If CSV doesn't exist: Create al_initial_samples samples
    - If CSV exists but has fewer samples: Create remaining samples to reach al_initial_samples
    - If CSV has enough samples: Skip data creation
    
    Args:
        config: Config object
        oracle: Oracle instance
        logger: Logger instance
    """
    csv_path = config.csv_path
    target_samples = int(getattr(config, "al_initial_samples", 0) or 0)
    
    if target_samples <= 0:
        raise RuntimeError(
            "Initial data creation requires 'al_initial_samples' > 0 in config."
        )
    
    # Check current database state
    db_stats = get_database_stats(csv_path)
    current_samples = db_stats['n_samples']
    
    # Case 1: Enough data already exists
    if current_samples >= target_samples:
        logger.info("="*70)
        logger.info("INITIAL DATA CHECK")
        logger.info("="*70)
        logger.info(f"Target samples: {target_samples}")
        logger.info(f"Current samples: {current_samples}")
        logger.info("? Sufficient data available, skipping initial data creation")
        logger.info("="*70)
        return
    
    # Case 2: Need to create (remaining) data
    remaining = target_samples - current_samples
    
    logger.info("="*70)
    if current_samples == 0:
        logger.info("INITIAL DATA CREATION (CSV empty or not found)")
    else:
        logger.info("RESUMING INITIAL DATA CREATION")
    logger.info("="*70)
    logger.info(f"Target samples: {target_samples}")
    logger.info(f"Current samples: {current_samples}")
    logger.info(f"Samples to create: {remaining}")
    logger.info(f"Target CSV: {csv_path}")
    
    elements = list(getattr(config, "elements", []))
    if not elements:
        raise RuntimeError("Initial data creation requires 'elements' in config.")
    
    logger.info(f"Elements: {elements}")
    
    # Generate compositions using Dirichlet sampling
    weights = sample_simplex_uniform(remaining, len(elements))
    compositions = []
    for row in weights:
        comp = {el: float(val) for el, val in zip(elements, row)}
        # Normalize to ensure sum = 1.0
        s = sum(comp.values())
        if abs(s - 1.0) > 1e-12:
            for el in comp:
                comp[el] /= s
        compositions.append(comp)
    
    # Calculate samples
    successes = 0
    for i, comp in enumerate(compositions, 1):
        try:
            logger.info(f"  [{current_samples + i}/{target_samples}] Calculating: {comp}")
            ok = oracle.calculate(comp)
            if ok is not False:
                successes += 1
        except Exception as e:
            logger.error(f"   Error at sample {i}: {e}")
    
    if successes == 0:
        raise RuntimeError("Initial data creation failed: no valid samples added.")
    
    # Final check
    final_stats = get_database_stats(csv_path)
    logger.info("="*70)
    logger.info(f"Initial data creation completed:")
    logger.info(f"  Added: {successes} samples")
    logger.info(f"  Total in database: {final_stats['n_samples']} samples")
    logger.info("="*70)


def train_cycle_model(config: Config, cycle: int, logger: logging.Logger, is_final: bool = False) -> dict:
    """
    Train the model for the given active learning cycle.
    
    Args:
        config: Config object
        cycle: Cycle number
        logger: Logger instance
        is_final: Whether this is final model training
    
    Returns:
        dict: Training results with best metrics
    """
    logger.info("="*70)
    if is_final:
        logger.info("TRAINING FINAL MODEL")
    else:
        logger.info(f"TRAINING MODEL - CYCLE {cycle}")
    logger.info("="*70)
    
    # ========== DETERMINISTIC INITIALIZATION ==========
    # Set seeds for reproducibility
    seed = config.random_seed + cycle  # Different seed per cycle, but deterministic
    logger.info(f"Setting random seed: {seed}")
    set_seed(seed)
    # ==================================================

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
    """
    Main active learning loop with convergence checking and final model training.
    
    Workflow per cycle:
    1. Train model (if not exists)
    2. Run inference
    3. Generate query data
    4. Check convergence
    
    Args:
        config: Config object
        logger: Logger instance
    """
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
        logger.info("\n" + "="*70)
        logger.info(f"CYCLE {cycle}/{config.al_max_cycles - 1}")
        logger.info("="*70)

        db_stats = get_database_stats(config.csv_path)
        logger.info(f"Current database: {db_stats['n_samples']} samples, "
                   f"{db_stats['n_compositions']} unique compositions")

        # ========== STEP 1: TRAIN MODEL ==========
        current_model_path = Path(config.checkpoint_dir) / f"cycle_{cycle}" / "best_model.pt"
        
        if current_model_path.exists():
            logger.info(f"? Model already exists for cycle {cycle}: {current_model_path}")
            logger.info("  Skipping training, proceeding to inference")
        else:
            logger.info(f"Training model for cycle {cycle} with current data ...")
            train_result = train_cycle_model(config, cycle, logger)
            if train_result is None:
                logger.error(f"? Training failed for cycle {cycle}, aborting AL loop")
                break
            logger.info(f"? Training completed: val MAE = {train_result['best_val_mae']:.4f} eV")
        
        # ========== STEP 2: INFERENCE ==========
        logger.info(f"\nStarting inference for cycle {cycle} ...")
        try:
            selected, predictions = run_inference_cycle(
                cycle,
                str(current_model_path),
                oracle,
                config,
                convergence_tracker=convergence_tracker,
                verbose=True
            )
            logger.info(f"? Inference completed: {len(selected)} high-error samples selected")

        except Exception as e:
            logger.error(f"? Inference failed for cycle {cycle}: {e}")
            traceback.print_exc()
            logger.warning("  Skipping to next cycle")
            last_cycle = cycle
            continue

        # ========== STEP 3: GENERATE QUERY DATA ==========
        logger.info(f"\nGenerating {config.al_n_query} query samples in high-error regions ...")
        db_stats_before = get_database_stats(config.csv_path)
        
        try:
            n_query_added = generate_and_calculate_query_data(
                selected_samples=selected,
                oracle=oracle,
                config=config,
                verbose=True
            )
            db_stats_after = get_database_stats(config.csv_path)
            logger.info(f"? Query generation completed: {n_query_added} samples added")
            logger.info(f"  Database: {db_stats_before['n_samples']} ? {db_stats_after['n_samples']} samples")
            
        except Exception as e:
            logger.error(f"? Query generation failed for cycle {cycle}: {e}")
            traceback.print_exc()
            logger.warning("  Skipping to next cycle")
            last_cycle = cycle
            continue

        # ========== STEP 4: CHECK CONVERGENCE ==========
        if convergence_tracker is not None:
            summary = convergence_tracker.get_summary()
            logger.info(f"\nConvergence status:")
            logger.info(f"  Best MAE: {summary['best_mae']:.4f} eV")
            logger.info(f"  Cycles without improvement: {summary['cycles_without_improvement']}/{config.al_convergence_patience}")
            
            if summary['converged']:
                logger.info("\n" + "="*70)
                logger.info("??? CONVERGENCE ACHIEVED! ???")
                logger.info("="*70)
                logger.info(f"  Metric: {config.al_convergence_metric.upper()}")
                logger.info(f"  Best value: {summary['best_mae'] if config.al_convergence_metric == 'mae' else summary['best_rel_mae']:.4f}")
                logger.info(f"  Stopped at cycle: {cycle}")
                converged = True
                last_cycle = cycle
                
                # Save convergence history
                save_convergence_history(
                    convergence_tracker,
                    Path(config.al_results_dir) / "convergence_history.json"
                )
                logger.info(f"  Convergence history saved")
                logger.info("="*70)
                break

        logger.info(f"\n? Cycle {cycle} completed successfully")
        logger.info("="*70)
        last_cycle = cycle
    
    # ========== FINAL MODEL TRAINING ==========
    logger.info("\n" + "="*70)
    if converged:
        logger.info("TRAINING FINAL MODEL (after convergence)")
        logger.info(f"  Converged at cycle {last_cycle}")
    else:
        logger.info("TRAINING FINAL MODEL (max cycles reached)")
        logger.info(f"  Completed {last_cycle + 1} cycles")
    logger.info("="*70)
    logger.info(f"Using higher patience: {config.final_model_patience} epochs")
    logger.info(f"Training on full dataset: {get_database_stats(config.csv_path)['n_samples']} samples")
    
    final_results = train_cycle_model(config, last_cycle, logger, is_final=True)
    
    if final_results:
        logger.info("\n" + "="*70)
        logger.info("FINAL MODEL RESULTS")
        logger.info("="*70)
        logger.info(f"  Best val MAE: {final_results['best_val_mae']:.4f} eV")
        logger.info(f"  Best val Rel MAE: {final_results['best_val_rel_mae']:.4f}")
        logger.info("="*70)
    
    logger.info("\n" + "="*70)
    logger.info("??? ACTIVE LEARNING COMPLETE ???")
    logger.info("="*70)
    logger.info(f"  Total cycles: {last_cycle + 1}")
    logger.info(f"  Final dataset size: {get_database_stats(config.csv_path)['n_samples']} samples")
    logger.info(f"  Converged: {'Yes' if converged else 'No'}")
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