"""
Active Learning Loop with Fixed Test Set

Key Changes:
- Fixed test set generated ONCE, used for all cycles
- Test data stored in SEPARATE CSV (not used for training)
- Training data grows with query samples each cycle
- Clear separation: test CSV vs training CSV
"""

import sys
import traceback
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# --- Project imports ---
from config import Config
from oracle import Oracle
from fixed_test_set import create_fixed_test_set, load_test_set, generate_uniform_compositions
from inference import (
    cleanup_gpu,
    ConvergenceTracker,
    save_convergence_history
)
from trainer import Trainer
from dataset import create_dataloaders
from graph_builder import GraphBuilder
from model import create_model_from_config, count_parameters
from utils import get_node_input_dim, save_model_for_inference, set_seed

# --- Optional Weights & Biases support ---
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ============================================================================
# Setup Simple Logger (ONE FILE ONLY)
# ============================================================================

def setup_simple_logger(config: Config):
    """Setup single logger for entire active learning loop."""
    logger = logging.getLogger("active_learning")
    logger.setLevel(logging.INFO)
    
    # Close and remove ALL existing handlers
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.handlers = []
    
    # File handler
    log_file = Path(config.log_dir) / "active_learning.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode='w')  # Overwrite each run
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Simple formatter
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# Helper functions
# ============================================================================

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


def initial_training_data_creation(config: Config, oracle: Oracle, logger: logging.Logger):
    """
    Create initial training dataset.
    
    This function:
    - Checks if enough training data exists
    - Generates remaining samples if needed
    - Writes to MAIN CSV (used for training!)
    """
    csv_path = config.csv_path
    target_samples = int(getattr(config, "al_initial_training_samples", 0) or 0)
    
    if target_samples <= 0:
        raise RuntimeError("Initial training data requires 'al_initial_training_samples' > 0 in config.")
    
    # Check current database state
    db_stats = get_database_stats(csv_path)
    current_samples = db_stats['n_samples']
    
    # Case 1: Enough data already exists
    if current_samples >= target_samples:
        logger.info(f"? Training data exists: {current_samples}/{target_samples} samples")
        logger.info(f"  Training CSV: {csv_path}")
        return
    
    # Case 2: Need to create (remaining) data
    remaining = target_samples - current_samples
    
    logger.info("="*70)
    logger.info(f"INITIAL TRAINING DATA GENERATION")
    logger.info(f"Target: {target_samples} | Current: {current_samples} | To generate: {remaining}")
    logger.info("="*70)
    
    elements = list(getattr(config, "elements", []))
    if not elements:
        raise RuntimeError("Initial data creation requires 'elements' in config.")
    
    # Generate training compositions (uniform sampling)
    # Use DIFFERENT seed than test set to avoid overlap!
    compositions = generate_uniform_compositions(
        elements, 
        remaining, 
        seed=config.al_seed + 999
    )
    
    logger.info(f"Generated {len(compositions)} training compositions")
    
    # Calculate samples (writes to MAIN CSV!)
    successes = 0
    for i, comp in enumerate(compositions, 1):
        try:
            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{remaining}")
            
            ok = oracle.calculate(comp)  # ? Writes to TRAINING CSV!
            if ok is not False:
                successes += 1
        except Exception as e:
            logger.error(f"   Error at sample {i}: {e}")
    
    if successes == 0:
        raise RuntimeError("Initial training data creation failed: no valid samples added.")
    
    # Final check
    final_stats = get_database_stats(csv_path)
    logger.info("="*70)
    logger.info(f"Initial training data complete: {final_stats['n_samples']} samples")
    logger.info(f"  Training CSV: {csv_path}")
    logger.info("="*70)


def train_cycle_model(config: Config, cycle: int, logger: logging.Logger, is_final: bool = False) -> dict:
    """
    Train the model for the given active learning cycle.
    
    This function:
    - Creates dataloaders from MAIN CSV (training data)
    - Trains model with early stopping
    - Saves best model checkpoint
    
    Args:
        config: Config object
        cycle: Cycle number
        logger: Logger instance
        is_final: If True, uses longer patience for final training
    
    Returns:
        results: Dict with best_val_mae and best_val_rel_mae
    """
    
    # Set seeds for reproducibility
    seed = config.random_seed + cycle
    set_seed(seed)

    outdir = Path(config.checkpoint_dir) / ("final_model" if is_final else f"cycle_{cycle}")
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        # Get current database size
        db_stats = get_database_stats(config.csv_path)
        n_samples = db_stats['n_samples']
        
        # Store dataset size in config for wandb naming
        config._current_dataset_size = n_samples
        
        # Create dataloaders from MAIN CSV (training data)
        result = create_dataloaders(config)
        
        if result is None:
            raise RuntimeError("create_dataloaders returned None")
        
        if isinstance(result, tuple) and len(result) == 2:
            train_loader, val_loader = result
        else:
            train_loader = result
            val_loader = None
        
        if train_loader is None:
            raise RuntimeError("train_loader is None")
        
        builder = GraphBuilder(config)
        node_input_dim = get_node_input_dim(builder)
        model = create_model_from_config(config, node_input_dim)
        
        # Load best model from previous cycle if this is final training
        if is_final:
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
                logger.info(f"  Loading best model from cycle {best_cycle} (val MAE: {best_val_mae:.4f})")
                checkpoint = torch.load(best_model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer
        trainer = Trainer(model, config, save_dir=str(outdir), cycle=cycle, is_final_model=is_final)
        
        # Suppress trainer logging to console
        trainer.logger.handlers = [h for h in trainer.logger.handlers if not isinstance(h, logging.StreamHandler)]
        
        trainer.train(train_loader, val_loader, verbose=False)
        
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
    Main active learning loop with fixed test set.
    
    Workflow:
    0. Create fixed test set (ONCE)
    1. Create initial training data
    2. FOR each cycle:
       a. Train on growing training set (MAIN CSV)
       b. Evaluate on FIXED test set (separate CSV, no Oracle!)
       c. Generate query samples near high-error regions
       d. Add query samples to training set (MAIN CSV)
    3. Train final model on all training data
    """
    
    # ========== TRAIN-ONLY MODE ==========
    if config.train_only_mode:
        logger.info("="*70)
        logger.info("TRAIN-ONLY MODE")
        logger.info("="*70)
        
        db_stats = get_database_stats(config.csv_path)
        if db_stats['n_samples'] == 0:
            logger.error("ERROR: No data found in training CSV!")
            return
        
        logger.info(f"Training dataset: {db_stats['n_samples']} samples")
        logger.info(f"Training CSV: {config.csv_path}")
        logger.info("Training final model...")
        
        final_results = train_cycle_model(config, cycle=0, logger=logger, is_final=True)
        
        if final_results:
            logger.info(f"? Final model: val MAE = {final_results['best_val_mae']:.4f} eV")
        
        logger.info("="*70)
        logger.info("TRAIN-ONLY COMPLETE")
        logger.info("="*70)
        return
    
    # ========== ACTIVE LEARNING MODE ==========
    logger.info("="*70)
    logger.info("ACTIVE LEARNING WITH FIXED TEST SET")
    logger.info("="*70)
    logger.info(f"Max cycles: {config.al_max_cycles}")
    logger.info(f"Query samples per cycle: {config.al_n_query}")
    logger.info(f"Elements: {config.elements}")
    logger.info("="*70)
    
    # ========== STEP 0: CREATE FIXED TEST SET (ONCE!) ==========
    logger.info("")
    logger.info("="*70)
    logger.info("STEP 0: FIXED TEST SET (SEPARATE FROM TRAINING)")
    logger.info("="*70)
    
    create_fixed_test_set(config, logger)
    
    # Load test set
    test_df = load_test_set(config)
    logger.info(f"? Test set loaded: {len(test_df)} samples")
    logger.info(f"  Test CSV: {config.al_test_set_csv}")
    logger.info(f"  Test structures: {config.al_test_set_dir}")
    logger.info(f"  ??  NOT used for training - evaluation only!")
    
    # ========== STEP 1: CREATE INITIAL TRAINING DATA ==========
    oracle = Oracle(config)
    
    # Suppress oracle console logging
    oracle.logger.handlers = [
        h for h in oracle.logger.handlers 
        if not isinstance(h, logging.StreamHandler)
    ]
    
    logger.info("")
    logger.info("="*70)
    logger.info("STEP 1: INITIAL TRAINING DATA (MAIN CSV)")
    logger.info("="*70)
    
    initial_training_data_creation(config, oracle, logger)
    
    # ========== ACTIVE LEARNING CYCLES ==========
    convergence_tracker = ConvergenceTracker(config) if config.al_convergence_check else None
    converged = False
    last_cycle = 0
    
    for cycle in range(config.al_max_cycles):
        logger.info("")
        logger.info("="*70)
        logger.info(f"CYCLE {cycle}")
        logger.info("="*70)
        
        train_stats = get_database_stats(config.csv_path)
        logger.info(f"Training data: {train_stats['n_samples']} samples (MAIN CSV)")
        logger.info(f"Test data: {len(test_df)} samples (FIXED, separate)")
        
        # ========== TRAIN MODEL ==========
        if config.train_only_skip_cycles:
            logger.info("Skipping training (train_only_skip_cycles=True)")
        else:
            current_model_path = Path(config.checkpoint_dir) / f"cycle_{cycle}" / "best_model.pt"
            
            if current_model_path.exists():
                logger.info(f"? Model exists for cycle {cycle} - skipping training")
            else:
                logger.info(f"Training model on {train_stats['n_samples']} training samples...")
                train_result = train_cycle_model(config, cycle, logger)
                if train_result is None:
                    logger.error(f"Training failed for cycle {cycle}")
                    break
                logger.info(f"? Training complete: val MAE = {train_result['best_val_mae']:.4f} eV")
        
        # ========== INFERENCE ON FIXED TEST SET (NO ORACLE CALLS!) ==========
        if config.train_only_skip_cycles:
            logger.info("Skipping inference")
        else:
            logger.info(f"Evaluating on fixed test set ({len(test_df)} samples)...")
            logger.info(f"  ??  No Oracle calls - using existing test data!")
            
            try:
                current_model_path = Path(config.checkpoint_dir) / f"cycle_{cycle}" / "best_model.pt"
                
                # Predict on FIXED test set (no Oracle calls!)
                from inference import predict_barriers_for_test_set, select_samples_by_error
                
                predictions = predict_barriers_for_test_set(
                    str(current_model_path),
                    test_df,
                    config,
                    verbose=False
                )
                
                # Calculate statistics
                mae = predictions['absolute_error'].mean()
                rel_mae = predictions['relative_error'].mean()
                max_error = predictions['absolute_error'].max()
                
                logger.info(f"? Test set evaluation:")
                logger.info(f"  MAE: {mae:.4f} eV")
                logger.info(f"  Rel MAE: {rel_mae:.4f}")
                logger.info(f"  Max error: {max_error:.4f} eV")
                
                # Update convergence tracker
                if convergence_tracker is not None:
                    converged = convergence_tracker.update(cycle, mae, rel_mae)
                    if converged:
                        logger.info(f"? CONVERGED! (no improvement for {config.al_convergence_patience} cycles)")
                
                # Select high-error samples for query
                selected = select_samples_by_error(
                    predictions=predictions,
                    n_query=config.al_n_query,
                    strategy=config.al_query_strategy,
                    seed=config.al_seed + cycle
                )
                
                logger.info(f"? Selected {len(selected)} high-error test samples")
                
                # Show top errors
                logger.info(f"  Top 5 errors:")
                for i, sample in enumerate(selected[:5], 1):
                    logger.info(
                        f"    {i}. {sample['composition_str']}: "
                        f"Oracle={sample['oracle_barrier']:.3f} eV, "
                        f"Pred={sample['predicted_barrier']:.3f} eV, "
                        f"RelErr={sample['relative_error']:.3f}"
                    )
                
            except Exception as e:
                logger.error(f"Inference failed for cycle {cycle}: {e}")
                traceback.print_exc()
                last_cycle = cycle
                continue
        
        # ========== GENERATE QUERY DATA (ADDED TO TRAINING SET!) ==========
        if config.train_only_skip_cycles:
            logger.info("Skipping query generation")
        else:
            logger.info(f"Generating {config.al_n_query} query samples...")
            logger.info(f"  ??  These will be ADDED to training set (MAIN CSV)!")
            
            train_stats_before = get_database_stats(config.csv_path)
            
            try:
                from inference import generate_query_compositions_from_selected
                
                # Generate query compositions near high-error samples
                query_compositions = generate_query_compositions_from_selected(
                    selected_samples=selected,
                    n_query=config.al_n_query,
                    elements=list(config.elements),
                    strategy='nearby',
                    noise_level=0.1,
                    seed=config.al_seed + cycle
                )
                
                logger.info(f"  Generated {len(query_compositions)} query compositions")
                
                # Calculate with Oracle (writes to MAIN CSV = TRAINING DATA!)
                n_successful = 0
                for i, comp in enumerate(query_compositions, 1):
                    if i % 100 == 0:
                        logger.info(f"    Progress: {i}/{len(query_compositions)}")
                    
                    success = oracle.calculate(comp)  # ? Writes to TRAINING CSV!
                    if success:
                        n_successful += 1
                
                train_stats_after = get_database_stats(config.csv_path)
                
                logger.info(f"? Query data generated and added to training set:")
                logger.info(f"  Successful: {n_successful}/{len(query_compositions)}")
                logger.info(f"  Training set: {train_stats_before['n_samples']} ? {train_stats_after['n_samples']} samples")
                logger.info(f"  Training CSV: {config.csv_path}")
                
            except Exception as e:
                logger.error(f"Query generation failed: {e}")
                traceback.print_exc()
                last_cycle = cycle
                continue
        
        # ========== CHECK CONVERGENCE ==========
        if converged:
            logger.info("="*70)
            logger.info(f"CONVERGED AT CYCLE {cycle}")
            logger.info("="*70)
            last_cycle = cycle
            break
        
        last_cycle = cycle
    
    # ========== FINAL MODEL TRAINING ==========
    logger.info("")
    logger.info("="*70)
    logger.info("TRAINING FINAL MODEL")
    logger.info("="*70)
    
    final_train_stats = get_database_stats(config.csv_path)
    logger.info(f"Final training data: {final_train_stats['n_samples']} samples")
    logger.info(f"Training CSV: {config.csv_path}")
    
    final_results = train_cycle_model(config, last_cycle, logger, is_final=True)
    
    if final_results:
        logger.info(f"? Final model: val MAE = {final_results['best_val_mae']:.4f} eV")
    
    # ========== SUMMARY ==========
    logger.info("")
    logger.info("="*70)
    logger.info("ACTIVE LEARNING COMPLETE")
    logger.info("="*70)
    logger.info(f"Cycles: {last_cycle + 1}")
    logger.info(f"Training data: {final_train_stats['n_samples']} samples (MAIN CSV)")
    logger.info(f"  CSV: {config.csv_path}")
    logger.info(f"Test data: {len(test_df)} samples (FIXED, separate)")
    logger.info(f"  CSV: {config.al_test_set_csv}")
    logger.info("="*70)
    
    # Cleanup
    oracle.cleanup()


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    config = Config()
    logger = setup_simple_logger(config)
    
    # Log important paths
    logger.info("="*70)
    logger.info("CONFIG - KEY PATHS")
    logger.info("="*70)
    logger.info(f"Training CSV (grows): {config.csv_path}")
    logger.info(f"Training structures:  {config.database_dir}")
    logger.info(f"Test CSV (fixed):     {config.al_test_set_csv}")
    logger.info(f"Test structures:      {config.al_test_set_dir}")
    logger.info(f"Checkpoints:          {config.checkpoint_dir}")
    logger.info("="*70)
    
    # Log key config values
    logger.info("")
    logger.info("="*70)
    logger.info("CONFIG - ACTIVE LEARNING")
    logger.info("="*70)
    logger.info(f"Elements: {config.elements}")
    logger.info(f"Initial training samples: {config.al_initial_training_samples}")
    logger.info(f"Fixed test samples: {config.al_test_set_size}")
    logger.info(f"Test strategy: {config.al_test_set_strategy}")
    logger.info(f"Query samples per cycle: {config.al_n_query}")
    logger.info(f"Max cycles: {config.al_max_cycles}")
    logger.info(f"Calculator: {config.calculator}")
    logger.info("="*70)
    
    try:
        active_learning_loop(config, logger)
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)