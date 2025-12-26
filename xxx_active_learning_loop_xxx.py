# === Active Learning Loop - SIMPLIFIED LOGGING ===
# Console: English; Comments: English

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
from inference import (
    run_inference_cycle,
    cleanup_gpu,
    ConvergenceTracker,
    save_convergence_history,
    generate_and_calculate_query_data
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


def sample_simplex_uniform(n: int, k: int) -> np.ndarray:
    """Sample n points uniformly on a (k-1)-simplex using Dirichlet distribution."""
    return np.random.dirichlet(alpha=np.ones(k), size=n)


def initial_data_creation_if_needed(config: Config, oracle: Oracle, logger: logging.Logger):
    """
    Create initial dataset with intelligent resume capability.
    """
    csv_path = config.csv_path
    target_samples = int(getattr(config, "al_initial_samples", 0) or 0)
    
    if target_samples <= 0:
        raise RuntimeError("Initial data creation requires 'al_initial_samples' > 0 in config.")
    
    # Check current database state
    db_stats = get_database_stats(csv_path)
    current_samples = db_stats['n_samples']
    
    # Case 1: Enough data already exists
    if current_samples >= target_samples:
        logger.info(f"Initial data: {current_samples}/{target_samples} samples available - skipping")
        return
    
    # Case 2: Need to create (remaining) data
    remaining = target_samples - current_samples
    
    logger.info("="*70)
    logger.info(f"INITIAL DATA GENERATION")
    logger.info(f"Target: {target_samples} | Current: {current_samples} | To generate: {remaining}")
    logger.info("="*70)
    
    elements = list(getattr(config, "elements", []))
    if not elements:
        raise RuntimeError("Initial data creation requires 'elements' in config.")
    
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
            comp_str = ", ".join([f"{k}:{v:.2f}" for k, v in comp.items()])
            logger.info(f"  Generating [{current_samples + i}/{target_samples}]: {comp_str}")
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
    logger.info(f"Initial data complete: {final_stats['n_samples']} samples total")
    logger.info("="*70)


def train_cycle_model(config: Config, cycle: int, logger: logging.Logger, is_final: bool = False) -> dict:
    """Train the model for the given active learning cycle (NO LOGGING)."""
    
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
        
        # Create dataloaders
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
                logger.info(f"Loading best model from cycle {best_cycle} (val MAE: {best_val_mae:.4f})")
                checkpoint = torch.load(best_model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer (with its own logger, but we ignore that)
        trainer = Trainer(model, config, save_dir=str(outdir), cycle=cycle, is_final_model=is_final)
        
        # Suppress trainer logging to console
        trainer.logger.handlers = [h for h in trainer.logger.handlers if not isinstance(h, logging.StreamHandler)]
        
        trainer.train(train_loader, val_loader, verbose=False)  # verbose=False!
        
        return {
            'best_val_mae': trainer.best_val_mae,
            'best_val_rel_mae': trainer.best_val_rel_mae
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return None


def active_learning_loop(config: Config, logger: logging.Logger):
    """Main active learning loop - SIMPLIFIED VERSION."""
    
    # ========== TRAIN-ONLY MODE ==========
    if config.train_only_mode:
        logger.info("="*70)
        logger.info("TRAIN-ONLY MODE")
        logger.info("="*70)
        
        db_stats = get_database_stats(config.csv_path)
        if db_stats['n_samples'] == 0:
            logger.error("ERROR: No data found in CSV!")
            return
        
        logger.info(f"Dataset: {db_stats['n_samples']} samples")
        logger.info("Training final model...")
        
        final_results = train_cycle_model(config, cycle=0, logger=logger, is_final=True)
        
        if final_results:
            logger.info(f"Final model: val MAE = {final_results['best_val_mae']:.4f} eV")
        
        logger.info("="*70)
        logger.info("TRAIN-ONLY COMPLETE")
        logger.info("="*70)
        return
    
    # ========== NORMAL ACTIVE LEARNING MODE ==========
    logger.info("="*70)
    logger.info("ACTIVE LEARNING START")
    logger.info("="*70)
    logger.info(f"Max cycles: {config.al_max_cycles}")
    logger.info(f"Test samples per cycle: {config.al_n_test}")
    logger.info(f"Query samples per cycle: {config.al_n_query}")
    logger.info(f"Elements: {config.elements}")
    logger.info("="*70)

    oracle = Oracle(config)
    
    # Suppress oracle logging to console
    oracle.logger.handlers = [h for h in oracle.logger.handlers if not isinstance(h, logging.StreamHandler)]
    
    initial_data_creation_if_needed(config, oracle, logger)
    
    # Initialize convergence tracker
    convergence_tracker = ConvergenceTracker(config) if config.al_convergence_check else None
    converged = False
    last_cycle = 0

    for cycle in range(config.al_max_cycles):
        logger.info("")
        logger.info("="*70)
        logger.info(f"CYCLE {cycle}")
        logger.info("="*70)

        db_stats = get_database_stats(config.csv_path)
        logger.info(f"Current database: {db_stats['n_samples']} samples")

        # ========== STEP 1: TRAIN MODEL ==========
        if config.train_only_skip_cycles:
            logger.info("Skipping cycle training (train_only_skip_cycles=True)")
        else:
            current_model_path = Path(config.checkpoint_dir) / f"cycle_{cycle}" / "best_model.pt"
            
            if current_model_path.exists():
                logger.info(f"Model exists for cycle {cycle} - skipping training")
            else:
                logger.info(f"Training model for cycle {cycle}...")
                train_result = train_cycle_model(config, cycle, logger)
                if train_result is None:
                    logger.error(f"Training failed for cycle {cycle}")
                    break
                logger.info(f"Training complete: val MAE = {train_result['best_val_mae']:.4f} eV")
        
        # ========== STEP 2: INFERENCE ==========
        if config.train_only_skip_cycles:
            logger.info("Skipping inference")
        else:
            logger.info(f"Running inference for cycle {cycle}...")
            try:
                current_model_path = Path(config.checkpoint_dir) / f"cycle_{cycle}" / "best_model.pt"
                
                # Suppress inference logging (create temporary logger)
                inference_logger = logging.getLogger(f"inference_cycle_{cycle}")
                inference_logger.handlers = []  # No handlers = no output
                inference_logger.setLevel(logging.CRITICAL)  # Ignore everything
                
                # Run inference with suppressed logging
                from inference import (
                    generate_test_compositions,
                    generate_test_data_with_oracle,
                    predict_barriers_for_test_set,
                    select_samples_by_error
                )
                
                # Generate test compositions
                test_compositions = generate_test_compositions(
                    elements=list(config.elements),
                    n_test=config.al_n_test,
                    strategy=config.al_test_strategy,
                    seed=config.al_seed + cycle
                )
                
                logger.info(f"  Generating {len(test_compositions)} test samples...")
                
                # Generate test data with Oracle (with progress in main log)
                test_results = []
                for i, comp in enumerate(test_compositions, 1):
                    if i % 100 == 0:  # Log every 100 samples
                        logger.info(f"    Progress: {i}/{len(test_compositions)} test samples")
                    
                    success = oracle.calculate(comp)
                    if success:
                        # Read last entry from CSV
                        df = pd.read_csv(config.csv_path)
                        last_entry = df.iloc[-1]
                        test_results.append({
                            'composition_string': last_entry['composition_string'],
                            'structure_folder': last_entry['structure_folder'],
                            'oracle_barrier': last_entry['backward_barrier_eV']
                        })
                
                test_data = pd.DataFrame(test_results)
                logger.info(f"  Generated {len(test_data)} test samples")
                
                # Predict with model
                logger.info(f"  Making predictions...")
                predictions = predict_barriers_for_test_set(
                    str(current_model_path),
                    test_data,
                    config,
                    verbose=False
                )
                
                # Calculate statistics
                mae = predictions['absolute_error'].mean()
                
                logger.info(f"  Prediction MAE: {mae:.4f} eV")
                
                # Update convergence tracker
                if convergence_tracker is not None:
                    rel_mae = predictions['relative_error'].mean()
                    converged = convergence_tracker.update(cycle, mae, rel_mae)
                    if converged:
                        logger.info(f"  CONVERGED! (no improvement for {config.al_convergence_patience} cycles)")
                
                # Select samples for training
                selected = select_samples_by_error(
                    predictions=predictions,
                    n_query=config.al_n_query,
                    strategy=config.al_query_strategy,
                    seed=config.al_seed + cycle
                )
                
                logger.info(f"  Selected {len(selected)} high-error samples")

            except Exception as e:
                logger.error(f"Inference failed for cycle {cycle}: {e}")
                traceback.print_exc()
                last_cycle = cycle
                continue

        # ========== STEP 3: GENERATE QUERY DATA ==========
        if config.train_only_skip_cycles:
            logger.info("Skipping query generation")
        else:
            logger.info(f"Generating {config.al_n_query} query samples...")
            db_stats_before = get_database_stats(config.csv_path)
            
            try:
                # Generate query compositions near high-error samples
                from inference import generate_query_compositions_from_selected
                
                query_compositions = generate_query_compositions_from_selected(
                    selected_samples=selected,
                    n_query=config.al_n_query,
                    elements=list(config.elements),
                    strategy='nearby',
                    noise_level=0.1,
                    seed=config.al_seed + cycle
                )
                
                # Calculate with Oracle (with progress)
                n_successful = 0
                for i, comp in enumerate(query_compositions, 1):
                    if i % 100 == 0:
                        logger.info(f"    Progress: {i}/{len(query_compositions)} query samples")
                    
                    success = oracle.calculate(comp)
                    if success:
                        n_successful += 1
                
                db_stats_after = get_database_stats(config.csv_path)
                logger.info(f"  Added {n_successful} query samples")
                logger.info(f"  Database: {db_stats_before['n_samples']} ? {db_stats_after['n_samples']} samples")
                
            except Exception as e:
                logger.error(f"Query generation failed: {e}")
                traceback.print_exc()
                last_cycle = cycle
                continue

        # ========== STEP 4: CHECK CONVERGENCE ==========
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
    
    final_results = train_cycle_model(config, last_cycle, logger, is_final=True)
    
    if final_results:
        logger.info(f"Final model: val MAE = {final_results['best_val_mae']:.4f} eV")
    
    logger.info("="*70)
    logger.info("ACTIVE LEARNING COMPLETE")
    logger.info(f"Total cycles: {last_cycle + 1}")
    logger.info(f"Final dataset: {get_database_stats(config.csv_path)['n_samples']} samples")
    logger.info("="*70)


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    config = Config()
    logger = setup_simple_logger(config)
    
    # Log config
    logger.info("="*70)
    logger.info("CONFIG")
    logger.info("="*70)
    for key, val in config.__dict__.items():
        if not key.startswith('_'):
            logger.info(f"{key:30s}: {val}")
    logger.info("="*70)
    
    try:
        active_learning_loop(config, logger)
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)