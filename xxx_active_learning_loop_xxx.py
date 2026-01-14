"""
Active Learning Loop (Clean Logging Version)

Features:
- Fixed test set generated ONCE
- Clean periodic logging (no tqdm)
- Robust error handling
- Uses unrelaxed .npz inference via updated inference.py
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
    save_convergence_history,
    predict_barriers_for_test_set,
    select_samples_by_error,
    generate_query_compositions_from_selected
)
from trainer import Trainer
from dataset import create_dataloaders
from graph_builder import GraphBuilder
from model import create_model_from_config
from utils import get_node_input_dim, set_seed

# ============================================================================
# Setup Simple Logger
# ============================================================================

def setup_simple_logger(config: Config):
    """Setup single logger for entire active learning loop."""
    logger = logging.getLogger("active_learning")
    logger.setLevel(logging.INFO)
    logger.handlers = [] # Clear existing
    
    # File handler
    log_file = Path(config.log_dir) / "active_learning.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
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
        return {
            "n_samples": len(df),
            "n_compositions": df['composition_string'].nunique() if 'composition_string' in df.columns else 0
        }
    except Exception:
        return {"n_samples": 0, "n_compositions": 0}


def initial_training_data_creation(config: Config, oracle: Oracle, logger: logging.Logger):
    """Create initial training dataset (No TQDM, clean logs)."""
    csv_path = config.csv_path
    target_samples = int(getattr(config, "al_initial_training_samples", 0) or 0)
    
    db_stats = get_database_stats(csv_path)
    current_samples = db_stats['n_samples']
    
    if current_samples >= target_samples:
        logger.info(f"Training data exists: {current_samples}/{target_samples} samples")
        return
    
    remaining = target_samples - current_samples
    logger.info("="*70)
    logger.info(f"INITIAL TRAINING DATA GENERATION")
    logger.info(f"Target: {target_samples} | Current: {current_samples} | To generate: {remaining}")
    logger.info("="*70)
    
    elements = list(getattr(config, "elements", []))
    compositions = generate_uniform_compositions(elements, remaining, seed=config.al_seed + 999)
    
    logger.info(f"Starting Oracle calculations for {len(compositions)} samples...")
    
    successes = 0
    total = len(compositions)
    
    for i, comp in enumerate(compositions, 1):
        if i % 50 == 0 or i == 1:
            logger.info(f"Initial Data: {i}/{total} ({i/total*100:.1f}%) | Successes: {successes}")
            for h in logger.handlers: h.flush()
            
        try:
            ok = oracle.calculate(comp)
            if ok: successes += 1
        except Exception as e:
            logger.error(f"Error at sample {i}: {e}")
    
    if successes == 0:
        raise RuntimeError("Initial training data creation failed.")
    
    final_stats = get_database_stats(csv_path)
    logger.info(f"Initial training data complete: {final_stats['n_samples']} samples")


def train_cycle_model(config: Config, cycle: int, logger: logging.Logger, is_final: bool = False) -> dict:
    """Train the model for the given active learning cycle."""
    seed = config.random_seed + cycle
    set_seed(seed)
    outdir = Path(config.checkpoint_dir) / ("final_model" if is_final else f"cycle_{cycle}")
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        db_stats = get_database_stats(config.csv_path)
        config._current_dataset_size = db_stats['n_samples']
        
        # --- FIX: Pass csv_path explicitly ---
        result = create_dataloaders(config, config.csv_path)
        train_loader, val_loader = result if isinstance(result, tuple) else (result, None)
        
        builder = GraphBuilder(config)
        node_input_dim = get_node_input_dim(builder)
        model = create_model_from_config(config, node_input_dim)
        
        # Load best model from previous cycle (Transfer Learning)
        if not is_final and cycle > 0:
            # Try to load best model from previous cycle
            prev_cycle = cycle - 1
            prev_path = Path(config.checkpoint_dir) / f"cycle_{prev_cycle}" / "best_model.pt"
            if prev_path.exists():
                logger.info(f"Transfer Learning: Loading model from cycle {prev_cycle}")
                checkpoint = torch.load(prev_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load best model from ANY previous cycle for final training
        if is_final:
            best_val_mae = float('inf')
            best_cycle = None
            for c in range(cycle + 1):
                cycle_path = Path(config.checkpoint_dir) / f"cycle_{c}" / "best_model.pt"
                if cycle_path.exists():
                    checkpoint = torch.load(cycle_path, map_location='cpu')
                    val_mae = checkpoint.get('best_val_mae', float('inf'))
                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        best_cycle = c
            
            if best_cycle is not None:
                path = Path(config.checkpoint_dir) / f"cycle_{best_cycle}" / "best_model.pt"
                logger.info(f"Loading best model from cycle {best_cycle} (MAE: {best_val_mae:.4f})")
                checkpoint = torch.load(path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
        
        trainer = Trainer(model, config, save_dir=str(outdir), cycle=cycle, is_final_model=is_final)
        
        # Clean trainer logs (remove stream handlers to avoid double logging in console)
        trainer.logger.handlers = [h for h in trainer.logger.handlers if not isinstance(h, logging.StreamHandler)]
        
        trainer.train(train_loader, val_loader, verbose=False)
        return {'best_val_mae': trainer.best_val_mae, 'best_val_rel_mae': trainer.best_val_rel_mae}
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return None

def active_learning_loop(config: Config, logger: logging.Logger):
    """Main active learning loop (Clean version)."""
    
    # TRAIN ONLY MODE
    if config.train_only_mode:
        logger.info("TRAIN-ONLY MODE SELECTED")
        train_cycle_model(config, cycle=0, logger=logger, is_final=True)
        return

    logger.info("="*70)
    logger.info("ACTIVE LEARNING STARTED")
    logger.info("="*70)

    # 0. FIXED TEST SET
    # Aufruf der cleanen Version (siehe unten)
    create_fixed_test_set(config, logger)
    test_df = load_test_set(config)
    logger.info(f"Test set loaded: {len(test_df)} samples")
    
    # 1. INITIAL DATA
    oracle = Oracle(config)
    # Silence oracle logger to avoid noise
    oracle.logger.handlers = [h for h in oracle.logger.handlers if not isinstance(h, logging.StreamHandler)]
    
    initial_training_data_creation(config, oracle, logger)
    
    # 2. CYCLES
    convergence_tracker = ConvergenceTracker(config) if config.al_convergence_check else None
    last_cycle = 0
    
    for cycle in range(config.al_max_cycles):
        logger.info(f"\n{'='*30} CYCLE {cycle} {'='*30}")
        
        # A. TRAIN
        if not config.train_only_skip_cycles:
            train_res = train_cycle_model(config, cycle, logger)
            if not train_res: break
            logger.info(f"Training done. Val MAE: {train_res['best_val_mae']:.4f}")
        
        # B. INFERENCE (Using UNRELAXED NPZs via inference.py)
        logger.info("Running inference on test set (unrelaxed)...")
        model_path = Path(config.checkpoint_dir) / f"cycle_{cycle}" / "best_model.pt"
        
        try:
            predictions = predict_barriers_for_test_set(str(model_path), test_df, config)
            mae = predictions['absolute_error'].mean()
            logger.info(f"Test MAE: {mae:.4f} eV")
            
            if convergence_tracker:
                if convergence_tracker.update(cycle, mae, predictions['relative_error'].mean()):
                    logger.info("CONVERGED!")
                    last_cycle = cycle
                    break
            
            selected = select_samples_by_error(
                predictions, config.al_n_query, config.al_query_strategy, 
                config.al_seed + cycle, config, logger
            )
            logger.info(f"Selected {len(selected)} samples for query.")
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            traceback.print_exc()
            continue

        # C. QUERY GENERATION
        if not config.train_only_skip_cycles:
            logger.info("Generating query data via Oracle...")
            query_comps = generate_query_compositions_from_selected(
                selected, config.al_n_query, list(config.elements), seed=config.al_seed + cycle
            )
            
            n_ok = 0
            total_q = len(query_comps)
            for i, comp in enumerate(query_comps, 1):
                # Periodisches Logging
                if i % 20 == 0: 
                    logger.info(f"Oracle Progress: {i}/{total_q}")
                    for h in logger.handlers: h.flush()
                
                if oracle.calculate(comp): n_ok += 1
            
            logger.info(f"Added {n_ok} new samples to training set.")
            
        last_cycle = cycle

    # FINAL MODEL
    logger.info("Training FINAL model...")
    train_cycle_model(config, last_cycle, logger, is_final=True)
    oracle.cleanup()
    logger.info("Active Learning Complete.")

if __name__ == "__main__":
    config = Config()
    logger = setup_simple_logger(config)
    try:
        active_learning_loop(config, logger)
    except Exception as e:
        logger.error(f"FATAL: {e}")
        traceback.print_exc()