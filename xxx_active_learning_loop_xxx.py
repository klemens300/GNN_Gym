"""
Active Learning Loop (Clean Logging Version)
UPDATED: Train from Scratch each cycle.
"""

import sys
import traceback
import logging
from pathlib import Path
import pandas as pd
import torch

from config import Config
from oracle import Oracle
from fixed_test_set import create_fixed_test_set, load_test_set, generate_uniform_compositions
from inference import (
    ConvergenceTracker,
    predict_barriers_for_test_set,
    select_samples_by_error,
    generate_query_compositions_from_selected
)
from trainer import Trainer
from dataset import create_dataloaders
from graph_builder import GraphBuilder
from model import create_model_from_config
from utils import get_node_input_dim, set_seed

def setup_simple_logger(config: Config):
    logger = logging.getLogger("active_learning")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    log_file = Path(config.log_dir) / "active_learning.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode='w')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_database_stats(csv_path: str) -> dict:
    p = Path(csv_path)
    if not p.exists(): return {"n_samples": 0}
    try:
        df = pd.read_csv(p)
        return {"n_samples": len(df)}
    except: return {"n_samples": 0}

def initial_training_data_creation(config: Config, oracle: Oracle, logger: logging.Logger):
    csv_path = config.csv_path
    target_samples = int(getattr(config, "al_initial_training_samples", 0) or 0)
    current_samples = get_database_stats(csv_path)['n_samples']
    
    if current_samples >= target_samples: return
    
    remaining = target_samples - current_samples
    logger.info(f"Generating {remaining} initial samples...")
    compositions = generate_uniform_compositions(list(config.elements), remaining, seed=config.al_seed + 999)
    
    successes = 0
    for i, comp in enumerate(compositions, 1):
        if i % 50 == 0: logger.info(f"Initial: {i}/{len(compositions)}")
        if oracle.calculate(comp): successes += 1

def train_cycle_model(config: Config, cycle: int, logger: logging.Logger, is_final: bool = False) -> dict:
    """Train model from SCRATCH for the given cycle."""
    seed = config.random_seed + cycle
    set_seed(seed)
    outdir = Path(config.checkpoint_dir) / ("final_model" if is_final else f"cycle_{cycle}")
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        config._current_dataset_size = get_database_stats(config.csv_path)['n_samples']
        result = create_dataloaders(config, config.csv_path)
        train_loader, val_loader = result if isinstance(result, tuple) else (result, None)
        
        builder = GraphBuilder(config)
        node_input_dim = get_node_input_dim(builder)
        
        # ALWAYS CREATE FRESH MODEL
        model = create_model_from_config(config, node_input_dim)
        
        # REMOVED: Loading of previous cycle model (Transfer Learning)
        
        trainer = Trainer(model, config, save_dir=str(outdir), cycle=cycle, is_final_model=is_final)
        trainer.logger.handlers = [h for h in trainer.logger.handlers if not isinstance(h, logging.StreamHandler)]
        
        trainer.train(train_loader, val_loader, verbose=False)
        return {'best_val_mae': trainer.best_val_mae}
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return None

def active_learning_loop(config: Config, logger: logging.Logger):
    if config.train_only_mode:
        train_cycle_model(config, cycle=0, logger=logger, is_final=True)
        return

    create_fixed_test_set(config, logger)
    test_df = load_test_set(config)
    
    oracle = Oracle(config)
    oracle.logger.handlers = [h for h in oracle.logger.handlers if not isinstance(h, logging.StreamHandler)]
    
    initial_training_data_creation(config, oracle, logger)
    
    convergence_tracker = ConvergenceTracker(config) if config.al_convergence_check else None
    last_cycle = 0
    
    for cycle in range(config.al_max_cycles):
        logger.info(f"\n{'='*30} CYCLE {cycle} {'='*30}")
        
        if not config.train_only_skip_cycles:
            res = train_cycle_model(config, cycle, logger)
            if not res: break
            logger.info(f"Training done. Best Val MAE: {res['best_val_mae']:.4f}")
        
        # Inference
        model_path = Path(config.checkpoint_dir) / f"cycle_{cycle}" / "best_model.pt"
        try:
            predictions = predict_barriers_for_test_set(str(model_path), test_df, config, logger)
            mae = predictions['absolute_error'].mean()
            logger.info(f"Test Set MAE: {mae:.4f} eV")
            
            if convergence_tracker and convergence_tracker.update(cycle, mae, predictions['relative_error'].mean()):
                logger.info("CONVERGED!")
                last_cycle = cycle
                break
            
            selected = select_samples_by_error(
                predictions, config.al_n_query, config.al_query_strategy, 
                config.al_seed + cycle, config, logger
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            traceback.print_exc()
            continue

        if not config.train_only_skip_cycles:
            logger.info("Oracle calculating new samples...")
            query_comps = generate_query_compositions_from_selected(
                selected, config.al_n_query, list(config.elements), seed=config.al_seed + cycle
            )
            for i, comp in enumerate(query_comps):
                oracle.calculate(comp)
            
        last_cycle = cycle

    logger.info("Training FINAL model...")
    train_cycle_model(config, last_cycle, logger, is_final=True)
    oracle.cleanup()

if __name__ == "__main__":
    config = Config()
    logger = setup_simple_logger(config)
    try: active_learning_loop(config, logger)
    except Exception as e: logger.error(f"FATAL: {e}")