"""
Module for generating and managing the fixed test set.
CLEAN VERSION: No tqdm, robust logging, compatible with updated Oracle.
FIXED: Uses paths from config.py instead of hardcoded paths.
"""

import os
import csv
import logging
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from config import Config
from oracle import Oracle

def generate_uniform_compositions(
    elements: List[str],
    n_samples: int,
    step: float = 0.0, # Kept for signature compatibility
    seed: int = 42
) -> List[Dict[str, float]]:
    """
    Generate random compositions using a Dirichlet distribution.
    This ensures uniform sampling over the compositional simplex.
    """
    np.random.seed(seed)
    
    # Alpha=1.0 corresponds to a uniform distribution over the simplex
    alpha = np.ones(len(elements))
    fractions = np.random.dirichlet(alpha, size=n_samples)
    
    compositions = []
    for frac in fractions:
        # Create dictionary mapping element -> fraction
        comp = {elem: float(f) for elem, f in zip(elements, frac)}
        compositions.append(comp)
        
    return compositions

def load_test_set(config: Config) -> pd.DataFrame:
    """
    Load the fixed test set dataframe.
    Uses path defined in config.al_test_set_csv.
    """
    # FIXED: Use config path instead of hardcoded subfolder
    test_csv = Path(config.al_test_set_csv)
    
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test set CSV not found at {test_csv}. "
            "Please run create_fixed_test_set() first."
        )
    
    return pd.read_csv(test_csv)

def create_fixed_test_set(config: Config, logger: logging.Logger):
    """
    Generates a fixed test set if it doesn't exist yet.
    Uses the Oracle to calculate barriers for random compositions.
    """
    # FIXED: Use paths from config
    test_csv = Path(config.al_test_set_csv)
    test_dir = Path(config.al_test_set_dir)
    
    # 1. Check if test set already exists
    if test_csv.exists():
        try:
            df = pd.read_csv(test_csv)
            if len(df) >= config.al_test_set_size:
                logger.info(f"Fixed test set already exists ({len(df)} samples).")
                logger.info(f"  Location: {test_csv}")
                return
            else:
                logger.warning(f"Existing test set is too small ({len(df)} < {config.al_test_set_size}). Extending...")
        except Exception:
            logger.warning("Existing test set file is corrupted. Recreating from scratch.")
    
    logger.info("="*70)
    logger.info("CREATING FIXED TEST SET")
    logger.info("="*70)
    logger.info(f"  Target CSV: {test_csv}")
    logger.info(f"  Target Dir: {test_dir}")
    
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate target compositions
    # We use a fixed seed to ensure the test set is reproducible across runs
    logger.info(f"Generating {config.al_test_set_size} target compositions...")
    compositions = generate_uniform_compositions(
        list(config.elements), 
        config.al_test_set_size, 
        seed=42 # Fixed seed strictly for test set
    )
    
    # 3. Setup Oracle for Test Set
    # We create a separate config to redirect Oracle output to the test_set directory
    test_config = copy.deepcopy(config)
    test_config.csv_path = str(test_csv)
    test_config.database_dir = str(test_dir)
    
    # Initialize Oracle
    oracle = Oracle(test_config)
    
    # Silence the Oracle's internal logger to avoid spamming the main log
    # We only want the high-level progress updates here
    oracle.logger.handlers = [h for h in oracle.logger.handlers if not isinstance(h, logging.StreamHandler)]
    
    logger.info(f"Starting calculations for {len(compositions)} samples...")
    
    successes = 0
    failures = 0
    total = len(compositions)
    
    # 4. Calculation Loop (No TQDM)
    for i, comp in enumerate(compositions, 1):
        # Periodic Logging (every 50 samples or first one)
        if i % 50 == 0 or i == 1:
            progress = (i / total) * 100
            logger.info(f"Test Set Gen: {i}/{total} ({progress:.1f}%) | OK: {successes} | Fail: {failures}")
            # Explicit flush to ensure logs appear in file immediately
            for h in logger.handlers:
                h.flush()
        
        try:
            # Oracle.calculate returns True on success, False on failure
            if oracle.calculate(comp):
                successes += 1
            else:
                failures += 1
        except Exception as e:
            logger.error(f"Error calculating test sample {i}: {e}")
            failures += 1
            
    # Cleanup
    oracle.cleanup()
    
    if successes == 0:
        raise RuntimeError("Test set generation failed: 0 samples successfully calculated.")
    
    logger.info(f"Test set generation complete.")
    logger.info(f"Saved to: {test_csv}")
    logger.info(f"Final Stats: {successes} Successes, {failures} Failures")
    logger.info("="*70)

if __name__ == "__main__":
    # Standalone execution for testing
    logging.basicConfig(level=logging.INFO)
    local_logger = logging.getLogger("test_set_standalone")
    
    try:
        cfg = Config()
        create_fixed_test_set(cfg, local_logger)
    except Exception as e:
        local_logger.error(f"Standalone execution failed: {e}")