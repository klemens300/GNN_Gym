"""
Fixed Test Set Generation

Creates a representative test set ONCE, used across all AL cycles.
Test data is stored in a SEPARATE CSV and NOT used for training.

Key Features:
- Uniform, Sobol, or Grid sampling strategies
- Separate CSV/directory from training data
- Guarantees representativeness across composition space
- Only calculated once, then reused
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict

from oracle import Oracle
from config import Config


def generate_uniform_compositions(elements: List[str], n_samples: int, seed: int = 42) -> List[Dict]:
    """
    Generate uniformly distributed compositions on simplex using Dirichlet.
    
    Dirichlet distribution with alpha=1 gives uniform distribution on the simplex.
    This ensures all compositions are equally likely.
    
    Args:
        elements: List of element symbols (e.g., ['Mo', 'Nb', 'Cr', 'V'])
        n_samples: Number of compositions to generate
        seed: Random seed for reproducibility
    
    Returns:
        compositions: List of composition dicts
        
    Example:
        >>> comps = generate_uniform_compositions(['Mo', 'W'], 100)
        >>> print(comps[0])
        {'Mo': 0.734, 'W': 0.266}
    """
    np.random.seed(seed)
    
    # Dirichlet with alpha=1 gives uniform distribution on simplex
    fractions = np.random.dirichlet(
        alpha=np.ones(len(elements)),
        size=n_samples
    )
    
    compositions = []
    for frac in fractions:
        comp = {el: float(f) for el, f in zip(elements, frac)}
        compositions.append(comp)
    
    return compositions


def generate_sobol_compositions(elements: List[str], n_samples: int, seed: int = 42) -> List[Dict]:
    """
    Generate compositions using Sobol quasi-random sequence.
    
    Sobol sequences provide better space coverage than random sampling.
    They are "low-discrepancy" sequences that fill space more uniformly.
    
    Advantages:
    - Better coverage than random sampling
    - Deterministic for given seed
    - No clustering or gaps
    
    Args:
        elements: List of element symbols
        n_samples: Number of compositions to generate
        seed: Random seed
    
    Returns:
        compositions: List of composition dicts
        
    Example:
        >>> comps = generate_sobol_compositions(['Mo', 'Nb', 'Ta', 'W'], 1000)
    """
    from scipy.stats import qmc
    
    n_dim = len(elements) - 1  # Simplex has n-1 degrees of freedom
    
    # Generate Sobol samples in unit hypercube [0,1]^(n-1)
    sampler = qmc.Sobol(d=n_dim, scramble=True, seed=seed)
    samples = sampler.random(n_samples)
    
    # Convert to simplex coordinates using stick-breaking construction
    # This maps uniform samples in hypercube to uniform samples on simplex
    compositions = []
    for sample in samples:
        comp_array = np.zeros(len(elements))
        remaining = 1.0
        
        # Stick-breaking process
        for i in range(len(elements) - 1):
            comp_array[i] = remaining * sample[i]
            remaining -= comp_array[i]
        comp_array[-1] = remaining
        
        comp = {el: float(val) for el, val in zip(elements, comp_array)}
        compositions.append(comp)
    
    return compositions


def generate_grid_compositions(elements: List[str], n_per_dim: int = 10) -> List[Dict]:
    """
    Generate compositions on a regular grid.
    
    Creates a grid with n_per_dim points per element, then filters
    to only include points on the simplex (sum = 1).
    
    Warning: 
    - Number of samples grows exponentially with n_elements!
    - For 4 elements, n_per_dim=10 gives ~220 valid points
    - For 5 elements, n_per_dim=10 gives ~715 valid points
    
    Args:
        elements: List of element symbols
        n_per_dim: Grid resolution per dimension
    
    Returns:
        compositions: List of composition dicts
        
    Example:
        >>> comps = generate_grid_compositions(['Mo', 'W'], n_per_dim=11)
        >>> # Returns 11 compositions: Mo0.0W1.0, Mo0.1W0.9, ..., Mo1.0W0.0
    """
    from itertools import product
    
    grid_points = np.linspace(0, 1, n_per_dim)
    compositions = []
    
    # Generate all combinations
    for combo in product(grid_points, repeat=len(elements)):
        # Filter: must sum to 1.0 (on simplex)
        if abs(sum(combo) - 1.0) < 0.01:
            comp = {el: float(val) for el, val in zip(elements, combo)}
            compositions.append(comp)
    
    return compositions


def create_fixed_test_set(config: Config, logger: logging.Logger):
    """
    Create fixed test set for active learning.
    
    This function:
    1. Checks if test set already exists (skip if complete)
    2. Generates representative compositions using chosen strategy
    3. Calculates barriers using Oracle
    4. Saves to SEPARATE CSV (not used for training!)
    
    The test set is:
    - Generated ONCE at the start
    - Stored in SEPARATE CSV/directory
    - NOT used for training
    - Used for evaluation across all cycles
    - Allows fair comparison across cycles
    
    Args:
        config: Config object with test set parameters
        logger: Logger instance
        
    Raises:
        RuntimeError: If test set creation fails
        ValueError: If unknown strategy specified
    """
    test_csv = Path(config.al_test_set_csv)
    test_dir = Path(config.al_test_set_dir)
    
    # ========================================
    # Check if test set already exists
    # ========================================
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        n_existing = len(df)
        
        if n_existing >= config.al_test_set_size:
            logger.info(f"? Fixed test set already exists: {n_existing} samples")
            logger.info(f"  CSV: {test_csv}")
            return
        else:
            logger.info(f"??  Test set incomplete: {n_existing}/{config.al_test_set_size}")
            logger.info("   Generating remaining samples...")
            n_needed = config.al_test_set_size - n_existing
    else:
        logger.info("="*70)
        logger.info("CREATING FIXED TEST SET")
        logger.info("="*70)
        n_needed = config.al_test_set_size
    
    # ========================================
    # Generate compositions
    # ========================================
    logger.info(f"Generating {n_needed} test compositions...")
    logger.info(f"Strategy: {config.al_test_set_strategy}")
    logger.info(f"Elements: {config.elements}")
    
    if config.al_test_set_strategy == "uniform":
        compositions = generate_uniform_compositions(
            config.elements,
            n_needed,
            seed=config.al_seed
        )
        logger.info(f"  Using Dirichlet uniform sampling")
        
    elif config.al_test_set_strategy == "sobol":
        compositions = generate_sobol_compositions(
            config.elements,
            n_needed,
            seed=config.al_seed
        )
        logger.info(f"  Using Sobol quasi-random sequence")
        
    elif config.al_test_set_strategy == "grid":
        compositions = generate_grid_compositions(
            config.elements,
            n_per_dim=10
        )
        logger.info(f"  Using regular grid")
        
        # Take subset if too many
        if len(compositions) > n_needed:
            logger.info(f"  Grid produced {len(compositions)} points, sampling {n_needed}")
            np.random.seed(config.al_seed)
            indices = np.random.choice(len(compositions), n_needed, replace=False)
            compositions = [compositions[i] for i in indices]
    else:
        raise ValueError(f"Unknown strategy: {config.al_test_set_strategy}")
    
    logger.info(f"? Generated {len(compositions)} compositions")
    
    # ========================================
    # Calculate barriers with Oracle
    # ========================================
    logger.info(f"Calculating barriers (this may take a while)...")
    logger.info(f"  Test CSV: {test_csv}")
    logger.info(f"  Test structures: {test_dir}")
    logger.info(f"  ??  This data will NOT be used for training!")
    
    # Create temporary config for test Oracle
    import copy
    test_config = copy.deepcopy(config)
    test_config.csv_path = str(test_csv)
    test_config.database_dir = str(test_dir)
    
    # Create Oracle for test data
    test_oracle = Oracle(test_config)
    
    # Suppress oracle console logging
    test_oracle.logger.handlers = [
        h for h in test_oracle.logger.handlers 
        if not isinstance(h, logging.StreamHandler)
    ]
    
    # Calculate test samples
    successes = 0
    failures = 0
    
    for i, comp in enumerate(compositions, 1):
        # Progress logging
        if i % 100 == 0 or i == 1:
            logger.info(f"  Progress: {i}/{len(compositions)} ({i/len(compositions)*100:.1f}%)")
        
        try:
            success = test_oracle.calculate(comp)
            if success:
                successes += 1
            else:
                failures += 1
        except Exception as e:
            failures += 1
            logger.error(f"  Failed sample {i}: {e}")
    
    # Cleanup
    test_oracle.cleanup()
    
    # ========================================
    # Verify final test set
    # ========================================
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        
        logger.info("="*70)
        logger.info("FIXED TEST SET CREATED")
        logger.info("="*70)
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Successful: {successes}/{len(compositions)}")
        logger.info(f"Failed: {failures}/{len(compositions)}")
        logger.info(f"CSV: {test_csv}")
        logger.info(f"Structures: {test_dir}")
        logger.info(f"??  NOT used for training - evaluation only!")
        
        # Show barrier statistics
        barriers = df['backward_barrier_eV']
        logger.info(f"\nBarrier statistics:")
        logger.info(f"  Mean: {barriers.mean():.3f} eV")
        logger.info(f"  Std:  {barriers.std():.3f} eV")
        logger.info(f"  Min:  {barriers.min():.3f} eV")
        logger.info(f"  Max:  {barriers.max():.3f} eV")
        
        logger.info("="*70)
    else:
        raise RuntimeError("Test set creation failed - no CSV created")


def load_test_set(config: Config) -> pd.DataFrame:
    """
    Load fixed test set from CSV.
    
    Args:
        config: Config object
    
    Returns:
        test_df: DataFrame with test data
    
    Raises:
        FileNotFoundError: If test set doesn't exist
        
    Example:
        >>> test_df = load_test_set(config)
        >>> print(f"Test set size: {len(test_df)}")
    """
    test_csv = Path(config.al_test_set_csv)
    
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test set not found: {test_csv}\n"
            f"Run create_fixed_test_set() first!"
        )
    
    df = pd.read_csv(test_csv)
    
    # Validate
    if len(df) == 0:
        raise ValueError(f"Test set is empty: {test_csv}")
    
    required_columns = ['composition_string', 'structure_folder', 'backward_barrier_eV']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Test set missing columns: {missing}")
    
    return df


if __name__ == "__main__":
    from config import Config
    
    print("="*70)
    print("FIXED TEST SET GENERATION")
    print("="*70)
    
    config = Config()
    
    # Setup logger
    logger = logging.getLogger("test_set")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Create test set
    create_fixed_test_set(config, logger)
    
    # Load and show statistics
    test_df = load_test_set(config)
    
    print("\n" + "="*70)
    print("TEST SET STATISTICS")
    print("="*70)
    print(f"Samples: {len(test_df)}")
    print(f"Compositions: {test_df['composition_string'].nunique()}")
    print(f"Barrier range: [{test_df['backward_barrier_eV'].min():.3f}, "
          f"{test_df['backward_barrier_eV'].max():.3f}] eV")
    print(f"Barrier mean: {test_df['backward_barrier_eV'].mean():.3f} eV")
    print(f"Barrier std: {test_df['backward_barrier_eV'].std():.3f} eV")
    print("="*70)