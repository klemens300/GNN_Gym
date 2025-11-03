"""
Inference and Active Learning Query Module

Handles:
- Test composition generation
- Oracle-based test data generation
- Model predictions
- Error-based query strategy
- GPU memory cleanup

Usage:
    from inference import run_inference_cycle
    
    selected_comps, predictions_df = run_inference_cycle(
        cycle=0,
        model_path='checkpoints/best_model.pt',
        oracle=oracle,
        config=config
    )
"""

import numpy as np
import pandas as pd
import torch
import gc
import logging
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from config import Config
from oracle import Oracle
from utils import load_model_for_inference
from template_graph_builder import TemplateGraphBuilder


# ============================================================================
# GPU CLEANUP
# ============================================================================

def cleanup_gpu():
    """
    Clean GPU memory after heavy operations.
    Call after Oracle and Model operations.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# LOGGER SETUP
# ============================================================================

def setup_inference_logger(cycle: int, config: Config):
    """
    Setup logger for inference cycle.
    
    Args:
        cycle: Cycle number
        config: Config object
    
    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(f"inference_cycle_{cycle}")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    log_file = Path(config.log_dir) / f"inference_cycle_{cycle}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, config.log_level.upper()))
    
    # Console handler (optional)
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
# 1. TEST COMPOSITION GENERATION
# ============================================================================

def generate_test_compositions(
    elements: List[str],
    n_test: int,
    strategy: str = 'uniform',
    seed: int = 42
) -> List[dict]:
    """
    Generate test compositions using different strategies.
    
    Args:
        elements: List of elements (e.g., ['Mo', 'Nb', 'O', 'Ta', 'W'])
        n_test: Number of test compositions to generate
        strategy: Generation strategy ('uniform', ...)
        seed: Random seed for reproducibility
    
    Returns:
        List of composition dicts: [{'Mo': 0.2, 'Nb': 0.3, ...}, ...]
    
    Strategies:
        - 'uniform': Uniform grid over composition simplex
        - (future: 'random', 'boundary', 'adaptive', ...)
    
    Example:
        >>> comps = generate_test_compositions(['Mo', 'Nb', 'O'], n_test=10)
        >>> print(comps[0])
        {'Mo': 0.333, 'Nb': 0.333, 'O': 0.334}
    """
    if strategy == 'uniform':
        return uniform_grid_sampling(elements, n_test, seed)
    else:
        raise ValueError(f"Unknown test generation strategy: {strategy}")


def uniform_grid_sampling(
    elements: List[str],
    n_test: int,
    seed: int = 42
) -> List[dict]:
    """
    Generate uniform grid over composition simplex.
    
    Uses Dirichlet sampling to create uniform coverage.
    Grid density is automatically determined from n_test.
    
    Args:
        elements: List of elements
        n_test: Number of test points
        seed: Random seed
    
    Returns:
        List of composition dicts
    """
    np.random.seed(seed)
    n_elements = len(elements)
    
    # Use Dirichlet distribution for uniform simplex sampling
    # Alpha = 1 gives uniform distribution over simplex
    alpha = np.ones(n_elements)
    samples = np.random.dirichlet(alpha, size=n_test)
    
    # Convert to list of dicts
    compositions = []
    for sample in samples:
        # Ensure positive and sum to 1
        sample = np.clip(sample, 0.0, 1.0)
        sample = sample / sample.sum()
        
        comp = {elem: float(frac) for elem, frac in zip(elements, sample)}
        compositions.append(comp)
    
    # Ensure equimolar composition is included (good reference point)
    if n_test >= 1:
        equimolar = {elem: 1.0 / n_elements for elem in elements}
        compositions[0] = equimolar
    
    return compositions


# ============================================================================
# 2. ORACLE TEST DATA GENERATION
# ============================================================================

def generate_test_data_with_oracle(
    compositions: List[dict],
    oracle: Oracle,
    config: Config,
    logger: logging.Logger = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate test data using Oracle (ground truth barriers).
    
    For each composition:
    1. Call oracle.calculate() to get real barrier
    2. Store results
    3. Clean GPU memory at the end
    
    Args:
        compositions: List of composition dicts
        oracle: Oracle instance
        config: Config object
        logger: Logger instance (optional)
        verbose: Print progress
    
    Returns:
        DataFrame with columns:
        - composition (str): Formatted composition string
        - oracle_barrier (float): Ground truth barrier (eV)
        - structure_folder (str): Path to structures
        - success (bool): Whether calculation succeeded
    
    Example:
        >>> test_data = generate_test_data_with_oracle(comps, oracle, config)
        >>> print(test_data.head())
    """
    if verbose and logger:
        logger.info("="*70)
        logger.info("GENERATING TEST DATA WITH ORACLE")
        logger.info("="*70)
        logger.info(f"Test compositions: {len(compositions)}")
    
    results = []
    
    iterator = tqdm(compositions, desc="Oracle calculations", disable=not verbose)
    
    for idx, comp in enumerate(iterator):
        # Format composition string
        comp_str = composition_to_string(comp)
        
        try:
            # Oracle calculation
            success = oracle.calculate(comp)
            
            if success:
                # Get barrier from last entry in CSV
                df = pd.read_csv(config.csv_path)
                last_row = df.iloc[-1]
                
                results.append({
                    'composition': comp_str,
                    'oracle_barrier': last_row['backward_barrier_eV'],
                    'structure_folder': last_row['structure_folder'],
                    'success': True
                })
            else:
                results.append({
                    'composition': comp_str,
                    'oracle_barrier': np.nan,
                    'structure_folder': None,
                    'success': False
                })
                
        except Exception as e:
            if verbose and logger:
                logger.error(f"Error for {comp_str}: {e}")
            
            results.append({
                'composition': comp_str,
                'oracle_barrier': np.nan,
                'structure_folder': None,
                'success': False
            })
    
    # Cleanup GPU once at the end (not after each calculation)
    cleanup_gpu()
    
    df = pd.DataFrame(results)
    
    if verbose and logger:
        success_rate = df['success'].sum() / len(df) * 100
        logger.info(f"✓ Test data generated")
        logger.info(f"  Success rate: {success_rate:.1f}% ({df['success'].sum()}/{len(df)})")
        if df['success'].any():
            successful = df[df['success']]
            logger.info(f"  Barrier range: [{successful['oracle_barrier'].min():.3f}, "
                  f"{successful['oracle_barrier'].max():.3f}] eV")
    
    return df


# ============================================================================
# 3. MODEL PREDICTIONS
# ============================================================================

def predict_barriers_for_test_set(
    model_path: str,
    test_data: pd.DataFrame,
    config: Config,
    logger: logging.Logger = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Make barrier predictions for test set.
    
    Args:
        model_path: Path to trained model checkpoint
        test_data: DataFrame from generate_test_data_with_oracle()
        config: Config object
        logger: Logger instance (optional)
        verbose: Print progress
    
    Returns:
        DataFrame with added columns:
        - predicted_barrier (float): Model prediction (eV)
        - relative_error (float): |pred - oracle| / oracle
        - absolute_error (float): |pred - oracle|
    
    Example:
        >>> predictions = predict_barriers_for_test_set(
        ...     'checkpoints/best_model.pt', test_data, config
        ... )
    """
    if verbose and logger:
        logger.info("="*70)
        logger.info("MODEL PREDICTIONS")
        logger.info("="*70)
        logger.info(f"Model: {model_path}")
        logger.info(f"Test samples: {len(test_data)}")
    
    # Load model
    model, checkpoint = load_model_for_inference(model_path, config, validate=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize builder for graph construction
    builder = TemplateGraphBuilder(config)
    
    # Filter only successful oracle calculations
    test_data_filtered = test_data[test_data['success']].copy()
    
    if len(test_data_filtered) == 0:
        if verbose:
            if logger: logger.info(f"\n⚠️  No successful oracle calculations to predict!")
        return pd.DataFrame(columns=[
            'composition', 'oracle_barrier', 'predicted_barrier',
            'absolute_error', 'relative_error', 'structure_folder'
        ])
    
    predictions = []
    
    iterator = tqdm(test_data_filtered.iterrows(), 
                   total=len(test_data_filtered),
                   desc="Predictions") if verbose else test_data_filtered.iterrows()
    
    for idx, row in iterator:
        try:
            # Get structure paths
            structure_folder = Path(row['structure_folder'])
            initial_cif = structure_folder / "initial_relaxed.cif"
            final_cif = structure_folder / "final_relaxed.cif"
            
            # Check if files exist
            if not initial_cif.exists() or not final_cif.exists():
                if verbose:
                    if logger: logger.info(f"\n  Structures not found for {row['composition']}")
                predictions.append({
                    'composition': row['composition'],
                    'oracle_barrier': row['oracle_barrier'],
                    'predicted_barrier': np.nan,
                    'absolute_error': np.nan,
                    'relative_error': np.nan,
                    'structure_folder': row['structure_folder']
                })
                continue
            
            # Build graphs
            initial_graph, final_graph = builder.build_pair_graph(
                str(initial_cif),
                str(final_cif),
                backward_barrier=0.0  # Dummy value
            )
            
            # Move to device
            from torch_geometric.data import Batch
            initial_batch = Batch.from_data_list([initial_graph]).to(device)
            final_batch = Batch.from_data_list([final_graph]).to(device)
            
            # Predict
            with torch.no_grad():
                prediction = model(initial_batch, final_batch)
            
            predicted_barrier = prediction.item()
            
            # Calculate errors
            oracle_barrier = row['oracle_barrier']
            absolute_error = abs(predicted_barrier - oracle_barrier)
            relative_error = absolute_error / oracle_barrier if oracle_barrier != 0 else np.inf
            
            predictions.append({
                'composition': row['composition'],
                'oracle_barrier': oracle_barrier,
                'predicted_barrier': predicted_barrier,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'structure_folder': row['structure_folder']
            })
            
        except Exception as e:
            if verbose:
                if logger: logger.info(f"\n  Prediction error for {row['composition']}: {e}")
            
            predictions.append({
                'composition': row['composition'],
                'oracle_barrier': row['oracle_barrier'],
                'predicted_barrier': np.nan,
                'absolute_error': np.nan,
                'relative_error': np.nan,
                'structure_folder': row['structure_folder']
            })
    
    # Clean GPU after all predictions
    cleanup_gpu()
    
    df = pd.DataFrame(predictions)
    
    if verbose:
        # Safe check for valid predictions
        if 'predicted_barrier' in df.columns and len(df) > 0:
            valid = df.dropna(subset=['predicted_barrier'])
        else:
            valid = pd.DataFrame()
        
        if logger: logger.info(f"\n✓ Predictions completed")
        if logger: logger.info(f"  Valid predictions: {len(valid)}/{len(df)}")
        if len(valid) > 0:
            if logger: logger.info(f"  Mean absolute error: {valid['absolute_error'].mean():.3f} eV")
            if logger: logger.info(f"  Mean relative error: {valid['relative_error'].mean():.3f}")
            if logger: logger.info(f"  Median relative error: {valid['relative_error'].median():.3f}")
        else:
            if logger: logger.info(f"  ⚠️  No valid predictions generated!")
    
    return df


# ============================================================================
# 4. ERROR-WEIGHTED QUERY STRATEGY
# ============================================================================

def select_samples_by_error(
    predictions: pd.DataFrame,
    n_query: int,
    strategy: str = 'error_weighted',
    seed: int = 42,
    logger: logging.Logger = None
) -> List[dict]:
    """
    Select samples based on prediction error.
    
    Strategy 'error_weighted':
    - Each sample's selection probability = its_error / sum(all_errors)
    - Higher error → Higher probability
    - No thresholds, no filters - pure raw data
    
    Args:
        predictions: DataFrame with relative_error column
        n_query: Number of samples to select
        strategy: Query strategy ('error_weighted', ...)
        seed: Random seed
    
    Returns:
        List of selected composition dicts
        Also adds 'selected_for_training' column to predictions
    
    Example:
        >>> selected = select_samples_by_error(predictions, n_query=20)
        >>> if logger: logger.info(f"Selected {len(selected)} samples")
    """
    if strategy == 'error_weighted':
        return error_weighted_sampling(predictions, n_query, seed)
    else:
        raise ValueError(f"Unknown query strategy: {strategy}")


def error_weighted_sampling(
    predictions: pd.DataFrame,
    n_query: int,
    seed: int = 42
) -> List[dict]:
    """
    Pure error-weighted sampling.
    
    Steps:
    1. total_error = sum(all relative_errors)
    2. probability_i = error_i / total_error
    3. Sample n_query without replacement using probabilities
    
    Args:
        predictions: DataFrame with relative_error column
        n_query: Number to select
        seed: Random seed
    
    Returns:
        List of selected composition dicts with metadata
    """
    np.random.seed(seed)
    
    # Filter valid predictions
    valid = predictions.dropna(subset=['relative_error']).copy()
    
    if len(valid) == 0:
        if logger: logger.info(f"Warning: No valid predictions for query selection!")
        return []
    
    # Adjust n_query if necessary
    n_query = min(n_query, len(valid))
    
    # Get relative errors
    errors = valid['relative_error'].values
    
    # Calculate probabilities (error proportional)
    total_error = errors.sum()
    
    if total_error == 0:
        if logger: logger.info(f"Warning: Total error is zero, using uniform sampling")
        probabilities = np.ones(len(errors)) / len(errors)
    else:
        probabilities = errors / total_error
    
    # Sample indices
    selected_indices = np.random.choice(
        len(valid),
        size=n_query,
        replace=False,
        p=probabilities
    )
    
    # Mark selected samples in DataFrame
    predictions['selected_for_training'] = False
    predictions.loc[valid.iloc[selected_indices].index, 'selected_for_training'] = True
    
    # Get selected compositions
    selected_rows = valid.iloc[selected_indices]
    
    selected_compositions = []
    for _, row in selected_rows.iterrows():
        # Parse composition string back to dict
        comp_dict = string_to_composition(row['composition'])
        
        selected_compositions.append({
            'composition': comp_dict,
            'composition_str': row['composition'],
            'oracle_barrier': row['oracle_barrier'],
            'predicted_barrier': row['predicted_barrier'],
            'relative_error': row['relative_error'],
            'structure_folder': row['structure_folder']
        })
    
    return selected_compositions


# ============================================================================
# 5. CSV MANAGEMENT (DISABLED - not needed anymore)
# ============================================================================

# def save_cycle_predictions(
#     predictions: pd.DataFrame,
#     cycle: int,
#     output_dir: str = "active_learning_results"
# ):
#     """
#     Save predictions CSV for this cycle.
#     DISABLED: Active learning reports not needed anymore.
#     """
#     pass


def load_cycle_predictions(
    cycle: int,
    output_dir: str = "active_learning_results"
) -> pd.DataFrame:
    """
    Load predictions from specific cycle.
    
    Args:
        cycle: Cycle number
        output_dir: Output directory
    
    Returns:
        Predictions DataFrame
    """
    filename = Path(output_dir) / f"cycle_{cycle}_predictions.csv"
    
    if not filename.exists():
        raise FileNotFoundError(f"Predictions file not found: {filename}")
    
    return pd.read_csv(filename)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def composition_to_string(comp: dict) -> str:
    """
    Format composition dict to string.
    
    Args:
        comp: {'Mo': 0.2, 'Nb': 0.3, ...}
    
    Returns:
        String: 'Mo0.200Nb0.300O0.200Ta0.150W0.150'
    
    Example:
        >>> composition_to_string({'Mo': 0.2, 'Nb': 0.3, 'O': 0.5})
        'Mo0.200Nb0.300O0.500'
    """
    return "".join(f"{elem}{frac:.3f}" for elem, frac in sorted(comp.items()))


def string_to_composition(comp_str: str) -> dict:
    """
    Parse composition string to dict.
    
    Args:
        comp_str: 'Mo0.200Nb0.300O0.500'
    
    Returns:
        dict: {'Mo': 0.2, 'Nb': 0.3, 'O': 0.5}
    
    Example:
        >>> string_to_composition('Mo0.200Nb0.300O0.500')
        {'Mo': 0.2, 'Nb': 0.3, 'O': 0.5}
    """
    import re
    pattern = r'([A-Z][a-z]?)([0-9.]+)'
    matches = re.findall(pattern, comp_str)
    return {elem: float(frac) for elem, frac in matches}


# ============================================================================
# COMPLETE CYCLE FUNCTION
# ============================================================================

def run_inference_cycle(
    cycle: int,
    model_path: str,
    oracle: Oracle,
    config: Config,
    verbose: bool = True
) -> Tuple[List[dict], pd.DataFrame]:
    """
    Complete inference cycle for active learning.
    
    Steps:
    1. Generate n_test compositions (uniform grid)
    2. Oracle generates ground truth barriers + GPU cleanup
    3. Model predicts barriers + GPU cleanup
    4. Calculate errors (relative & absolute)
    5. Select n_query samples (error-weighted)
    6. Return selected compositions for training
    
    Args:
        cycle: Cycle number (for logging/saving)
        model_path: Path to trained model checkpoint
        oracle: Oracle instance
        config: Config object with AL parameters
        verbose: Print detailed progress
    
    Returns:
        selected_compositions: List of dicts with selected samples
        predictions_df: Full predictions DataFrame
    
    Example:
        >>> selected, predictions = run_inference_cycle(
        ...     cycle=0,
        ...     model_path='checkpoints/best_model.pt',
        ...     oracle=oracle,
        ...     config=config
        ... )
        >>> logger.info(f"Selected {len(selected)} samples for next training")
    """
    # Setup logger for this cycle
    logger = setup_inference_logger(cycle, config)
    
    if verbose:
        logger.info("="*70)
        logger.info(f"INFERENCE CYCLE {cycle}")
        logger.info("="*70)
        logger.info(f"Test samples (n_test): {config.al_n_test}")
        logger.info(f"Query samples (n_query): {config.al_n_query}")
        logger.info(f"Test strategy: {config.al_test_strategy}")
        logger.info(f"Query strategy: {config.al_query_strategy}")
    
    # 1. Generate test compositions
    builder = TemplateGraphBuilder(config)
    elements = builder.elements
    
    test_compositions = generate_test_compositions(
        elements=elements,
        n_test=config.al_n_test,
        strategy=config.al_test_strategy,
        seed=config.al_seed + cycle  # Different seed per cycle
    )
    
    # 2. Oracle generates test data (ground truth)
    test_data = generate_test_data_with_oracle(
        compositions=test_compositions,
        oracle=oracle,
        config=config,
        logger=logger,
        verbose=verbose
    )
    
    # 3. Model predictions
    predictions = predict_barriers_for_test_set(
        model_path=model_path,
        test_data=test_data,
        config=config,
        logger=logger,
        verbose=verbose
    )
    
    # 4. Select samples by error
    selected_compositions = select_samples_by_error(
        predictions=predictions,
        n_query=config.al_n_query,
        strategy=config.al_query_strategy,
        seed=config.al_seed + cycle,
        logger=logger
    )
    
    if verbose:
        logger.info(f"✓ Selected {len(selected_compositions)} samples for training")
        if len(selected_compositions) > 0:
            errors = [s['relative_error'] for s in selected_compositions]
            logger.info(f"  Error range: [{min(errors):.3f}, {max(errors):.3f}]")
            logger.info(f"  Mean error: {np.mean(errors):.3f}")
    
    # Note: save_cycle_predictions() removed (Issue #4)
    
    if verbose:
        logger.info("="*70)
    
    return selected_compositions, predictions


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    if logger: logger.info(f"inference.py - Import and use run_inference_cycle()")
    if logger: logger.info(f"\nExample:")
    if logger: logger.info(f"  from inference import run_inference_cycle")
    if logger: logger.info(f"  selected, predictions = run_inference_cycle(")
    if logger: logger.info(f"      cycle=0,")
    if logger: logger.info(f"      model_path='checkpoints/best_model.pt',")
    if logger: logger.info(f"      oracle=oracle,")
    if logger: logger.info(f"      config=config")
    if logger: logger.info(f"  )")