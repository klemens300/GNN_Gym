"""
Inference and Active Learning Query Module

Handles:
- Test composition generation
- Oracle-based test data generation
- Model predictions
- Error-based query strategy
- Convergence checking for Active Learning
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
from typing import List, Tuple, Optional
from tqdm import tqdm

from config import Config
from oracle import Oracle
from utils import load_model_for_inference
from graph_builder import GraphBuilder


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
    n_test: int = 100,
    strategy: str = 'uniform',
    seed: int = 42
) -> List[dict]:
    """
    Generate test compositions for prediction.
    
    Args:
        elements: List of element symbols
        n_test: Number of test compositions
        strategy: Sampling strategy ('uniform')
        seed: Random seed
    
    Returns:
        compositions: List of composition dicts {element: fraction}
    
    Example:
        >>> comps = generate_test_compositions(['Mo', 'W'], n_test=10)
        >>> print(comps[0])
        {'Mo': 0.234, 'W': 0.766}
    """
    np.random.seed(seed)
    
    if strategy == 'uniform':
        # Uniform sampling on simplex using Dirichlet
        alpha = np.ones(len(elements))
        fractions = np.random.dirichlet(alpha, size=n_test)
        
        compositions = []
        for frac in fractions:
            comp = {elem: float(f) for elem, f in zip(elements, frac)}
            compositions.append(comp)
        
        return compositions
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# 2. ORACLE-BASED TEST DATA GENERATION
# ============================================================================

def generate_test_data_with_oracle(
    compositions: List[dict],
    oracle: Oracle,
    config: Config,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate test data using Oracle (NEB calculations).
    
    Args:
        compositions: List of composition dicts
        oracle: Oracle instance
        config: Config object
        verbose: Print progress
    
    Returns:
        test_data: DataFrame with columns:
            - composition_string
            - structure_folder
            - backward_barrier_eV
    
    Example:
        >>> test_data = generate_test_data_with_oracle(
        ...     compositions=[{'Mo': 0.5, 'W': 0.5}],
        ...     oracle=oracle,
        ...     config=config
        ... )
    """
    results = []
    
    iterator = tqdm(compositions, desc="Generating test data") if verbose else compositions
    
    for comp in iterator:
        # Run Oracle calculation
        success = oracle.calculate(comp)
        
        if success:
            # Read last entry from CSV
            df = pd.read_csv(config.csv_path)
            last_entry = df.iloc[-1]
            
            results.append({
                'composition_string': last_entry['composition_string'],
                'structure_folder': last_entry['structure_folder'],
                'backward_barrier_eV': last_entry['backward_barrier_eV']
            })
    
    # Cleanup after Oracle operations
    cleanup_gpu()
    
    return pd.DataFrame(results)


# ============================================================================
# 3. MODEL PREDICTIONS
# ============================================================================

def predict_barriers_for_test_set(
    model_path: str,
    test_data: pd.DataFrame,
    config: Config,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Predict barriers for test set using trained model.
    
    Args:
        model_path: Path to trained model checkpoint
        test_data: DataFrame with structure_folder and backward_barrier_eV columns
        config: Config object
        verbose: Print progress
    
    Returns:
        predictions: DataFrame with additional columns:
            - predicted_barrier
            - absolute_error
            - relative_error
    
    Example:
        >>> predictions = predict_barriers_for_test_set(
        ...     'checkpoints/best_model.pt',
        ...     test_data,
        ...     config
        ... )
    """
    # Load model
    model, checkpoint = load_model_for_inference(model_path, config)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Build graph builder
    builder = GraphBuilder(config, csv_path=config.csv_path)
    
    predictions = []
    
    iterator = tqdm(test_data.iterrows(), total=len(test_data), 
                   desc="Predicting barriers") if verbose else test_data.iterrows()
    
    for idx, row in iterator:
        structure_folder = Path(row['structure_folder'])
        
        # ? FIX: Use correct column name from CSV
        oracle_barrier = row['backward_barrier_eV']
        
        # Make path absolute if needed
        if not structure_folder.is_absolute():
            csv_parent = Path(config.csv_path).parent
            structure_folder = csv_parent / structure_folder
        
        # Build graphs
        initial_cif = structure_folder / "initial_relaxed.cif"
        final_cif = structure_folder / "final_relaxed.cif"
        
        initial_graph, final_graph = builder.build_pair_graph(
            str(initial_cif),
            str(final_cif),
            backward_barrier=oracle_barrier
        )
        
        # Move to device
        initial_graph = initial_graph.to(device)
        final_graph = final_graph.to(device)
        
        # Predict
        with torch.no_grad():
            from torch_geometric.data import Batch
            initial_batch = Batch.from_data_list([initial_graph])
            final_batch = Batch.from_data_list([final_graph])
            
            prediction = model(initial_batch, final_batch)
            predicted_barrier = prediction.item()
        
        # Calculate errors
        abs_error = abs(predicted_barrier - oracle_barrier)
        rel_error = abs_error / (oracle_barrier + 1e-8)
        
        predictions.append({
            'composition': row['composition_string'],
            'structure_folder': str(structure_folder),
            'oracle_barrier': oracle_barrier,
            'predicted_barrier': predicted_barrier,
            'absolute_error': abs_error,
            'relative_error': rel_error
        })
    
    # Cleanup after predictions
    cleanup_gpu()
    
    return pd.DataFrame(predictions)


# ============================================================================
# 4. QUERY STRATEGY (ERROR-WEIGHTED)
# ============================================================================

def select_samples_by_error(
    predictions: pd.DataFrame,
    n_query: int = 10,
    strategy: str = 'error_weighted',
    seed: int = 42
) -> List[dict]:
    """
    Select samples for training based on prediction error.
    
    Args:
        predictions: DataFrame with relative_error column
        n_query: Number of samples to select
        strategy: Selection strategy ('error_weighted')
        seed: Random seed
    
    Returns:
        selected_samples: List of dicts with composition info
    
    Example:
        >>> selected = select_samples_by_error(predictions, n_query=5)
        >>> print(selected[0]['composition_str'])
    """
    np.random.seed(seed)
    
    if strategy == 'error_weighted':
        # Sample proportional to relative error
        weights = predictions['relative_error'].values
        weights = weights / weights.sum()
        
        # Sample without replacement
        n_query = min(n_query, len(predictions))
        indices = np.random.choice(
            len(predictions),
            size=n_query,
            replace=False,
            p=weights
        )
        
        selected_samples = []
        for idx in indices:
            row = predictions.iloc[idx]
            selected_samples.append({
                'composition_str': row['composition'],
                'structure_folder': row['structure_folder'],
                'oracle_barrier': row['oracle_barrier'],
                'predicted_barrier': row['predicted_barrier'],
                'relative_error': row['relative_error']
            })
        
        return selected_samples
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# 4.5. QUERY COMPOSITION GENERATION
# ============================================================================

def parse_composition_string(comp_str: str) -> dict:
    """
    Parse composition string to dict.

    Args:
        comp_str: Composition string like 'Mo0.50W0.50'

    Returns:
        composition: Dict {element: fraction}

    Example:
        >>> comp = parse_composition_string('Mo0.50W0.50')
        >>> print(comp)
        {'Mo': 0.50, 'W': 0.50}
    """
    import re
    pattern = r'([A-Z][a-z]?)(\d+\.\d+)'
    matches = re.findall(pattern, comp_str)

    composition = {}
    for element, fraction in matches:
        composition[element] = float(fraction)

    return composition


def generate_query_compositions_from_selected(
    selected_samples: List[dict],
    n_query: int,
    elements: List[str],
    strategy: str = 'nearby',
    noise_level: float = 0.1,
    seed: int = 42
) -> List[dict]:
    """
    Generate new query compositions based on selected high-error samples.

    Strategy:
    - 'nearby': Generate compositions near high-error samples with random noise
    - 'interpolate': Generate compositions by interpolating between high-error samples

    Args:
        selected_samples: List of selected samples from select_samples_by_error
        n_query: Number of query compositions to generate
        elements: List of element symbols
        strategy: Generation strategy ('nearby', 'interpolate')
        noise_level: Amount of noise to add (0.0 to 1.0)
        seed: Random seed

    Returns:
        query_compositions: List of composition dicts

    Example:
        >>> query_comps = generate_query_compositions_from_selected(
        ...     selected_samples=[{'composition_str': 'Mo0.50W0.50', ...}],
        ...     n_query=100,
        ...     elements=['Mo', 'W']
        ... )
    """
    np.random.seed(seed)

    if not selected_samples:
        raise ValueError("No selected samples provided")

    query_compositions = []

    if strategy == 'nearby':
        # Generate compositions near high-error samples
        # Sample selected_samples with replacement to get n_query base compositions
        n_selected = len(selected_samples)
        base_indices = np.random.choice(n_selected, size=n_query, replace=True)

        for idx in base_indices:
            sample = selected_samples[idx]
            # Parse composition string
            base_comp = parse_composition_string(sample['composition_str'])

            # Add noise to composition
            comp_array = np.array([base_comp.get(elem, 0.0) for elem in elements])

            # Add Dirichlet noise (keeps on simplex)
            # Higher alpha = less noise
            alpha = comp_array / noise_level + 1e-6
            alpha = np.maximum(alpha, 0.1)  # Ensure positive
            noisy_comp = np.random.dirichlet(alpha)

            # Create composition dict
            new_comp = {elem: float(frac) for elem, frac in zip(elements, noisy_comp)}
            query_compositions.append(new_comp)

    elif strategy == 'interpolate':
        # Generate compositions by interpolating between pairs of high-error samples
        n_selected = len(selected_samples)

        for i in range(n_query):
            # Pick two random samples
            idx1, idx2 = np.random.choice(n_selected, size=2, replace=True)
            sample1 = selected_samples[idx1]
            sample2 = selected_samples[idx2]

            # Parse compositions
            comp1 = parse_composition_string(sample1['composition_str'])
            comp2 = parse_composition_string(sample2['composition_str'])

            # Random interpolation weight
            weight = np.random.random()

            # Interpolate
            new_comp = {}
            for elem in elements:
                frac1 = comp1.get(elem, 0.0)
                frac2 = comp2.get(elem, 0.0)
                new_comp[elem] = weight * frac1 + (1 - weight) * frac2

            # Normalize to sum to 1.0
            total = sum(new_comp.values())
            if total > 0:
                new_comp = {elem: frac / total for elem, frac in new_comp.items()}

            query_compositions.append(new_comp)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return query_compositions


def generate_and_calculate_query_data(
    selected_samples: List[dict],
    oracle: Oracle,
    config: Config,
    verbose: bool = False
) -> int:
    """
    Generate and calculate query compositions based on high-error samples.

    This function:
    1. Generates new compositions near high-error samples
    2. Calculates barriers with Oracle (adds to CSV)
    3. Returns number of successful calculations

    Args:
        selected_samples: List of selected high-error samples
        oracle: Oracle instance
        config: Config object
        verbose: Print progress

    Returns:
        n_successful: Number of successfully calculated query samples

    Example:
        >>> n_added = generate_and_calculate_query_data(
        ...     selected_samples=selected,
        ...     oracle=oracle,
        ...     config=config
        ... )
        >>> print(f"Added {n_added} query samples to CSV")
    """
    elements = list(config.elements)
    n_query = config.al_n_query

    # Generate query compositions
    query_compositions = generate_query_compositions_from_selected(
        selected_samples=selected_samples,
        n_query=n_query,
        elements=elements,
        strategy='nearby',
        noise_level=0.1,
        seed=config.al_seed
    )

    # Calculate with Oracle
    n_successful = 0
    iterator = tqdm(query_compositions, desc="Calculating query data") if verbose else query_compositions

    for comp in iterator:
        success = oracle.calculate(comp)
        if success:
            n_successful += 1

    # Cleanup after Oracle operations
    cleanup_gpu()

    return n_successful


# ============================================================================
# 5. CONVERGENCE CHECKING
# ============================================================================

class ConvergenceTracker:
    """
    Track convergence metrics across Active Learning cycles.
    
    Tracks MAE and relative MAE to determine if model has converged.
    """
    
    def __init__(self, config: Config):
        """
        Initialize convergence tracker.
        
        Args:
            config: Config object with convergence parameters
        """
        self.config = config
        self.history = []
        self.best_mae = float('inf')
        self.best_rel_mae = float('inf')
        self.cycles_without_improvement = 0
    
    def update(self, cycle: int, mae: float, rel_mae: float) -> bool:
        """
        Update tracker with new cycle metrics.
        
        Args:
            cycle: Cycle number
            mae: Mean Absolute Error (eV)
            rel_mae: Relative MAE
        
        Returns:
            converged: True if convergence criteria met
        """
        self.history.append({
            'cycle': cycle,
            'mae': mae,
            'rel_mae': rel_mae
        })
        
        # Check for improvement based on selected metric
        metric = self.config.al_convergence_metric
        
        if metric == "mae":
            current_value = mae
            best_value = self.best_mae
            threshold = self.config.al_convergence_threshold_mae
        elif metric == "rel_mae":
            current_value = rel_mae
            best_value = self.best_rel_mae
            threshold = self.config.al_convergence_threshold_rel_mae
        else:
            raise ValueError(f"Unknown convergence metric: {metric}")
        
        # Check if improved
        if current_value < best_value - threshold:
            # Significant improvement
            if metric == "mae":
                self.best_mae = current_value
            else:
                self.best_rel_mae = current_value
            
            self.cycles_without_improvement = 0
            return False
        else:
            # No significant improvement
            self.cycles_without_improvement += 1
            
            if self.cycles_without_improvement >= self.config.al_convergence_patience:
                return True  # Converged
            else:
                return False
    
    def get_summary(self) -> dict:
        """Get convergence summary."""
        return {
            'history': self.history,
            'best_mae': self.best_mae,
            'best_rel_mae': self.best_rel_mae,
            'cycles_without_improvement': self.cycles_without_improvement,
            'converged': self.cycles_without_improvement >= self.config.al_convergence_patience
        }


def save_convergence_history(tracker: ConvergenceTracker, save_path: str):
    """
    Save convergence history to file.
    
    Args:
        tracker: ConvergenceTracker instance
        save_path: Path to save JSON file
    """
    import json
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = tracker.get_summary()
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# 6. COMPLETE INFERENCE CYCLE
# ============================================================================

def run_inference_cycle(
    cycle: int,
    model_path: str,
    oracle: Oracle,
    config: Config,
    convergence_tracker: Optional[ConvergenceTracker] = None,
    verbose: bool = True
) -> Tuple[List[dict], pd.DataFrame]:
    """
    Run complete inference cycle for Active Learning.
    
    Workflow:
    1. Generate test compositions
    2. Calculate barriers with Oracle
    3. Predict with model
    4. Select high-error samples for training
    5. Check convergence (optional)
    
    Args:
        cycle: Cycle number
        model_path: Path to trained model
        oracle: Oracle instance
        config: Config object
        convergence_tracker: ConvergenceTracker instance (optional)
        verbose: Print progress
    
    Returns:
        selected_samples: List of selected samples for next training
        predictions: Full predictions DataFrame
    
    Example:
        >>> tracker = ConvergenceTracker(config)
        >>> selected, preds = run_inference_cycle(
        ...     cycle=0,
        ...     model_path='checkpoints/cycle_0/best_model.pt',
        ...     oracle=oracle,
        ...     config=config,
        ...     convergence_tracker=tracker
        ... )
    """
    logger = setup_inference_logger(cycle, config)
    
    logger.info("="*70)
    logger.info(f"INFERENCE CYCLE {cycle}")
    logger.info("="*70)
    
    # Get elements from config
    elements = list(config.elements)
    
    # 1. Generate test compositions
    logger.info(f"Step 1: Generating {config.al_n_test} test compositions")
    test_compositions = generate_test_compositions(
        elements=elements,
        n_test=config.al_n_test,
        strategy=config.al_test_strategy,
        seed=config.al_seed + cycle
    )
    logger.info(f"  Generated {len(test_compositions)} compositions")
    
    # 2. Generate test data with Oracle
    logger.info(f"Step 2: Calculating barriers with Oracle")
    test_data = generate_test_data_with_oracle(
        compositions=test_compositions,
        oracle=oracle,
        config=config,
        verbose=verbose
    )
    logger.info(f"  Calculated {len(test_data)} barriers")
    
    # 3. Predict with model
    logger.info(f"Step 3: Predicting barriers with model")
    predictions = predict_barriers_for_test_set(
        model_path=model_path,
        test_data=test_data,
        config=config,
        verbose=verbose
    )
    logger.info(f"  Made {len(predictions)} predictions")
    
    # Calculate statistics
    mae = predictions['absolute_error'].mean()
    rel_mae = predictions['relative_error'].mean()
    max_error = predictions['absolute_error'].max()
    
    logger.info(f"\nPrediction Statistics:")
    logger.info(f"  MAE: {mae:.4f} eV")
    logger.info(f"  Relative MAE: {rel_mae:.4f}")
    logger.info(f"  Max error: {max_error:.4f} eV")
    
    # 4. Update convergence tracker
    if convergence_tracker is not None:
        converged = convergence_tracker.update(cycle, mae, rel_mae)
        logger.info(f"\nConvergence Status:")
        logger.info(f"  Metric: {config.al_convergence_metric}")
        logger.info(f"  Best MAE: {convergence_tracker.best_mae:.4f} eV")
        logger.info(f"  Best Rel MAE: {convergence_tracker.best_rel_mae:.4f}")
        logger.info(f"  Cycles without improvement: {convergence_tracker.cycles_without_improvement}")
        logger.info(f"  Converged: {converged}")
    
    # 5. Select samples for training
    logger.info(f"\nStep 4: Selecting {config.al_n_query} samples for training")
    selected_samples = select_samples_by_error(
        predictions=predictions,
        n_query=config.al_n_query,
        strategy=config.al_query_strategy,
        seed=config.al_seed + cycle
    )
    logger.info(f"  Selected {len(selected_samples)} samples")
    
    # Show top errors
    logger.info(f"\nTop 5 selected samples by error:")
    for i, sample in enumerate(selected_samples[:5], 1):
        logger.info(
            f"  {i}. {sample['composition_str']}: "
            f"Oracle={sample['oracle_barrier']:.3f} eV, "
            f"Pred={sample['predicted_barrier']:.3f} eV, "
            f"RelErr={sample['relative_error']:.3f}"
        )
    
    # 6. Save predictions
    results_dir = Path(config.al_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_file = results_dir / f"cycle_{cycle}_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    logger.info(f"\nPredictions saved: {predictions_file}")
    
    logger.info("="*70)
    
    return selected_samples, predictions


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("INFERENCE MODULE")
    print("="*70)
    
    print("\nFeatures:")
    print("  1. Test composition generation")
    print("  2. Oracle-based test data generation")
    print("  3. Model predictions")
    print("  4. Error-weighted query strategy")
    print("  5. Convergence checking")
    print("  6. Complete inference cycle")
    
    print("\nUsage:")
    print("  from inference import run_inference_cycle, ConvergenceTracker")
    print("")
    print("  tracker = ConvergenceTracker(config)")
    print("  selected, predictions = run_inference_cycle(")
    print("      cycle=0,")
    print("      model_path='checkpoints/cycle_0/best_model.pt',")
    print("      oracle=oracle,")
    print("      config=config,")
    print("      convergence_tracker=tracker")
    print("  )")
    
    print("\n" + "="*70)