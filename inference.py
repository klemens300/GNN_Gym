"""
Inference and Active Learning Query Module
UPDATED: Uses unrelaxed .npz files for realistic KMC-like prediction.
"""

import numpy as np
import pandas as pd
import torch
import gc
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from config import Config
from oracle import Oracle
from utils import load_model_for_inference
from graph_builder import GraphBuilder

# ============================================================================
# GPU CLEANUP
# ============================================================================

def cleanup_gpu():
    """Clean GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# LOGGER SETUP
# ============================================================================

def setup_inference_logger(cycle: int, config: Config):
    """Setup logger for inference cycle."""
    logger = logging.getLogger(f"inference_cycle_{cycle}")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    # No handler reset logic here to avoid messing with main loop logging
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
    """Generate test compositions."""
    np.random.seed(seed)
    
    if strategy == 'uniform':
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
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Generate test data using Oracle (No TQDM, clean logging)."""
    results = []
    total = len(compositions)
    
    if logger:
        logger.info(f"Generating test data for {total} compositions...")

    for i, comp in enumerate(compositions, 1):
        if logger and (i % 50 == 0 or i == 1):
            logger.info(f"  Test Data Gen: {i}/{total}")
            for h in logger.handlers: h.flush()
            
        success = oracle.calculate(comp)
        
        if success:
            df = pd.read_csv(config.csv_path)
            last_entry = df.iloc[-1]
            results.append({
                'composition_string': last_entry['composition_string'],
                'structure_folder': last_entry['structure_folder'],
                'backward_barrier_eV': last_entry['backward_barrier_eV']
            })
    
    cleanup_gpu()
    return pd.DataFrame(results)

# ============================================================================
# 3. MODEL PREDICTIONS (UPDATED FOR UNRELAXED NPZ)
# ============================================================================

def predict_barriers_for_test_set(
    model_path: str,
    test_data: pd.DataFrame,
    config: Config,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Predict barriers using UNRELAXED structures from .npz files.
    This simulates the real KMC use-case.
    """
    model, checkpoint = load_model_for_inference(model_path, config)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    builder = GraphBuilder(config, csv_path=config.csv_path)
    predictions = []
    
    total = len(test_data)
    if logger:
        logger.info(f"Predicting {total} barriers using unrelaxed structures...")

    for i, (idx, row) in enumerate(test_data.iterrows(), 1):
        if logger and (i % 100 == 0):
            logger.info(f"  Prediction: {i}/{total}")
        
        structure_folder = Path(row['structure_folder'])
        oracle_barrier = row['backward_barrier_eV']
        
        if not structure_folder.is_absolute():
            csv_parent = Path(config.csv_path).parent
            structure_folder = csv_parent / structure_folder
        
        # ?? CRITICAL UPDATE: Load UNRELAXED .npz files
        initial_file = structure_folder / "initial_unrelaxed.npz"
        final_file = structure_folder / "final_unrelaxed.npz"
        
        # Fallback to CIF if NPZ missing (legacy data)
        if not initial_file.exists(): initial_file = structure_folder / "initial_unrelaxed.cif"
        if not final_file.exists(): final_file = structure_folder / "final_unrelaxed.cif"
        
        # Build graphs with forced progress=0.0 (Unrelaxed)
        initial_graph, final_graph = builder.build_pair_graph(
            str(initial_file),
            str(final_file),
            backward_barrier=oracle_barrier,
            progress_initial=0.0,  # Force unrelaxed state
            progress_final=0.0     # Force unrelaxed state
        )
        
        initial_graph = initial_graph.to(device)
        final_graph = final_graph.to(device)
        
        with torch.no_grad():
            from torch_geometric.data import Batch
            initial_batch = Batch.from_data_list([initial_graph])
            final_batch = Batch.from_data_list([final_graph])
            
            prediction = model(initial_batch, final_batch)
            predicted_barrier = prediction.item()
        
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
    
    cleanup_gpu()
    return pd.DataFrame(predictions)

# ============================================================================
# 4. QUERY STRATEGY
# ============================================================================

def select_samples_by_error(
    predictions: pd.DataFrame,
    n_query: int = 10,
    strategy: str = 'mixed',
    seed: int = 42,
    config: Optional[Config] = None,
    logger: Optional[logging.Logger] = None
) -> List[dict]:
    """Select samples (Mixed or Error Weighted)."""
    np.random.seed(seed)
    
    if strategy == 'mixed':
        if config is None: raise ValueError("Config required for mixed sampling")
        return select_samples_mixed(predictions, n_query, config.al_mixed_exploitation_ratio, config.al_error_cap_multiplier, seed, logger)
    
    elif strategy == 'error_weighted':
        weights = predictions['relative_error'].values
        weights = weights / weights.sum()
        indices = np.random.choice(len(predictions), size=min(n_query, len(predictions)), replace=False, p=weights)
        
        selected = []
        for idx in indices:
            row = predictions.iloc[idx]
            selected.append({
                'composition_str': row['composition'],
                'structure_folder': row['structure_folder'],
                'oracle_barrier': row['oracle_barrier'],
                'predicted_barrier': row['predicted_barrier'],
                'relative_error': row['relative_error']
            })
        return selected
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def select_samples_mixed(
    predictions: pd.DataFrame,
    n_query: int,
    exploitation_ratio: float,
    error_cap_multiplier: float,
    seed: int,
    logger: Optional[logging.Logger]
) -> List[dict]:
    """Mixed sampling: Exploitation + Exploration."""
    np.random.seed(seed)
    n_query = min(n_query, len(predictions))
    n_exploit = int(n_query * exploitation_ratio)
    n_explore = n_query - n_exploit
    
    # Exploitation
    if n_exploit > 0:
        rel_errors = predictions['relative_error'].values
        median_error = np.median(rel_errors)
        capped_errors = np.minimum(rel_errors, error_cap_multiplier * median_error)
        weights = capped_errors / capped_errors.sum()
        exploit_indices = np.random.choice(len(predictions), size=n_exploit, replace=False, p=weights)
    else:
        exploit_indices = np.array([], dtype=int)
        
    # Exploration
    if n_explore > 0:
        remaining = list(set(range(len(predictions))) - set(exploit_indices))
        if len(remaining) < n_explore:
            explore_indices = np.array(remaining)
        else:
            explore_indices = np.random.choice(remaining, size=n_explore, replace=False)
    else:
        explore_indices = np.array([], dtype=int)
        
    selected_indices = np.concatenate([exploit_indices, explore_indices])
    
    selected = []
    for idx in selected_indices:
        row = predictions.iloc[idx]
        selected.append({
            'composition_str': row['composition'],
            'structure_folder': row['structure_folder'],
            'oracle_barrier': row['oracle_barrier'],
            'predicted_barrier': row['predicted_barrier'],
            'relative_error': row['relative_error']
        })
    return selected

# ============================================================================
# 5. QUERY GENERATION
# ============================================================================

def parse_composition_string(comp_str: str) -> dict:
    import re
    matches = re.findall(r'([A-Z][a-z]?)(\d+\.\d+)', comp_str)
    return {el: float(frac) for el, frac in matches}

def generate_query_compositions_from_selected(
    selected_samples: List[dict],
    n_query: int,
    elements: List[str],
    strategy: str = 'nearby',
    noise_level: float = 0.1,
    seed: int = 42
) -> List[dict]:
    np.random.seed(seed)
    query_compositions = []
    n_selected = len(selected_samples)
    base_indices = np.random.choice(n_selected, size=n_query, replace=True)

    for idx in base_indices:
        sample = selected_samples[idx]
        base_comp = parse_composition_string(sample['composition_str'])
        comp_array = np.array([base_comp.get(elem, 0.0) for elem in elements])
        
        alpha = comp_array / noise_level + 1e-6
        alpha = np.maximum(alpha, 0.1)
        noisy_comp = np.random.dirichlet(alpha)
        
        query_compositions.append({elem: float(frac) for elem, frac in zip(elements, noisy_comp)})

    return query_compositions

# ============================================================================
# 6. CONVERGENCE TRACKER
# ============================================================================

class ConvergenceTracker:
    def __init__(self, config: Config):
        self.config = config
        self.history = []
        self.best_mae = float('inf')
        self.best_rel_mae = float('inf')
        self.cycles_without_improvement = 0
    
    def update(self, cycle: int, mae: float, rel_mae: float) -> bool:
        self.history.append({'cycle': cycle, 'mae': mae, 'rel_mae': rel_mae})
        
        metric = self.config.al_convergence_metric
        current = mae if metric == 'mae' else rel_mae
        best = self.best_mae if metric == 'mae' else self.best_rel_mae
        threshold = self.config.al_convergence_threshold_mae if metric == 'mae' else self.config.al_convergence_threshold_rel_mae
        
        if current < best - threshold:
            if metric == 'mae': self.best_mae = current
            else: self.best_rel_mae = current
            self.cycles_without_improvement = 0
            return False
        else:
            self.cycles_without_improvement += 1
            return self.cycles_without_improvement >= self.config.al_convergence_patience
    
    def get_summary(self) -> dict:
        return {
            'history': self.history,
            'best_mae': self.best_mae,
            'converged': self.cycles_without_improvement >= self.config.al_convergence_patience
        }

def save_convergence_history(tracker: ConvergenceTracker, save_path: str):
    import json
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(tracker.get_summary(), f, indent=2)

# ============================================================================
# 7. INFERENCE CYCLE
# ============================================================================

def run_inference_cycle(
    cycle: int,
    model_path: str,
    oracle: Oracle,
    config: Config,
    convergence_tracker: Optional[ConvergenceTracker] = None,
    verbose: bool = True
) -> Tuple[List[dict], pd.DataFrame]:
    """Run complete inference cycle."""
    logger = setup_inference_logger(cycle, config)
    logger.info(f"INFERENCE CYCLE {cycle}")
    
    # 1. Test compositions
    test_compositions = generate_test_compositions(
        config.elements, config.al_test_set_size, 
        config.al_test_set_strategy, config.al_seed + cycle
    )
    
    # 2. Test data (Oracle)
    test_data = generate_test_data_with_oracle(test_compositions, oracle, config, logger)
    
    # 3. Predict (Unrelaxed NPZ)
    predictions = predict_barriers_for_test_set(model_path, test_data, config, logger)
    
    mae = predictions['absolute_error'].mean()
    rel_mae = predictions['relative_error'].mean()
    logger.info(f"Stats: MAE={mae:.4f} eV, RelMAE={rel_mae:.4f}")
    
    # 4. Convergence
    if convergence_tracker:
        converged = convergence_tracker.update(cycle, mae, rel_mae)
        logger.info(f"Converged: {converged}")
    
    # 5. Select
    selected = select_samples_by_error(
        predictions, config.al_n_query, config.al_query_strategy,
        config.al_seed + cycle, config, logger
    )
    logger.info(f"Selected {len(selected)} samples")
    
    # 6. Save
    results_dir = Path(config.al_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results_dir / f"cycle_{cycle}_predictions.csv", index=False)
    
    return selected, predictions