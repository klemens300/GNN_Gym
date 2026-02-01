"""
Inference and Active Learning Query Module
UPDATED: Handles Target Normalization & Removed Progress Input.
"""

import numpy as np
import pandas as pd
import torch
import gc
import logging
from pathlib import Path
from typing import List, Optional

from config import Config
from oracle import Oracle
from utils import load_model_for_inference
from graph_builder import GraphBuilder

def cleanup_gpu():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

def setup_inference_logger(cycle: int, config: Config):
    logger = logging.getLogger(f"inference_cycle_{cycle}")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    return logger

def predict_barriers_for_test_set(
    model_path: str,
    test_data: pd.DataFrame,
    config: Config,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Predict barriers using UNRELAXED structures from .npz files.
    Denormalizes predictions using stats from checkpoint.
    """
    # Load Model & Checkpoint (contains normalization stats)
    model, checkpoint = load_model_for_inference(model_path, config)
    model.eval()
    
    # Extract Normalization Stats
    target_mean = checkpoint.get('target_mean', 0.0)
    target_std = checkpoint.get('target_std', 1.0)
    
    if logger:
        logger.info(f"Loaded model. Normalization: Mean={target_mean:.4f}, Std={target_std:.4f}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    builder = GraphBuilder(config, csv_path=config.csv_path)
    predictions = []
    
    total = len(test_data)
    if logger: logger.info(f"Predicting {total} barriers...")

    for i, (idx, row) in enumerate(test_data.iterrows(), 1):
        if logger and i % 100 == 0: logger.info(f"  Prediction: {i}/{total}")
        
        structure_folder = Path(row['structure_folder'])
        oracle_barrier = row['backward_barrier_eV']
        
        if not structure_folder.is_absolute():
            structure_folder = Path(config.csv_path).parent / structure_folder
        
        # Use Unrelaxed (traj_0) or Fallback
        initial_file = structure_folder / "initial_traj_0.npz"
        final_file = structure_folder / "final_traj_0.npz"
        
        if not initial_file.exists(): initial_file = structure_folder / "initial_relaxed.npz"
        if not final_file.exists(): final_file = structure_folder / "final_relaxed.npz"
        if not initial_file.exists(): initial_file = structure_folder / "initial_relaxed.cif"
        if not final_file.exists(): final_file = structure_folder / "final_relaxed.cif"
        
        # Build graphs
        initial_graph, final_graph = builder.build_pair_graph(
            str(initial_file), str(final_file), backward_barrier=oracle_barrier
        )
        
        initial_graph = initial_graph.to(device)
        final_graph = final_graph.to(device)
        
        with torch.no_grad():
            from torch_geometric.data import Batch
            initial_batch = Batch.from_data_list([initial_graph])
            final_batch = Batch.from_data_list([final_graph])
            
            # Prediction (Normalized)
            pred_norm = model(initial_batch, final_batch)
            
            # De-normalize: Pred = Norm * Std + Mean
            predicted_barrier = pred_norm.item() * target_std + target_mean
        
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

# --- RE-USE EXISTING HELPER FUNCTIONS ---

def select_samples_by_error(
    predictions: pd.DataFrame,
    n_query: int = 10,
    strategy: str = 'mixed',
    seed: int = 42,
    config: Optional[Config] = None,
    logger: Optional[logging.Logger] = None
) -> List[dict]:
    np.random.seed(seed)
    
    if strategy == 'mixed':
        return select_samples_mixed(predictions, n_query, config.al_mixed_exploitation_ratio, config.al_error_cap_multiplier, seed, logger)
    
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

def select_samples_mixed(predictions, n_query, exploitation_ratio, error_cap_multiplier, seed, logger):
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
    else: exploit_indices = np.array([], dtype=int)
        
    # Exploration
    if n_explore > 0:
        remaining = list(set(range(len(predictions))) - set(exploit_indices))
        if len(remaining) < n_explore: explore_indices = np.array(remaining)
        else: explore_indices = np.random.choice(remaining, size=n_explore, replace=False)
    else: explore_indices = np.array([], dtype=int)
        
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

def generate_query_compositions_from_selected(selected_samples, n_query, elements, strategy='nearby', noise_level=0.1, seed=42):
    np.random.seed(seed)
    query_compositions = []
    base_indices = np.random.choice(len(selected_samples), size=n_query, replace=True)

    for idx in base_indices:
        sample = selected_samples[idx]
        import re
        matches = re.findall(r'([A-Z][a-z]?)(\d+\.\d+)', sample['composition_str'])
        base_comp = {el: float(frac) for el, frac in matches}
        
        comp_array = np.array([base_comp.get(elem, 0.0) for elem in elements])
        alpha = comp_array / noise_level + 1e-6
        alpha = np.maximum(alpha, 0.1)
        noisy_comp = np.random.dirichlet(alpha)
        
        query_compositions.append({elem: float(frac) for elem, frac in zip(elements, noisy_comp)})
    return query_compositions

class ConvergenceTracker:
    def __init__(self, config):
        self.config = config
        self.best_metric = float('inf')
        self.patience_counter = 0
    
    def update(self, cycle, mae, rel_mae):
        metric = mae if self.config.al_convergence_metric == 'mae' else rel_mae
        threshold = self.config.al_convergence_threshold_mae if self.config.al_convergence_metric == 'mae' else self.config.al_convergence_threshold_rel_mae
        
        if metric < self.best_metric - threshold:
            self.best_metric = metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.al_convergence_patience