"""
Dataset for Diffusion Barrier Prediction.

Handles loading of molecular structures and energy barriers from CSV index.
New features:
- Trajectory Sampling: In training mode, randomizes between unrelaxed/relaxed frames.
- Progress Tracking: Returns relaxation progress for weighted loss.
- Robust Filtering: Handles barriers and element types.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
# --- FIX: Import Subset and DataLoader ---
from torch.utils.data import Dataset, DataLoader, Subset
from graph_builder import GraphBuilder

class DiffusionBarrierDataset(Dataset):
    """
    PyTorch Dataset for barrier prediction.
    
    Modes:
    - 'train': Enables trajectory sampling (data augmentation via intermediate states).
    - 'val'/'test': Uses standard relaxed structures for consistent evaluation.
    """
    
    def __init__(self, csv_path, config, mode='train'):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to the data summary CSV
            config: Configuration object
            mode: 'train', 'val', or 'test'
        """
        self.config = config
        self.mode = mode
        self.csv_path = csv_path
        
        # Load Dataframe
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV database not found at {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        original_len = len(self.df)
        
        # ---------------------------------------------------------------------
        # FILTERING (Preserve original logic)
        # ---------------------------------------------------------------------
        
        # 1. Filter by barrier limits (remove outliers/failed runs)
        if hasattr(config, 'min_barrier') and config.min_barrier is not None:
            self.df = self.df[self.df['backward_barrier_eV'] >= config.min_barrier]
        
        if hasattr(config, 'max_barrier') and config.max_barrier is not None:
            self.df = self.df[self.df['backward_barrier_eV'] <= config.max_barrier]
            
        # 2. Filter by allowed elements (sanity check)
        if hasattr(config, 'elements'):
            allowed_elements = set(config.elements)
            valid_indices = []
            for idx, row in self.df.iterrows():
                # Check if diffusing element is allowed
                if row['diffusing_element'] in allowed_elements:
                    valid_indices.append(idx)
            
            self.df = self.df.loc[valid_indices].reset_index(drop=True)
            
        print(f"Dataset ({mode}) loaded: {len(self.df)} samples (filtered from {original_len})")
        
        # Initialize Graph Builder
        self.graph_builder = GraphBuilder(config, csv_path=csv_path, profile=False, use_cache=True)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            initial_graph: PyG Data object (random frame from trajectory if train)
            final_graph: PyG Data object (random frame from trajectory if train)
            barrier: float (target label)
            progress: float (0.0 = unrelaxed, 1.0 = relaxed)
        """
        row = self.df.iloc[idx]
        structure_folder = Path(row['structure_folder'])
        
        # Handle relative paths if necessary
        if not structure_folder.is_absolute():
            structure_folder = Path(self.csv_path).parent / structure_folder
            
        # Use backward barrier as target (standard for vacancy diffusion)
        barrier = torch.tensor([row['backward_barrier_eV']], dtype=torch.float32)
        
        # ---------------------------------------------------------------------
        # TRAJECTORY SAMPLING STRATEGY
        # ---------------------------------------------------------------------
        
        # Default: Use unrelaxed structures (traj_0)
        initial_file = structure_folder / "initial_traj_0.npz"
        final_file = structure_folder / "final_traj_0.npz"
        
        # TRAINING MODE: Random trajectory sampling
        if self.mode == 'train':
            traj_files = list(structure_folder.glob("initial_traj_*.npz"))
            
            if len(traj_files) > 1:  # If multiple frames exist
                n_frames = len(traj_files)
                chosen_idx = np.random.randint(0, n_frames)
                
                initial_file = structure_folder / f"initial_traj_{chosen_idx}.npz"
                potential_final = structure_folder / f"final_traj_{chosen_idx}.npz"
                if potential_final.exists():
                    final_file = potential_final
        
        # VAL/TEST MODE: Always use traj_0 (unrelaxed)
        # Already set above as default
        
        # Fallback to relaxed if traj_0 missing (legacy data)
        if not initial_file.exists():
            initial_file = structure_folder / "initial_relaxed.npz"
        if not final_file.exists():
            final_file = structure_folder / "final_relaxed.npz"
        
        # Fallback to CIF if NPZ missing
        if not initial_file.exists():
            initial_file = structure_folder / "initial_relaxed.cif"
        if not final_file.exists():
            final_file = structure_folder / "final_relaxed.cif"

        # ---------------------------------------------------------------------
        # GRAPH CONSTRUCTION
        # ---------------------------------------------------------------------
        
        try:
            # Build graphs
            # Note: GraphBuilder automatically extracts 'relax_progress' from the NPZ file
            initial_graph = self.graph_builder.cif_to_graph(initial_file)
            final_graph = self.graph_builder.cif_to_graph(final_file)
            
        except Exception as e:
            # Emergency Fallback if specific file is corrupted/missing
            print(f"Warning: Failed to load {initial_file} or {final_file}. Error: {e}")
            print("Falling back to standard relaxed structures.")
            
            initial_file = structure_folder / "initial_relaxed.cif"
            final_file = structure_folder / "final_relaxed.cif"
            
            initial_graph = self.graph_builder.cif_to_graph(initial_file)
            final_graph = self.graph_builder.cif_to_graph(final_file)

        # Extract progress for Trainer (needed for loss weighting)
        # We take it from the initial graph (usually identical for final)
        # GraphBuilder puts it in 'relax_progress' with shape [1, 1]
        progress = initial_graph.relax_progress
        
        # Flatten for simpler handling in collate/trainer if needed, 
        # but Trainer expects tensor for batching.
        # Shape is [1, 1], so it's fine.

        return initial_graph, final_graph, barrier, progress


def create_dataloaders(config, csv_path):
    """
    Create train and validation dataloaders.
    
    Strategy:
    1. Instantiate TWO datasets: 
       - One in 'train' mode (trajectory sampling enabled)
       - One in 'val' mode (standard relaxed structures)
    2. Split indices randomly.
    3. Create Subsets using the split indices.
       - Train Subset points to 'train' mode dataset
       - Val Subset points to 'val' mode dataset
    """
    # 1. Instantiate datasets
    # Note: They share the same underlying CSV data, just behave differently
    train_dataset_full = DiffusionBarrierDataset(csv_path, config, mode='train')
    val_dataset_full = DiffusionBarrierDataset(csv_path, config, mode='val')
    
    dataset_size = len(train_dataset_full)
    indices = list(range(dataset_size))
    
    # 2. Split indices
    # Use fixed seed for reproducibility of splits if desired, 
    # but shuffling is good for active learning.
    np.random.shuffle(indices)
    
    val_split = getattr(config, 'val_split', 0.2)
    split = int(np.floor(val_split * dataset_size))
    
    val_indices = indices[:split]
    train_indices = indices[split:]
    
    # 3. Create Subsets
    # This ensures train_loader gets data from the 'trajectory sampling' dataset
    # and val_loader gets data from the 'standard' dataset, consistent with the indices.
    # --- NOW WORKS BECAUSE Subset IS IMPORTED ---
    train_subset = Subset(train_dataset_full, train_indices)
    val_subset = Subset(val_dataset_full, val_indices)
    
    # 4. Create Loaders
    # PyG Data objects work fine with standard torch DataLoader 
    # (they will be collated into Batch objects automatically if using PyG > 2.0, 
    # otherwise might need torch_geometric.loader.DataLoader. 
    # Assuming standard PyG installation which patches torch DataLoader or provides its own)
    
    # Safer: try to import from torch_geometric if available, else standard
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        LoaderClass = PyGDataLoader
    except ImportError:
        LoaderClass = DataLoader

    train_loader = LoaderClass(
        train_subset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True
    )
    
    val_loader = LoaderClass(
        val_subset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True
    )
    
    print(f"Data split: {len(train_subset)} train, {len(val_subset)} val")
    
    return train_loader, val_loader