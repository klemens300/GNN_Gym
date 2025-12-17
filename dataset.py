"""
PyTorch Dataset and DataLoaders for Diffusion Barrier Prediction

Simple workflow:
1. Load CSV
2. Filter barriers (cleanup)
3. Build graphs on-the-fly
4. Return train/val loaders
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from pathlib import Path
import numpy as np

from template_graph_builder import TemplateGraphBuilder


class DiffusionBarrierDataset(Dataset):
    """
    Dataset for diffusion barriers with on-the-fly graph construction.
    
    Features:
    - Template-based graph building (fast!)
    - Barrier filtering for data quality
    - Simple and clean
    """
    
    def __init__(
        self,
        csv_path: str,
        config,
        min_barrier: float = None,
        max_barrier: float = None,
        indices: list = None
    ):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV database
        config : Config
            Configuration object
        min_barrier : float, optional
            Minimum barrier cutoff (eV)
        max_barrier : float, optional
            Maximum barrier cutoff (eV)
        indices : list, optional
            Subset indices (for train/val split)
        """
        self.csv_path = csv_path
        self.config = config
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        initial_count = len(self.df)
        
        # Barrier filtering (data cleanup)
        if min_barrier is not None:
            self.df = self.df[self.df['backward_barrier_eV'] >= min_barrier]
        
        if max_barrier is not None:
            self.df = self.df[self.df['backward_barrier_eV'] <= max_barrier]
        
        # Reset index after filtering
        self.df = self.df.reset_index(drop=True)
        
        # Apply subset (for train/val split)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
        final_count = len(self.df)
        
        # Create graph builder ONCE
        self.graph_builder = TemplateGraphBuilder(config, csv_path=csv_path)
        
        # Print info
        if indices is None:  # Only print for full dataset
            print(f"Dataset loaded: {final_count} samples")
            if initial_count != final_count:
                removed = initial_count - final_count
                print(f"  Removed {removed} samples (barrier filtering)")
    
    def __len__(self) -> int:
        """Number of samples in dataset"""
        return len(self.df)
    def __getitem__(self, idx: int) -> tuple:
        """
        Get single sample.
        
        Returns:
        --------
        initial_graph : Data
            Initial structure graph with label
        final_graph : Data
            Final structure graph with label
        barrier : float
            Backward barrier (for easy access)
        """
        # Get row from CSV
        row = self.df.iloc[idx]
        
        # Get CIF paths
        structure_folder = Path(row['structure_folder'])
        
        # ✅ FIX: Make path absolute if it's relative
        if not structure_folder.is_absolute():
            # Get parent directory of CSV file
            csv_parent = Path(self.csv_path).parent
            structure_folder = csv_parent / structure_folder
        
        initial_cif = structure_folder / "initial_relaxed.cif"
        final_cif = structure_folder / "final_relaxed.cif"
        
        # Get barrier
        barrier = row['backward_barrier_eV']
        
        # Build graphs on-the-fly (fast with template!)
        initial_graph, final_graph = self.graph_builder.build_pair_graph(
            str(initial_cif),
            str(final_cif),
            backward_barrier=barrier
        )
        
        return initial_graph, final_graph, barrier
    
    def get_statistics(self) -> dict:
        """Get barrier statistics"""
        barriers = self.df['backward_barrier_eV']
        return {
            'count': len(barriers),
            'min': barriers.min(),
            'max': barriers.max(),
            'mean': barriers.mean(),
            'median': barriers.median(),
            'std': barriers.std()
        }


def collate_fn(batch: list) -> tuple:
    """
    Collate function for batching graph pairs.
    
    Combines individual samples into batches.
    
    Parameters:
    -----------
    batch : list
        List of (initial_graph, final_graph, barrier) tuples
    
    Returns:
    --------
    initial_batch : Batch
        Batched initial graphs
    final_batch : Batch
        Batched final graphs
    barriers : Tensor [batch_size]
        Barriers
    """
    # Separate components
    initial_graphs = [item[0] for item in batch]
    final_graphs = [item[1] for item in batch]
    barriers = [item[2] for item in batch]
    
    # Batch graphs using PyG
    initial_batch = Batch.from_data_list(initial_graphs)
    final_batch = Batch.from_data_list(final_graphs)
    
    # Convert barriers to tensor
    barriers = torch.tensor(barriers, dtype=torch.float32)
    
    return initial_batch, final_batch, barriers


def create_dataloaders(
    config,
    val_split: float = 0.1,
    random_seed: int = 42
) -> tuple:
    """
    Create train and validation dataloaders.
    
    Simple workflow:
    1. Load all data from CSV
    2. Apply barrier filtering
    3. Random split into train/val
    4. Create dataloaders
    
    Parameters:
    -----------
    config : Config
        Configuration object with:
        - csv_path
        - batch_size
        - min_barrier (optional)
        - max_barrier (optional)
    val_split : float
        Validation split ratio (default: 0.1 = 10%)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    train_loader : DataLoader
    val_loader : DataLoader (or None if val_split=0)
    """
    csv_path = config.csv_path
    
    # Get barrier filtering from config
    min_barrier = getattr(config, 'min_barrier', None)
    max_barrier = getattr(config, 'max_barrier', None)
    
    print("\n" + "="*70)
    print("CREATING DATALOADERS")
    print("="*70)
    
    # Load full dataset with filtering
    print(f"\nLoading data from: {csv_path}")
    if min_barrier or max_barrier:
        barrier_range = f"[{min_barrier if min_barrier else 'any'}, "
        barrier_range += f"{max_barrier if max_barrier else 'any'}] eV"
        print(f"Barrier filter: {barrier_range}")
    
    full_dataset = DiffusionBarrierDataset(
        csv_path,
        config,
        min_barrier=min_barrier,
        max_barrier=max_barrier
    )
    
    total_samples = len(full_dataset)
    
    if total_samples == 0:
        raise ValueError(f"No data found in {csv_path}")
    
    # Print statistics
    stats = full_dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Samples: {stats['count']}")
    print(f"  Barrier range: [{stats['min']:.3f}, {stats['max']:.3f}] eV")
    print(f"  Barrier mean: {stats['mean']:.3f} ± {stats['std']:.3f} eV")
    
    # Train/Val split
    if val_split > 0 and total_samples >= 10:
        train_size = int((1 - val_split) * total_samples)
        val_size = total_samples - train_size
        
        print(f"\nSplitting data:")
        print(f"  Train: {train_size} samples ({(1-val_split)*100:.0f}%)")
        print(f"  Val: {val_size} samples ({val_split*100:.0f}%)")
        
        # Random indices
        indices = torch.randperm(
            total_samples,
            generator=torch.Generator().manual_seed(random_seed)
        ).tolist()
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets with subsets
        train_dataset = DiffusionBarrierDataset(
            csv_path,
            config,
            min_barrier=min_barrier,
            max_barrier=max_barrier,
            indices=train_indices
        )
        
        val_dataset = DiffusionBarrierDataset(
            csv_path,
            config,
            min_barrier=min_barrier,
            max_barrier=max_barrier,
            indices=val_indices
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        print(f"\n✓ Dataloaders created:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val: {len(val_loader)} batches")
        print("="*70 + "\n")
        
        return train_loader, val_loader
    
    else:
        # No validation split
        print(f"\nNo validation split")
        
        train_loader = DataLoader(
            full_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        print(f"\n✓ Train loader created:")
        print(f"  Train: {len(train_loader)} batches")
        print("="*70 + "\n")
        
        return train_loader, None


# Test script
if __name__ == "__main__":
    from config import Config
    import time
    
    print("\n" + "="*70)
    print("DATASET TEST")
    print("="*70)
    
    # Setup config
    config = Config()
    #config.min_barrier = 0.1
    #config.max_barrier = 666.0
    config.batch_size = 4
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config,
        val_split=0.2
    )
    
    # Test iteration
    print("\nTesting batch iteration:")
    for i, (initial_batch, final_batch, barriers) in enumerate(train_loader):
        print(f"\nBatch {i+1}:")
        print(f"  Initial graphs:")
        print(f"    Total nodes: {initial_batch.num_nodes}")
        print(f"    Num graphs: {initial_batch.num_graphs}")
        print(f"    Node features: {initial_batch.x.shape}")
        print(f"  Final graphs:")
        print(f"    Total nodes: {final_batch.num_nodes}")
        print(f"    Num graphs: {final_batch.num_graphs}")
        print(f"  Barriers: {barriers.shape}")
        print(f"    Range: [{barriers.min():.3f}, {barriers.max():.3f}] eV")
        
        if i >= 2:  # Only show first 3 batches
            break
    
    # Speed test
    print("\nSpeed test (10 batches):")
    start = time.time()
    for i, (initial_batch, final_batch, barriers) in enumerate(train_loader):
        if i >= 10:
            break
    elapsed = time.time() - start
    print(f"  Loaded 10 batches in {elapsed:.2f}s ({elapsed/10*1000:.1f}ms per batch)")
    
    print("\n" + "="*70)