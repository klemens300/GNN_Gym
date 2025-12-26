"""
PyTorch Dataset and DataLoaders for Diffusion Barrier Prediction

Uses GraphBuilder with atom embeddings (element indices + optional properties).
Graphs are built on-the-fly from CIF files with real geometry.
NPZ caching is handled automatically by GraphBuilder.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from pathlib import Path
import numpy as np

from graph_builder import GraphBuilder


class DiffusionBarrierDataset(Dataset):
    """
    Dataset for diffusion barriers with on-the-fly graph construction.
    
    Uses GraphBuilder with atom embeddings:
    - Element indices for embedding lookup
    - Optional atomic properties
    - Real geometry from CIF files
    - Automatic NPZ caching for 10x speedup
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
        
        Args:
            csv_path: Path to CSV database
            config: Configuration object
            min_barrier: Minimum barrier cutoff in eV (optional)
            max_barrier: Maximum barrier cutoff in eV (optional)
            indices: Subset indices for train/val split (optional)
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
        
        # Create graph builder (with atom embeddings + NPZ caching)
        self.graph_builder = GraphBuilder(config, csv_path=csv_path)
        
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
        
        Builds graphs from CIF files with:
        - Element indices for embedding lookup
        - Optional atomic properties
        - Real geometry (edges and distances)
        - Automatic NPZ caching (10x faster loading)
        
        Returns:
            Tuple of (initial_graph, final_graph, barrier)
        """
        # Get row from CSV
        row = self.df.iloc[idx]
        
        # Get CIF paths
        structure_folder = Path(row['structure_folder'])
        
        # Make path absolute if relative
        if not structure_folder.is_absolute():
            csv_parent = Path(self.csv_path).parent
            structure_folder = csv_parent / structure_folder
        
        initial_cif = structure_folder / "initial_relaxed.cif"
        final_cif = structure_folder / "final_relaxed.cif"
        
        # Get barrier
        barrier = row['backward_barrier_eV']
        
        # Build graphs from CIF files
        # GraphBuilder automatically:
        # - Tries NPZ first (fast!)
        # - Falls back to CIF if needed
        # - Creates NPZ for next time
        # - Caches in RAM if enabled
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
    
    Combines individual samples into batches for efficient GPU processing.
    
    Args:
        batch: List of (initial_graph, final_graph, barrier) tuples
    
    Returns:
        Tuple of (initial_batch, final_batch, barriers)
    """
    # Separate components
    initial_graphs = [item[0] for item in batch]
    final_graphs = [item[1] for item in batch]
    barriers = [item[2] for item in batch]
    
    # Batch graphs using PyG
    # This creates sparse block-diagonal adjacency matrix
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
    2. Apply barrier filtering (if configured)
    3. Random split into train/val
    4. Create optimized dataloaders with:
       - Multi-worker loading
       - Pin memory for GPU transfer
       - Persistent workers to avoid respawning
       - Prefetching for pipeline parallelism
    
    Args:
        config: Configuration object with paths and parameters
        val_split: Validation split ratio (default: 0.1 = 10%)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader) or (train_loader, None) if no validation
    """
    csv_path = config.csv_path
    
    # Get barrier filtering from config
    min_barrier = getattr(config, 'min_barrier', None)
    max_barrier = getattr(config, 'max_barrier', None)
    
    print("\n" + "="*70)
    print("CREATING DATALOADERS (ATOM EMBEDDINGS + NPZ CACHING)")
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
    print(f"  Barrier mean: {stats['mean']:.3f} +/- {stats['std']:.3f} eV")
    
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
        
        # Create datasets
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
        
        # DataLoader settings
        num_workers = getattr(config, 'num_workers', 0)
        batch_size_train = config.batch_size
        batch_size_val = getattr(config, 'batch_size_val', config.batch_size)
        prefetch_factor = getattr(config, 'prefetch_factor', 2)
        drop_last = getattr(config, 'drop_last', False)
        
        print(f"\nDataLoader optimization:")
        print(f"  num_workers: {num_workers}")
        print(f"  pin_memory: {num_workers > 0}")
        print(f"  persistent_workers: {num_workers > 0}")
        print(f"  prefetch_factor: {prefetch_factor if num_workers > 0 else 'N/A'}")
        print(f"  drop_last: {drop_last}")
        print(f"  batch_size (train): {batch_size_train}")
        print(f"  batch_size (val): {batch_size_val}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=drop_last
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size_val,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=False  # Never drop last for validation
        )
        
        print("="*70)
        print()
        
        return train_loader, val_loader
    
    else:
        # No validation split
        print(f"\nNo validation split (total samples: {total_samples})")
        
        num_workers = getattr(config, 'num_workers', 0)
        batch_size = config.batch_size
        prefetch_factor = getattr(config, 'prefetch_factor', 2)
        drop_last = getattr(config, 'drop_last', False)
        
        print(f"\nDataLoader optimization:")
        print(f"  num_workers: {num_workers}")
        print(f"  pin_memory: {num_workers > 0}")
        print(f"  persistent_workers: {num_workers > 0}")
        print(f"  prefetch_factor: {prefetch_factor if num_workers > 0 else 'N/A'}")
        print(f"  drop_last: {drop_last}")
        print(f"  batch_size: {batch_size}")
        
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=drop_last
        )
        
        print("="*70)
        print()
        
        return train_loader, None


# Test function
if __name__ == "__main__":
    from config import Config
    
    print("Testing Dataset with AtomEmbeddings and NPZ caching...")
    
    # Create config
    config = Config()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config,
        val_split=0.1,
        random_seed=42
    )
    
    # Get one batch
    print("\nFetching one batch...")
    initial_batch, final_batch, barriers = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  Initial graphs: {initial_batch.num_graphs} graphs")
    print(f"  Initial atoms: {initial_batch.num_nodes} total atoms")
    print(f"  Initial edges: {initial_batch.num_edges} total edges")
    print(f"  Element indices shape: {initial_batch.x_element.shape}")
    print(f"  Barriers shape: {barriers.shape}")
    print(f"  Barriers: {barriers.tolist()}")
    
    print("\nDataset test successful! ?")