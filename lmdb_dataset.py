"""
LMDB Dataset for Diffusion Barrier Prediction

Fast PyTorch Dataset that loads pre-computed graphs from LMDB.

Usage:
    from lmdb_dataset import LMDBDataset, create_lmdb_dataloaders
    
    train_loader, val_loader = create_lmdb_dataloaders(config)

Benefits over on-the-fly graph building:
- 10-20x faster data loading
- No CPU overhead during training
- Perfect for Active Learning (repeated training)
"""

import lmdb
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


class LMDBDataset(Dataset):
    """
    PyTorch Dataset that loads pre-computed graphs from LMDB.
    
    Much faster than on-the-fly graph building because:
    - Graphs are pre-computed
    - Memory-mapped I/O (very fast)
    - No deserialization overhead
    """
    
    def __init__(
        self,
        lmdb_path: str,
        indices: Optional[list] = None
    ):
        """
        Initialize LMDB dataset.
        
        Args:
            lmdb_path: Path to LMDB directory (e.g., 'MoNbTaW_lmdb/graphs.lmdb')
            indices: Subset indices for train/val split (optional)
        """
        self.lmdb_path = Path(lmdb_path)
        
        if not self.lmdb_path.exists():
            raise FileNotFoundError(
                f"LMDB not found: {self.lmdb_path}\n"
                f"Run build_lmdb_dataset.py first!"
            )
        
        # 🔥 CRITICAL: Don't open LMDB here! (causes segfault with multiprocessing)
        # Each DataLoader worker will open its own environment in _init_env()
        self.env = None
        self._env_initialized = False
        
        # Get total length from metadata (temporarily open/close)
        meta_path = self.lmdb_path.parent / "meta.lmdb"
        if meta_path.exists():
            meta_env = lmdb.open(
                str(meta_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            try:
                with meta_env.begin() as meta_txn:
                    total_len = int(meta_txn.get(b'__len__').decode('ascii'))
            finally:
                meta_env.close()
        else:
            # Fallback: temporarily open graphs.lmdb to count
            env_temp = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            try:
                with env_temp.begin() as txn:
                    total_len = txn.stat()['entries']
            finally:
                env_temp.close()
        
        # Apply subset if specified
        if indices is not None:
            self.indices = indices
            self.length = len(indices)
        else:
            self.indices = list(range(total_len))
            self.length = total_len
        
        print(f"LMDBDataset initialized: {self.length} samples")
        print(f"  LMDB: {self.lmdb_path}")
    
    def __len__(self) -> int:
        """Number of samples in dataset."""
        return self.length
    
    def _init_env(self):
        """
        Lazy LMDB environment initialization.
        
        CRITICAL: Opens LMDB in each worker process separately.
        This avoids segfaults when using num_workers > 0.
        """
        if not self._env_initialized:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            self._env_initialized = True
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get single sample from LMDB.
        
        Returns:
            Tuple of (initial_graph, final_graph, barrier)
        """
        # Lazy init: open LMDB on first access (each worker gets its own env)
        self._init_env()
        
        # Map to actual index if subset
        actual_idx = self.indices[idx]
        
        # Get from LMDB
        key = f"{actual_idx}".encode('ascii')
        
        with self.env.begin() as txn:
            serialized_data = txn.get(key)
        
        if serialized_data is None:
            raise KeyError(f"Sample {actual_idx} not found in LMDB")
        
        # Deserialize
        data = pickle.loads(serialized_data)
        
        initial_graph = data['initial']
        final_graph = data['final']
        barrier = data['barrier']
        
        return initial_graph, final_graph, barrier
    
    def get_statistics(self) -> dict:
        """Get barrier statistics from dataset."""
        barriers = []
        
        for idx in range(len(self)):
            _, _, barrier = self[idx]
            barriers.append(barrier)
        
        barriers = np.array(barriers)
        
        return {
            'count': len(barriers),
            'min': barriers.min(),
            'max': barriers.max(),
            'mean': barriers.mean(),
            'median': np.median(barriers),
            'std': barriers.std()
        }
    
    def __del__(self):
        """Close LMDB environment on deletion."""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


def collate_fn(batch: list) -> Tuple:
    """
    Collate function for batching graph pairs.
    
    Same as in dataset.py - combines graphs into batches.
    
    CRITICAL FIX: Removes any pre-existing batch/ptr attributes
    that could cause CUDA index errors during batching.
    This includes BOTH atom graph and line graph attributes!
    """
    initial_graphs = []
    final_graphs = []
    barriers = []
    
    for initial, final, barrier in batch:
        # 🔥 CRITICAL: Remove ALL batch-related attributes before batching!
        # This includes both atom graph and line graph attributes
        batch_attrs = [
            'batch', 'ptr',           # Atom graph batching
            'line_batch', 'line_ptr'  # Line graph batching
        ]
        
        for attr in batch_attrs:
            if hasattr(initial, attr):
                setattr(initial, attr, None)  # Set to None (safer than delattr)
            if hasattr(final, attr):
                setattr(final, attr, None)
        
        initial_graphs.append(initial)
        final_graphs.append(final)
        barriers.append(barrier)
    
    # Batch graphs using PyG (creates new batch indices)
    initial_batch = Batch.from_data_list(initial_graphs)
    final_batch = Batch.from_data_list(final_graphs)
    
    # Convert barriers to tensor
    barriers = torch.tensor(barriers, dtype=torch.float32)
    
    return initial_batch, final_batch, barriers


def create_lmdb_dataloaders(
    config,
    val_split: float = 0.1,
    random_seed: int = 42
) -> Tuple:
    """
    Create train and validation dataloaders from LMDB.
    
    Args:
        config: Configuration object
        val_split: Validation split ratio (default: 0.1 = 10%)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader) or (train_loader, None)
    """
    # Determine LMDB path
    lmdb_path = getattr(config, 'lmdb_path', None)
    
    if lmdb_path is None:
        # Auto-generate path
        lmdb_path = str(Path(config.database_dir).parent / f"{config.system_name}_lmdb" / "graphs.lmdb")
    
    lmdb_path = Path(lmdb_path)
    
    if not lmdb_path.exists():
        raise FileNotFoundError(
            f"LMDB not found: {lmdb_path}\n"
            f"Run build_lmdb_dataset.py first or set use_lmdb=False"
        )
    
    print("\n" + "="*70)
    print("CREATING LMDB DATALOADERS")
    print("="*70)
    print(f"LMDB path: {lmdb_path}")
    
    # Load full dataset
    full_dataset = LMDBDataset(str(lmdb_path))
    
    total_samples = len(full_dataset)
    
    if total_samples == 0:
        raise ValueError(f"LMDB is empty: {lmdb_path}")
    
    # 🔥 Fast statistics from CSV (instead of loading all LMDB samples)
    try:
        import pandas as pd
        df = pd.read_csv(config.csv_path)
        barriers = df['backward_barrier_eV']
        print(f"\nDataset: {total_samples} samples")
        print(f"Barrier range: [{barriers.min():.3f}, {barriers.max():.3f}] eV")
        print(f"Barrier mean: {barriers.mean():.3f} ± {barriers.std():.3f} eV")
    except Exception:
        print(f"\nDataset: {total_samples} samples")
    
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
        
        # Create datasets with subset indices
        train_dataset = LMDBDataset(str(lmdb_path), indices=train_indices)
        val_dataset = LMDBDataset(str(lmdb_path), indices=val_indices)
        
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
            drop_last=False
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


if __name__ == "__main__":
    from config import Config
    
    print("="*70)
    print("LMDB DATASET - TEST MODE")
    print("="*70)
    
    config = Config()
    
    try:
        # Test LMDB dataloaders
        train_loader, val_loader = create_lmdb_dataloaders(
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
        print(f"  Barriers shape: {barriers.shape}")
        print(f"  Barriers: {barriers.tolist()[:5]}...")
        
        print("\n✅ LMDB dataset test successful!")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nRun build_lmdb_dataset.py first to create LMDB!")
    
    print("="*70)