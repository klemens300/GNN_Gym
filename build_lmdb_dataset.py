"""
Build LMDB Dataset - Memory-Safe Sequential Version (FIXED)

Builds LMDB database from CSV with minimal output (single progress bar).
Sequential processing to avoid memory crashes.

CRITICAL FIX: Removes batch attributes before saving to prevent CUDA errors!

Usage:
    python build_lmdb_dataset.py
"""

import lmdb
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

from config import Config
from graph_builder import GraphBuilder


def build_lmdb_sequential(config: Config, force_rebuild: bool = True):
    """
    Build LMDB dataset sequentially (memory-safe).
    
    Shows only a single progress bar during processing.
    """
    # Paths
    output_dir = Path(config.database_dir).parent / f"{config.system_name}_lmdb"
    graphs_lmdb_path = output_dir / "graphs.lmdb"
    meta_lmdb_path = output_dir / "meta.lmdb"
    map_size = 100 * 1024**3  # 100 GB
    
    # Check if exists
    if not force_rebuild and graphs_lmdb_path.exists():
        print(f"LMDB exists: {graphs_lmdb_path}")
        print("Skipping build (use force_rebuild=True to rebuild)")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(config.csv_path)
    total_samples = len(df)
    
    # Create graph builder
    builder = GraphBuilder(
        config,
        csv_path=config.csv_path,
        profile=False,
        use_cache=False
    )
    
    # Open LMDB environments
    graphs_env = lmdb.open(
        str(graphs_lmdb_path),
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True
    )
    
    meta_env = lmdb.open(
        str(meta_lmdb_path),
        map_size=map_size // 10,
        readonly=False,
        meminit=False,
        map_async=True
    )
    
    # Process samples with single progress bar
    valid_count = 0
    failed_count = 0
    start_time = time.time()
    
    with graphs_env.begin(write=True) as graph_txn:
        with meta_env.begin(write=True) as meta_txn:
            
            # Single progress bar
            pbar = tqdm(total=total_samples, desc="Building LMDB", unit="sample")
            
            for idx, row in df.iterrows():
                try:
                    # Get paths
                    structure_folder = Path(row['structure_folder'])
                    if not structure_folder.is_absolute():
                        structure_folder = Path(config.csv_path).parent / structure_folder
                    
                    initial_cif = structure_folder / "initial_relaxed.cif"
                    final_cif = structure_folder / "final_relaxed.cif"
                    barrier = row['backward_barrier_eV']
                    
                    # Build graphs
                    initial_graph, final_graph = builder.build_pair_graph(
                        str(initial_cif),
                        str(final_cif),
                        backward_barrier=barrier
                    )
                    
                    # 🔥 CRITICAL FIX: Remove ALL batch-related attributes before saving!
                    # These attributes will interfere with PyTorch Geometric batching later
                    # and cause CUDA "index out of bounds" errors during training
                    batch_attrs = [
                        'batch', 'ptr',           # Atom graph batching
                        'line_batch', 'line_ptr', # Line graph batching (for angle features)
                        '_slice_dict', '_inc_dict' # PyG internal batching state
                    ]
                    
                    for attr in batch_attrs:
                        # Remove from initial graph
                        if hasattr(initial_graph, attr):
                            delattr(initial_graph, attr)
                        # Remove from final graph
                        if hasattr(final_graph, attr):
                            delattr(final_graph, attr)
                    
                    # Serialize graph pair
                    graph_data = {
                        'initial': initial_graph,
                        'final': final_graph,
                        'barrier': float(barrier)
                    }
                    serialized_graph = pickle.dumps(graph_data, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    # Serialize metadata
                    metadata = {
                        'composition_string': row['composition_string'],
                        'backward_barrier_eV': float(barrier),
                        'forward_barrier_eV': float(row.get('forward_barrier_eV', 0.0)),
                        'E_initial_eV': float(row.get('E_initial_eV', 0.0)),
                        'E_final_eV': float(row.get('E_final_eV', 0.0)),
                        'structure_folder': str(structure_folder),
                        'run_number': int(row.get('run_number', 1))
                    }
                    serialized_meta = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    # Write immediately
                    key = f"{valid_count}".encode('ascii')
                    graph_txn.put(key, serialized_graph)
                    meta_txn.put(key, serialized_meta)
                    
                    valid_count += 1
                    
                except Exception:
                    failed_count += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'valid': valid_count,
                    'failed': failed_count,
                    'rate': f'{valid_count/(time.time()-start_time):.1f}/s'
                })
            
            # Store total count
            meta_txn.put(b'__len__', str(valid_count).encode('ascii'))
            
            pbar.close()
    
    # Close environments
    graphs_env.close()
    meta_env.close()
    
    # Summary
    total_time = time.time() - start_time
    graphs_size = sum(f.stat().st_size for f in graphs_lmdb_path.parent.glob("graphs.lmdb*"))
    meta_size = sum(f.stat().st_size for f in meta_lmdb_path.parent.glob("meta.lmdb*"))
    
    print(f"\nComplete: {valid_count}/{total_samples} valid ({failed_count} failed)")
    print(f"Time: {total_time:.1f}s ({valid_count/total_time:.1f} samples/s)")
    print(f"Size: {(graphs_size + meta_size) / 1024**2:.1f} MB")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    config = Config()
    
    if not Path(config.csv_path).exists():
        print(f"ERROR: CSV not found: {config.csv_path}")
        exit(1)
    
    print("="*70)
    print("BUILDING LMDB WITH BATCH ATTRIBUTE FIX")
    print("="*70)
    print("This will:")
    print("  1. Remove old LMDB if exists")
    print("  2. Build new LMDB with cleaned graphs")
    print("  3. Prevent CUDA index errors during training")
    print("="*70)
    print()
    
    build_lmdb_sequential(config, force_rebuild=True)