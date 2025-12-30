"""
Find corrupted batch attributes in LMDB

Scans ALL samples to find ones with non-None batch attributes.
"""

import lmdb
import pickle
import torch
from pathlib import Path
from tqdm import tqdm

lmdb_path = Path("/home/klemens/databases/MoNbTaW_lmdb/graphs.lmdb")

print("="*70)
print("SCANNING ALL SAMPLES FOR CORRUPTED BATCH ATTRIBUTES")
print("="*70)
print(f"LMDB: {lmdb_path}")
print()

env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)

# Get total count
with env.begin() as txn:
    total = txn.stat()['entries']

print(f"Total samples: {total}")
print("Scanning for non-None batch attributes...")
print()

corrupted_samples = []
batch_none_count = 0
batch_exists_count = 0

with env.begin() as txn:
    for i in tqdm(range(total), desc="Scanning"):
        key = f"{i}".encode('ascii')
        serialized = txn.get(key)
        
        if serialized is None:
            continue
        
        try:
            data = pickle.loads(serialized)
            initial = data['initial']
            final = data['final']
            
            # Check initial graph
            if hasattr(initial, 'batch'):
                if initial.batch is None:
                    batch_none_count += 1
                else:
                    batch_exists_count += 1
                    corrupted_samples.append({
                        'sample_id': i,
                        'graph': 'initial',
                        'batch_shape': initial.batch.shape,
                        'batch_min': initial.batch.min().item(),
                        'batch_max': initial.batch.max().item(),
                        'num_nodes': initial.num_nodes,
                        'barrier': data['barrier']
                    })
            
            # Check final graph
            if hasattr(final, 'batch'):
                if final.batch is None:
                    batch_none_count += 1
                else:
                    batch_exists_count += 1
                    corrupted_samples.append({
                        'sample_id': i,
                        'graph': 'final',
                        'batch_shape': final.batch.shape,
                        'batch_min': final.batch.min().item(),
                        'batch_max': final.batch.max().item(),
                        'num_nodes': final.num_nodes,
                        'barrier': data['barrier']
                    })
                    
        except Exception as e:
            print(f"\n❌ Error reading sample {i}: {e}")

env.close()

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Samples with batch=None: {batch_none_count}")
print(f"Samples with batch!=None: {batch_exists_count} 🔥")
print()

if corrupted_samples:
    print(f"❌ Found {len(corrupted_samples)} corrupted graphs!")
    print("\nShowing first 20:")
    for item in corrupted_samples[:20]:
        print(f"  Sample {item['sample_id']} ({item['graph']}):")
        print(f"    batch shape: {item['batch_shape']}")
        print(f"    batch range: [{item['batch_min']}, {item['batch_max']}]")
        print(f"    num_nodes: {item['num_nodes']}")
        print(f"    Expected: batch should be all zeros for single graph")
        
        # Check if batch is valid
        if item['batch_max'] > 0:
            print(f"    🔥 CRITICAL: batch.max() = {item['batch_max']} (should be 0!)")
        if item['batch_min'] < 0:
            print(f"    🔥 CRITICAL: batch.min() = {item['batch_min']} (negative!)")
        if item['batch_shape'][0] != item['num_nodes']:
            print(f"    🔥 CRITICAL: batch size mismatch!")
        print()
    
    if len(corrupted_samples) > 20:
        print(f"... and {len(corrupted_samples) - 20} more")
    
    print("\n🔧 FIX REQUIRED:")
    print("="*70)
    print("These samples have non-None batch attributes that will cause")
    print("CUDA index errors during training.")
    print()
    print("OPTION 1 - Quick Fix (recommended):")
    print("  Add to lmdb_dataset.py collate_fn:")
    print()
    print("  for initial, final, barrier in batch:")
    print("      if hasattr(initial, 'batch'):")
    print("          initial.batch = None")
    print("      if hasattr(final, 'batch'):")
    print("          final.batch = None")
    print()
    print("OPTION 2 - Clean Fix:")
    print("  Rebuild LMDB with batch attributes removed")
    print()
    
else:
    print("✅ No corrupted samples found!")
    print()
    print("The CUDA error might be from a different source.")
    print("Possible other causes:")
    print("  - Edge indices out of bounds")
    print("  - Invalid node indices")
    print("  - Memory corruption during training")

print("="*70)