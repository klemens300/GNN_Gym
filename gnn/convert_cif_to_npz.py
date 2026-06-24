"""
Convert CIF files to NPZ format for faster loading.

Usage:
    python convert_cif_to_npz.py

What it does:
- Reads all CIF files from your database
- Creates .npz files next to each .cif
- Original CIF files remain untouched
- Shows progress bar with ETA
- NPZ is numpy's compressed format (smaller + faster than CIF)
"""

import numpy as np
from ase.io import read
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time


def convert_cif_to_npz(csv_path: str):
    """
    Convert all CIF files in database to NPZ format.
    
    Parameters:
    -----------
    csv_path : str
        Path to your CSV database
    """
    print("="*70)
    print("CIF → NPZ CONVERSION")
    print("="*70)
    
    # Check CSV exists
    if not Path(csv_path).exists():
        print(f"❌ CSV not found: {csv_path}")
        return
    
    # Load CSV
    print(f"\n📂 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Found {len(df)} entries")
    
    # Count total CIF files to convert
    total_cifs = 0
    for idx, row in df.iterrows():
        folder = Path(row['structure_folder'])
        if not folder.is_absolute():
            folder = Path(csv_path).parent / folder
        
        for name in ['initial_relaxed', 'final_relaxed']:
            cif_file = folder / f"{name}.cif"
            if cif_file.exists():
                total_cifs += 1
    
    print(f"\n🔍 Found {total_cifs} CIF files to convert")
    
    if total_cifs == 0:
        print("❌ No CIF files found!")
        return
    
    # Convert with progress bar
    print("\n🔄 Converting CIF → NPZ...")
    
    converted = 0
    skipped = 0
    failed = 0
    
    # Create progress bar
    pbar = tqdm(total=total_cifs, desc="Converting", unit="file")
    
    for idx, row in df.iterrows():
        folder = Path(row['structure_folder'])
        
        # Make absolute path if needed
        if not folder.is_absolute():
            folder = Path(csv_path).parent / folder
        
        for name in ['initial_relaxed', 'final_relaxed']:
            cif_file = folder / f"{name}.cif"
            npz_file = folder / f"{name}.npz"
            
            if not cif_file.exists():
                continue
            
            # Skip if NPZ already exists
            if npz_file.exists():
                skipped += 1
                pbar.update(1)
                pbar.set_postfix({
                    'converted': converted,
                    'skipped': skipped,
                    'failed': failed
                })
                continue
            
            # Convert
            try:
                # Read CIF
                atoms = read(str(cif_file))
                
                # Save as NPZ (compressed numpy format)
                np.savez_compressed(
                    npz_file,
                    positions=atoms.positions,
                    numbers=atoms.numbers,
                    cell=atoms.cell,
                    pbc=atoms.pbc
                )
                
                converted += 1
                
            except Exception as e:
                failed += 1
                tqdm.write(f"⚠️  Failed: {cif_file.name} - {e}")
            
            pbar.update(1)
            pbar.set_postfix({
                'converted': converted,
                'skipped': skipped,
                'failed': failed
            })
    
    pbar.close()
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"✅ Converted:  {converted} files")
    print(f"⏭️  Skipped:    {skipped} files (already existed)")
    print(f"❌ Failed:     {failed} files")
    print(f"📊 Total:      {total_cifs} files")
    
    # Check file sizes
    if converted > 0:
        print("\n📏 File size comparison (first converted file):")
        for idx, row in df.iterrows():
            folder = Path(row['structure_folder'])
            if not folder.is_absolute():
                folder = Path(csv_path).parent / folder
            
            cif_file = folder / "initial_relaxed.cif"
            npz_file = folder / "initial_relaxed.npz"
            
            if cif_file.exists() and npz_file.exists():
                cif_size = cif_file.stat().st_size
                npz_size = npz_file.stat().st_size
                
                print(f"   CIF: {cif_size:,} bytes")
                print(f"   NPZ: {npz_size:,} bytes")
                print(f"   Ratio: {npz_size/cif_size:.2f}x")
                break
    
    print("\n✅ Conversion complete!")
    print("="*70)


def benchmark_loading_speed(csv_path: str, n_samples: int = 20):
    """
    Benchmark CIF vs NPZ loading speed.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV database
    n_samples : int
        Number of files to benchmark
    """
    print("\n" + "="*70)
    print("LOADING SPEED BENCHMARK")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    cif_times = []
    npz_times = []
    
    print(f"\n⏱️  Testing {n_samples} files...")
    
    for idx in range(min(n_samples, len(df))):
        row = df.iloc[idx]
        folder = Path(row['structure_folder'])
        
        if not folder.is_absolute():
            folder = Path(csv_path).parent / folder
        
        cif_file = folder / "initial_relaxed.cif"
        npz_file = folder / "initial_relaxed.npz"
        
        if not (cif_file.exists() and npz_file.exists()):
            continue
        
        # Benchmark CIF
        t0 = time.time()
        _ = read(str(cif_file))
        cif_times.append(time.time() - t0)
        
        # Benchmark NPZ
        t0 = time.time()
        from ase import Atoms
        data = np.load(npz_file)
        _ = Atoms(
            numbers=data['numbers'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc']
        )
        npz_times.append(time.time() - t0)
    
    if len(cif_times) > 0:
        cif_mean = np.mean(cif_times) * 1000
        npz_mean = np.mean(npz_times) * 1000
        speedup = cif_mean / npz_mean
        
        print(f"\n📊 Results ({len(cif_times)} files):")
        print(f"   CIF loading: {cif_mean:.2f} ms")
        print(f"   NPZ loading: {npz_mean:.2f} ms")
        print(f"   🔥 Speedup:  {speedup:.1f}x faster!")
    else:
        print("⚠️  No valid files found for benchmark")
    
    print("="*70)


if __name__ == "__main__":
    # Configuration
    csv_path = "/home/klemens/databases/MoNbTaW.csv"
    
    print("\n🚀 Starting CIF to NPZ conversion...")
    print(f"   Database: {csv_path}")
    print(f"   NPZ = Numpy compressed format (fast + small)")
    
    # Convert all CIFs
    convert_cif_to_npz(csv_path)
    
    # Benchmark loading speed
    benchmark_loading_speed(csv_path, n_samples=20)
    
    print("\n💡 Next steps:")
    print("   1. Your CIF files are unchanged (still there)")
    print("   2. New .npz files created next to each .cif")
    print("   3. Update GraphBuilder to use NPZ (see below)")
    print("\n📝 To use NPZ in GraphBuilder, add this method:")
    print("""
    def _read_structure_from_npz(self, npz_path: str) -> Atoms:
        import numpy as np
        from ase import Atoms
        
        data = np.load(npz_path)
        atoms = Atoms(
            numbers=data['numbers'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc']
        )
        return atoms
    
    def _read_structure_from_cif(self, cif_path: str) -> Atoms:
        # Try NPZ first (fast!)
        npz_path = Path(cif_path).with_suffix('.npz')
        if npz_path.exists():
            return self._read_structure_from_npz(str(npz_path))
        
        # Fallback to CIF
        return ase_read(cif_path)
    """)