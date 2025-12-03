#!/usr/bin/env python3
"""
Test script for Oracle NEB calculations

Tests both CHGNet and FAIRChem calculators with a simple composition.
"""

import sys
from pathlib import Path
import pandas as pd

from config import Config
from oracle import Oracle


def test_single_calculation(calculator: str = "fairchem", model: str = "uma-s-1p1"):
    """
    Test a single NEB calculation.
    
    Args:
        calculator: "chgnet" or "fairchem"
        model: Model name (e.g., "uma-s-1p1")
    """
    print("="*70)
    print(f"ORACLE TEST - {calculator.upper()}")
    if calculator == "fairchem":
        print(f"Model: {model}")
    print("="*70)
    
    # Setup config
    config = Config()
    config.calculator = calculator
    
    if calculator == "fairchem":
        config.fairchem_model = model
    
    # Use smaller settings for faster testing
    config.neb_images = 5  # 5 total images (initial + 3 intermediate + final)
    config.neb_max_steps = 200
    config.relax_max_steps = 200
    config.relax_fmax = 0.05  # Reasonable convergence
    config.neb_fmax = 0.05
    
    # Test database
    config.csv_path = f"test_oracle_{calculator}.csv"
    config.database_dir = f"test_oracle_{calculator}_db"
    
    print(f"\nConfiguration:")
    print(f"  Calculator: {config.calculator}")
    if calculator == "fairchem":
        print(f"  FAIRChem model: {config.fairchem_model}")
    print(f"  CSV: {config.csv_path}")
    print(f"  Database: {config.database_dir}")
    print(f"  NEB images: {config.neb_images}")
    print(f"  Max steps (relax/NEB): {config.relax_max_steps}/{config.neb_max_steps}")
    print(f"  Force convergence: {config.relax_fmax} eV/Å")
    
    # Test composition: 50% Mo, 50% W
    test_composition = {
        'Mo': 0.5,
        'W': 0.5
    }
    
    print(f"\nTest composition: {test_composition}")
    print("\nInitializing Oracle...")
    
    # Create Oracle with context manager (auto-cleanup)
    try:
        with Oracle(config) as oracle:
            print("\n? Oracle initialized successfully!")
            
            print(f"\nRunning NEB calculation...")
            print("  This may take a few minutes...")
            
            success = oracle.calculate(test_composition)
            
            if success:
                print("\n" + "="*70)
                print("? CALCULATION SUCCESSFUL!")
                print("="*70)
                
                # Read results from CSV
                df = pd.read_csv(config.csv_path)
                last_entry = df.iloc[-1]
                
                print(f"\nResults:")
                print(f"  Composition: {last_entry['composition_string']}")
                print(f"  Run number: {last_entry['run_number']}")
                print(f"  Calculator: {last_entry['calculator']}")
                print(f"  Model: {last_entry['model']}")
                print(f"  Diffusing element: {last_entry['diffusing_element']}")
                print(f"\nEnergies:")
                print(f"  E_initial: {last_entry['E_initial_eV']:.6f} eV")
                print(f"  E_final: {last_entry['E_final_eV']:.6f} eV")
                print(f"  ?E: {last_entry['E_final_eV'] - last_entry['E_initial_eV']:.6f} eV")
                print(f"\nBarriers:")
                print(f"  Forward: {last_entry['forward_barrier_eV']:.6f} eV")
                print(f"  Backward: {last_entry['backward_barrier_eV']:.6f} eV")
                print(f"\nTiming:")
                print(f"  Initial relax: {last_entry['initial_relax_time_s']:.2f} s")
                print(f"  Final relax: {last_entry['final_relax_time_s']:.2f} s")
                print(f"  NEB: {last_entry['neb_time_s']:.2f} s")
                print(f"  Total: {last_entry['total_time_s']:.2f} s")
                print(f"\nOutput:")
                print(f"  Structure folder: {last_entry['structure_folder']}")
                
                # Check if structure files exist
                structure_folder = Path(last_entry['structure_folder'])
                if structure_folder.exists():
                    files = list(structure_folder.glob("*.cif"))
                    print(f"  Files created: {len(files)}")
                    print(f"    - initial_unrelaxed.cif")
                    print(f"    - initial_relaxed.cif")
                    print(f"    - final_unrelaxed.cif")
                    print(f"    - final_relaxed.cif")
                    print(f"    - neb_image_*.cif ({config.neb_images} images)")
                
                print("\n" + "="*70)
                return True
            else:
                print("\n" + "="*70)
                print("? CALCULATION FAILED!")
                print("="*70)
                print("\nCheck the log file for details:")
                print(f"  {config.log_dir}/oracle.log")
                return False
    
    except Exception as e:
        print("\n" + "="*70)
        print("? ORACLE INITIALIZATION FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_calculations(calculator: str = "fairchem", n_tests: int = 3):
    """
    Test multiple calculations with different compositions.
    
    Args:
        calculator: "chgnet" or "fairchem"
        n_tests: Number of test calculations
    """
    print("="*70)
    print(f"ORACLE STRESS TEST - {calculator.upper()}")
    print(f"Running {n_tests} calculations")
    print("="*70)
    
    # Setup config
    config = Config()
    config.calculator = calculator
    
    # Fast settings for stress test
    config.neb_images = 3
    config.neb_max_steps = 100
    config.relax_max_steps = 100
    config.relax_fmax = 0.1
    config.neb_fmax = 0.1
    
    config.csv_path = f"test_oracle_stress_{calculator}.csv"
    config.database_dir = f"test_oracle_stress_{calculator}_db"
    
    # Test compositions
    test_compositions = [
        {'Mo': 0.5, 'W': 0.5},
        {'Mo': 0.25, 'Nb': 0.25, 'Ta': 0.25, 'W': 0.25},
        {'Mo': 0.0, 'Nb': 0.33, 'Ta': 0.33, 'W': 0.34},
    ][:n_tests]
    
    successes = 0
    failures = 0
    
    try:
        with Oracle(config) as oracle:
            for i, comp in enumerate(test_compositions, 1):
                print(f"\n{'='*70}")
                print(f"Test {i}/{n_tests}: {comp}")
                print('='*70)
                
                success = oracle.calculate(comp)
                
                if success:
                    successes += 1
                    print(f"? Test {i} passed")
                else:
                    failures += 1
                    print(f"? Test {i} failed")
    
    except Exception as e:
        print(f"\n? Oracle crashed: {e}")
        import traceback
        traceback.print_exc()
        failures = n_tests - successes
    
    print("\n" + "="*70)
    print("STRESS TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {n_tests}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    if n_tests > 0:
        print(f"Success rate: {successes/n_tests*100:.1f}%")
    print("="*70)
    
    return failures == 0


def test_calculator_import():
    """Test if calculators can be imported."""
    print("="*70)
    print("CALCULATOR IMPORT TEST")
    print("="*70)
    
    # Test CHGNet
    print("\nTesting CHGNet import...")
    try:
        from chgnet.model.dynamics import CHGNetCalculator
        print("  ? CHGNet available")
        chgnet_available = True
    except ImportError as e:
        print(f"  ? CHGNet not available: {e}")
        chgnet_available = False
    
    # Test FAIRChem
    print("\nTesting FAIRChem import...")
    try:
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        print("  ? FAIRChem core available")
        
        # Try loading a predictor
        print("  Testing predictor loading...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Device: {device}")
        
        try:
            predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
            print("  ? Predictor loaded successfully (uma-s-1p1)")
            del predictor
            fairchem_available = True
        except Exception as e:
            print(f"  ? Predictor loading failed: {e}")
            fairchem_available = False
            
    except ImportError as e:
        print(f"  ? FAIRChem not available: {e}")
        fairchem_available = False
    
    print("\n" + "="*70)
    print("IMPORT SUMMARY")
    print("="*70)
    print(f"CHGNet: {'? Available' if chgnet_available else '? Not available'}")
    print(f"FAIRChem: {'? Available' if fairchem_available else '? Not available'}")
    print("="*70)
    
    return chgnet_available, fairchem_available


def cleanup_test_files(calculator: str = "all"):
    """Clean up test files."""
    import shutil
    
    print("\n" + "="*70)
    print("CLEANUP TEST FILES")
    print("="*70)
    
    patterns = []
    if calculator == "all":
        patterns = ["test_oracle_*"]
    else:
        patterns = [f"test_oracle_{calculator}*", f"test_oracle_stress_{calculator}*"]
    
    removed_count = 0
    
    for pattern in patterns:
        # Clean CSV files
        for csv_file in Path(".").glob(f"{pattern}.csv"):
            print(f"Removing: {csv_file}")
            csv_file.unlink()
            removed_count += 1
        
        # Clean database directories
        for db_dir in Path(".").glob(f"{pattern}_db"):
            print(f"Removing: {db_dir}/")
            shutil.rmtree(db_dir)
            removed_count += 1
    
    # Clean log directories (only if they contain oracle test logs)
    log_dir = Path("logs")
    if log_dir.exists():
        oracle_log = log_dir / "oracle.log"
        if oracle_log.exists():
            # Check if it's a test log
            try:
                with open(oracle_log, 'r') as f:
                    first_lines = f.read(500)
                    if "test_oracle" in first_lines.lower():
                        print(f"Removing: {log_dir}/")
                        shutil.rmtree(log_dir)
                        removed_count += 1
            except:
                pass
    
    if removed_count == 0:
        print("No test files found to clean up")
    else:
        print(f"\n? Cleanup complete ({removed_count} items removed)")
    print("="*70)


def test_fairchem_models():
    """Test different FAIRChem models."""
    print("="*70)
    print("FAIRCHEM MODEL COMPARISON TEST")
    print("="*70)
    
    models_to_test = [
        "uma-s-1p1",  # Small, fast
        # "uma-m-1p1",  # Medium (comment out for quick test)
        # "uma-l-1p1",  # Large (comment out for quick test)
    ]
    
    config = Config()
    config.calculator = "fairchem"
    config.neb_images = 3
    config.neb_max_steps = 50
    config.relax_max_steps = 50
    config.relax_fmax = 0.2
    config.neb_fmax = 0.2
    
    test_composition = {'Mo': 0.5, 'W': 0.5}
    
    results = []
    
    for model in models_to_test:
        print(f"\n{'='*70}")
        print(f"Testing model: {model}")
        print('='*70)
        
        config.fairchem_model = model
        config.csv_path = f"test_fairchem_{model}.csv"
        config.database_dir = f"test_fairchem_{model}_db"
        
        try:
            with Oracle(config) as oracle:
                success = oracle.calculate(test_composition)
                
                if success:
                    df = pd.read_csv(config.csv_path)
                    last = df.iloc[-1]
                    
                    results.append({
                        'model': model,
                        'success': True,
                        'barrier': last['backward_barrier_eV'],
                        'time': last['total_time_s']
                    })
                    
                    print(f"  ? Success")
                    print(f"    Barrier: {last['backward_barrier_eV']:.4f} eV")
                    print(f"    Time: {last['total_time_s']:.1f} s")
                else:
                    results.append({
                        'model': model,
                        'success': False,
                        'barrier': None,
                        'time': None
                    })
                    print(f"  ? Failed")
        
        except Exception as e:
            results.append({
                'model': model,
                'success': False,
                'barrier': None,
                'time': None
            })
            print(f"  ? Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Status':<10} {'Barrier (eV)':<15} {'Time (s)':<10}")
    print("-"*70)
    for r in results:
        status = "? Pass" if r['success'] else "? Fail"
        barrier = f"{r['barrier']:.4f}" if r['barrier'] is not None else "N/A"
        time = f"{r['time']:.1f}" if r['time'] is not None else "N/A"
        print(f"{r['model']:<15} {status:<10} {barrier:<15} {time:<10}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Oracle NEB calculations")
    parser.add_argument(
        "--mode",
        choices=["import", "single", "stress", "cleanup", "compare"],
        default="single",
        help="Test mode (default: single)"
    )
    parser.add_argument(
        "--calculator",
        choices=["chgnet", "fairchem", "all"],
        default="fairchem",
        help="Calculator to test (default: fairchem)"
    )
    parser.add_argument(
        "--model",
        default="uma-s-1p1",
        help="FAIRChem model name (default: uma-s-1p1)"
    )
    parser.add_argument(
        "--n-tests",
        type=int,
        default=3,
        help="Number of tests for stress mode (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Run tests
    if args.mode == "import":
        # Test imports
        chgnet_ok, fairchem_ok = test_calculator_import()
        sys.exit(0 if (chgnet_ok and fairchem_ok) else 1)
    
    elif args.mode == "single":
        # Single calculation test
        if args.calculator == "all":
            print("Testing both calculators...\n")
            chgnet_ok = test_single_calculation("chgnet")
            print("\n")
            fairchem_ok = test_single_calculation("fairchem", args.model)
            
            print("\n" + "="*70)
            print("OVERALL RESULTS")
            print("="*70)
            print(f"CHGNet: {'? Pass' if chgnet_ok else '? Fail'}")
            print(f"FAIRChem: {'? Pass' if fairchem_ok else '? Fail'}")
            print("="*70)
            
            sys.exit(0 if (chgnet_ok and fairchem_ok) else 1)
        else:
            success = test_single_calculation(args.calculator, args.model)
            sys.exit(0 if success else 1)
    
    elif args.mode == "stress":
        # Stress test
        if args.calculator == "all":
            print("Stress testing both calculators...\n")
            chgnet_ok = test_multiple_calculations("chgnet", args.n_tests)
            print("\n")
            fairchem_ok = test_multiple_calculations("fairchem", args.n_tests)
            
            print("\n" + "="*70)
            print("OVERALL RESULTS")
            print("="*70)
            print(f"CHGNet: {'? Pass' if chgnet_ok else '? Fail'}")
            print(f"FAIRChem: {'? Pass' if fairchem_ok else '? Fail'}")
            print("="*70)
            
            sys.exit(0 if (chgnet_ok and fairchem_ok) else 1)
        else:
            success = test_multiple_calculations(args.calculator, args.n_tests)
            sys.exit(0 if success else 1)
    
    elif args.mode == "compare":
        # Compare different FAIRChem models
        test_fairchem_models()
    
    elif args.mode == "cleanup":
        # Cleanup test files
        cleanup_test_files(args.calculator)