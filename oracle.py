"""
Oracle for NEB calculations on BCC structures with vacancies.

Handles:
- BCC supercell creation with arbitrary compositions
- Vacancy creation and neighbor selection
- Structure relaxation using CHGNet or FAIRChem
- NEB calculations
- Data storage with run numbering per composition
- NPZ file creation for fast loading

Supports multiple calculators:
- CHGNet: Pre-trained universal model
- FAIRChem: Universal Materials Accelerator (UMA) models for inorganic materials
"""

import os
import sys
import csv
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from ase.io import write
from ase.build import bulk
from ase.mep import NEB
from ase.optimize import FIRE
import warnings
import gc
import torch

warnings.filterwarnings('ignore')


@contextmanager
def suppress_output():
    """Context manager to suppress stdout/stderr during calculator operations"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class Oracle:
    """
    Oracle for vacancy diffusion NEB calculations.
    
    Workflow:
    1. Create BCC supercell with random element distribution
    2. Create vacancy at center
    3. Relax initial structure
    4. Pick random neighbor, move to vacancy
    5. Relax final structure
    6. Run NEB between initial and final
    7. Save everything (structures, energies, barriers)
    8. Save NPZ files for fast loading
    """
    
    def __init__(self, config):
        """
        Initialize Oracle.
        
        Parameters:
        -----------
        config : Config
            Configuration object with all parameters
        """
        self.config = config
        self.database_dir = Path(config.database_dir)
        self.csv_path = Path(config.csv_path)
        
        # Setup logger
        log_file = Path(config.log_dir) / "oracle.log"
        self.logger = self._setup_logger(log_file, config.log_level, config.log_to_console)
        
        # Create database directory
        self.database_dir.mkdir(exist_ok=True)
        
        # Initialize CSV if needed
        self._init_csv()
        
        # Initialize calculator-specific components
        self.calculator_name = config.calculator
        self._init_calculator()
        
        self.logger.info("Oracle initialized")
        self.logger.info(f"  Database: {self.database_dir}")
        self.logger.info(f"  CSV: {self.csv_path}")
        self.logger.info(f"  Calculator: {self.calculator_name}")
    
    def _setup_logger(self, log_file: Path, level: str = "INFO", also_console: bool = True):
        """Setup logger for Oracle."""
        logger = logging.getLogger("oracle")
        logger.setLevel(getattr(logging, level.upper()))
        logger.handlers = []  # Clear existing handlers
        
        # File handler
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, level.upper()))
        
        # Console handler (optional)
        if also_console:
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        if also_console:
            ch.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(fh)
        if also_console:
            logger.addHandler(ch)
        
        return logger
    
    def _init_calculator(self):
        """
        Initialize calculator-specific components.
        Sets up model name from config and loads predictor for FAIRChem.
        """
        if self.calculator_name == "chgnet":
            self.model_name = f"CHGNet-{self.config.chgnet_model}"
            self.predictor = None
            self.logger.info(f"  CHGNet model: {self.model_name}")
            
        elif self.calculator_name == "fairchem":
            # Get model name from config
            self.model_name = self.config.fairchem_model
            
            # Initialize predictor ONCE (shared, but calculators are separate)
            from fairchem.core import pretrained_mlip
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"  Loading FAIRChem predictor: {self.model_name}")
            self.logger.info(f"  Device: {device}")
            
            with suppress_output():
                self.predictor = pretrained_mlip.get_predict_unit(
                    self.model_name,
                    device=device
                )
            
            self.logger.info(f"  ? FAIRChem predictor loaded successfully")
            
        else:
            raise ValueError(f"Unknown calculator: {self.calculator_name}")
    
    def _create_calculator(self):
        """
        Create calculator instance.
        
        Returns:
        --------
        calculator : ASE Calculator
            Configured calculator (CHGNet or FAIRChem)
        """
        if self.calculator_name == "chgnet":
            from chgnet.model.dynamics import CHGNetCalculator
            return CHGNetCalculator(verbose=False)
        
        elif self.calculator_name == "fairchem":
            from fairchem.core import FAIRChemCalculator
            
            # Create NEW calculator instance for each call
            # CRITICAL: Each NEB image needs its own calculator!
            # BUT they can share the same predictor (loaded once in _init_calculator)
            return FAIRChemCalculator(
                self.predictor,  # Shared predictor
                task_name="omat"  # For inorganic materials (OMat24 dataset)
            )
        
        else:
            raise ValueError(f"Unknown calculator: {self.calculator_name}")
    
    def _init_csv(self):
        """Initialize CSV with headers if it doesn't exist"""
        if not self.csv_path.exists():
            # Build header dynamically based on elements
            header = ['composition_string']
            header.extend(self.config.elements)  # Add element columns
            header.extend([
                'run_number',
                'calculator',
                'model',
                'diffusing_element',
                'forward_barrier_eV',
                'backward_barrier_eV',
                'E_initial_eV',
                'E_final_eV',
                'initial_relax_time_s',
                'final_relax_time_s',
                'neb_time_s',
                'total_time_s',
                'structure_folder',
                'timestamp'
            ])
            
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
    
    def _composition_to_string(self, composition: dict) -> str:
        """
        Convert composition dict to string like 'Mo25Nb25Ta25W25'.
        
        Parameters:
        -----------
        composition : dict
            Element composition, e.g., {'Mo': 0.25, 'Nb': 0.25, 'Ta': 0.25, 'W': 0.25}
        
        Returns:
        --------
        comp_str : str
            Composition string
        """
        sorted_items = sorted(composition.items())
        comp_str = ""
        for element, fraction in sorted_items:
            percentage = int(round(fraction * 100))
            comp_str += f"{element}{percentage}"
        return comp_str
    
    def _get_next_run_number(self, composition_string: str) -> int:
        """
        Get next run number for a given composition.
        Run numbers start at 1 and increment for each composition independently.
        
        Parameters:
        -----------
        composition_string : str
            Composition string (e.g., 'Mo25Nb25Ta25W25')
        
        Returns:
        --------
        run_number : int
            Next available run number for this composition
        """
        if not self.csv_path.exists():
            return 1
        
        df = pd.read_csv(self.csv_path)
        
        # Filter for this composition
        comp_df = df[df['composition_string'] == composition_string]
        
        if len(comp_df) == 0:
            return 1
        
        return int(comp_df['run_number'].max()) + 1
    
    def _create_bcc_supercell(self, composition: dict):
        """
        Create BCC supercell with random element distribution.
        
        Parameters:
        -----------
        composition : dict
            Element composition, e.g., {'Mo': 0.25, 'Nb': 0.25, 'Ta': 0.25, 'W': 0.25}
        
        Returns:
        --------
        supercell : ase.Atoms
            BCC supercell with random element distribution
        """
        size = self.config.supercell_size
        lattice_param = self.config.lattice_parameter
        
        # Create BCC supercell (using Fe as template, will replace atoms)
        crystal = bulk('Fe', crystalstructure='bcc', a=lattice_param, cubic=True)
        supercell = crystal.repeat([size, size, size])
        
        # Get total number of atoms
        total_atoms = len(supercell)
        
        # Convert composition to atom counts
        elements = []
        for element, fraction in sorted(composition.items()):
            count = int(round(fraction * total_atoms))
            elements.extend([element] * count)
        
        # Adjust for rounding errors
        while len(elements) < total_atoms:
            elements.append(list(composition.keys())[0])
        while len(elements) > total_atoms:
            elements.pop()
        
        # Randomly shuffle element assignment
        np.random.shuffle(elements)
        
        # Assign elements to atoms
        supercell.set_chemical_symbols(elements)
        
        return supercell
    
    def _create_vacancy_at_center(self, structure):
        """
        Create vacancy by removing atom closest to center.
        
        Parameters:
        -----------
        structure : ase.Atoms
            Structure to create vacancy in
        
        Returns:
        --------
        structure_with_vacancy : ase.Atoms
            Structure with vacancy
        center_index : int
            Index of removed atom (in original structure)
        center_position : np.ndarray
            Position of the vacancy (center position)
        """
        structure = structure.copy()
        
        # Calculate center position
        cell = structure.cell
        supercell_center = np.diagonal(cell) / 2.0
        
        # Find closest atom to center
        positions = structure.positions
        distances = []
        for pos in positions:
            diff = pos - supercell_center
            # Apply minimum image convention
            for i in range(3):
                while diff[i] > cell[i, i] / 2:
                    diff[i] -= cell[i, i]
                while diff[i] < -cell[i, i] / 2:
                    diff[i] += cell[i, i]
            distances.append(np.linalg.norm(diff))
        
        center_index = np.argmin(distances)
        center_position = positions[center_index].copy()
        
        # Remove center atom
        del structure[center_index]
        
        return structure, center_index, center_position
    
    def _get_nearest_neighbors(self, structure, reference_position: np.ndarray, n_neighbors: int = 8):
        """
        Get n nearest neighbors to a reference position.
        
        Parameters:
        -----------
        structure : ase.Atoms
            Structure to search in
        reference_position : np.ndarray
            Position to find neighbors around
        n_neighbors : int
            Number of neighbors to return
        
        Returns:
        --------
        neighbor_indices : list
            Indices of nearest neighbors
        """
        cell = structure.cell
        positions = structure.positions
        
        distances_and_indices = []
        
        for i, pos in enumerate(positions):
            diff = pos - reference_position
            # Apply minimum image convention
            for j in range(3):
                while diff[j] > cell[j, j] / 2:
                    diff[j] -= cell[j, j]
                while diff[j] < -cell[j, j] / 2:
                    diff[j] += cell[j, j]
            
            distance = np.linalg.norm(diff)
            distances_and_indices.append((distance, i))
        
        # Sort by distance and return nearest n
        distances_and_indices.sort()
        neighbor_indices = [idx for _, idx in distances_and_indices[:n_neighbors]]
        
        return neighbor_indices
    
    def _save_structure_as_npz(self, atoms, npz_path):
        """
        Save ASE atoms as NPZ for fast loading.
        
        Parameters:
        -----------
        atoms : ase.Atoms
            Structure to save
        npz_path : str or Path
            Path for NPZ file
        """
        np.savez_compressed(
            npz_path,
            positions=atoms.positions,
            numbers=atoms.numbers,
            cell=atoms.cell.array,
            pbc=atoms.pbc
        )
    
    def _relax_structure(self, atoms):
        """
        Relax structure using configured calculator (SILENT MODE).
        
        Parameters:
        -----------
        atoms : ase.Atoms
            Structure to relax
        
        Returns:
        --------
        relaxed_atoms : ase.Atoms
            Relaxed structure
        energy : float
            Final energy in eV
        relax_time : float
            Relaxation time in seconds
        """
        relax_start = time.time()
        
        if self.calculator_name == "chgnet":
            # CHGNet uses StructOptimizer
            from chgnet.model.dynamics import StructOptimizer
            
            with suppress_output():
                relaxer = StructOptimizer()
                result = relaxer.relax(
                    atoms,
                    relax_cell=self.config.relax_cell,
                    fmax=self.config.relax_fmax,
                    steps=self.config.relax_max_steps,
                    verbose=False
                )
            
            relaxed_structure = result['final_structure']
            relaxed_atoms = relaxed_structure.to_ase_atoms()
            energy = result['trajectory'].energies[-1]  # Final energy
            
            # Cleanup
            del relaxer
            del result
            gc.collect()
        
        elif self.calculator_name == "fairchem":
            # FAIRChem uses standard ASE optimization
            atoms.calc = self._create_calculator()
            
            with suppress_output():
                optimizer = FIRE(atoms, logfile=os.devnull)
                optimizer.run(
                    fmax=self.config.relax_fmax,
                    steps=self.config.relax_max_steps
                )
            
            relaxed_atoms = atoms.copy()
            energy = atoms.get_potential_energy()
            
            # Cleanup
            del optimizer
            if hasattr(atoms, 'calc') and atoms.calc is not None:
                del atoms.calc
            gc.collect()
        
        else:
            raise ValueError(f"Unknown calculator: {self.calculator_name}")
        
        relax_time = time.time() - relax_start
        
        # Cleanup GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return relaxed_atoms, energy, relax_time
    
    def _run_neb(self, initial_relaxed, final_relaxed):
        """
        Run NEB calculation between initial and final structures (SILENT MODE).
        
        Parameters:
        -----------
        initial_relaxed : ase.Atoms
            Initial relaxed structure
        final_relaxed : ase.Atoms
            Final relaxed structure
        
        Returns:
        --------
        forward_barrier : float
            Forward energy barrier (eV)
        backward_barrier : float
            Backward energy barrier (eV)
        neb_energies : list
            Energies of all NEB images (eV)
        images : list
            All NEB images (ase.Atoms objects)
        neb_time : float
            NEB calculation time in seconds
        """
        neb_start = time.time()
        
        n_images = self.config.neb_images
        
        # Attach calculators
        initial_relaxed.calc = self._create_calculator()
        final_relaxed.calc = self._create_calculator()
        
        # Create image chain
        images = [initial_relaxed]
        
        with suppress_output():
            # Create intermediate images by linear interpolation
            for i in range(1, n_images - 1):
                image = initial_relaxed.copy()
                fraction = i / (n_images - 1)
                image.positions = ((1 - fraction) * initial_relaxed.positions +
                                   fraction * final_relaxed.positions)
                image.calc = self._create_calculator()
                images.append(image)
        
        images.append(final_relaxed)
        
        # Run NEB (SILENT)
        with suppress_output():
            neb = NEB(
                images,
                k=self.config.neb_spring_constant,
                climb=self.config.neb_climb
            )
            optimizer = FIRE(neb, logfile=os.devnull)
            optimizer.run(
                fmax=self.config.neb_fmax,
                steps=self.config.neb_max_steps
            )
        
        neb_time = time.time() - neb_start
        
        # Calculate energies and barriers
        neb_energies = [img.get_potential_energy() for img in images]
        max_energy = max(neb_energies)
        forward_barrier = max_energy - neb_energies[0]
        backward_barrier = max_energy - neb_energies[-1]
        
        # Store images for return (before cleanup)
        images_copy = [img.copy() for img in images]
        
        # Cleanup
        for img in images:
            if hasattr(img, 'calc') and img.calc is not None:
                del img.calc
        
        del neb
        del optimizer
        del images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return forward_barrier, backward_barrier, neb_energies, images_copy, neb_time
    
    def calculate(self, composition: dict):
        """
        Calculate NEB barrier for given composition.
        
        Workflow:
        1. Create BCC supercell
        2. Create vacancy at center
        3. Relax initial structure
        4. Pick random neighbor
        5. Move neighbor to vacancy ? final structure
        6. Relax final structure
        7. Run NEB
        8. Save everything (CIF + NPZ)
        
        Parameters:
        -----------
        composition : dict
            Element composition, e.g., {'Mo': 0.25, 'Nb': 0.25, 'Ta': 0.25, 'W': 0.25}
        
        Returns:
        --------
        success : bool
            True if calculation successful
        """
        calc_start = time.time()
        
        comp_string = self._composition_to_string(composition)
        run_number = self._get_next_run_number(comp_string)
        
        self.logger.info(f"Calculating {comp_string} (run {run_number})...")
        
        try:
            # 1. Generate BCC structure
            structure = self._create_bcc_supercell(composition)
            
            # 2. Create vacancy at center
            initial_unrelaxed, center_idx, center_pos = self._create_vacancy_at_center(structure)
            
            # 3. Relax initial structure
            initial_relaxed, E_initial, initial_relax_time = self._relax_structure(initial_unrelaxed.copy())
            
            # 4. Pick random neighbor
            neighbors = self._get_nearest_neighbors(initial_unrelaxed, center_pos, n_neighbors=8)
            chosen_neighbor = np.random.choice(neighbors)
            
            # Get the element of the diffusing atom
            diffusing_element = initial_unrelaxed.get_chemical_symbols()[chosen_neighbor]
            
            # 5. Create final structure (neighbor jumps to vacancy)
            final_unrelaxed = initial_unrelaxed.copy()
            final_unrelaxed.positions[chosen_neighbor] = center_pos
            
            # 6. Relax final structure
            final_relaxed, E_final, final_relax_time = self._relax_structure(final_unrelaxed.copy())
            
            # 7. Run NEB
            forward_barrier, backward_barrier, neb_energies, neb_images, neb_time = self._run_neb(
                initial_relaxed.copy(),
                final_relaxed.copy()
            )
            
            total_time = time.time() - calc_start
            
            # 8. Save everything
            run_dir = self.database_dir / comp_string / f"run_{run_number}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save structures (CIF)
            write(run_dir / "initial_unrelaxed.cif", initial_unrelaxed)
            write(run_dir / "initial_relaxed.cif", initial_relaxed)
            write(run_dir / "final_unrelaxed.cif", final_unrelaxed)
            write(run_dir / "final_relaxed.cif", final_relaxed)
            
            # Save NEB images (CIF)
            for i, img in enumerate(neb_images):
                write(run_dir / f"neb_image_{i}.cif", img)
            
            # NEW: Save structures (NPZ) for fast loading
            self._save_structure_as_npz(initial_unrelaxed, run_dir / "initial_unrelaxed.npz")
            self._save_structure_as_npz(initial_relaxed, run_dir / "initial_relaxed.npz")
            self._save_structure_as_npz(final_unrelaxed, run_dir / "final_unrelaxed.npz")
            self._save_structure_as_npz(final_relaxed, run_dir / "final_relaxed.npz")
            
            # Save NEB images (NPZ)
            for i, img in enumerate(neb_images):
                self._save_structure_as_npz(img, run_dir / f"neb_image_{i}.npz")
            
            # Save results.json
            results = {
                'composition': composition,
                'composition_string': comp_string,
                'run_number': run_number,
                'calculator': self.calculator_name,
                'model': self.model_name,
                'diffusing_element': diffusing_element,
                'E_initial_eV': float(E_initial),
                'E_final_eV': float(E_final),
                'forward_barrier_eV': float(forward_barrier),
                'backward_barrier_eV': float(backward_barrier),
                'neb_energies_eV': [float(e) for e in neb_energies],
                'neb_converged': True,
                'timing': {
                    'initial_relax_s': float(initial_relax_time),
                    'final_relax_s': float(final_relax_time),
                    'neb_s': float(neb_time),
                    'total_s': float(total_time)
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(run_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Append to CSV
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [comp_string]
                # Add element fractions in order
                for elem in self.config.elements:
                    row.append(composition.get(elem, 0.0))
                row.extend([
                    run_number,
                    self.calculator_name,
                    self.model_name,
                    diffusing_element,
                    f"{forward_barrier:.6f}",
                    f"{backward_barrier:.6f}",
                    f"{E_initial:.6f}",
                    f"{E_final:.6f}",
                    f"{initial_relax_time:.2f}",
                    f"{final_relax_time:.2f}",
                    f"{neb_time:.2f}",
                    f"{total_time:.2f}",
                    str(run_dir),
                    timestamp
                ])
                writer.writerow(row)
            
            self.logger.info(f"? Completed in {total_time:.1f}s")
            self.logger.info(f"  Diffusing element: {diffusing_element}")
            self.logger.info(f"  Forward barrier: {forward_barrier:.3f} eV")
            self.logger.info(f"  Backward barrier: {backward_barrier:.3f} eV")
            self.logger.info(f"  Timing: relax={initial_relax_time+final_relax_time:.1f}s, neb={neb_time:.1f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"? Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """
        Cleanup models and GPU memory.
        Call after completing calculations.
        """
        self.logger.info("Cleaning up models and GPU memory...")
        
        # Cleanup predictor if FAIRChem
        if self.calculator_name == "fairchem" and self.predictor is not None:
            del self.predictor
            self.predictor = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear CHGNet cache if using CHGNet
        if self.calculator_name == "chgnet":
            try:
                from chgnet.model import CHGNet
                if hasattr(CHGNet, '_models'):
                    CHGNet._models.clear()
            except:
                pass
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("? Cleanup complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup automatically"""
        self.cleanup()
        return False


if __name__ == "__main__":
    from config import Config
    
    print("="*70)
    print("ORACLE TEST")
    print("="*70)
    
    # This is just to show the structure
    # Don't run actual calculations here - use test_oracle.py instead
    
    config = Config()
    oracle = Oracle(config)
    
    print("\n? Oracle initialized successfully!")
    print("\nTo run actual calculations, use test_oracle.py")