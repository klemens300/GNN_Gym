"""
Oracle for NEB calculations on BCC structures with vacancies.
ROBUST VERSION: No stdout hijacking, clean logging, saves unrelaxed NPZ.
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
from typing import Tuple, List, Optional
import gc
import warnings

# ASE imports
from ase import Atoms
from ase.io import write
from ase.build import bulk
from ase.mep import NEB
from ase.optimize import FIRE
import torch

warnings.filterwarnings('ignore')

class Oracle:
    """
    Oracle for vacancy diffusion NEB calculations.
    """
    
    def __init__(self, config):
        """Initialize Oracle."""
        self.config = config
        self.database_dir = Path(config.database_dir)
        self.csv_path = Path(config.csv_path)
        
        # Logging: Get logger but do not configure handlers here.
        # This prevents conflict with the main script's logging setup.
        self.logger = logging.getLogger("oracle")
        
        self.database_dir.mkdir(exist_ok=True, parents=True)
        self._init_csv()
        
        self.calculator_name = config.calculator
        self._init_calculator()
        
        self.logger.info("Oracle initialized")
    
    def _init_calculator(self):
        """Initialize calculator-specific components."""
        if self.calculator_name == "chgnet":
            self.model_name = f"CHGNet-{self.config.chgnet_model}"
            self.predictor = None
            self.logger.info(f"  CHGNet model: {self.model_name}")
            
        elif self.calculator_name == "fairchem":
            self.model_name = self.config.fairchem_model
            from fairchem.core import pretrained_mlip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"  Loading FAIRChem predictor: {self.model_name}")
            
            # Load directly without suppressing output to see potential errors
            self.predictor = pretrained_mlip.get_predict_unit(self.model_name, device=device)
            self.logger.info(f"  FAIRChem predictor loaded")
            
        else:
            raise ValueError(f"Unknown calculator: {self.calculator_name}")
    
    def _create_calculator(self):
        """Create calculator instance."""
        if self.calculator_name == "chgnet":
            from chgnet.model.dynamics import CHGNetCalculator
            return CHGNetCalculator(verbose=False)
        
        elif self.calculator_name == "fairchem":
            from fairchem.core import FAIRChemCalculator
            return FAIRChemCalculator(self.predictor, task_name="omat")
        
        else:
            raise ValueError(f"Unknown calculator: {self.calculator_name}")
    
    def _init_csv(self):
        """Initialize CSV with headers if it doesn't exist"""
        if not self.csv_path.exists():
            header = ['composition_string'] + self.config.elements + [
                'run_number', 'calculator', 'model', 'diffusing_element',
                'forward_barrier_eV', 'backward_barrier_eV', 'E_initial_eV', 'E_final_eV',
                'initial_relax_time_s', 'final_relax_time_s', 'neb_time_s', 'total_time_s',
                'structure_folder', 'timestamp'
            ]
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(header)
    
    def _composition_to_string(self, composition: dict) -> str:
        """Convert composition dict to string like 'Mo25Nb25Ta25W25'."""
        comp_str = ""
        for element, fraction in sorted(composition.items()):
            comp_str += f"{element}{int(round(fraction * 100))}"
        return comp_str
    
    def _get_next_run_number(self, composition_string: str) -> int:
        """Get next run number for a given composition."""
        if not self.csv_path.exists(): return 1
        try:
            df = pd.read_csv(self.csv_path)
            if 'composition_string' not in df.columns: return 1
            comp_df = df[df['composition_string'] == composition_string]
            return int(comp_df['run_number'].max()) + 1 if len(comp_df) > 0 else 1
        except Exception:
            return 1
    
    def _create_bcc_supercell(self, composition: dict):
        """Create BCC supercell with random element distribution."""
        size = self.config.supercell_size
        lattice_param = self.config.lattice_parameter
        
        crystal = bulk('Fe', crystalstructure='bcc', a=lattice_param, cubic=True)
        supercell = crystal.repeat([size, size, size])
        total_atoms = len(supercell)
        
        elements = []
        for element, fraction in sorted(composition.items()):
            elements.extend([element] * int(round(fraction * total_atoms)))
        
        while len(elements) < total_atoms: elements.append(list(composition.keys())[0])
        while len(elements) > total_atoms: elements.pop()
        
        np.random.shuffle(elements)
        supercell.set_chemical_symbols(elements)
        return supercell
    
    def _create_vacancy_at_center(self, structure):
        """Create vacancy by removing atom closest to center."""
        structure = structure.copy()
        cell = structure.cell
        supercell_center = np.diagonal(cell) / 2.0
        
        positions = structure.positions
        distances = []
        for pos in positions:
            diff = pos - supercell_center
            for i in range(3):
                while diff[i] > cell[i, i] / 2: diff[i] -= cell[i, i]
                while diff[i] < -cell[i, i] / 2: diff[i] += cell[i, i]
            distances.append(np.linalg.norm(diff))
        
        center_index = np.argmin(distances)
        center_position = positions[center_index].copy()
        del structure[center_index]
        return structure, center_index, center_position
    
    def _get_nearest_neighbors(self, structure, reference_position: np.ndarray, n_neighbors: int = 8):
        """Get n nearest neighbors to a reference position."""
        cell = structure.cell
        positions = structure.positions
        distances_and_indices = []
        
        for i, pos in enumerate(positions):
            diff = pos - reference_position
            for j in range(3):
                while diff[j] > cell[j, j] / 2: diff[j] -= cell[j, j]
                while diff[j] < -cell[j, j] / 2: diff[j] += cell[j, j]
            distances_and_indices.append((np.linalg.norm(diff), i))
        
        distances_and_indices.sort()
        return [idx for _, idx in distances_and_indices[:n_neighbors]]
    
    def _save_structure_as_npz(self, atoms, npz_path, progress: float = 1.0):
        """Save ASE atoms as NPZ for fast loading, including progress scalar."""
        np.savez_compressed(
            npz_path,
            positions=atoms.positions,
            numbers=atoms.numbers,
            cell=atoms.cell.array,
            pbc=atoms.pbc,
            progress=np.array([progress], dtype=np.float32)
        )
    
    def _relax_structure(self, atoms: Atoms, relax_cell: Optional[bool] = None) -> Tuple[Atoms, float, float, List[Atoms]]:
        """Relax structure and capture trajectory."""
        relax_start = time.time()
        if relax_cell is None: relax_cell = self.config.relax_cell
        
        trajectory_frames = [atoms.copy()]
        def capture_frame(a=atoms): trajectory_frames.append(a.copy())
        
        if self.calculator_name == "chgnet":
            from chgnet.model.dynamics import StructOptimizer
            # verbose=False suppresses output
            relaxer = StructOptimizer()
            result = relaxer.relax(atoms, relax_cell=relax_cell, fmax=self.config.relax_fmax, steps=self.config.relax_max_steps, verbose=False)
            
            relaxed_atoms = result['final_structure'].to_ase_atoms()
            energy = result['trajectory'].energies[-1]
            if hasattr(result['trajectory'], 'structures'):
                trajectory_frames = [s.to_ase_atoms() for s in result['trajectory'].structures]
            else:
                trajectory_frames.append(relaxed_atoms.copy())
            del relaxer, result
        
        elif self.calculator_name == "fairchem":
            atoms.calc = self._create_calculator()
            if relax_cell:
                from ase.constraints import ExpCellFilter
                atoms_to_optimize = ExpCellFilter(atoms)
            else:
                atoms_to_optimize = atoms
            
            # Use logfile=None to silence output without hijacking stdout
            optimizer = FIRE(atoms_to_optimize, logfile=None)
            optimizer.attach(capture_frame, interval=1)
            optimizer.run(fmax=self.config.relax_fmax, steps=self.config.relax_max_steps)
            
            relaxed_atoms = atoms.copy()
            energy = atoms.get_potential_energy()
            del optimizer
            if hasattr(atoms, 'calc') and atoms.calc is not None: del atoms.calc
        
        else:
            raise ValueError(f"Unknown calculator: {self.calculator_name}")
        
        relax_time = time.time() - relax_start
        trajectory_frames.append(relaxed_atoms.copy())
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        
        return relaxed_atoms, energy, relax_time, trajectory_frames
    
    def _run_neb(self, initial_relaxed, final_relaxed):
        """Run NEB calculation between initial and final structures."""
        neb_start = time.time()
        n_images = self.config.neb_images
        
        initial_relaxed.calc = self._create_calculator()
        final_relaxed.calc = self._create_calculator()
        
        images = [initial_relaxed]
        # Linear interpolation
        for i in range(1, n_images - 1):
            image = initial_relaxed.copy()
            fraction = i / (n_images - 1)
            image.positions = ((1 - fraction) * initial_relaxed.positions + fraction * final_relaxed.positions)
            image.calc = self._create_calculator()
            images.append(image)
        images.append(final_relaxed)
        
        # NEB Optimization with logfile=None
        neb = NEB(images, k=self.config.neb_spring_constant, climb=self.config.neb_climb)
        optimizer = FIRE(neb, logfile=None)
        optimizer.run(fmax=self.config.neb_fmax, steps=self.config.neb_max_steps)
        
        neb_time = time.time() - neb_start
        neb_energies = [img.get_potential_energy() for img in images]
        forward_barrier = max(neb_energies) - neb_energies[0]
        backward_barrier = max(neb_energies) - neb_energies[-1]
        
        images_copy = [img.copy() for img in images]
        for img in images:
            if hasattr(img, 'calc') and img.calc is not None: del img.calc
        
        del neb, optimizer, images
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        
        return forward_barrier, backward_barrier, neb_energies, images_copy, neb_time
    
    def calculate(self, composition: dict):
        """Calculate NEB barrier for given composition with trajectory capture."""
        calc_start = time.time()
        comp_string = self._composition_to_string(composition)
        run_number = self._get_next_run_number(comp_string)
        
        # Log less to keep main log clean, or delegate to main loop
        # self.logger.info(f"Calculating {comp_string} (run {run_number})...")
        
        try:
            # 1-3. Create & Relax Initial (WITH Cell Relax)
            structure = self._create_bcc_supercell(composition)
            initial_unrelaxed, center_idx, center_pos = self._create_vacancy_at_center(structure)
            initial_relaxed, E_initial, t_init, initial_traj = self._relax_structure(
                initial_unrelaxed.copy(),
                relax_cell=True
            )
            
            # Capture relaxed cell parameters
            relaxed_cell = initial_relaxed.get_cell()
            
            # 4-5. Create Final in SAME Cell
            neighbors = self._get_nearest_neighbors(initial_unrelaxed, center_pos)
            chosen_neighbor = np.random.choice(neighbors)
            diffusing_element = initial_unrelaxed.get_chemical_symbols()[chosen_neighbor]
            
            final_unrelaxed = initial_unrelaxed.copy()
            final_unrelaxed.positions[chosen_neighbor] = center_pos
            final_unrelaxed.set_cell(relaxed_cell, scale_atoms=True) # Apply relaxed cell
            
            # 6. Relax Final (WITHOUT Cell Relax)
            final_relaxed, E_final, t_final, final_traj = self._relax_structure(final_unrelaxed.copy(), relax_cell=False)
            
            # 7. NEB
            f_barrier, b_barrier, energies, images, t_neb = self._run_neb(initial_relaxed.copy(), final_relaxed.copy())
            
            # 8. Validate
            total_time = time.time() - calc_start
            b_min = getattr(self.config, 'barrier_min_cutoff', 0.0)
            b_max = getattr(self.config, 'barrier_max_cutoff', 5.0)
            
            if not (b_min <= f_barrier <= b_max and b_min <= b_barrier <= b_max):
                self.logger.warning(f"REJECTED: Barriers {f_barrier:.2f}/{b_barrier:.2f} outside [{b_min}, {b_max}]")
                del initial_unrelaxed, initial_relaxed, final_unrelaxed, final_relaxed
                gc.collect()
                return False
            
            # 9. Save
            run_dir = self.database_dir / comp_string / f"run_{run_number}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            write(run_dir / "initial_unrelaxed.cif", initial_unrelaxed)
            write(run_dir / "initial_relaxed.cif", initial_relaxed)
            write(run_dir / "final_unrelaxed.cif", final_unrelaxed)
            write(run_dir / "final_relaxed.cif", final_relaxed)
            for i, img in enumerate(images): write(run_dir / f"neb_image_{i}.cif", img)
            
            # --- WICHTIG: Speichern der unrelaxierten NPZ Dateien (Progress=0.0) ---
            # Das wird für die neue Inference benötigt
            self._save_structure_as_npz(initial_unrelaxed, run_dir / "initial_unrelaxed.npz", 0.0)
            self._save_structure_as_npz(final_unrelaxed, run_dir / "final_unrelaxed.npz", 0.0)
            
            # Save Standard NPZs
            self._save_structure_as_npz(initial_relaxed, run_dir / "initial_relaxed.npz", 1.0)
            self._save_structure_as_npz(final_relaxed, run_dir / "final_relaxed.npz", 1.0)
            for i, img in enumerate(images): self._save_structure_as_npz(img, run_dir / f"neb_image_{i}.npz", 1.0)

            # Save Trajectories (Sampled)
            def save_traj(traj, prefix):
                if not traj: return
                n_save = 5
                indices = np.linspace(0, len(traj)-1, n_save, dtype=int) if len(traj) >= n_save else range(len(traj))
                for i, idx in enumerate(indices):
                    progress = i / (len(indices) - 1) if len(indices) > 1 else 1.0
                    self._save_structure_as_npz(traj[idx], run_dir / f"{prefix}_traj_{i}.npz", progress)
            
            save_traj(initial_traj, "initial")
            save_traj(final_traj, "final")
            
            # JSON & CSV
            results = {
                'composition': composition, 'run_number': run_number, 'model': self.model_name,
                'diffusing_element': diffusing_element, 'E_initial': float(E_initial), 'E_final': float(E_final),
                'forward_barrier': float(f_barrier), 'backward_barrier': float(b_barrier), 'neb_energies': [float(e) for e in energies],
                'timing': {'total': total_time, 'neb': t_neb}
            }
            with open(run_dir / "results.json", 'w') as f: json.dump(results, f, indent=2)
            
            with open(self.csv_path, 'a', newline='') as f:
                row = [comp_string] + [composition.get(e, 0.0) for e in self.config.elements] + [
                    run_number, self.calculator_name, self.model_name, diffusing_element,
                    f"{f_barrier:.6f}", f"{b_barrier:.6f}", f"{E_initial:.6f}", f"{E_final:.6f}",
                    f"{t_init:.2f}", f"{t_final:.2f}", f"{t_neb:.2f}", f"{total_time:.2f}",
                    str(run_dir), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
                csv.writer(f).writerow(row)
            
            # Optional: Log success (debug level or controlled by main loop)
            # self.logger.info(f"Completed in {total_time:.1f}s | Barrier: {b_barrier:.3f} eV")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed {comp_string}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Cleanup models and GPU memory."""
        try: self.logger.info("Cleaning up...")
        except: pass
        
        if self.calculator_name == "fairchem": self.predictor = None
        if self.calculator_name == "chgnet":
            try: 
                from chgnet.model import CHGNet
                if hasattr(CHGNet, '_models'): CHGNet._models.clear()
            except: pass
            
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.cleanup(); return False

if __name__ == "__main__":
    from config import Config
    print("ORACLE INITIALIZED")
    Oracle(Config())