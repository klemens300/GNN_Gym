"""
Oracle for diffusion coefficient calculations on BCC high-entropy alloys.

Workflow:
1. Create BCC supercell with moving atom at center
2. Relax full structure (cell + atoms)
3. Calculate elastic tensor on relaxed structure
4. Calculate vacancy formation energy
5. Pick random nearest neighbor of center atom
6. Build initial structure (vacancy at center)
7. Build final structure (vacancy at neighbor)
8. Relax initial & final (no cell relaxation)
9. Run NEB between initial and final
10. Return DiffusionResult with all raw values

Supports CHGNet and FAIRChem (UMA) calculators.
"""

import csv
import json
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict

from ase import Atoms
from ase.io import write
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.mep import NEB
from ase.units import GPa

from matscipy.elasticity import fit_elastic_constants, Voigt_6x6_to_cubic

import gc
import torch
import warnings

from diffusion.config import DiffusionConfig
from diffusion.results import DiffusionResult

warnings.filterwarnings("ignore")


class DiffusionOracle:
    """
    Compute engine for vacancy diffusion in BCC alloys.

    Usage:
        config = DiffusionConfig()
        with DiffusionOracle(config) as oracle:
            result = oracle.calculate(composition, moving_atom="W")
            if result is not None:
                oracle.save(result)
    """

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.database_dir = Path(config.database_dir)
        self.csv_path = Path(config.csv_path)

        self.logger = self._setup_logger(
            Path(config.log_dir) / "diffusion_oracle.log",
            config.log_level,
            config.log_to_console,
        )

        self.database_dir.mkdir(exist_ok=True, parents=True)

        self.calculator_name = config.calculator
        self._init_calculator()

        self.logger.info("DiffusionOracle initialized")
        self.logger.info(f"  Calculator: {self.calculator_name}")
        self.logger.info(f"  Database:   {self.database_dir}")

    # ------------------------------------------------------------------ #
    #  Setup helpers                                                       #
    # ------------------------------------------------------------------ #

    def _setup_logger(self, log_file: Path, level: str = "INFO", console: bool = True):
        logger = logging.getLogger("diffusion_oracle")
        logger.setLevel(getattr(logging, level.upper()))
        logger.handlers = []
        logger.propagate = False

        log_file.parent.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, level.upper()))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        if console:
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, level.upper()))
            ch.setFormatter(fmt)
            logger.addHandler(ch)

        return logger

    def _init_calculator(self):
        """Store calculator metadata (predictors loaded per-use)."""
        if self.calculator_name == "chgnet":
            self.model_name = f"CHGNet-{self.config.chgnet_model}"
        elif self.calculator_name == "fairchem":
            self.model_name = self.config.fairchem_model
        else:
            raise ValueError(f"Unknown calculator: {self.calculator_name}")

    def _create_calculator(self):
        """Create a fresh ASE calculator instance."""
        if self.calculator_name == "chgnet":
            from chgnet.model.dynamics import CHGNetCalculator
            return CHGNetCalculator(verbose=False)

        elif self.calculator_name == "fairchem":
            from fairchem.core import pretrained_mlip, FAIRChemCalculator
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = pretrained_mlip.get_predict_unit(
                self.model_name, device=device
            )
            return FAIRChemCalculator(predictor, task_name=self.config.fairchem_task)

        raise ValueError(f"Unknown calculator: {self.calculator_name}")

    # ------------------------------------------------------------------ #
    #  Composition helpers                                                 #
    # ------------------------------------------------------------------ #

    def _composition_to_string(self, composition: dict) -> str:
        return "".join(
            f"{el}{int(round(frac * 100))}"
            for el, frac in sorted(composition.items())
        )

    def _get_next_run_number(self, comp_string: str) -> int:
        comp_dir = self.database_dir / comp_string
        if not comp_dir.exists():
            return 1
        runs = []
        for d in comp_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                try:
                    runs.append(int(d.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return max(runs) + 1 if runs else 1

    # ------------------------------------------------------------------ #
    #  Structure creation                                                  #
    # ------------------------------------------------------------------ #

    def _create_bcc_supercell(
        self, composition: dict, moving_atom: str, seed: int = None
    ) -> Tuple[Atoms, int]:
        """
        Create BCC supercell with moving atom placed at center site.

        1. Build cubic BCC supercell with Vegard lattice parameter
        2. Distribute elements randomly according to composition
        3. Override atom closest to cell center with moving_atom
        """
        if seed is not None:
            np.random.seed(seed)

        # Lattice parameter via Vegard's law
        try:
            from atomic_properties import get_vegard_lattice_parameter
            a = get_vegard_lattice_parameter(
                composition, fallback=self.config.lattice_parameter
            )
        except ImportError:
            a = self.config.lattice_parameter

        size = self.config.supercell_size
        crystal = bulk("Fe", crystalstructure="bcc", a=a, cubic=True)
        supercell = crystal.repeat([size, size, size])
        n_atoms = len(supercell)

        # Random element assignment according to composition
        elements = []
        for element, fraction in sorted(composition.items()):
            elements.extend([element] * int(round(fraction * n_atoms)))
        while len(elements) < n_atoms:
            elements.append(max(composition, key=composition.get))
        while len(elements) > n_atoms:
            elements.pop()
        np.random.shuffle(elements)
        supercell.set_chemical_symbols(elements)

        # Place moving atom at center
        center_idx = self._find_center_index(supercell)
        symbols = list(supercell.get_chemical_symbols())
        symbols[center_idx] = moving_atom
        supercell.set_chemical_symbols(symbols)

        self.logger.debug(
            f"Supercell: {n_atoms} atoms, a={a:.4f} A, "
            f"center_idx={center_idx} -> {moving_atom}"
        )
        return supercell, center_idx

    def _find_center_index(self, atoms: Atoms) -> int:
        """Return index of atom closest to cell center."""
        cell = atoms.cell.array
        center = np.diagonal(cell) / 2.0
        diffs = atoms.positions - center
        for i in range(3):
            diffs[:, i] -= np.round(diffs[:, i] / cell[i, i]) * cell[i, i]
        return int(np.argmin(np.linalg.norm(diffs, axis=1)))

    def _get_nearest_neighbors(
        self, atoms: Atoms, ref_pos: np.ndarray, n: int = 8
    ) -> List[int]:
        """Return indices of n nearest neighbors to ref_pos (min-image)."""
        cell = atoms.cell.array
        diffs = atoms.positions - ref_pos
        for i in range(3):
            diffs[:, i] -= np.round(diffs[:, i] / cell[i, i]) * cell[i, i]
        return list(np.argsort(np.linalg.norm(diffs, axis=1))[:n])

    def _calculate_jump_distance(
        self, initial: Atoms, final: Atoms
    ) -> float:
        """
        Compute actual jump distance from relaxed NEB endpoints.

        Finds the atom with the largest displacement between initial
        and final (minimum image convention). This is the jumping atom.

        Returns jump distance in Angstrom.
        """
        cell = initial.cell.array
        diffs = final.positions - initial.positions
        # Apply minimum image convention
        for i in range(3):
            diffs[:, i] -= np.round(diffs[:, i] / cell[i, i]) * cell[i, i]
        displacements = np.linalg.norm(diffs, axis=1)
        return float(np.max(displacements))

    # ------------------------------------------------------------------ #
    #  Initial / Final structure builders                                  #
    # ------------------------------------------------------------------ #

    def _build_initial_structure(
        self, original, center_idx, neighbor_idx, moving_atom, relaxed_cell
    ) -> Atoms:
        """
        Build initial NEB endpoint: vacancy at center.

        1. Replace neighbor element with moving_atom
        2. Remove center atom (vacancy at center)
        3. Impose relaxed cell
        """
        atoms = original.copy()
        symbols = list(atoms.get_chemical_symbols())
        symbols[neighbor_idx] = moving_atom
        atoms.set_chemical_symbols(symbols)
        del atoms[center_idx]
        atoms.set_cell(relaxed_cell, scale_atoms=True)
        return atoms

    def _build_final_structure(
        self, original, center_idx, neighbor_idx, moving_atom, relaxed_cell
    ) -> Atoms:
        """
        Build final NEB endpoint: vacancy at neighbor site.

        Same atom ordering as initial (remove center_idx), but move
        neighbor atom to center position first.
        """
        atoms = original.copy()
        symbols = list(atoms.get_chemical_symbols())
        symbols[neighbor_idx] = moving_atom
        atoms.set_chemical_symbols(symbols)
        atoms.positions[neighbor_idx] = atoms.positions[center_idx].copy()
        del atoms[center_idx]
        atoms.set_cell(relaxed_cell, scale_atoms=True)
        return atoms

    # ------------------------------------------------------------------ #
    #  Relaxation                                                          #
    # ------------------------------------------------------------------ #

    def _relax_structure(
        self, atoms: Atoms, relax_cell: bool = True
    ) -> Tuple[Atoms, float, float]:
        """
        Relax structure using FIRE optimizer.

        Returns (relaxed_atoms, energy_eV, time_seconds).
        """
        t0 = time.time()
        atoms.calc = self._create_calculator()

        if relax_cell:
            opt_target = FrechetCellFilter(atoms)
        else:
            opt_target = atoms

        optimizer = FIRE(opt_target, logfile=None)
        optimizer.run(fmax=self.config.relax_fmax, steps=self.config.relax_max_steps)

        energy = atoms.get_potential_energy()
        dt = time.time() - t0

        del optimizer
        if hasattr(atoms, "calc") and atoms.calc is not None:
            del atoms.calc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return atoms, energy, dt

    # ------------------------------------------------------------------ #
    #  Elastic tensor                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_elastic_tensor(
        self, relaxed_atoms: Atoms
    ) -> Tuple[Dict[str, float], float]:
        """
        Calculate elastic tensor using matscipy on relaxed structure.

        Returns ({"C11": ..., "C12": ..., "C44": ...} in GPa, time_seconds).
        """
        t0 = time.time()
        atoms = relaxed_atoms.copy()
        atoms.calc = self._create_calculator()

        try:
            Cij, Cij_err = fit_elastic_constants(
                atoms,
                symmetry="cubic",
                N_steps=self.config.elastic_n_steps,
                delta=self.config.elastic_delta,
                optimizer=None,
            )
            C11, C12, C44 = Voigt_6x6_to_cubic(Cij)
            result = {"C11": C11 / GPa, "C12": C12 / GPa, "C44": C44 / GPa}
        finally:
            if hasattr(atoms, "calc") and atoms.calc is not None:
                del atoms.calc
            del atoms
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result, time.time() - t0

    # ------------------------------------------------------------------ #
    #  Vacancy formation energy                                            #
    # ------------------------------------------------------------------ #

    def _calculate_vacancy_formation_energy(
        self, relaxed_structure: Atoms, center_idx: int, E_original: float
    ) -> Tuple[float, float, float]:
        """
        Calculate vacancy formation energy E_f^V.

        Creates a clean vacancy (remove center atom, no chemical swap),
        relaxes without cell relaxation, and computes:

            E_f^V = E_vacancy - (N-1)/N * E_original

        Returns (E_f_eV, E_vacancy_eV, time_seconds).
        """
        n_atoms = len(relaxed_structure)

        # Create vacancy structure from relaxed perfect crystal
        vacancy_structure = relaxed_structure.copy()
        del vacancy_structure[center_idx]

        # Relax atomic positions only (cell fixed)
        vacancy_relaxed, E_vacancy, dt = self._relax_structure(
            vacancy_structure, relax_cell=False
        )

        # Vacancy formation energy
        E_f = E_vacancy - (n_atoms - 1) / n_atoms * E_original

        self.logger.info(
            f"  E_f^V = {E_f:.4f} eV "
            f"(E_vac={E_vacancy:.4f}, E_orig={E_original:.4f}, N={n_atoms}) "
            f"({dt:.1f}s)"
        )

        return E_f, E_vacancy, dt

    # ------------------------------------------------------------------ #
    #  NEB                                                                 #
    # ------------------------------------------------------------------ #

    def _run_neb(
        self, initial: Atoms, final: Atoms
    ) -> Tuple[float, float, List[float], List[Atoms], float]:
        """
        Run NEB between initial and final structures.

        Returns (forward_barrier, backward_barrier, energies, images, time_s).
        """
        t0 = time.time()
        n = self.config.neb_images

        initial = initial.copy()
        final = final.copy()
        initial.calc = self._create_calculator()
        final.calc = self._create_calculator()

        # Interpolate intermediate images
        images = [initial]
        for i in range(1, n - 1):
            img = initial.copy()
            frac = i / (n - 1)
            img.positions = (1 - frac) * initial.positions + frac * final.positions
            img.calc = self._create_calculator()
            images.append(img)
        images.append(final)

        neb = NEB(
            images, k=self.config.neb_spring_constant, climb=self.config.neb_climb
        )
        optimizer = FIRE(neb, logfile=None)
        optimizer.run(fmax=self.config.neb_fmax, steps=self.config.neb_max_steps)

        energies = [img.get_potential_energy() for img in images]
        fwd = max(energies) - energies[0]
        bwd = max(energies) - energies[-1]
        images_copy = [img.copy() for img in images]

        # Cleanup
        for img in images:
            if hasattr(img, "calc") and img.calc is not None:
                del img.calc
        del neb, optimizer, images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return fwd, bwd, energies, images_copy, time.time() - t0

    # ------------------------------------------------------------------ #
    #  Save helpers                                                        #
    # ------------------------------------------------------------------ #

    def _save_structure_as_npz(self, atoms: Atoms, path: Path):
        np.savez_compressed(
            path,
            positions=atoms.positions,
            numbers=atoms.numbers,
            cell=atoms.cell.array,
            pbc=atoms.pbc,
        )

    # ------------------------------------------------------------------ #
    #  Main calculation                                                    #
    # ------------------------------------------------------------------ #

    def calculate(
        self, composition: dict, moving_atom: str
    ) -> Optional[DiffusionResult]:
        """
        Run the full diffusion workflow for one composition and moving atom.

        Returns a DiffusionResult on success, None on failure.
        """
        calc_start = time.time()
        comp_string = self._composition_to_string(composition)
        run_number = self._get_next_run_number(comp_string)

        self.logger.info(
            f"[{comp_string}] run {run_number} | moving_atom={moving_atom}"
        )

        try:
            # 1. Create structure
            structure, center_idx = self._create_bcc_supercell(
                composition, moving_atom, seed=run_number
            )
            n_atoms = len(structure)
            center_pos = structure.positions[center_idx].copy()
            self.logger.info(
                f"  Structure: {n_atoms} atoms, center_idx={center_idx} ({moving_atom})"
            )

            # 2. Relax full structure (with cell relaxation)
            self.logger.info("  Relaxing original structure (cell + atoms)...")
            relaxed_structure, E_original, t_relax_orig = self._relax_structure(
                structure.copy(), relax_cell=True
            )
            relaxed_cell = relaxed_structure.get_cell()
            self.logger.info(
                f"  E_original = {E_original:.4f} eV ({t_relax_orig:.1f}s)"
            )

            # 3. Elastic tensor
            self.logger.info("  Calculating elastic tensor...")
            Cij, t_elastic = self._calculate_elastic_tensor(relaxed_structure)
            self.logger.info(
                f"  C11={Cij['C11']:.1f}, C12={Cij['C12']:.1f}, "
                f"C44={Cij['C44']:.1f} GPa ({t_elastic:.1f}s)"
            )

            # 4. Vacancy formation energy
            self.logger.info("  Calculating vacancy formation energy...")
            E_f, E_vacancy, t_relax_vac = self._calculate_vacancy_formation_energy(
                relaxed_structure, center_idx, E_original
            )

            # 5. Pick random nearest neighbor
            neighbors = self._get_nearest_neighbors(structure, center_pos, n=8)
            neighbors = [i for i in neighbors if i != center_idx]
            neighbor_idx = int(np.random.choice(neighbors))
            neighbor_element = structure.get_chemical_symbols()[neighbor_idx]
            self.logger.info(f"  Neighbor: idx={neighbor_idx} ({neighbor_element})")

            # 6. Build & relax initial structure (vacancy at center)
            self.logger.info("  Building initial structure (vacancy at center)...")
            initial_unrelaxed = self._build_initial_structure(
                structure, center_idx, neighbor_idx, moving_atom, relaxed_cell
            )
            initial_relaxed, E_initial, t_relax_init = self._relax_structure(
                initial_unrelaxed.copy(), relax_cell=False
            )
            self.logger.info(
                f"  E_initial = {E_initial:.4f} eV ({t_relax_init:.1f}s)"
            )

            # 7. Build & relax final structure (vacancy at neighbor)
            self.logger.info("  Building final structure (vacancy at neighbor)...")
            final_unrelaxed = self._build_final_structure(
                structure, center_idx, neighbor_idx, moving_atom, relaxed_cell
            )
            final_relaxed, E_final, t_relax_final = self._relax_structure(
                final_unrelaxed.copy(), relax_cell=False
            )
            self.logger.info(
                f"  E_final = {E_final:.4f} eV ({t_relax_final:.1f}s)"
            )

            # 8. NEB
            self.logger.info("  Running NEB...")
            fwd_barrier, bwd_barrier, neb_energies, neb_images, t_neb = (
                self._run_neb(initial_relaxed, final_relaxed)
            )
            self.logger.info(
                f"  Barriers: fwd={fwd_barrier:.4f}, bwd={bwd_barrier:.4f} eV "
                f"({t_neb:.1f}s)"
            )

            # Validate barriers
            b_min = self.config.barrier_min_cutoff
            b_max = self.config.barrier_max_cutoff
            if not (b_min <= fwd_barrier <= b_max and b_min <= bwd_barrier <= b_max):
                self.logger.warning(
                    f"  REJECTED: barriers {fwd_barrier:.3f}/{bwd_barrier:.3f} "
                    f"outside [{b_min}, {b_max}]"
                )
                return None

            total_time = time.time() - calc_start

            # Structure properties from relaxed perfect crystal
            volume = relaxed_structure.get_volume()
            mass = sum(relaxed_structure.get_masses())
            density = mass * 1.66054e-27 / (volume * 1e-30)
            lattice_param = np.mean(np.diagonal(relaxed_structure.cell.array))

            # Jump distance from relaxed NEB endpoints (minimum image convention)
            jump_distance = self._calculate_jump_distance(
                initial_relaxed, final_relaxed
            )
            self.logger.info(f"  Jump distance: {jump_distance:.4f} A")

            # 9. Build result
            result = DiffusionResult(
                composition=composition,
                composition_string=comp_string,
                moving_atom=moving_atom,
                neighbor_element=neighbor_element,
                run_number=run_number,
                E_original_eV=float(E_original),
                E_vacancy_eV=float(E_vacancy),
                E_initial_eV=float(E_initial),
                E_final_eV=float(E_final),
                vacancy_formation_energy_eV=float(E_f),
                forward_barrier_eV=float(fwd_barrier),
                backward_barrier_eV=float(bwd_barrier),
                neb_energies_eV=[float(e) for e in neb_energies],
                C11_GPa=Cij["C11"],
                C12_GPa=Cij["C12"],
                C44_GPa=Cij["C44"],
                n_atoms=n_atoms,
                lattice_parameter_A=float(lattice_param),
                jump_distance_A=float(jump_distance),
                volume_A3=float(volume),
                density_kg_m3=float(density),
                calculator=self.calculator_name,
                model=self.model_name,
                time_relax_original_s=float(t_relax_orig),
                time_relax_vacancy_s=float(t_relax_vac),
                time_elastic_s=float(t_elastic),
                time_relax_initial_s=float(t_relax_init),
                time_relax_final_s=float(t_relax_final),
                time_neb_s=float(t_neb),
                time_total_s=float(total_time),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            # 10. Save structures if configured
            if self.config.save_structures:
                run_dir = self.database_dir / comp_string / f"run_{run_number}"
                run_dir.mkdir(parents=True, exist_ok=True)
                result.structure_folder = str(run_dir)

                write(run_dir / "original_unrelaxed.cif", structure)
                write(run_dir / "original_relaxed.cif", relaxed_structure)
                write(run_dir / "initial_unrelaxed.cif", initial_unrelaxed)
                write(run_dir / "initial_relaxed.cif", initial_relaxed)
                write(run_dir / "final_unrelaxed.cif", final_unrelaxed)
                write(run_dir / "final_relaxed.cif", final_relaxed)
                for i, img in enumerate(neb_images):
                    write(run_dir / f"neb_image_{i}.cif", img)

                self._save_structure_as_npz(structure, run_dir / "original_unrelaxed.npz")
                self._save_structure_as_npz(relaxed_structure, run_dir / "original_relaxed.npz")
                self._save_structure_as_npz(initial_unrelaxed, run_dir / "initial_unrelaxed.npz")
                self._save_structure_as_npz(initial_relaxed, run_dir / "initial_relaxed.npz")
                self._save_structure_as_npz(final_unrelaxed, run_dir / "final_unrelaxed.npz")
                self._save_structure_as_npz(final_relaxed, run_dir / "final_relaxed.npz")
                for i, img in enumerate(neb_images):
                    self._save_structure_as_npz(img, run_dir / f"neb_image_{i}.npz")

                try:
                    write(run_dir / "neb_path_movie.xyz", neb_images)
                except Exception as e:
                    self.logger.warning(f"  Could not save NEB movie: {e}")

                result.to_json(str(run_dir / "results.json"))

            self.logger.info(f"  DONE in {total_time:.1f}s")
            return result

        except Exception as e:
            self.logger.error(f"  FAILED {comp_string}: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    #  CSV export                                                          #
    # ------------------------------------------------------------------ #

    def save_to_csv(self, result: DiffusionResult):
        """Append a DiffusionResult as one row to the CSV file."""
        write_header = not self.csv_path.exists()

        header = (
            # --- Identification ---
            ["composition_string"]
            + self.config.elements
            + ["moving_atom", "neighbor_element", "run_number"]
            # --- Key results (most important columns first) ---
            + [
                "activation_energy_eV", "activation_energy_kJ_per_mol",
                "vacancy_formation_energy_eV",
                "forward_barrier_eV", "backward_barrier_eV",
            ]
            # --- Elastic constants ---
            + ["C11_GPa", "C12_GPa", "C44_GPa"]
            # --- Structure properties ---
            + ["n_atoms", "lattice_parameter_A", "jump_distance_A", "volume_A3", "density_kg_m3"]
            # --- Raw energies ---
            + ["E_original_eV", "E_vacancy_eV", "E_initial_eV", "E_final_eV"]
            # --- Calculator ---
            + ["calculator", "model"]
            # --- Timings ---
            + [
                "time_relax_original_s", "time_relax_vacancy_s",
                "time_elastic_s", "time_relax_initial_s",
                "time_relax_final_s", "time_neb_s", "time_total_s",
            ]
            # --- Meta ---
            + ["structure_folder", "timestamp"]
        )

        row = [result.composition_string]
        for el in self.config.elements:
            row.append(result.composition.get(el, 0.0))
        row.extend([
            result.moving_atom, result.neighbor_element, result.run_number,
            # Key results
            f"{result.activation_energy_eV:.6f}",
            f"{result.activation_energy_kJ_per_mol:.3f}",
            f"{result.vacancy_formation_energy_eV:.6f}",
            f"{result.forward_barrier_eV:.6f}",
            f"{result.backward_barrier_eV:.6f}",
            # Elastic constants
            f"{result.C11_GPa:.3f}",
            f"{result.C12_GPa:.3f}",
            f"{result.C44_GPa:.3f}",
            # Structure
            result.n_atoms,
            f"{result.lattice_parameter_A:.6f}",
            f"{result.jump_distance_A:.6f}",
            f"{result.volume_A3:.3f}",
            f"{result.density_kg_m3:.3f}",
            # Raw energies
            f"{result.E_original_eV:.6f}",
            f"{result.E_vacancy_eV:.6f}",
            f"{result.E_initial_eV:.6f}",
            f"{result.E_final_eV:.6f}",
            # Calculator
            result.calculator, result.model,
            # Timings
            f"{result.time_relax_original_s:.2f}",
            f"{result.time_relax_vacancy_s:.2f}",
            f"{result.time_elastic_s:.2f}",
            f"{result.time_relax_initial_s:.2f}",
            f"{result.time_relax_final_s:.2f}",
            f"{result.time_neb_s:.2f}",
            f"{result.time_total_s:.2f}",
            # Meta
            result.structure_folder,
            result.timestamp,
        ])

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

    # ------------------------------------------------------------------ #
    #  Cleanup / context manager                                           #
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """Release GPU memory and cached models."""
        self.logger.info("Cleaning up...")
        if self.calculator_name == "chgnet":
            try:
                from chgnet.model import CHGNet
                if hasattr(CHGNet, "_models"):
                    CHGNet._models.clear()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self.logger.info("Cleanup complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False