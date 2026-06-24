"""
Data containers for the diffusion coefficient calculation pipeline.

All results are stored as dataclasses for easy inspection, serialization,
and downstream use in physics calculations.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass
class DiffusionResult:
    """
    Complete output of a single diffusion calculation run.

    Contains all raw values needed to compute the diffusion coefficient D(T):
    - NEB barriers (migration energy E_m)
    - Elastic constants (for attempt frequency k_0)
    - Vacancy formation energy E_f^V (for vacancy concentration c_v)
    - Structural properties (lattice parameter -> jump distance)
    """

    # --- Input parameters ---
    composition: Dict[str, float]
    composition_string: str
    moving_atom: str
    neighbor_element: str
    run_number: int

    # --- Energies (eV) ---
    E_original_eV: float          # Relaxed perfect supercell (N atoms)
    E_vacancy_eV: float           # Relaxed supercell with vacancy (N-1 atoms)
    E_initial_eV: float           # NEB initial endpoint (N-1 atoms)
    E_final_eV: float             # NEB final endpoint (N-1 atoms)

    # --- Vacancy formation energy (eV) ---
    vacancy_formation_energy_eV: float  # E_vac - (N-1)/N * E_original

    # --- NEB barriers (eV) ---
    forward_barrier_eV: float
    backward_barrier_eV: float
    neb_energies_eV: List[float]

    # --- Elastic constants (GPa) ---
    C11_GPa: float
    C12_GPa: float
    C44_GPa: float

    # --- Structure properties ---
    n_atoms: int                  # Number of atoms in perfect supercell
    lattice_parameter_A: float    # Mean diagonal of relaxed cell (Angstrom)
    jump_distance_A: float        # Actual jump distance from relaxed NEB endpoints (Angstrom)
    volume_A3: float              # Cell volume (Angstrom^3)
    density_kg_m3: float          # Mass density (kg/m^3)

    # --- Calculator metadata ---
    calculator: str
    model: str

    # --- Timing (seconds) ---
    time_relax_original_s: float
    time_relax_vacancy_s: float
    time_elastic_s: float
    time_relax_initial_s: float
    time_relax_final_s: float
    time_neb_s: float
    time_total_s: float

    # --- Meta ---
    timestamp: str = ""
    structure_folder: str = ""

    def to_dict(self) -> dict:
        """Convert to plain dictionary (JSON-serializable)."""
        return asdict(self)

    def to_json(self, path: str, indent: int = 2):
        """Save as JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_json(cls, path: str) -> "DiffusionResult":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @property
    def migration_barrier_eV(self) -> float:
        """Convenience alias: forward barrier as migration energy."""
        return self.forward_barrier_eV

    @property
    def activation_energy_eV(self) -> float:
        """Total activation energy Q = E_f^V + E_m (eV)."""
        return self.vacancy_formation_energy_eV + self.forward_barrier_eV

    @property
    def activation_energy_kJ_per_mol(self) -> float:
        """Total activation energy Q = E_f^V + E_m (kJ/mol)."""
        return self.activation_energy_eV * 96.485

    def summary(self) -> str:
        """Human-readable summary of key results."""
        lines = [
            f"Composition:  {self.composition_string}",
            f"Moving atom:  {self.moving_atom} -> {self.neighbor_element} site",
            f"Run:          {self.run_number}",
            f"",
            f"E_f^V:        {self.vacancy_formation_energy_eV:.4f} eV",
            f"E_m (fwd):    {self.forward_barrier_eV:.4f} eV",
            f"E_m (bwd):    {self.backward_barrier_eV:.4f} eV",
            f"Q (E_f+E_m):  {self.activation_energy_eV:.4f} eV = {self.activation_energy_kJ_per_mol:.2f} kJ/mol",
            f"",
            f"C11:          {self.C11_GPa:.1f} GPa",
            f"C12:          {self.C12_GPa:.1f} GPa",
            f"C44:          {self.C44_GPa:.1f} GPa",
            f"",
            f"Lattice:      {self.lattice_parameter_A:.4f} A",
            f"Jump dist:    {self.jump_distance_A:.4f} A",
            f"N atoms:      {self.n_atoms}",
            f"Total time:   {self.time_total_s:.1f} s",
        ]
        return "\n".join(lines)