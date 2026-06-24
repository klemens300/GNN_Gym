"""
Configuration for diffusion coefficient calculations on BCC high-entropy alloys.

All parameters in one dataclass with sensible defaults. Override by passing
keyword arguments:

    config = DiffusionConfig(supercell_size=5, composition_step=0.10)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DiffusionConfig:
    """Unified configuration for the diffusion calculation pipeline."""

    # --- Elements & Runs ---
    elements: List[str] = field(default_factory=lambda: ["Mo", "Nb", "Ta", "W"])
    runs_per_composition: int = 1

    # --- Composition space sweep ---
    composition_step: float = 0.10  # Simplex grid spacing as fraction (0.05 = 5%)
    confirm_threshold: int = 500    # Ask for confirmation if pending runs exceed this

    # --- Structure ---
    supercell_size: int = 4         # NxNxN BCC supercell
    lattice_parameter: float = 3.2  # Angstrom (fallback for Vegard's law)

    # --- Calculator ---
    calculator: str = "fairchem"    # "fairchem" or "chgnet"
    fairchem_model: str = "uma-s-1p1"
    fairchem_task: str = "omat"
    chgnet_model: str = "0.3.0"
    device: str = "cuda"

    # --- Relaxation ---
    relax_fmax: float = 0.01        # eV/Angstrom
    relax_max_steps: int = 2000

    # --- NEB ---
    neb_images: int = 7             # Total images including endpoints
    neb_fmax: float = 0.05          # eV/Angstrom
    neb_max_steps: int = 500
    neb_spring_constant: float = 0.1  # eV/Angstrom^2
    neb_climb: bool = True

    # --- Elastic tensor ---
    elastic_n_steps: int = 5
    elastic_delta: float = 0.02     # Strain magnitude (+/- 2%)

    # --- Validation ---
    barrier_min_cutoff: float = 0.0  # eV
    barrier_max_cutoff: float = 5.0  # eV

    # --- Output ---
    save_structures: bool = True    # Save CIF/NPZ files
    save_csv: bool = True           # Append to CSV
    database_dir: str = ""          # Auto-generated if empty
    csv_path: str = ""              # Auto-generated if empty

    # --- Logging ---
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_console: bool = True

    def __post_init__(self):
        """Auto-generate paths from element list if not set, validate step."""
        element_string = "".join(self.elements)
        if not self.database_dir:
            self.database_dir = f"database_diffusion_{element_string}"
        if not self.csv_path:
            self.csv_path = f"diffusion_results_{element_string}.csv"

        # Validate composition_step: 1/step must be (close to) an integer
        inv = 1.0 / self.composition_step
        if abs(inv - round(inv)) > 1e-6:
            raise ValueError(
                f"composition_step={self.composition_step} does not divide 1.0 evenly. "
                f"Use values like 0.5, 0.25, 0.2, 0.1, 0.05, 0.04, 0.025, 0.02, 0.01."
            )

    def summary(self) -> str:
        """Human-readable config summary."""
        lines = [
            f"Elements:     {self.elements}",
            f"Supercell:    {self.supercell_size}x{self.supercell_size}x{self.supercell_size}",
            f"Calculator:   {self.calculator} ({self.fairchem_model})",
            f"Comp. step:   {self.composition_step*100:.1f}%",
            f"Runs/point:   {self.runs_per_composition}",
            f"NEB images:   {self.neb_images}",
            f"Elastic:      +/-{self.elastic_delta*100:.0f}%",
            f"Database:     {self.database_dir}",
            f"CSV:          {self.csv_path}",
        ]
        return "\n".join(lines)
