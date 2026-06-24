"""
Trajectory I/O: write a KMC run to disk for visualization and post-processing.

Two outputs are supported:

- write_extxyz_trajectory(result, path): the periodic-snapshot trajectory as
  Extended XYZ, with the vacancy rendered as a pseudo-atom of a chosen
  symbol so that OVITO and similar viewers display it. Each frame carries
  the simulated time and the vacancy site index in its info dict.

- write_event_log(result, path): a compact CSV with one row per BKL step,
  capturing time, vacancy site before/after, hopper species index, and the
  local jump direction. From this plus the initial species array the full
  trajectory can be replayed exactly.
"""

import csv
from pathlib import Path
from typing import Union

from ase import Atoms
from ase.io import write as ase_write

from KMC.state import VACANCY_SPECIES


def write_extxyz_trajectory(
    result,
    path: Union[str, Path],
    vacancy_symbol: str = "X",
) -> int:
    """Write the snapshot trajectory of a KMC run as Extended XYZ.

    Args:
        result: KMCResult that was created with snapshot_every_n_steps > 0.
        path: output filename (.extxyz or .xyz).
        vacancy_symbol: pseudo-atom symbol used at the vacancy site so that
            it renders in OVITO. Default "X".

    Returns:
        Number of frames written.

    Raises:
        ValueError: if result has no snapshot data.
    """
    if result.snapshot_species is None:
        raise ValueError(
            "Result has no snapshots recorded; rerun with "
            "snapshot_every_n_steps > 0 to enable trajectory output."
        )

    state = result.final_state
    site_positions = state.positions
    cell = state.cell
    element_symbols = state.element_symbols

    frames = []
    for k in range(result.n_snapshots):
        species = result.snapshot_species[k]
        symbols = []
        for sp in species:
            if sp == VACANCY_SPECIES:
                symbols.append(vacancy_symbol)
            else:
                symbols.append(element_symbols[int(sp)])

        atoms = Atoms(
            symbols=symbols,
            positions=site_positions,
            cell=cell,
            pbc=True,
        )
        atoms.info["time_s"] = float(result.snapshot_times_s[k])
        atoms.info["vacancy_index"] = int(result.snapshot_vacancy_indices[k])
        atoms.info["snapshot_index"] = int(k)
        frames.append(atoms)

    ase_write(str(path), frames, format="extxyz")
    return len(frames)


def write_event_log(result, path: Union[str, Path]) -> int:
    """Write a compact CSV event log of every BKL step.

    Columns:
        step                     index of the step (0-based)
        time_s                   cumulative simulated time AFTER this step
        delta_t_s                time advance of this step
        vacancy_idx_before       vacancy site index before the step
        vacancy_idx_after        vacancy site index after the step
        hopper_species_idx       element index of the atom that hopped
        hopper_symbol            element symbol of the atom that hopped
        chosen_jump_local        which of the 8 NN was chosen (0..7)

    Args:
        result: KMCResult.
        path: output filename (.csv).

    Returns:
        Number of rows written (= number of BKL steps).
    """
    state = result.final_state
    element_symbols = state.element_symbols

    with open(str(path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "time_s",
            "delta_t_s",
            "vacancy_idx_before",
            "vacancy_idx_after",
            "hopper_species_idx",
            "hopper_symbol",
            "chosen_jump_local",
        ])
        for k in range(result.n_steps):
            sp_idx = int(result.hopper_species[k])
            symbol = element_symbols[sp_idx] if 0 <= sp_idx < len(element_symbols) else "?"
            writer.writerow([
                k,
                f"{float(result.times_s[k]):.6e}",
                f"{float(result.delta_t_s[k]):.6e}",
                int(result.vacancy_indices[k]),
                int(result.vacancy_indices[k + 1]),
                sp_idx,
                symbol,
                int(result.chosen_jump_local[k]),
            ])

    return int(result.n_steps)
