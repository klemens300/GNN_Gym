"""
KMC-specific configuration for vacancy diffusion in BCC RHEAs.

This module is standalone — it does not import from the GNN training Config.
A KMCConfig instance fully specifies a single KMC run, including the GNN
predictor paths used in Phase 3+ and the temperature-sweep parameters used
by the Phase 5c entry point (main.py).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union


@dataclass
class KMCConfig:
    """Unified configuration for a single KMC run."""

    # --- Material system ---
    elements: List[str] = field(
        default_factory=lambda: ["Mo", "Nb", "Ta", "W"]
    )
    composition: Dict[str, float] = field(
        default_factory=lambda: {"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25}
    )

    # --- Lattice ---
    # BCC supercell with `2 * supercell_size^3` sites (cubic conventional cell).
    supercell_size: int = 8
    lattice_parameter_A: float = 3.22

    # --- Initial state ---
    initial_state_strategy: Literal[
        "random", "bicrystal", "slabs", "custom"
    ] = "random"
    bicrystal_axis: Literal["x", "y", "z"] = "x"
    bicrystal_elements: Tuple[str, str] = ("Ta", "W")
    # `slabs` is a generalisation of bicrystal: split the box along
    # `slab_axis` into len(slab_elements) equal-thickness layers and assign
    # each element to one layer in the listed order. The vacancy is placed
    # on the most central slab boundary (overrides vacancy_initial_position
    # for this strategy).
    slab_axis: Literal["x", "y", "z"] = "x"
    slab_elements: Optional[List[str]] = None
    vacancy_initial_position: Literal["center", "random", "explicit"] = "center"
    # Required when vacancy_initial_position == "explicit". Coordinates are
    # in Angstrom; the closest BCC site (with PBC) is chosen as the vacancy.
    # Currently honoured only by KMCState.from_random_composition; bicrystal
    # always places the vacancy at the interface plane, and custom passes
    # custom_vacancy_index directly.
    vacancy_initial_xyz: Optional[Tuple[float, float, float]] = None
    custom_symbols: Optional[List[str]] = None
    custom_vacancy_index: Optional[int] = None

    # --- Thermodynamics ---
    temperature_K: float = 1500.0
    # Phase-5 DiffusionLookup will override this with composition-dependent values.
    attempt_frequency_Hz: float = 1e13

    # --- Run control ---
    n_steps: Optional[int] = 1000
    t_max_sim_s: Optional[float] = None
    random_seed: int = 42

    # --- GNN predictor (used in Phase 3+) ---
    gnn_model_path: str = "/home/klemens/doctor/gnn_kmc/model/mo_nb_ta_w.pt"
    gnn_device: Literal["cuda", "cpu", "auto"] = "auto"
    # If True (default) the predictor uses the Phase-6 static-lattice graph
    # cache: ~5-8x faster but exercises a PyG batching path that has been
    # observed to trigger CUDA index-out-of-bounds asserts on larger
    # supercells (>= 8^3) due to non-offset line_graph_batch_mapping.
    # Set to False to fall back to the per-step Phase-3 rebuild, which is
    # slower but battle-tested. See project_state.md "Bekannter Bug".
    use_static_cache: bool = True
    # If > 0, the static-cache fast path splits the per-step forward pass
    # into chunks of this size instead of doing all 8 candidate jumps in
    # one batched call. Shorter individual GPU kernels reduce TDR pressure
    # on Blackwell (RTX 5090) at the cost of a small launch overhead.
    # Numerics are unchanged. Default None = no splitting (single 8-graph
    # batch as before).
    inference_subbatch_size: Optional[int] = None
    # If > 0, call torch.cuda.empty_cache() every k completed BKL steps to
    # keep the VRAM footprint flat across multi-hour runs (counters
    # gradual fragmentation). Costs ~50-100 ms per call so use sparingly
    # (e.g. 1000). Default None disables it. CPU runs ignore the field.
    empty_cuda_cache_every_n_steps: Optional[int] = None

    # --- DiffusionLookup (used in Phase 5b+) ---
    # JSON cache of pre-computed (or on-the-fly computed) diffusion data.
    diffusion_cache_path: str = (
        "/home/klemens/doctor/gnn_kmc/data/diffusion_cache.json"
    )
    # k-nearest-neighbour interpolation parameters for the lookup.
    lookup_n_neighbors: int = 3
    # Maximum L2 distance (in composition-vector space, fractions in [0,1])
    # at which a cache entry still counts as "close enough". If the closest
    # n_neighbors are farther than this, the lookup falls back to the oracle.
    lookup_max_distance: float = 0.10

    # --- Temperature sweep (used in Phase 5c, main.py) ---
    # If None or empty: a single ensemble at temperature_K. Otherwise: an
    # Arrhenius sweep across the listed temperatures.
    temperatures_K_sweep: Optional[List[float]] = None
    n_realizations_per_T: int = 5

    # --- Output (used in Phase 4+) ---
    snapshot_every_n_steps: int = 100
    output_dir: str = "kmc_output"
    # If True, main.py writes an ExtXYZ trajectory of the first realisation
    # at each temperature (vacancy rendered as pseudo-atom "X" for OVITO).
    # Requires snapshot_every_n_steps > 0.
    write_trajectory_extxyz: bool = False
    # If True, main.py writes a multi-page PDF summary per temperature
    # covering vacancy/tracer MSD, Warren-Cowley alpha_ij(t) trajectories
    # and final SRO matrix, plus the sim->real time conversion when a
    # diffusion cache is available. Default on.
    write_summary_pdf: bool = True
    # If > 0, the engine prints a status line every k BKL steps (step count,
    # simulated time, elapsed wall-clock, ETA, ms/step). Useful for long runs.
    progress_every_n_steps: Optional[int] = None
    # If > 0, atomically write a rolling .npz checkpoint every k steps so a
    # crash on a multi-hour run loses at most k steps of progress. Each
    # checkpoint can be reloaded as a full KMCResult via
    # KMC.checkpoint.load_run_checkpoint(...). Default None disables it.
    checkpoint_every_n_steps: Optional[int] = None

    def __post_init__(self):
        # Composition consistency checks
        comp_sum = sum(self.composition.values())
        if abs(comp_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Composition fractions must sum to 1.0, got {comp_sum}"
            )
        for el in self.composition:
            if el not in self.elements:
                raise ValueError(
                    f"Composition element '{el}' not in elements list "
                    f"{self.elements}"
                )
        # Run-control sanity
        if self.n_steps is None and self.t_max_sim_s is None:
            raise ValueError(
                "At least one of n_steps or t_max_sim_s must be specified"
            )
        # Vacancy-position consistency
        if self.vacancy_initial_position == "explicit":
            if self.vacancy_initial_xyz is None:
                raise ValueError(
                    "vacancy_initial_position='explicit' requires "
                    "vacancy_initial_xyz to be set (x, y, z) in Angstrom"
                )
            if len(self.vacancy_initial_xyz) != 3:
                raise ValueError(
                    "vacancy_initial_xyz must be a 3-tuple (x, y, z), "
                    f"got length {len(self.vacancy_initial_xyz)}"
                )
        # Slabs-strategy consistency
        if self.initial_state_strategy == "slabs":
            if self.slab_elements is None or len(self.slab_elements) < 2:
                raise ValueError(
                    "initial_state_strategy='slabs' requires slab_elements "
                    "to be a list with at least 2 entries"
                )
            for el in self.slab_elements:
                if el not in self.elements:
                    raise ValueError(
                        f"slab_elements entry '{el}' not in elements list "
                        f"{self.elements}"
                    )

    @property
    def n_sites(self) -> int:
        """Number of BCC sites (2 atoms per cubic conventional cell)."""
        return 2 * self.supercell_size ** 3

    # ------- JSON roundtrip (Phase 5c CLI / config-driven runs) -------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Persist the config as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "KMCConfig":
        # JSON load turns tuples into lists. Coerce typed-tuple fields back
        # to tuples so downstream code that pattern-matches them keeps
        # working uniformly.
        d = dict(d)
        if "bicrystal_elements" in d and d["bicrystal_elements"] is not None:
            d["bicrystal_elements"] = tuple(d["bicrystal_elements"])
        if (
            "vacancy_initial_xyz" in d
            and d["vacancy_initial_xyz"] is not None
        ):
            d["vacancy_initial_xyz"] = tuple(d["vacancy_initial_xyz"])
        return cls(**d)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "KMCConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
