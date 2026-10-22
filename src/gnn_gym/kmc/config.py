"""Settings of a KMC run and of a campaign of runs.

A campaign file (configs/kmc/*.yaml) has three sections:

    campaign   runs_dir and the two model checkpoints
    defaults   RunConfig fields shared by all points
    points     one entry per run; its fields override the defaults

Every point is one trajectory in its own directory
<runs_dir>/kmc_<system>_T<T>. The seed of a point is the seed of that
trajectory: it fixes the random start configuration and the random-number
stream of the BKL loop (gnn_gym.kmc.runner.seed_streams).
"""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..paths import load_yaml, resolve
from ..systems import get_system


@dataclass
class EscapeConfig:
    """Absorbing-Markov-chain escape layer (gnn_gym.kmc.escape)."""
    revisit_threshold: int = 3     # flag a basin when a configuration recurs
    window: int = 200              # more often than this within `window` steps
    internal_n_kT: float = 1.0     # internal jumps: barrier within this many
                                   # k_B T of the lowest barrier of the entry
    max_basin_size: int = 100      # configurations; a basin still open here
                                   # is diffusive and left to the BKL loop
    cooldown_steps: int = 25       # no detection for this many steps after
                                   # such an open basin


@dataclass
class RunConfig:
    system: str                    # entry of configs/systems.yaml
    T_K: float
    seed: int
    n_steps: int
    supercell_size: int
    lattice_parameter_A: float = None   # None: the value of the system
    attempt_frequency_Hz: float = 1e13
    escape: bool = True
    escape_params: EscapeConfig = field(default_factory=EscapeConfig)
    snapshot_every: int = 500      # store the configuration every k steps
                                   # (and after every escape)
    checkpoint_every: int = 2000
    progress_every: int = 5000

    def __post_init__(self):
        if isinstance(self.escape_params, dict):
            self.escape_params = EscapeConfig(**self.escape_params)
        self.T_K = float(self.T_K)
        self.attempt_frequency_Hz = float(self.attempt_frequency_Hz)
        if self.n_steps <= 0 or self.checkpoint_every <= 0:
            raise ValueError("n_steps and checkpoint_every must be positive")
        if self.lattice_parameter_A is None:
            self.lattice_parameter_A = self.material.lattice_parameter_A

    @property
    def material(self):
        return get_system(self.system)

    @property
    def name(self):
        return f"kmc_{self.system}_T{int(round(float(self.T_K)))}"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown run keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class Campaign:
    runs_dir: Path
    de_model: str
    bsym_model: str
    points: list

    def run_dir(self, point):
        return self.runs_dir / point.name


def load_campaign(path):
    data = load_yaml(path)
    unknown = set(data) - {"campaign", "defaults", "points"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    head = data["campaign"]
    defaults = data.get("defaults", {})
    points = [RunConfig.from_dict({**defaults, **p}) for p in data["points"]]
    names = [p.name for p in points]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: two points share a run directory")
    return Campaign(runs_dir=resolve(head["runs_dir"]),
                    de_model=str(resolve(head["de_model"])),
                    bsym_model=str(resolve(head["bsym_model"])),
                    points=points)
