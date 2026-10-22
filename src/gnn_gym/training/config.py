"""Training settings and the YAML run files that combine them with a model.

A run file has two sections:

    model:     ModelConfig fields (architecture, see gnn_gym.model.config)
    training:  TrainConfig fields (data, optimiser, schedule, output)
"""
from dataclasses import asdict, dataclass, fields

from ..model.config import ModelConfig
from ..paths import load_yaml


@dataclass
class TrainConfig:
    train_csv: str = "data/jumps/MoNbTaW_7cube_train.csv"
    referee_csv: str = "data/jumps/MoNbTaW_7cube_referee.csv"
    out_dir: str = "runs/models"
    run_name: str = ""

    batch_size: int = 32
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-6
    lr_factor: float = 0.5          # ReduceLROnPlateau on the validation MAE
    lr_patience: int = 12
    patience: int = 40              # early stopping on the validation MAE
    huber_delta_meV: float = 50.0   # 0 = mean squared error
    grad_clip: float = 5.0
    val_frac: float = 0.1
    seed: int = 0
    # Use only the first `limit` training rows (0 = all). The rows were
    # generated in random order, so a prefix is a random subset; the referee
    # set is never limited. This is how the learning curve is taken.
    limit: int = 0

    device: str = "cuda"
    # TensorFloat-32 matmuls: about 1e-3 relative precision on activations,
    # far below the label noise, and most of the training speed.
    tf32: bool = True
    transfer_samples: int = 8       # jumps for the size-transfer check
    lattice_parameter_A: float = 3.239121   # only for reporting distances

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown training keys: {sorted(unknown)}")
        return cls(**data)


def load_run(path):
    """(ModelConfig, TrainConfig) from a run file."""
    data = load_yaml(path)
    unknown = set(data) - {"model", "training"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    return (ModelConfig.from_dict(data.get("model", {})),
            TrainConfig.from_dict(data.get("training", {})))
