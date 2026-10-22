"""Architecture of the two jump models.

The site-energy model (target "de") and the barrier model (target "bsym") are
separate networks with separate checkpoints. The site-energy model carries
detailed balance and must stay strictly local; the barrier model carries the
accuracy and only has to be symmetric. Their receptive fields, depths and
read-outs are therefore chosen independently. One config class describes
both, `target` says which network a checkpoint holds.

The defaults are the models used in the paper.
"""
from dataclasses import asdict, dataclass, field, fields


@dataclass
class ModelConfig:
    target: str = "de"                       # "de" | "bsym"
    elements: list = field(default_factory=lambda: ["Mo", "Nb", "Ta", "W"])
    lattice: str = "bcc"
    size: int = 7                            # training cell, size^3 conv. cells
    shells: list = field(default_factory=lambda: [1, 2])   # graph edges

    # site-energy model: atom-centred, strictly local
    de_layers: int = 2
    de_hidden: int = 96
    de_site_hidden: list = field(default_factory=lambda: [96, 48])
    de_line_graph: bool = False

    # barrier model: bond-aware (line graph), read out around the jump
    bsym_layers: int = 2
    bsym_hidden: int = 128
    bsym_pool_hops: int = 2                  # read-out ball around the bond
    bsym_head_hidden: list = field(default_factory=lambda: [256, 128])
    bsym_line_graph: bool = True
    # True: the head runs on every site of the read-out ball and the outputs
    # are averaged. False: site features are averaged first and the head runs
    # once on the mean (the "pooled" read-out of the comparison in the paper).
    bsym_site_sum: bool = True

    def __post_init__(self):
        if self.target not in ("de", "bsym"):
            raise ValueError(f"target must be 'de' or 'bsym', not {self.target!r}")

    @property
    def n_species(self):
        """Elements plus the vacancy, which is a species of the model."""
        return len(self.elements) + 1

    @property
    def vac_index(self):
        return len(self.elements)

    @property
    def line_graph(self):
        return self.de_line_graph if self.target == "de" else self.bsym_line_graph

    @property
    def layers(self):
        return self.de_layers if self.target == "de" else self.bsym_layers

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**data)
