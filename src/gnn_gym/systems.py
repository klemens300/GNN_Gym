"""Material systems: elements, composition, lattice and lattice parameter.

Defined in configs/systems.yaml. The jump models know only the element set
they were trained on; a simulation of a subsystem (Ta-W, or a pure element)
uses the same models with a subset of the elements.
"""
from dataclasses import dataclass, field

from .lattice import get_lattice
from .paths import load_yaml

SYSTEMS_FILE = "configs/systems.yaml"


@dataclass(frozen=True)
class MaterialSystem:
    name: str
    elements: tuple
    lattice: str = "bcc"
    lattice_parameter_A: float = 0.0
    composition: dict = field(default_factory=dict)   # element -> fraction

    def __post_init__(self):
        get_lattice(self.lattice)
        if not self.composition:
            object.__setattr__(self, "composition",
                               {e: 1.0 / len(self.elements) for e in self.elements})
        if set(self.composition) != set(self.elements):
            raise ValueError(f"{self.name}: composition and elements differ")
        if abs(sum(self.composition.values()) - 1.0) > 1e-9:
            raise ValueError(f"{self.name}: fractions do not sum to 1")

    @property
    def lattice_obj(self):
        return get_lattice(self.lattice)


def load_systems(path=SYSTEMS_FILE):
    data = load_yaml(path)
    return {name: MaterialSystem(name=name, elements=tuple(d["elements"]),
                                 lattice=d.get("lattice", "bcc"),
                                 lattice_parameter_A=float(d["lattice_parameter_A"]),
                                 composition=dict(d.get("composition") or {}))
            for name, d in data.items()}


def get_system(name, path=SYSTEMS_FILE):
    systems = load_systems(path)
    try:
        return systems[name]
    except KeyError:
        raise ValueError(f"unknown system {name!r}; defined: "
                         f"{sorted(systems)}") from None
