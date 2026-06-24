"""Import-smoke tests guarding the package refactor.

After moving the formerly flat scripts into the ``gnn``, ``KMC`` and
``diffusion`` packages, every intra-package import was rewritten from a bare
``from config import ...`` style to an absolute ``from gnn.config import ...``
style. A single missed line would leave a module unimportable. These tests
import every refactored module so that any leftover bare import surfaces
immediately as a clear ImportError instead of crashing a real run later.

Modules whose *module-level* imports pull in optional heavy dependencies
(FairChem / UMA, matscipy) are guarded with ``pytest.importorskip`` so the
suite still runs in the lightweight training/KMC environment.
"""

import importlib

import pytest


# Modules importable with the core stack (torch, torch_geometric, ase, scipy).
GNN_MODULES = [
    "gnn",
    "gnn.config",
    "gnn.atomic_properties",
    "gnn.model",
    "gnn.graph_builder",
    "gnn.utils",
    "gnn.dataset",
    "gnn.inference",
    "gnn.oracle",
    "gnn.fixed_test_set",
    "gnn.active_learning",
]

KMC_MODULES = [
    "KMC",
    "KMC.config",
    "KMC.state",
    "KMC.observables",
    "KMC.engine",
    "KMC.runner",
    "KMC.result",
    "KMC.analysis",
    "KMC.barrier_predictor",
]

DIFFUSION_MODULES = [
    "diffusion",
    "diffusion.config",
    "diffusion.results",
    "diffusion.diffusion_physics",
]

# These pull matscipy / FairChem at module import time.
DIFFUSION_HEAVY_MODULES = [
    "diffusion.diffusion_oracle",
    "diffusion.diffusion_calc",
]


@pytest.mark.parametrize("module_name", GNN_MODULES + KMC_MODULES + DIFFUSION_MODULES)
def test_module_imports(module_name):
    """Every refactored module imports without a leftover bare import."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", DIFFUSION_HEAVY_MODULES)
def test_heavy_module_imports(module_name):
    """Diffusion oracle modules import once matscipy is available."""
    pytest.importorskip("matscipy")
    importlib.import_module(module_name)
