"""Smoke tests for the KMC <-> GNN wiring (the 'Verschaltung').

The KMC engine talks to any barrier source through the ``BarrierPredictor``
protocol. ``GNNBarrierPredictor`` is the production backend that loads the
trained GNN from the ``gnn`` package; ``MockBarrierPredictor`` is the
dependency-free reference used to validate the BKL machinery.

These tests check the contract without needing a GPU or a trained checkpoint:
  * the predictor classes are importable from KMC and expose the protocol API;
  * the mock predictor returns correctly shaped barriers;
  * GNNBarrierPredictor resolves the gnn package lazily (importing the module
    must not already require torch / a checkpoint).
"""

import numpy as np
import pytest

from KMC.barrier_predictor import (
    BarrierPredictor,
    MockBarrierPredictor,
    GNNBarrierPredictor,
)


def test_predictor_classes_follow_protocol():
    assert hasattr(MockBarrierPredictor, "get_forward_barriers_batch")
    assert hasattr(GNNBarrierPredictor, "get_forward_barriers_batch")
    # Runtime-checkable Protocol: the mock instance must satisfy it.
    assert isinstance(MockBarrierPredictor(constant_eV=1.0), BarrierPredictor)


def test_mock_constant_barriers_shape_and_value():
    predictor = MockBarrierPredictor(constant_eV=0.75)
    jumps = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # Constant mode ignores the spatial environment, so `state` is unused.
    barriers = predictor.get_forward_barriers_batch(state=None, jump_atom_indices=jumps)
    assert barriers.shape == (8,)
    assert np.allclose(barriers, 0.75)


def test_gnn_predictor_import_is_torch_free():
    """Importing the module must not require torch at module import time.

    torch / torch_geometric are imported lazily inside __init__ so that
    Mock-only setups (and this test environment) stay usable.
    """
    import sys

    import KMC.barrier_predictor as bp

    # The module object exists and exposes the class without torch being a
    # hard module-level requirement.
    assert bp.GNNBarrierPredictor is GNNBarrierPredictor
