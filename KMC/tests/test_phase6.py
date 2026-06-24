"""
Phase-6 tests for GNNBarrierPredictor static-lattice graph cache.

Verifies that the fast path (use_static_cache=True) is a drop-in equivalent
of the legacy per-step rebuild (use_static_cache=False):

- The 8 forward barriers must agree to within float-noise on identical states
- Equivalence holds across supercell sizes (4x4x4, 6x6x6) and initial setups
- Cache rebuild on geometry change does not break correctness
- Cache survives multiple BKL steps (state mutation)
- Atomic-properties lookup matches GraphBuilder's per-atom dict pathway

These tests skip if torch / torch_geometric / the trained model are missing,
matching the convention in test_phase3.py.

Run from /home/klemens/doctor/gnn_kmc/scipts:

    pytest -v KMC/tests/test_phase6.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `scipts/` importable so that `from KMC.* import ...` works
_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.state import KMCState, VACANCY_SPECIES
from KMC.engine import bkl_step


# ---------------------------------------------------------------------------
# Reference: legacy graphs run through the encoder PER GRAPH (no batching).
# This avoids the PyG line_graph_edge_index offset bug (see
# barrier_predictor.py module docstring) and gives the cross-graph-
# contamination-free output. The fast path with the _BatchSafeData subclass
# should match this reference within float32 noise.
# ---------------------------------------------------------------------------

def _legacy_per_graph_reference(predictor, state, jump_atom_indices):
    """Compute barriers via the legacy graph builder + per-graph encoding."""
    torch = predictor._torch
    Batch = predictor._Batch

    current_atoms = state.to_atoms(include_vacancy=False)
    non_vac_mask = state.species != VACANCY_SPECIES
    non_vac_site_indices = np.where(non_vac_mask)[0]
    site_to_atom_idx = np.full(state.n_sites, -1, dtype=np.int64)
    site_to_atom_idx[non_vac_site_indices] = np.arange(
        len(non_vac_site_indices)
    )
    vacancy_pos = state.positions[state.vacancy_index].copy()

    current_graph = predictor.builder.atoms_to_graph(current_atoms)
    post_graphs = []
    for atom_site_idx in jump_atom_indices:
        atoms_idx = int(site_to_atom_idx[int(atom_site_idx)])
        post_atoms = current_atoms.copy()
        post_atoms.positions[atoms_idx] = vacancy_pos
        post_graphs.append(predictor.builder.atoms_to_graph(post_atoms))

    predictor.model.eval()
    current_batch = Batch.from_data_list([current_graph]).to(predictor.device)
    with torch.no_grad():
        emb_current = predictor.model.encoder(current_batch)

    barriers = np.empty(len(jump_atom_indices), dtype=np.float64)
    with torch.no_grad():
        for i, post_graph in enumerate(post_graphs):
            post_batch = Batch.from_data_list([post_graph]).to(predictor.device)
            emb_post = predictor.model.encoder(post_batch)
            pred_norm = predictor.model.predictor(emb_post, emb_current)
            barriers[i] = float(pred_norm.cpu().numpy().reshape(-1)[0])

    return barriers * predictor.target_std + predictor.target_mean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(supercell_size: int) -> KMCConfig:
    return KMCConfig(
        elements=["Mo", "Nb", "Ta", "W"],
        composition={"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25},
        supercell_size=supercell_size,
        lattice_parameter_A=3.22,
        random_seed=42,
        temperature_K=1500.0,
        attempt_frequency_Hz=1e13,
        n_steps=10,
    )


@pytest.fixture(scope="module")
def small_config():
    return _make_config(supercell_size=4)


def _gnn_available(config) -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except Exception:
        return False
    return Path(config.gnn_model_path).exists()


@pytest.fixture(scope="module")
def predictor(small_config):
    """Module-scoped: build a SINGLE predictor and toggle use_static_cache.

    Why a single instance instead of two: each GNNBarrierPredictor instance
    gets its own cuDNN kernel cache. With cudnn.benchmark=True (set by the
    GNN training package's set_seed), two separate instances can pick
    different algorithms on first call and diverge by up to a few % per
    barrier - which is fine for the predictor's purpose but breaks naive
    1:1 numerical equivalence asserts. By using ONE predictor and flipping
    the use_static_cache flag at the Python level, we keep the model +
    cuDNN state fixed across both calls so that any remaining difference
    actually comes from the input-graph code path.
    """
    if not _gnn_available(small_config):
        pytest.skip(
            "torch / torch_geometric / trained model not available; "
            "Phase-6 tests require a runnable GNN predictor."
        )
    from KMC.barrier_predictor import GNNBarrierPredictor

    return GNNBarrierPredictor(small_config, use_static_cache=True)


def _fast_vs_reference(predictor, state, jump_atom_indices):
    """Compute fast (subclass-batched) and the per-graph legacy reference."""
    predictor.use_static_cache = True
    b_fast = predictor.get_forward_barriers_batch(state, jump_atom_indices)
    b_ref = _legacy_per_graph_reference(predictor, state, jump_atom_indices)
    return b_fast, b_ref


# ---------------------------------------------------------------------------
# Numerical equivalence
# ---------------------------------------------------------------------------

# Tolerance: float32 sum-order noise in graph construction (RBF expansion,
# atomic-props lookup) plus cuDNN noise in the encoder forward pass. The
# fast path with the _BatchSafeData subclass produces the same
# contamination-free output as a per-graph legacy reference, so the residual
# difference is dominated by float-noise. atol=5e-4 = 0.5 meV is comfortably
# below any physically meaningful barrier scale.
_BARRIER_ATOL = 5e-4
_BARRIER_RTOL = 5e-4


def test_equivalence_random_4x4x4(small_config, predictor):
    """Fast (subclass-batched) matches the per-graph legacy reference on 4x4x4."""
    state = KMCState.from_random_composition(small_config)
    nn = state.get_neighbor_atom_indices()

    b_fast, b_ref = _fast_vs_reference(predictor, state, nn)

    np.testing.assert_allclose(
        b_fast, b_ref,
        atol=_BARRIER_ATOL, rtol=_BARRIER_RTOL,
        err_msg=f"Fast vs reference diverged: fast={b_fast}, ref={b_ref}",
    )


def test_equivalence_after_swap(small_config, predictor):
    """Equivalence holds after the vacancy moves (state mutation)."""
    state = KMCState.from_random_composition(small_config)

    # Move the vacancy a few times to exercise different environments
    rng = np.random.default_rng(7)
    for _ in range(3):
        nn = state.get_neighbor_atom_indices()
        state.swap_vacancy(int(rng.choice(nn)))

    nn = state.get_neighbor_atom_indices()
    b_fast, b_ref = _fast_vs_reference(predictor, state, nn)

    np.testing.assert_allclose(
        b_fast, b_ref, atol=_BARRIER_ATOL, rtol=_BARRIER_RTOL
    )


def test_equivalence_bicrystal(small_config, predictor):
    """Equivalence on a Ta|W bicrystal initial state."""
    bi_config = KMCConfig(
        elements=small_config.elements,
        composition=small_config.composition,
        supercell_size=4,
        lattice_parameter_A=3.22,
        initial_state_strategy="bicrystal",
        bicrystal_axis="x",
        bicrystal_elements=("Ta", "W"),
        random_seed=42,
        temperature_K=1500.0,
        n_steps=1,
    )
    state = KMCState.from_bicrystal(bi_config)
    nn = state.get_neighbor_atom_indices()

    b_fast, b_ref = _fast_vs_reference(predictor, state, nn)

    np.testing.assert_allclose(
        b_fast, b_ref, atol=_BARRIER_ATOL, rtol=_BARRIER_RTOL
    )


def test_equivalence_6x6x6():
    """Equivalence on the 6x6x6 supercell."""
    cfg = _make_config(supercell_size=6)
    if not _gnn_available(cfg):
        pytest.skip("GNN model not available")

    from KMC.barrier_predictor import GNNBarrierPredictor

    pred = GNNBarrierPredictor(cfg, use_static_cache=True)

    state = KMCState.from_random_composition(cfg)
    nn = state.get_neighbor_atom_indices()

    b_fast, b_ref = _fast_vs_reference(pred, state, nn)

    np.testing.assert_allclose(
        b_fast, b_ref, atol=_BARRIER_ATOL, rtol=_BARRIER_RTOL,
        err_msg=f"6x6x6 mismatch: fast={b_fast}, ref={b_ref}",
    )


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

def test_cache_built_lazily(small_config):
    """The static cache is None until the first call, then populated."""
    if not _gnn_available(small_config):
        pytest.skip("GNN model not available")
    from KMC.barrier_predictor import GNNBarrierPredictor

    pred = GNNBarrierPredictor(small_config, use_static_cache=True)
    assert pred._cache is None

    state = KMCState.from_random_composition(small_config)
    nn = state.get_neighbor_atom_indices()
    pred.get_forward_barriers_batch(state, nn)

    assert pred._cache is not None
    assert pred._cache.n_sites == state.n_sites
    # 4x4x4 BCC has 128 sites, each with 8 1-NN @ 2.79 A and 6 2-NN @ 3.22 A
    # within a 3.5 A cutoff: 14 neighbours per atom * 128 atoms = 1792 edges
    assert pred._cache.edge_src.shape[0] == 128 * 14


def test_cache_rebuild_on_geometry_change(small_config):
    """Switching to a different lattice geometry triggers a cache rebuild."""
    if not _gnn_available(small_config):
        pytest.skip("GNN model not available")
    from KMC.barrier_predictor import GNNBarrierPredictor

    pred = GNNBarrierPredictor(small_config, use_static_cache=True)

    state_4 = KMCState.from_random_composition(small_config)
    pred.get_forward_barriers_batch(
        state_4, state_4.get_neighbor_atom_indices()
    )
    cache_id_4 = id(pred._cache)
    n_sites_4 = pred._cache.n_sites

    # Different lattice -> different positions array id() -> rebuild
    cfg_6 = _make_config(supercell_size=6)
    state_6 = KMCState.from_random_composition(cfg_6)
    pred.get_forward_barriers_batch(
        state_6, state_6.get_neighbor_atom_indices()
    )

    assert id(pred._cache) != cache_id_4
    assert pred._cache.n_sites != n_sites_4
    assert pred._cache.n_sites == state_6.n_sites


def test_cache_survives_state_mutation(small_config, predictor):
    """The cache should stay alive across many BKL steps on the same lattice.

    The species array changes every step, but positions/cell don't, so the
    cache must NOT be rebuilt - that would defeat the whole optimisation.
    """
    state = KMCState.from_random_composition(small_config)
    rng = np.random.default_rng(0)

    # Trigger the first build
    predictor.get_forward_barriers_batch(
        state, state.get_neighbor_atom_indices()
    )
    cache_id_after_first = id(predictor._cache)

    for _ in range(5):
        nn = state.get_neighbor_atom_indices()
        state.swap_vacancy(int(rng.choice(nn)))
        predictor.get_forward_barriers_batch(state, state.get_neighbor_atom_indices())

    assert id(predictor._cache) == cache_id_after_first


# ---------------------------------------------------------------------------
# Drop-in compatibility (sanity that the engine still runs)
# ---------------------------------------------------------------------------

def test_bkl_step_runs_with_fast_predictor(small_config, predictor):
    """A BKL step using the fast predictor must complete normally."""
    state = KMCState.from_random_composition(small_config)
    rng = np.random.default_rng(0)
    initial_vac = int(state.vacancy_index)
    nn_before = set(state.get_neighbor_atom_indices().tolist())

    info = bkl_step(
        state, predictor,
        T_K=1500.0,
        attempt_frequency_Hz=1e13,
        rng=rng,
    )

    assert info["delta_t_sim_s"] > 0.0
    assert info["forward_barriers_eV"].shape == (8,)
    assert int(state.vacancy_index) != initial_vac
    assert int(state.vacancy_index) in nn_before
    assert int((state.species == VACANCY_SPECIES).sum()) == 1


# ---------------------------------------------------------------------------
# Atomic-properties table parity
# ---------------------------------------------------------------------------

def test_atomic_props_table_matches_dict_lookup(small_config, predictor):
    """The cached props table must match the per-atom dict pathway used in
    GraphBuilder.atoms_to_graph (column order matters).
    """
    from gnn.atomic_properties import get_atomic_properties

    table = predictor._atomic_props_table
    column_keys = [
        "atomic_number", "atomic_mass", "atomic_radius",
        "electronegativity", "first_ionization", "electron_affinity",
        "melting_point", "density",
    ]
    for el, idx in predictor.builder.element_to_idx.items():
        props = get_atomic_properties(el)
        for col, key in enumerate(column_keys):
            assert table[idx, col] == pytest.approx(float(props[key]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
