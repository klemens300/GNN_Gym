"""
Barrier predictor protocol and implementations.

The KMC engine queries this layer for the eight forward barriers around the
current vacancy. The engine doesn't know whether the values come from a mock,
the GNN, or a future hybrid backend - it just gets eight floats in eV.

Naming note: 'Predictor' is used rather than 'Oracle' because 'Oracle' is
already reserved in this project for the NEB ground-truth source
(scipts/GNN/oracle.py and scipts/diffusion_coefficient/diffusion_oracle.py).

Performance note (Phase 6): GNNBarrierPredictor uses a static lattice graph
cache. The BCC lattice geometry never changes during a KMC run, so the full
N-site cutoff graph (edges, RBF features, line graph topology, edge angles)
is computed exactly once on first call. Each subsequent BKL step derives the
9 (N-1)-site subgraphs (1 current + 8 post-jump) by edge masking + node
re-indexing, which is O(E) numpy work. This eliminates the per-step ASE
NeighborList Python loop that dominated the runtime on supercells > 4x4x4.
The legacy per-step rebuild remains accessible via use_static_cache=False
for benchmarking and equivalence testing.

Batching note (Phase 6 bug fix): PyG's default Data.__inc__ increments any
attribute name containing 'index' by num_nodes when graphs are concatenated
via Batch.from_data_list. For our line_graph_edge_index this is wrong - line
graph nodes correspond to atom-graph EDGES (not atoms), so the correct
increment is num_edges. The legacy GraphBuilder pathway uses vanilla Data
and inherits this bug, which causes cross-graph contamination of line-graph
message passing during batched inference. The fast path here uses a custom
_BatchSafeData subclass with a corrected __inc__, so its outputs differ from
the legacy batched outputs by ~30-50 meV per barrier. Since the trained
model was fitted under the buggy regime, this is a real change in behaviour
that needs to be re-validated against FairChem-NEB. See project_state.md
for full discussion and action items.

Follow-up fix (8^3 CUDA OOB crash, 2026-05-05): the same _BatchSafeData also
needs a custom rule for line_graph_batch_mapping. That tensor maps each
line-graph node (= atom-graph bond) to the atom-graph node id of its source
atom, so when graphs are concatenated those values must shift by num_nodes
per subgraph. PyG's default __inc__ does NOT do this for us: the name does
not end in 'index' (so the num_nodes branch does not match) and although
the name contains 'batch', PyG's batch-branch returns int(value.max())+1
which is silently almost-but-not-quite num_nodes and produces out-of-bounds
atom-graph indices once subgraphs are stitched. On 4^3 / 6^3 these bad
indices happened to land in valid memory; on 8^3 (1024 sites) they overrun
and trigger an asynchronous CUDA index-OOB assert (surfaced by the GPU as a
failure inside global_mean_pool). The explicit return self.num_nodes below
fixes both. Same caveat as for line_graph_edge_index: the fix changes the
numerical output relative to the buggy training regime, so the FairChem-NEB
re-validation should be redone before quoting MAE in the paper.
"""

from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BarrierPredictor(Protocol):
    """Interface every barrier source must implement."""

    def get_forward_barriers_batch(
        self, state, jump_atom_indices: np.ndarray
    ) -> np.ndarray:
        """Forward barriers for the candidate jumps from the current vacancy.

        Args:
            state: KMCState (current configuration; vacancy at
                state.vacancy_index).
            jump_atom_indices: site indices of atoms that may hop into the
                vacancy. For BCC vacancy KMC these are the 8 NN of
                state.vacancy_index, typically obtained via
                state.get_neighbor_atom_indices().

        Returns:
            np.ndarray of shape (len(jump_atom_indices),) with forward
            barriers in eV.
        """
        ...


class MockBarrierPredictor:
    """Constant or element-dependent barriers; ignores the spatial environment.

    Used in Phase 1-2 to verify the BKL machinery against analytical
    expectations (linear MSD, correct D from constant-barrier theory).
    """

    def __init__(
        self,
        constant_eV: float = 1.0,
        element_barriers_eV: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            constant_eV: barrier returned for all jumps (default mode).
            element_barriers_eV: if provided, returns the barrier looked up by
                the species of the hopping atom; constant_eV is ignored.
        """
        self.constant_eV = float(constant_eV)
        self.element_barriers_eV = element_barriers_eV

    def get_forward_barriers_batch(
        self, state, jump_atom_indices: np.ndarray
    ) -> np.ndarray:
        if self.element_barriers_eV is None:
            return np.full(
                len(jump_atom_indices), self.constant_eV, dtype=np.float64
            )

        out = np.empty(len(jump_atom_indices), dtype=np.float64)
        for k, atom_idx in enumerate(jump_atom_indices):
            sp = int(state.species[int(atom_idx)])
            symbol = state.element_symbols[sp]
            if symbol not in self.element_barriers_eV:
                raise KeyError(
                    f"No barrier defined for element '{symbol}' in "
                    f"element_barriers_eV"
                )
            out[k] = self.element_barriers_eV[symbol]
        return out


# ---------------------------------------------------------------------------
# Static lattice graph cache (Phase 6 fast path)
# ---------------------------------------------------------------------------

# Order of atomic-properties columns must match GraphBuilder.atoms_to_graph,
# otherwise the trained model receives shifted features and predictions break.
_ATOMIC_PROPS_COLUMN_ORDER = (
    "atomic_number",
    "atomic_mass",
    "atomic_radius",
    "electronegativity",
    "first_ionization",
    "electron_affinity",
    "melting_point",
    "density",
)


def _rbf_expand(distances: np.ndarray, n_gaussians: int, cutoff: float) -> np.ndarray:
    """RBF expansion identical to graph_builder.rbf_expansion (numpy form)."""
    centers = np.linspace(0.0, cutoff, n_gaussians)
    width = centers[1] - centers[0] if n_gaussians > 1 else cutoff
    diff = distances[:, np.newaxis] - centers[np.newaxis, :]
    return np.exp(-(diff ** 2) / (width ** 2))


def _apply_max_neighbors(
    src: np.ndarray, dist: np.ndarray, max_neighbors: int
) -> np.ndarray:
    """Keep at most max_neighbors closest edges per source atom.

    Returns a boolean keep mask. Mirrors the per-source argsort + slice that
    GraphBuilder._compute_edges_from_positions does.
    """
    n_edges = len(src)
    keep = np.ones(n_edges, dtype=bool)
    if n_edges == 0:
        return keep
    # Sort primary by source, secondary by distance
    order = np.lexsort((dist, src))
    sorted_src = src[order]
    # Boundary indices where source changes
    change = np.where(np.diff(sorted_src) != 0)[0] + 1
    boundaries = np.concatenate(([0], change, [n_edges]))
    drop = []
    for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
        if b_end - b_start > max_neighbors:
            drop.extend(order[b_start + max_neighbors:b_end].tolist())
    if drop:
        keep[np.asarray(drop, dtype=np.int64)] = False
    return keep


def _build_full_line_graph(
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_vectors: np.ndarray,
):
    """Build line graph (bond-bond angles) on the full N-site graph in numpy.

    Same logic as GraphBuilder._build_line_graph_from_vectors:
    - Line graph nodes = atom-graph edges.
    - For each source atom, all pairs of incident edges become a pair of
      line-graph edges (both directions). Edge attribute = angle between
      bond vectors.

    Returns
    -------
    line_src, line_dst : np.int64 [n_line_edges]
    line_angles        : np.float32 [n_line_edges]
    """
    n_edges = len(edge_src)
    if n_edges == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    # Avoid division by zero on degenerate edges (shouldn't happen in BCC)
    safe_len = np.where(edge_lengths > 1e-8, edge_lengths, 1.0)
    edge_vec_norm = edge_vectors / safe_len[:, None]

    # Group edges by source atom. Sort once; iterate over runs.
    order = np.argsort(edge_src, kind="stable")
    sorted_src = edge_src[order]
    change = np.where(np.diff(sorted_src) != 0)[0] + 1
    boundaries = np.concatenate(([0], change, [n_edges]))

    line_src_chunks = []
    line_dst_chunks = []
    line_angle_chunks = []

    for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
        n_local = b_end - b_start
        if n_local < 2:
            continue
        edge_idx = order[b_start:b_end]  # original-edge indices for this source
        vecs = edge_vec_norm[edge_idx]
        cos_angles = vecs @ vecs.T
        np.clip(cos_angles, -1.0, 1.0, out=cos_angles)
        angles = np.arccos(cos_angles)

        i_idx, j_idx = np.triu_indices(n_local, k=1)
        e_i = edge_idx[i_idx]
        e_j = edge_idx[j_idx]
        a_vals = angles[i_idx, j_idx]

        # Both directions (matches the original implementation)
        line_src_chunks.append(np.concatenate([e_i, e_j]))
        line_dst_chunks.append(np.concatenate([e_j, e_i]))
        line_angle_chunks.append(np.concatenate([a_vals, a_vals]))

    if line_src_chunks:
        line_src = np.concatenate(line_src_chunks).astype(np.int64)
        line_dst = np.concatenate(line_dst_chunks).astype(np.int64)
        line_angles = np.concatenate(line_angle_chunks).astype(np.float32)
    else:
        line_src = np.empty(0, dtype=np.int64)
        line_dst = np.empty(0, dtype=np.int64)
        line_angles = np.empty(0, dtype=np.float32)
    return line_src, line_dst, line_angles


class _StaticLatticeGraphCache:
    """Cache the full N-site cutoff graph for a fixed lattice geometry.

    The KMC lattice (positions, cell, cutoff) never changes during a run, so
    everything that does not depend on per-step species can be computed once:

    - edge connectivity (using ase.neighborlist.neighbor_list, vectorised)
    - per-edge distances and edge vectors
    - per-edge RBF features
    - line graph topology and per-line-edge bond angles
    - line graph node features (normalized direction + length)

    Per-step the predictor derives the 9 (N-1)-site subgraphs by edge
    masking on these arrays, which is O(E) numpy work.

    Atomic properties for the configured element list are also cached as a
    [n_elements, 8] lookup table so that node features are a single index
    operation rather than a per-atom Python dict lookup.
    """

    def __init__(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        cutoff_radius: float,
        max_neighbors: int,
        rbf_num_gaussians: int,
        rbf_cutoff: float,
        atomic_props_table: np.ndarray,
        use_line_graph: bool,
    ):
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        self.n_sites = int(positions.shape[0])
        self.use_line_graph = bool(use_line_graph)
        self.atomic_props_table = atomic_props_table.astype(np.float32)

        # Build the N-site cutoff graph in a single vectorised call.
        # Element identity is irrelevant here - the cutoff graph depends only
        # on positions, cell and pbc - so we use a placeholder species.
        atoms = Atoms(
            symbols=["H"] * self.n_sites,
            positions=np.asarray(positions, dtype=np.float64),
            cell=np.asarray(cell, dtype=np.float64),
            pbc=True,
        )
        i_arr, j_arr, vec_arr = neighbor_list(
            "ijD", atoms, cutoff=float(cutoff_radius)
        )
        i_arr = np.asarray(i_arr, dtype=np.int64)
        j_arr = np.asarray(j_arr, dtype=np.int64)
        vec_arr = np.asarray(vec_arr, dtype=np.float32)

        # Strict-cutoff filter (parity with GraphBuilder which uses dists < cutoff)
        dist_arr = np.linalg.norm(vec_arr, axis=1).astype(np.float32)
        strict = dist_arr < float(cutoff_radius)
        i_arr = i_arr[strict]
        j_arr = j_arr[strict]
        vec_arr = vec_arr[strict]
        dist_arr = dist_arr[strict]

        # Optional max-neighbours cap (matches the original GraphBuilder)
        if max_neighbors is not None and max_neighbors > 0:
            keep = _apply_max_neighbors(i_arr, dist_arr, int(max_neighbors))
            i_arr = i_arr[keep]
            j_arr = j_arr[keep]
            vec_arr = vec_arr[keep]
            dist_arr = dist_arr[keep]

        # Sort edges by (source, dest) for a deterministic ordering. The model
        # is permutation-invariant on edges, but a deterministic order makes
        # the line-graph remap below cleaner.
        order = np.lexsort((j_arr, i_arr))
        i_arr = i_arr[order]
        j_arr = j_arr[order]
        vec_arr = vec_arr[order]
        dist_arr = dist_arr[order]

        # Per-edge RBF features
        rbf_features = _rbf_expand(
            dist_arr, int(rbf_num_gaussians), float(rbf_cutoff)
        ).astype(np.float32)

        # Line-graph node features: normalised bond direction + length
        edge_lengths_col = dist_arr.reshape(-1, 1)
        safe_len = np.where(edge_lengths_col > 1e-8, edge_lengths_col, 1.0)
        edge_vec_norm = vec_arr / safe_len
        line_node_features = np.concatenate(
            [edge_vec_norm, edge_lengths_col], axis=1
        ).astype(np.float32)

        # Line graph topology + per-line-edge angles
        if self.use_line_graph:
            line_src, line_dst, line_angles = _build_full_line_graph(
                i_arr, j_arr, vec_arr
            )
        else:
            line_src = np.empty(0, dtype=np.int64)
            line_dst = np.empty(0, dtype=np.int64)
            line_angles = np.empty(0, dtype=np.float32)

        # Store everything; numpy arrays stay on CPU, conversion to torch is
        # done per call inside derive_subgraph (cheap; small for a single
        # subgraph and avoids holding GPU memory permanently).
        self.edge_src = i_arr
        self.edge_dst = j_arr
        self.edge_vectors = vec_arr
        self.edge_distances = dist_arr
        self.rbf_features = rbf_features
        self.line_node_features = line_node_features
        self.line_edge_src = line_src
        self.line_edge_dst = line_dst
        self.line_edge_angles = line_angles


# Lazy-cached BatchSafeData subclass. Defined on first use because importing
# torch_geometric at module level breaks Mock-only setups.
_BATCH_SAFE_DATA_CLS = None


def _get_batch_safe_data_class():
    """Return (and cache) a Data subclass with corrected __inc__.

    The default torch_geometric.data.Data.__inc__ has two failure modes for
    our line-graph attributes:

    1. line_graph_edge_index: PyG returns num_nodes for any 'index' key, but
       line-graph nodes correspond to atom-graph EDGES, not atoms - the
       correct increment is the number of edges.
    2. line_graph_batch_mapping: maps each line-graph node (= bond) to the
       atom-graph node id of its source atom. PyG's default returns
       int(value.max())+1 because the key contains 'batch'; that is
       silently close to num_nodes but produces out-of-bounds atom-graph
       indices when subgraphs are concatenated, manifesting as a CUDA
       index-OOB assert on large supercells (e.g. 8^3).

    Without these overrides, Batch.from_data_list either silently produces
    cross-graph contamination of line-graph message passing (bug 1) or
    crashes the GPU at the global_mean_pool gather step (bug 2).
    """
    global _BATCH_SAFE_DATA_CLS
    if _BATCH_SAFE_DATA_CLS is None:
        from torch_geometric.data import Data

        class _BatchSafeData(Data):
            def __inc__(self, key, value, *args, **kwargs):
                if key == "line_graph_edge_index":
                    # Line-graph node ids = atom-graph edge ids, so offset
                    # by num_edges of the current subgraph.
                    if self.edge_index is None:
                        return 0
                    return int(self.edge_index.shape[1])
                if key == "line_graph_batch_mapping":
                    # Maps each line-graph node to the atom-graph node id
                    # of its source atom, so offset by num_nodes (= atoms)
                    # of the current subgraph. This is the fix for the
                    # 8^3 CUDA index-OOB crash; see module docstring.
                    return self.num_nodes
                return super().__inc__(key, value, *args, **kwargs)

        _BATCH_SAFE_DATA_CLS = _BatchSafeData
    return _BATCH_SAFE_DATA_CLS


def _derive_subgraph_data(
    cache: _StaticLatticeGraphCache,
    excluded_site: int,
    species_for_subgraph: np.ndarray,
    torch_module,
):
    """Derive a PyG Data object for a (N-1)-site subgraph from the cache.

    Parameters
    ----------
    cache
        Pre-built _StaticLatticeGraphCache.
    excluded_site
        The lattice site that is empty in this subgraph (the vacancy site
        for the current state, or the hopping atom's origin site for a
        post-jump state).
    species_for_subgraph
        Length-N integer array giving the element index at every lattice
        site in this subgraph (the value at index ``excluded_site`` is
        ignored because that site is masked out).

    Returns
    -------
    _BatchSafeData
        Graph for this subgraph in CPU memory; transfer to the GPU happens
        once per BKL step via Batch.from_data_list(...).to(device). The
        custom subclass carries a corrected __inc__ so that line graph edge
        indices are offset by num_edges (not num_nodes) during batching.
    """
    DataCls = _get_batch_safe_data_class()

    torch = torch_module
    e = int(excluded_site)
    n_sites = cache.n_sites

    # Edge mask: drop every edge incident to the excluded site
    keep_edge = (cache.edge_src != e) & (cache.edge_dst != e)

    new_src = cache.edge_src[keep_edge]
    new_dst = cache.edge_dst[keep_edge]
    # Re-index: every node id > e shifts down by one because site e is gone.
    new_src = new_src - (new_src > e).astype(new_src.dtype)
    new_dst = new_dst - (new_dst > e).astype(new_dst.dtype)
    edge_index_t = torch.from_numpy(
        np.stack([new_src, new_dst], axis=0).astype(np.int64)
    )
    edge_attr_t = torch.from_numpy(cache.rbf_features[keep_edge].copy())

    # Node features: drop site e, then look up element/properties
    keep_node = np.ones(n_sites, dtype=bool)
    keep_node[e] = False
    species_sub = species_for_subgraph[keep_node].astype(np.int64)
    x_element_t = torch.from_numpy(species_sub)
    x_props_t = torch.from_numpy(cache.atomic_props_table[species_sub].copy())

    n_atoms_sub = int(species_sub.shape[0])

    data_kwargs = dict(
        x_element=x_element_t,
        x_props=x_props_t,
        edge_index=edge_index_t,
        edge_attr=edge_attr_t,
        num_nodes=n_atoms_sub,
        relax_progress=torch.tensor([[1.0]], dtype=torch.float32),
    )

    if cache.use_line_graph:
        # Line-graph nodes correspond 1-1 with atom-graph edges, so the same
        # edge mask gives us the kept line-graph nodes.
        line_node_features_sub = cache.line_node_features[keep_edge].copy()

        # Line-graph edges connect bonds that share a source atom. Keep only
        # the line edges where both endpoints (= bond ids in the FULL graph)
        # survived the edge mask.
        keep_line = keep_edge[cache.line_edge_src] & keep_edge[cache.line_edge_dst]

        # Re-index line-graph node ids: cumsum over keep_edge gives, for each
        # original edge id that survived, its new (sub-graph) edge id.
        cum = np.cumsum(keep_edge, dtype=np.int64) - 1
        new_line_src = cum[cache.line_edge_src[keep_line]]
        new_line_dst = cum[cache.line_edge_dst[keep_line]]
        line_edge_index_t = torch.from_numpy(
            np.stack([new_line_src, new_line_dst], axis=0).astype(np.int64)
        )
        line_edge_attr_t = torch.from_numpy(
            cache.line_edge_angles[keep_line].reshape(-1, 1).copy()
        )

        data_kwargs["line_graph_x"] = torch.from_numpy(line_node_features_sub)
        data_kwargs["line_graph_edge_index"] = line_edge_index_t
        data_kwargs["line_graph_edge_attr"] = line_edge_attr_t
        # batch_mapping = source atom of each bond in re-indexed node space
        data_kwargs["line_graph_batch_mapping"] = edge_index_t[0].clone()

    return DataCls(**data_kwargs)


# ---------------------------------------------------------------------------
# GNN-backed predictor
# ---------------------------------------------------------------------------


class GNNBarrierPredictor:
    """GNN-backed predictor: drop-in replacement for MockBarrierPredictor.

    Loads the trained model + GraphBuilder once at construction. On the first
    call the lattice geometry is captured into a _StaticLatticeGraphCache;
    subsequent calls derive the 9 per-step subgraphs from that cache without
    rebuilding the cutoff graph.

    Aufruf-Konvention (see project_state.md section 2):
        The model was trained as model(initial=pre_jump, final=post_jump) ->
        backward_barrier = E_TS - E_post_jump. To get the forward barrier of
        the current state A jumping into a candidate state B (which is what
        the BKL rate needs), we call model(initial=B, final=A): the model
        internally treats B as the pre-jump and A as the post-jump and so
        outputs E_TS - E_A = forward barrier of A->B. The known limitation
        (the model never saw this swapped input order during training) is
        deliberately accepted and documented in the paper's Discussion.
    """

    def __init__(
        self,
        config,
        use_static_cache: bool = True,
        inference_subbatch_size: Optional[int] = None,
    ):
        """Load the GNN model, GraphBuilder, and normalisation stats.

        Args:
            config: KMCConfig with gnn_model_path and gnn_device.
            use_static_cache: if True (default), use the Phase-6 fast path;
                if False, fall back to the Phase-3 per-step rebuild (used
                for benchmarking and equivalence testing).
            inference_subbatch_size: if not None and > 0, the fast path
                splits the per-step forward pass into chunks of this size
                instead of doing all 8 candidate jumps in one batched call.
                Used to keep individual GPU kernel durations short enough
                to stay below the Windows TDR threshold on RTX 5090 /
                Blackwell. Numerics are unchanged. Ignored on the legacy
                path. Default None = single 8-graph batch (legacy
                behaviour).
        """
        # Lazy imports so that importing barrier_predictor.py does not
        # require torch / torch_geometric (the Mock variants stay usable).
        # The GNN training package is reused for its GraphBuilder, model
        # loader, and Config dataclass. It is a sibling package installed
        # alongside KMC (pip install -e .), so a plain absolute import works.
        import torch
        from torch_geometric.data import Batch
        from gnn.config import Config as GNNConfig
        from gnn.graph_builder import GraphBuilder
        from gnn.utils import load_model_for_inference

        self._torch = torch
        self._Batch = Batch

        # GNN training Config (only graph-construction parameters are read,
        # the auto-generated paths are not accessed).
        gnn_config = GNNConfig()
        self.builder = GraphBuilder(
            gnn_config, csv_path=None, profile=False, use_cache=False
        )

        self.model, checkpoint = load_model_for_inference(
            config.gnn_model_path, gnn_config
        )
        self.model.eval()
        self.target_mean = float(checkpoint.get("target_mean", 0.0))
        self.target_std = float(checkpoint.get("target_std", 1.0))

        if config.gnn_device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(config.gnn_device)
        # Warm up the CUDA context before moving model parameters. On some
        # vGPU profiles (e.g. NVIDIA GRID A100 / driver 525) the very first
        # GPU touch must be a CPU->GPU copy of a trivial tensor. Direct GPU
        # allocations (torch.zeros(..., device='cuda')) and large model
        # transfers as the first CUDA op both crash with
        #   RuntimeError: CUDA driver error: operation not supported
        # The .cuda() form below works because it allocates on CPU first
        # and then issues a tiny memcpy, which the vGPU profile accepts.
        # On bare-metal GPUs this is a no-op of ~1 ms.
        if self.device.type == "cuda":
            _ = torch.zeros(1).cuda()
            torch.cuda.synchronize()
        self.model = self.model.to(self.device)

        self.model_path = config.gnn_model_path

        # Phase-6 cache plumbing
        self.use_static_cache = bool(use_static_cache)
        self.inference_subbatch_size = (
            int(inference_subbatch_size)
            if inference_subbatch_size is not None and inference_subbatch_size > 0
            else None
        )
        self._cache: Optional[_StaticLatticeGraphCache] = None
        self._cache_geometry_id = None  # id() of the positions array used
        self._atomic_props_table = self._build_atomic_props_table()

    # ------- Atomic properties lookup table -------

    def _build_atomic_props_table(self) -> np.ndarray:
        """Pre-compute [n_elements, 8] atomic-properties table for fast lookup.

        Column order MUST match graph_builder.atoms_to_graph(); otherwise the
        trained model receives shifted features. The order is asserted in
        _ATOMIC_PROPS_COLUMN_ORDER above.
        """
        from gnn.atomic_properties import get_atomic_properties

        # The GraphBuilder's element_to_idx is the source of truth for the
        # element ordering used at training time, so we mirror it here.
        elements = list(self.builder.elements)
        n_elements = len(elements)
        table = np.zeros((n_elements, len(_ATOMIC_PROPS_COLUMN_ORDER)),
                         dtype=np.float32)
        for el in elements:
            idx = self.builder.element_to_idx[el]
            props = get_atomic_properties(el)
            for col, key in enumerate(_ATOMIC_PROPS_COLUMN_ORDER):
                table[idx, col] = float(props[key])
        return table

    # ------- Cache management -------

    def _ensure_cache(self, state) -> _StaticLatticeGraphCache:
        """Build (or rebuild) the static graph cache for this state's lattice.

        The cache key is id(state.positions). The geometry cache in
        KMC.state shares the read-only positions array across all states
        with the same (supercell_size, lattice_parameter), so this id is
        stable across states for a single run.

        Cache construction can take seconds-to-tens-of-seconds on larger
        supercells (8^3 ~ 1024 sites). To avoid the impression of a hang,
        progress is logged to stdout when the cache is built (or rebuilt).
        """
        import sys
        import time as _time

        geo_id = id(state.positions)
        if self._cache is None or self._cache_geometry_id != geo_id:
            n_sites = int(np.asarray(state.positions).shape[0])
            print(
                f"  [predictor] Building static lattice graph cache for "
                f"{n_sites} sites (cutoff = "
                f"{float(self.builder.cutoff_radius):.2f} A)...",
                flush=True,
            )
            sys.stdout.flush()
            t0 = _time.time()
            self._cache = _StaticLatticeGraphCache(
                positions=np.asarray(state.positions),
                cell=np.asarray(state.cell),
                cutoff_radius=self.builder.cutoff_radius,
                max_neighbors=self.builder.max_neighbors,
                rbf_num_gaussians=self.builder.rbf_num_gaussians,
                rbf_cutoff=self.builder.rbf_cutoff,
                atomic_props_table=self._atomic_props_table,
                use_line_graph=self.builder.use_line_graph,
            )
            self._cache_geometry_id = geo_id
            n_edges = int(self._cache.edge_src.shape[0])
            n_line_edges = int(self._cache.line_edge_src.shape[0])
            elapsed = _time.time() - t0
            print(
                f"  [predictor] Cache built in {elapsed:.1f} s "
                f"({n_edges} atom-graph edges, {n_line_edges} line-graph "
                "edges).",
                flush=True,
            )
            sys.stdout.flush()
        return self._cache

    # ------- Public entry point -------

    def get_forward_barriers_batch(
        self, state, jump_atom_indices: np.ndarray
    ) -> np.ndarray:
        """Forward barriers for the candidate jumps via a single GNN forward pass."""
        if self.use_static_cache:
            return self._fast_get_forward_barriers_batch(
                state, jump_atom_indices
            )
        return self._legacy_get_forward_barriers_batch(
            state, jump_atom_indices
        )

    # ------- Phase-6 fast path -------

    def _fast_get_forward_barriers_batch(
        self, state, jump_atom_indices: np.ndarray
    ) -> np.ndarray:
        torch = self._torch
        Batch = self._Batch

        cache = self._ensure_cache(state)

        v = int(state.vacancy_index)
        species_full = np.asarray(state.species, dtype=np.int64)
        n_jumps = len(jump_atom_indices)

        # Subgraph 0: current state. Excluded site = v. Species at v is
        # VACANCY_SPECIES but we mask that site out anyway, so the value
        # there does not enter the lookup.
        # Subgraphs 1..n_jumps: post-jump states. Excluded site = k. The
        # hopping atom (originally at k) now sits at v, so we set
        # species[v] = species[k] before masking out site k.
        current_species = species_full.copy()
        current_data = _derive_subgraph_data(
            cache, excluded_site=v,
            species_for_subgraph=current_species,
            torch_module=torch,
        )

        post_data_list = []
        for k in jump_atom_indices:
            k_int = int(k)
            post_species = species_full.copy()
            post_species[v] = species_full[k_int]  # hopper now occupies v
            post_data_list.append(
                _derive_subgraph_data(
                    cache, excluded_site=k_int,
                    species_for_subgraph=post_species,
                    torch_module=torch,
                )
            )

        # Forward pass.
        # Convention: model(initial=post_jump, final=current) -> forward barrier.
        # Optional sub-batching: split the n_jumps forward passes into
        # smaller chunks so that each individual GPU kernel runs shorter,
        # which avoids Windows TDR triggers on RTX 5090 / Blackwell. The
        # numerical result is identical to a single big batch.
        subbatch = (
            self.inference_subbatch_size
            if self.inference_subbatch_size is not None
            else n_jumps
        )
        # Targeted stream sync between H2D copy and forward dispatch.
        # On RTX 5090 / Driver 591.86 / CUDA 13.1 (and likely related
        # Blackwell stacks), the model occasionally launches with
        # partially-staged inputs, which surfaces as a CUDA index-OOB
        # assert in vectorized_gather_kernel (asynchronously reported via
        # global_mean_pool). A global CUDA_LAUNCH_BLOCKING=1 also masks
        # the bug but adds blocking on every single kernel; this targeted
        # synchronize() blocks only after the full per-subgraph H2D
        # transfer is staged, just before the forward pass.
        _is_cuda = self.device.type == "cuda"

        if subbatch >= n_jumps:
            # Single-batch path (unchanged behaviour).
            initial_batch = Batch.from_data_list(post_data_list).to(
                self.device
            )
            final_batch = Batch.from_data_list(
                [current_data] * n_jumps
            ).to(self.device)
            if _is_cuda:
                torch.cuda.synchronize()
            with torch.no_grad():
                pred_norm = self.model(initial_batch, final_batch)
                pred = pred_norm.detach().cpu().numpy().reshape(-1)
        else:
            pred_chunks = []
            for chunk_start in range(0, n_jumps, subbatch):
                chunk_end = min(chunk_start + subbatch, n_jumps)
                chunk_post = post_data_list[chunk_start:chunk_end]
                chunk_count = len(chunk_post)
                initial_batch = Batch.from_data_list(chunk_post).to(
                    self.device
                )
                final_batch = Batch.from_data_list(
                    [current_data] * chunk_count
                ).to(self.device)
                if _is_cuda:
                    torch.cuda.synchronize()
                with torch.no_grad():
                    pred_norm = self.model(initial_batch, final_batch)
                    pred_chunks.append(
                        pred_norm.detach().cpu().numpy().reshape(-1)
                    )
            pred = np.concatenate(pred_chunks)

        barriers_eV = pred * self.target_std + self.target_mean
        return barriers_eV.astype(np.float64)

    # ------- Phase-3 legacy path (kept for benchmarking and tests) -------

    def _legacy_get_forward_barriers_batch(
        self, state, jump_atom_indices: np.ndarray
    ) -> np.ndarray:
        """Old per-step rebuild via GraphBuilder.atoms_to_graph.

        Kept as an explicit reference so the static-cache fast path can be
        verified for numerical equivalence in tests, and so the speedup
        delta can be measured cleanly.
        """
        # Local import to avoid module-level dependency cycles
        from KMC.state import VACANCY_SPECIES

        torch = self._torch
        Batch = self._Batch

        # Build the current ASE Atoms (N-1 atoms, vacancy site omitted) once.
        current_atoms = state.to_atoms(include_vacancy=False)

        # Map state-site-index -> position in current_atoms (only non-vacancy sites).
        non_vac_mask = state.species != VACANCY_SPECIES
        non_vac_site_indices = np.where(non_vac_mask)[0]
        site_to_atom_idx = np.full(state.n_sites, -1, dtype=np.int64)
        site_to_atom_idx[non_vac_site_indices] = np.arange(
            len(non_vac_site_indices)
        )

        vacancy_pos = state.positions[state.vacancy_index].copy()

        # The current state's graph is the same for every candidate jump,
        # so build it once and reuse it 8x in the final batch.
        current_graph = self.builder.atoms_to_graph(current_atoms)

        # For each candidate jump, build the post-jump Atoms by moving the
        # hopping atom from its NN site to the (currently empty) vacancy site.
        post_graphs = []
        for atom_site_idx in jump_atom_indices:
            atoms_idx = int(site_to_atom_idx[int(atom_site_idx)])
            post_atoms = current_atoms.copy()
            post_atoms.positions[atoms_idx] = vacancy_pos
            post_graphs.append(self.builder.atoms_to_graph(post_atoms))

        # Forward pass.
        n_jumps = len(jump_atom_indices)
        initial_batch = Batch.from_data_list(post_graphs).to(self.device)
        final_batch = Batch.from_data_list(
            [current_graph] * n_jumps
        ).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(initial_batch, final_batch)
            pred = pred_norm.detach().cpu().numpy().reshape(-1)

        barriers_eV = pred * self.target_std + self.target_mean
        return barriers_eV.astype(np.float64)
