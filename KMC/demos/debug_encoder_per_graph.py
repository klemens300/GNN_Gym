"""Debug: compare encoder output per-graph (no batching) for fast vs legacy.

If single-graph embeddings agree but the batched barriers don't, the issue
is in PyG's Batch.from_data_list (likely the line_graph_edge_index increment).
If single-graph embeddings already disagree, the issue is in the graph data
itself (something we missed in the topology comparison).

Run from /path/to/GNN_Gym:
    python -m KMC.demos.debug_encoder_per_graph
"""

import sys
from pathlib import Path
import numpy as np

_SCIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIPTS_DIR))

from KMC.config import KMCConfig
from KMC.state import KMCState, VACANCY_SPECIES
from KMC.barrier_predictor import GNNBarrierPredictor, _derive_subgraph_data


def main():
    cfg = KMCConfig(supercell_size=4, lattice_parameter_A=3.22, random_seed=42, n_steps=1)
    if not Path(cfg.gnn_model_path).exists():
        print(f"Model missing at {cfg.gnn_model_path}")
        return

    pred = GNNBarrierPredictor(cfg, use_static_cache=True)
    torch = pred._torch
    Batch = pred._Batch

    state = KMCState.from_random_composition(cfg)
    nn = state.get_neighbor_atom_indices()
    v = int(state.vacancy_index)

    # ---- Build both fast and legacy versions of every post graph ----
    current_atoms = state.to_atoms(include_vacancy=False)
    legacy_current = pred.builder.atoms_to_graph(current_atoms)

    non_vac_mask = state.species != VACANCY_SPECIES
    non_vac_site_indices = np.where(non_vac_mask)[0]
    site_to_atom_idx = np.full(state.n_sites, -1, dtype=np.int64)
    site_to_atom_idx[non_vac_site_indices] = np.arange(len(non_vac_site_indices))
    vacancy_pos = state.positions[v].copy()

    legacy_posts = []
    for atom_site_idx in nn:
        atoms_idx = int(site_to_atom_idx[int(atom_site_idx)])
        post_atoms = current_atoms.copy()
        post_atoms.positions[atoms_idx] = vacancy_pos
        legacy_posts.append(pred.builder.atoms_to_graph(post_atoms))

    cache = pred._ensure_cache(state)
    species_full = np.asarray(state.species, dtype=np.int64)
    fast_current = _derive_subgraph_data(
        cache, excluded_site=v,
        species_for_subgraph=species_full.copy(),
        torch_module=torch,
    )
    fast_posts = []
    for k in nn:
        k_int = int(k)
        post_species = species_full.copy()
        post_species[v] = species_full[k_int]
        fast_posts.append(_derive_subgraph_data(
            cache, excluded_site=k_int,
            species_for_subgraph=post_species,
            torch_module=torch,
        ))

    # ---- Encoder per single graph (no batching) ----
    pred.model.eval()
    print("\n=== Per-graph encoder embeddings (no batching) ===")
    for i in range(8):
        single_legacy = Batch.from_data_list([legacy_posts[i]]).to(pred.device)
        single_fast = Batch.from_data_list([fast_posts[i]]).to(pred.device)
        with torch.no_grad():
            emb_legacy = pred.model.encoder(single_legacy).cpu().numpy().reshape(-1)
            emb_fast = pred.model.encoder(single_fast).cpu().numpy().reshape(-1)
        diff = np.linalg.norm(emb_fast - emb_legacy)
        max_abs = np.max(np.abs(emb_fast - emb_legacy))
        print(f"  POST_{i} (k=site_{int(nn[i])}): "
              f"||fast-legacy||={diff:.2e}, max|diff|={max_abs:.2e}")

    # ---- Encoder for the FULL batched call ----
    print("\n=== Batched encoder embeddings (8 post graphs in one batch) ===")
    legacy_batch = Batch.from_data_list(legacy_posts).to(pred.device)
    fast_batch = Batch.from_data_list(fast_posts).to(pred.device)
    with torch.no_grad():
        emb_legacy_batch = pred.model.encoder(legacy_batch).cpu().numpy()
        emb_fast_batch = pred.model.encoder(fast_batch).cpu().numpy()
    for i in range(8):
        diff = np.linalg.norm(emb_fast_batch[i] - emb_legacy_batch[i])
        max_abs = np.max(np.abs(emb_fast_batch[i] - emb_legacy_batch[i]))
        print(f"  POST_{i} (k=site_{int(nn[i])}): "
              f"||fast-legacy||={diff:.2e}, max|diff|={max_abs:.2e}")

    # ---- Compare line_graph_edge_index after batching ----
    print("\n=== line_graph_edge_index sanity check ===")
    print(f"  legacy_batch.line_graph_edge_index.max()  = {legacy_batch.line_graph_edge_index.max().item()}")
    print(f"  legacy_batch.line_graph_x.shape[0]        = {legacy_batch.line_graph_x.shape[0]}")
    print(f"  fast_batch.line_graph_edge_index.max()    = {fast_batch.line_graph_edge_index.max().item()}")
    print(f"  fast_batch.line_graph_x.shape[0]          = {fast_batch.line_graph_x.shape[0]}")
    print("  (max line_edge_index should be < line_graph_x.shape[0]; "
          "if not, PyG offsets are wrong)")
    print(f"  legacy_batch.edge_index.shape[1]          = {legacy_batch.edge_index.shape[1]}")
    print(f"  legacy_batch.num_nodes                    = {legacy_batch.num_nodes}")


if __name__ == "__main__":
    main()
