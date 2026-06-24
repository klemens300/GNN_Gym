"""Debug: compare legacy vs fast predictor graphs side-by-side.

Run from /path/to/GNN_Gym:
    python -m KMC.demos.debug_predictor_equivalence
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


def _canonical_edges(edge_index, positions, atol=1e-4):
    """Edges as frozenset of (sorted position-tuple pairs)."""
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    out = set()
    for s, d in zip(src, dst):
        ps = tuple(np.round(positions[s], 4))
        pd = tuple(np.round(positions[d], 4))
        out.add(tuple(sorted([ps, pd])))
    return out


def _per_atom_summary(data, positions):
    """Map position-tuple -> (x_element, n_incident_edges, sorted_neighbor_positions)."""
    src = data.edge_index[0].cpu().numpy()
    dst = data.edge_index[1].cpu().numpy()
    summary = {}
    for i in range(int(data.num_nodes)):
        pos = tuple(np.round(positions[i], 4))
        neighbors = []
        for s, d in zip(src, dst):
            if s == i:
                neighbors.append(tuple(np.round(positions[d], 4)))
        neighbors.sort()
        summary[pos] = (
            int(data.x_element[i].item()),
            len(neighbors),
            neighbors,
        )
    return summary


def compare_subgraphs(legacy_data, legacy_positions, fast_data, fast_positions, label):
    print(f"\n=== {label} ===")
    print(f"  num_nodes: legacy={legacy_data.num_nodes} fast={fast_data.num_nodes}")
    print(f"  num_edges: legacy={legacy_data.edge_index.shape[1]} fast={fast_data.edge_index.shape[1]}")

    if hasattr(legacy_data, "line_graph_x"):
        print(f"  line_nodes: legacy={legacy_data.line_graph_x.shape[0]} fast={fast_data.line_graph_x.shape[0]}")
        print(f"  line_edges: legacy={legacy_data.line_graph_edge_index.shape[1]} fast={fast_data.line_graph_edge_index.shape[1]}")

    # Position sets
    pos_l = {tuple(np.round(p, 4)) for p in legacy_positions}
    pos_f = {tuple(np.round(p, 4)) for p in fast_positions}
    if pos_l == pos_f:
        print(f"  position sets: IDENTICAL ({len(pos_l)} positions)")
    else:
        print(f"  position sets: DIFFER -- only legacy={len(pos_l - pos_f)}, only fast={len(pos_f - pos_l)}")

    # x_element histogram per position
    sum_l = _per_atom_summary(legacy_data, legacy_positions)
    sum_f = _per_atom_summary(fast_data, fast_positions)

    # Element identity at each position
    mismatches_elem = []
    for pos in pos_l & pos_f:
        if sum_l[pos][0] != sum_f[pos][0]:
            mismatches_elem.append((pos, sum_l[pos][0], sum_f[pos][0]))
    if mismatches_elem:
        print(f"  ELEMENT MISMATCHES: {len(mismatches_elem)}")
        for pos, el_l, el_f in mismatches_elem[:5]:
            print(f"    at {pos}: legacy={el_l} fast={el_f}")
    else:
        print(f"  element-at-position: identical")

    # Per-atom degree
    mismatches_deg = []
    for pos in pos_l & pos_f:
        if sum_l[pos][1] != sum_f[pos][1]:
            mismatches_deg.append((pos, sum_l[pos][1], sum_f[pos][1]))
    if mismatches_deg:
        print(f"  DEGREE MISMATCHES: {len(mismatches_deg)}")
        for pos, d_l, d_f in mismatches_deg[:5]:
            print(f"    at {pos}: legacy={d_l} fast={d_f}")
    else:
        print(f"  per-atom degree: identical")

    # Edge sets (as canonicalized position pairs)
    e_l = _canonical_edges(legacy_data.edge_index, legacy_positions)
    e_f = _canonical_edges(fast_data.edge_index, fast_positions)
    if e_l == e_f:
        print(f"  edge sets: IDENTICAL ({len(e_l)} unique pairs)")
    else:
        only_l = e_l - e_f
        only_f = e_f - e_l
        print(f"  EDGE SET DIFFERS: only_legacy={len(only_l)}, only_fast={len(only_f)}")
        for pair in list(only_l)[:3]:
            print(f"    only legacy: {pair}")
        for pair in list(only_f)[:3]:
            print(f"    only fast:   {pair}")

    # Edge attr stats
    ea_l = legacy_data.edge_attr.cpu().numpy()
    ea_f = fast_data.edge_attr.cpu().numpy()
    print(f"  edge_attr stats: legacy sum={ea_l.sum():.6f} fast sum={ea_f.sum():.6f} diff={ea_f.sum()-ea_l.sum():.2e}")

    # x_props stats
    if hasattr(legacy_data, "x_props"):
        xp_l = legacy_data.x_props.cpu().numpy()
        xp_f = fast_data.x_props.cpu().numpy()
        print(f"  x_props stats:   legacy sum={xp_l.sum():.6f} fast sum={xp_f.sum():.6f} diff={xp_f.sum()-xp_l.sum():.2e}")


def main():
    cfg = KMCConfig(supercell_size=4, lattice_parameter_A=3.22, random_seed=42, n_steps=1)
    if not Path(cfg.gnn_model_path).exists():
        print(f"Model missing at {cfg.gnn_model_path}")
        return

    fast = GNNBarrierPredictor(cfg, use_static_cache=True)
    legacy = GNNBarrierPredictor(cfg, use_static_cache=False)

    state = KMCState.from_random_composition(cfg)
    nn = state.get_neighbor_atom_indices()
    v = int(state.vacancy_index)

    # ----- Build legacy graphs -----
    current_atoms = state.to_atoms(include_vacancy=False)
    legacy_current = legacy.builder.atoms_to_graph(current_atoms)
    legacy_current_pos = current_atoms.positions.copy()

    non_vac_mask = state.species != VACANCY_SPECIES
    non_vac_site_indices = np.where(non_vac_mask)[0]
    site_to_atom_idx = np.full(state.n_sites, -1, dtype=np.int64)
    site_to_atom_idx[non_vac_site_indices] = np.arange(len(non_vac_site_indices))
    vacancy_pos = state.positions[v].copy()

    legacy_posts = []
    legacy_post_positions = []
    for atom_site_idx in nn:
        atoms_idx = int(site_to_atom_idx[int(atom_site_idx)])
        post_atoms = current_atoms.copy()
        post_atoms.positions[atoms_idx] = vacancy_pos
        legacy_posts.append(legacy.builder.atoms_to_graph(post_atoms))
        legacy_post_positions.append(post_atoms.positions.copy())

    # ----- Build fast graphs -----
    cache = fast._ensure_cache(state)
    species_full = np.asarray(state.species, dtype=np.int64)

    fast_current = _derive_subgraph_data(
        cache, excluded_site=v,
        species_for_subgraph=species_full.copy(),
        torch_module=fast._torch,
    )
    keep_node_v = np.ones(state.n_sites, dtype=bool)
    keep_node_v[v] = False
    fast_current_pos = state.positions[keep_node_v].copy()

    fast_posts = []
    fast_post_positions = []
    for k in nn:
        k_int = int(k)
        post_species = species_full.copy()
        post_species[v] = species_full[k_int]
        fast_posts.append(_derive_subgraph_data(
            cache, excluded_site=k_int,
            species_for_subgraph=post_species,
            torch_module=fast._torch,
        ))
        keep_node_k = np.ones(state.n_sites, dtype=bool)
        keep_node_k[k_int] = False
        fast_post_positions.append(state.positions[keep_node_k].copy())

    # ----- Compare -----
    compare_subgraphs(legacy_current, legacy_current_pos, fast_current, fast_current_pos, "CURRENT")
    for i in range(len(nn)):
        compare_subgraphs(
            legacy_posts[i], legacy_post_positions[i],
            fast_posts[i], fast_post_positions[i],
            f"POST_{i} (jump to NN[{i}]=site_{int(nn[i])})",
        )


if __name__ == "__main__":
    main()
