"""
Graph Builder for Diffusion Barrier Prediction with Atom Embeddings

Builds PyG graphs from CIF files with:
- Element indices for learned embeddings (instead of one-hot)
- Atomic properties as separate features
- Real geometry from structure files
- Optimized with NPZ loading and ASE NeighborList
- NEW: Reads relaxation progress for weighted loss training
"""

import numpy as np
import torch
from torch_geometric.data import Data
from ase import Atoms
from ase.io import read as ase_read
from ase.neighborlist import NeighborList
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings
import time

from atomic_properties import get_atomic_properties


class GraphBuilder:
    """
    Build PyG graphs from CIF files with learned atom embeddings.
    
    Uses element indices for embedding lookup instead of one-hot encoding.
    Optionally includes atomic properties as additional features.
    """
    
    def __init__(self, config, csv_path: str = None, profile: bool = False, use_cache: bool = True):
        """
        Initialize graph builder.
        
        Args:
            config: Config object with graph construction parameters
            csv_path: Path to CSV database (for element detection)
            profile: Print timing information for debugging
            use_cache: Cache structure files in RAM for faster loading
        """
        self.config = config
        self.csv_path = csv_path
        self.profile = profile
        self.use_cache = use_cache
        
        # Graph construction parameters
        self.cutoff_radius = config.cutoff_radius
        self.max_neighbors = config.max_neighbors
        self.use_line_graph = config.use_line_graph
        self.line_graph_cutoff = getattr(config, 'line_graph_cutoff', config.cutoff_radius)
        self.use_atomic_properties = getattr(config, 'use_atomic_properties', True)
        
        # RAM cache for structure files
        self._structure_cache = {} if use_cache else None
        
        # Detect elements from CSV or use config
        if csv_path and Path(csv_path).exists():
            self.elements = self._detect_elements_from_csv(csv_path)
        else:
            self.elements = getattr(config, 'elements', ['Mo', 'Nb', 'Ta', 'W'])
        
        print(f"Detected elements: {self.elements}")
        
        # Element to index mapping for embeddings
        self.element_to_idx = {el: idx for idx, el in enumerate(self.elements)}
        
        print("\n" + "="*70)
        print("GRAPH BUILDER INITIALIZED (ATOM EMBEDDINGS + PROGRESS)")
        print("="*70)
        print(f"Elements: {self.elements}")
        print(f"Element indices: {self.element_to_idx}")
        print(f"Cutoff radius: {self.cutoff_radius} Å")
        print(f"Max neighbors: {self.max_neighbors}")
        print(f"Line graph: {self.use_line_graph}")
        if self.use_line_graph:
            print(f"Line graph cutoff: {self.line_graph_cutoff} Å")
        print(f"Atomic properties: {self.use_atomic_properties}")
        print(f"RAM caching: {self.use_cache}")
        print(f"Profiling: {self.profile}")
        print("="*70 + "\n")
    
    def _detect_elements_from_csv(self, csv_path: str) -> List[str]:
        """Detect unique elements from CSV database."""
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        # Try different column names
        for col in ['elements', 'composition', 'alloy']:
            if col in df.columns:
                elements_set = set()
                for entry in df[col].dropna():
                    if isinstance(entry, str):
                        for el in entry.replace(',', ' ').replace(';', ' ').split():
                            el = el.strip()
                            if len(el) <= 2 and el[0].isupper():
                                elements_set.add(el)
                
                if elements_set:
                    return sorted(list(elements_set))
        
        # Fallback: detect from structure files
        return self._detect_elements_from_structures(csv_path)
    
    def _detect_elements_from_structures(self, csv_path: str) -> List[str]:
        """Detect elements by reading a sample CIF file."""
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        if 'structure_folder' in df.columns:
            folder = Path(df.iloc[0]['structure_folder'])
            if not folder.is_absolute():
                folder = Path(csv_path).parent / folder
            
            cif_file = folder / "initial_relaxed.cif"
            if cif_file.exists():
                atoms = self._read_structure_from_file(str(cif_file))
                elements = sorted(set(atoms.get_chemical_symbols()))
                return elements
        
        # Final fallback to config
        warnings.warn("Could not detect elements, using config")
        return getattr(self.config, 'elements', ['Mo', 'Nb', 'Ta', 'W'])
    
    def _read_structure_from_npz(self, npz_path: str) -> Atoms:
        """
        Read structure from NPZ file (fast).
        NEW: Also extracts 'progress' scalar if available.
        """
        data = np.load(npz_path)
        atoms = Atoms(
            numbers=data['numbers'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc']
        )
        
        # NEW: Extract progress metadata and store in atoms.info
        if 'progress' in data:
            atoms.info['relax_progress'] = float(data['progress'][0])
        else:
            # Default to 1.0 (fully relaxed) if not present (legacy files)
            atoms.info['relax_progress'] = 1.0
            
        return atoms
    
    def _read_structure_from_file(self, file_path: str) -> Atoms:
        """
        Read structure with NPZ fallback and optional caching.
        """
        # Check RAM cache
        if self.use_cache and file_path in self._structure_cache:
            return self._structure_cache[file_path].copy()
        
        # Try NPZ first (fast)
        npz_path = Path(file_path).with_suffix('.npz')
        if npz_path.exists():
            try:
                atoms = self._read_structure_from_npz(str(npz_path))
                
                # Cache in RAM
                if self.use_cache:
                    self._structure_cache[file_path] = atoms
                
                return atoms.copy()
            except Exception as e:
                warnings.warn(f"Failed to read NPZ {npz_path}: {e}, falling back to CIF")
        
        # Fallback to CIF
        atoms = ase_read(file_path)
        
        # Set default progress for CIF files (assumed standard/relaxed)
        atoms.info['relax_progress'] = 1.0
        
        # Create NPZ for next time
        try:
            self._save_structure_as_npz(atoms, str(npz_path))
        except Exception as e:
            warnings.warn(f"Could not create NPZ file {npz_path}: {e}")
        
        # Cache in RAM
        if self.use_cache:
            self._structure_cache[file_path] = atoms
        
        return atoms.copy()

    def _save_structure_as_npz(self, atoms: Atoms, npz_path: str):
        """Save ASE atoms as NPZ for faster loading."""
        # Check if progress info exists, default to 1.0
        progress = atoms.info.get('relax_progress', 1.0)
        
        np.savez_compressed(
            npz_path,
            positions=atoms.positions,
            numbers=atoms.numbers,
            cell=atoms.cell.array,
            pbc=atoms.pbc,
            progress=np.array([progress], dtype=np.float32) # Save progress
        )
    
    def _compute_edges_from_positions(
        self,
        atoms: Atoms,
        cutoff: float
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute edges with real distances using ASE NeighborList.
        """
        n_atoms = len(atoms)
        
        # Build ASE NeighborList
        cutoffs = [cutoff / 2] * n_atoms
        nl = NeighborList(
            cutoffs,
            skin=0.0,
            self_interaction=False,
            bothways=True
        )
        nl.update(atoms)
        
        # Pre-allocate lists
        edge_list = []
        distance_list = []
        vector_list = []
        
        # Iterate over atoms
        for i in range(n_atoms):
            # Get neighbors with PBC offsets
            indices, offsets = nl.get_neighbors(i)
            
            if len(indices) == 0:
                continue
            
            # Vectorized computation of neighbor positions
            pos_i = atoms.positions[i]
            pos_j_all = atoms.positions[indices] + offsets @ atoms.cell
            vecs = pos_j_all - pos_i
            dists = np.linalg.norm(vecs, axis=1)
            
            # Filter by cutoff
            mask = dists < cutoff
            indices_filtered = indices[mask]
            dists_filtered = dists[mask]
            vecs_filtered = vecs[mask]
            
            if len(indices_filtered) == 0:
                continue
            
            # Sort by distance and keep max_neighbors closest
            if len(indices_filtered) > self.max_neighbors:
                sorted_idx = np.argsort(dists_filtered)[:self.max_neighbors]
                indices_filtered = indices_filtered[sorted_idx]
                dists_filtered = dists_filtered[sorted_idx]
                vecs_filtered = vecs_filtered[sorted_idx]
            
            # Append to lists
            for j, dist, vec in zip(indices_filtered, dists_filtered, vecs_filtered):
                edge_list.append([i, j])
                distance_list.append(dist)
                vector_list.append(vec)
        
        if len(edge_list) == 0:
            # No edges found - create dummy edge
            warnings.warn(f"No edges found with cutoff {cutoff} Å! Using dummy edge.")
            edge_index = torch.LongTensor([[0], [1]])
            edge_attr = torch.FloatTensor([[cutoff]])
            edge_vectors = torch.FloatTensor([[0, 0, cutoff]])
        else:
            # Convert to tensors
            edge_index = torch.from_numpy(np.array(edge_list, dtype=np.int64)).T
            edge_attr = torch.from_numpy(np.array(distance_list, dtype=np.float32)).unsqueeze(1)
            edge_vectors = torch.from_numpy(np.array(vector_list, dtype=np.float32))
        
        return edge_index, edge_attr, edge_vectors
    
    def _build_line_graph_from_vectors(
        self,
        edge_index: torch.LongTensor,
        edge_vectors: torch.FloatTensor
    ) -> Dict:
        """
        Build line graph from edge vectors with vectorized computation.
        """
        n_edges = edge_index.shape[1]
        
        if n_edges == 0:
            return {
                'node_features': torch.zeros(0, 4),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'edge_attr': torch.zeros(0, 1)
            }
        
        # Line graph node features: normalized direction + length
        edge_lengths = torch.norm(edge_vectors, dim=1, keepdim=True)
        edge_vectors_norm = edge_vectors / (edge_lengths + 1e-8)
        
        line_node_features = torch.cat([
            edge_vectors_norm,  # [n_edges, 3] normalized direction
            edge_lengths        # [n_edges, 1] length
        ], dim=1)  # [n_edges, 4]
        
        # Build line graph edges
        source_atoms = edge_index[0]  # Source atom of each bond
        unique_sources = torch.unique(source_atoms)
        
        line_edge_list = []
        line_angle_list = []
        
        # For each atom, connect all bonds originating from it
        for source in unique_sources:
            mask = source_atoms == source
            edge_idx = torch.where(mask)[0]
            
            n_local = len(edge_idx)
            if n_local < 2:
                continue
            
            # Get vectors for these bonds
            vecs = edge_vectors[edge_idx]
            norms = torch.norm(vecs, dim=1, keepdim=True)
            vecs_norm = vecs / (norms + 1e-8)
            
            # Compute all pairwise angles
            cos_angles = vecs_norm @ vecs_norm.T  # [n_local, n_local]
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            angles = torch.acos(cos_angles)  # [n_local, n_local]
            
            # Get upper triangular indices (no self-loops, no duplicates)
            i_idx, j_idx = torch.triu_indices(n_local, n_local, offset=1)
            
            # Convert to global edge indices
            edge_i = edge_idx[i_idx]
            edge_j = edge_idx[j_idx]
            angle_vals = angles[i_idx, j_idx]
            
            # Add both directions
            line_edge_list.append(torch.stack([edge_i, edge_j], dim=0))
            line_edge_list.append(torch.stack([edge_j, edge_i], dim=0))
            
            line_angle_list.append(angle_vals)
            line_angle_list.append(angle_vals)
        
        if len(line_edge_list) == 0:
            # No line edges - create minimal dummy
            line_edge_index = torch.LongTensor([[0], [0]])
            line_edge_attr = torch.FloatTensor([[0.0]])
        else:
            # Concatenate all edges
            line_edge_index = torch.cat(line_edge_list, dim=1)
            line_edge_attr = torch.cat(line_angle_list, dim=0).unsqueeze(1)
        
        return {
            'node_features': line_node_features,
            'edge_index': line_edge_index,
            'edge_attr': line_edge_attr
        }
    
    def cif_to_graph(
        self,
        cif_path: str,
        backward_barrier: float = None,
        progress_override: Optional[float] = None  # NEW: Allow overriding progress
    ) -> Data:
        """
        Build PyG graph from CIF file with atom embeddings.
        
        Creates graph with:
        - x_element: Element indices for embedding lookup
        - x_props: Atomic properties (optional)
        - edge_index, edge_attr: Connectivity and distances
        - line_graph_*: Line graph data (optional)
        - relax_progress: NEW - Progress scalar [1, 1]
        
        Args:
            cif_path: Path to CIF file
            backward_barrier: Target barrier value (optional)
            progress_override: Force progress value (0.0 to 1.0)
        
        Returns:
            PyG Data object with graph representation
        """
        # Profiling setup
        if self.profile:
            t_total = time.time()
            t0 = time.time()
        
        # Read structure (this now also fetches progress info into atoms.info)
        atoms = self._read_structure_from_file(cif_path)
        
        # --- NEW: PROGRESS HANDLING ---
        if progress_override is not None:
            progress = float(progress_override)
        else:
            progress = atoms.info.get('relax_progress', 1.0) # Default to 1.0 if missing
            
        if self.profile:
            t_read = time.time() - t0
            t0 = time.time()
        
        # Get structure data
        positions = atoms.get_positions()
        elements = atoms.get_chemical_symbols()
        n_atoms = len(atoms)
        
        # Build node features
        element_indices = []
        atomic_props_list = []
        
        for element in elements:
            # Get element index for embedding
            if element in self.element_to_idx:
                element_indices.append(self.element_to_idx[element])
            else:
                warnings.warn(f"Unknown element {element}, using index 0")
                element_indices.append(0)
            
            # Get atomic properties
            if self.use_atomic_properties:
                props = get_atomic_properties(element)
                atomic_props_list.append([
                    props['atomic_number'],
                    props['atomic_mass'],
                    props['atomic_radius'],
                    props['electronegativity'],
                    props['first_ionization'],
                    props['electron_affinity'],
                    props['melting_point'],
                    props['density']
                ])
        
        # Convert to tensors
        x_element = torch.LongTensor(element_indices)  # For embedding lookup
        
        if self.use_atomic_properties:
            x_props = torch.FloatTensor(atomic_props_list)
        else:
            x_props = None
        
        if self.profile:
            t_features = time.time() - t0
            t0 = time.time()
        
        # Compute edges with real distances
        edge_index, edge_attr, edge_vectors = self._compute_edges_from_positions(
            atoms, self.cutoff_radius
        )
        
        if self.profile:
            t_edges = time.time() - t0
            t0 = time.time()
        
        # Build line graph if enabled
        if self.use_line_graph:
            line_graph = self._build_line_graph_from_vectors(edge_index, edge_vectors)
            line_graph_batch_mapping = edge_index[0].clone()
        else:
            line_graph = None
            line_graph_batch_mapping = None
        
        if self.profile:
            t_line = time.time() - t0
            t_total = time.time() - t_total
            
            npz_path = Path(cif_path).with_suffix('.npz')
            format_used = "NPZ" if npz_path.exists() else "CIF"
            
            print(f"Graph build timing ({Path(cif_path).name}) [{format_used}]:")
            print(f"  File reading:   {t_read*1000:6.2f} ms ({t_read/t_total*100:4.1f}%)")
            print(f"  Node features:  {t_features*1000:6.2f} ms ({t_features/t_total*100:4.1f}%)")
            print(f"  Edge compute:   {t_edges*1000:6.2f} ms ({t_edges/t_total*100:4.1f}%)")
            if self.use_line_graph:
                print(f"  Line graph:     {t_line*1000:6.2f} ms ({t_line/t_total*100:4.1f}%)")
            print(f"  TOTAL:          {t_total*1000:6.2f} ms")
        
        # Create PyG Data object
        data = Data(
            x_element=x_element,     # Element indices for embeddings
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_atoms,
            relax_progress=torch.tensor([[progress]], dtype=torch.float32) # NEW: Add progress scalar
        )
        
        # Add atomic properties if enabled
        if x_props is not None:
            data.x_props = x_props
        
        # Add line graph data if enabled
        if line_graph is not None:
            data.line_graph_x = line_graph['node_features']
            data.line_graph_edge_index = line_graph['edge_index']
            data.line_graph_edge_attr = line_graph['edge_attr']
            data.line_graph_batch_mapping = line_graph_batch_mapping
        
        # Add label if provided
        if backward_barrier is not None:
            data.y = torch.tensor([backward_barrier], dtype=torch.float32)
        
        return data
    
    def build_pair_graph(
        self,
        initial_cif: str,
        final_cif: str,
        backward_barrier: float = None,
        progress_initial: float = None, # NEW
        progress_final: float = None    # NEW
    ) -> Tuple[Data, Data]:
        """
        Build graph pair from initial and final structures.
        """
        initial_graph = self.cif_to_graph(initial_cif, backward_barrier, progress_override=progress_initial)
        final_graph = self.cif_to_graph(final_cif, backward_barrier, progress_override=progress_final)
        
        return initial_graph, final_graph
    
    def clear_cache(self):
        """Clear RAM cache to free memory."""
        if self._structure_cache is not None:
            self._structure_cache.clear()
            print("Structure cache cleared")


def test_graph_builder():
    """Test the graph builder with atom embeddings."""
    from config import Config
    import pandas as pd
    
    print("\n" + "="*70)
    print("TESTING GRAPH BUILDER (ATOM EMBEDDINGS)")
    print("="*70)
    
    config = Config()
    
    # Get CIF path from CSV
    csv_path = config.csv_path
    
    if not Path(csv_path).exists():
        print(f"\nCSV not found: {csv_path}")
        print("Cannot test without data.")
        print("="*70 + "\n")
        return
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print(f"\nCSV is empty: {csv_path}")
        print("="*70 + "\n")
        return
    
    # Get first structure
    first_row = df.iloc[0]
    structure_folder = Path(first_row['structure_folder'])
    
    if not structure_folder.is_absolute():
        structure_folder = Path(csv_path).parent / structure_folder
    
    # Try to find one of our NEW trajectory files to test progress reading
    traj_file = structure_folder / "initial_traj_0.npz"
    if traj_file.exists():
        test_file = traj_file
        print(f"Testing with TRAJECTORY file (should have progress=0.0): {test_file}")
    else:
        test_file = structure_folder / "initial_relaxed.cif"
        print(f"Testing with STANDARD file (should have progress=1.0): {test_file}")
    
    if not test_file.exists():
        print(f"\nTest file not found: {test_file}")
        return
    
    # Create builder with profiling
    builder = GraphBuilder(config, csv_path=csv_path, profile=True, use_cache=True)
    
    print("\n" + "-"*70)
    print("Single graph build:")
    print("-"*70)
    
    graph = builder.cif_to_graph(str(test_file), backward_barrier=1.5)
    
    print("\n" + "-"*70)
    print("Graph statistics:")
    print("-"*70)
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    # Verify Progress Feature
    if hasattr(graph, 'relax_progress'):
        print(f"  Relax Progress: {graph.relax_progress.item():.2f}")
    else:
        print("  Relax Progress: MISSING!")
        
    print(f"  Element indices: {graph.x_element.shape}")
    
    if hasattr(graph, 'line_graph_x'):
        print(f"  Line graph nodes: {graph.line_graph_x.shape[0]}")
    
    print("\nGraph builder test successful!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_graph_builder()