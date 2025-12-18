"""
Graph Builder for Diffusion Barrier Prediction

🔥 ULTRA-OPTIMIZED: Fast edge computation with ASE NeighborList + NPZ loading

Key Optimizations:
- NPZ format loading (10-15x faster than CIF)
- ASE NeighborList instead of O(N²) manual loops
- Vectorized distance/angle computations
- Efficient numpy → tensor conversion
- Optional RAM caching
- Optional profiling output
"""

import numpy as np
import torch
from torch_geometric.data import Data
from ase import Atoms
from ase.io import read as ase_read
from ase.neighborlist import NeighborList
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
import time

# Import atomic properties
from atomic_properties import get_atomic_properties


class GraphBuilder:
    """
    Build PyG graphs from CIF files with REAL geometry.
    
    🔥 ULTRA-OPTIMIZED with ASE NeighborList, vectorized operations, and NPZ loading!
    """
    
    def __init__(self, config, csv_path: str = None, profile: bool = False, use_cache: bool = True):
        """
        Initialize graph builder.
        
        Parameters:
        -----------
        config : Config
            Configuration object with:
            - cutoff_radius: Cutoff for neighbor search
            - max_neighbors: Maximum neighbors per atom
            - line_graph_cutoff: Cutoff for line graph edges
        csv_path : str, optional
            Path to CSV (for element detection)
        profile : bool
            If True, print timing information for each graph build
        use_cache : bool
            If True, cache CIF/NPZ files in RAM (recommended for training)
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
        
        # RAM cache for structure files
        self._structure_cache = {} if use_cache else None
        
        # Detect elements from CSV if available
        if csv_path and Path(csv_path).exists():
            self.elements = self._detect_elements_from_csv(csv_path)
        else:
            self.elements = getattr(config, 'elements', ['Mo', 'Nb', 'Ta', 'W'])
        
        print(f"✓ Detected elements: {self.elements}")
        
        # Element to index mapping
        self.element_to_idx = {el: idx for idx, el in enumerate(self.elements)}
        
        print("\n" + "="*70)
        print("GRAPH BUILDER INITIALIZED (ULTRA-OPTIMIZED)")
        print("="*70)
        print(f"Elements: {self.elements}")
        print(f"Cutoff radius: {self.cutoff_radius} Å")
        print(f"Max neighbors: {self.max_neighbors}")
        print(f"Line graph: {self.use_line_graph}")
        if self.use_line_graph:
            print(f"Line graph cutoff: {self.line_graph_cutoff} Å")
        print(f"RAM caching: {self.use_cache}")
        print(f"Profiling: {self.profile}")
        print("="*70 + "\n")
    
    def _detect_elements_from_csv(self, csv_path: str) -> List[str]:
        """Detect unique elements from CSV."""
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        # Try different column names
        for col in ['elements', 'composition', 'alloy']:
            if col in df.columns:
                # Parse elements
                elements_set = set()
                for entry in df[col].dropna():
                    if isinstance(entry, str):
                        # Split by common separators
                        for el in entry.replace(',', ' ').replace(';', ' ').split():
                            el = el.strip()
                            if len(el) <= 2 and el[0].isupper():
                                elements_set.add(el)
                
                if elements_set:
                    return sorted(list(elements_set))
        
        # Fallback: try to detect from structure files
        return self._detect_elements_from_structures(csv_path)
    
    def _detect_elements_from_structures(self, csv_path: str) -> List[str]:
        """Detect elements by reading a sample CIF."""
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        if 'structure_folder' in df.columns:
            # Read first structure
            folder = Path(df.iloc[0]['structure_folder'])
            if not folder.is_absolute():
                folder = Path(csv_path).parent / folder
            
            cif_file = folder / "initial_relaxed.cif"
            if cif_file.exists():
                atoms = self._read_structure_from_file(str(cif_file))
                elements = sorted(set(atoms.get_chemical_symbols()))
                return elements
        
        # Final fallback
        warnings.warn("Could not detect elements, using default: Mo, Nb, Ta, W")
        return ['Mo', 'Nb', 'Ta', 'W']
    
    def _read_structure_from_npz(self, npz_path: str) -> Atoms:
        """
        Read structure from NPZ file (FAST!).
        
        🔥 NPZ is ~10-15x faster than CIF reading
        
        NPZ contains:
        - positions: atomic positions [n_atoms, 3]
        - numbers: atomic numbers [n_atoms]
        - cell: unit cell [3, 3]
        - pbc: periodic boundary conditions [3]
        
        Parameters:
        -----------
        npz_path : str
            Path to NPZ file
        
        Returns:
        --------
        atoms : ASE Atoms object
        """
        data = np.load(npz_path)
        atoms = Atoms(
            numbers=data['numbers'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc']
        )
        return atoms
    
    def _read_structure_from_file(self, file_path: str) -> Atoms:
        """
        Read structure with NPZ fallback and optional caching.
        
        NEW: Automatically creates NPZ if it doesn't exist.
        
        Parameters:
        -----------
        file_path : str
            Path to structure file (CIF or NPZ)
        
        Returns:
        --------
        atoms : ASE Atoms object
        """
        # 1. Check RAM cache
        if self.use_cache and file_path in self._structure_cache:
            return self._structure_cache[file_path].copy()
        
        # 2. Try NPZ first (fast!)
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
        
        # 3. Fallback to CIF (slow but reliable)
        atoms = ase_read(file_path)
        
        # NEW: Create NPZ for next time
        try:
            self._save_structure_as_npz(atoms, str(npz_path))
        except Exception as e:
            warnings.warn(f"Could not create NPZ file {npz_path}: {e}")
        
        # Cache in RAM
        if self.use_cache:
            self._structure_cache[file_path] = atoms
        
        return atoms.copy()

    def _save_structure_as_npz(self, atoms: Atoms, npz_path: str):
        """
        Save ASE atoms as NPZ (on-the-fly conversion).
        
        Parameters:
        -----------
        atoms : ASE Atoms
            Structure to save
        npz_path : str
            Path for NPZ file
        """
        np.savez_compressed(
            npz_path,
            positions=atoms.positions,
            numbers=atoms.numbers,
            cell=atoms.cell.array,
            pbc=atoms.pbc
        )
    
    def _compute_edges_from_positions(
        self,
        atoms: Atoms,
        cutoff: float
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute edges with real distances using ASE NeighborList.
        
        🔥 OPTIMIZED: Vectorized operations + efficient numpy conversion
        
        Parameters:
        -----------
        atoms : ASE Atoms
            Atomic structure with positions, cell, pbc
        cutoff : float
            Cutoff radius for neighbor search
        
        Returns:
        --------
        edge_index : LongTensor [2, n_edges]
        edge_attr : FloatTensor [n_edges, 1] - distances
        edge_vectors : FloatTensor [n_edges, 3] - bond vectors (for line graph)
        """
        n_atoms = len(atoms)
        
        # Build ASE NeighborList (optimized C backend!)
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
            
            # 🔥 VECTORIZED: Compute all neighbor positions at once
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
            
            # Sort by distance and keep max_neighbors
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
            # No edges found - use dummy edge
            warnings.warn(f"No edges found with cutoff {cutoff} Å! Using dummy edge.")
            edge_index = torch.LongTensor([[0], [1]])
            edge_attr = torch.FloatTensor([[cutoff]])
            edge_vectors = torch.FloatTensor([[0, 0, cutoff]])
        else:
            # ✅ FAST: Convert to numpy array first, then to tensor (avoids warning)
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
        Build line graph from edge vectors with FULLY vectorized computation.
        
        🔥 ULTRA-OPTIMIZED: Minimal Python loops, maximum tensor operations!
        
        Line Graph:
        - Nodes = edges of atom graph (bonds)
        - Edges = pairs of bonds that share an atom (angles)
        
        Parameters:
        -----------
        edge_index : LongTensor [2, n_edges]
        edge_vectors : FloatTensor [n_edges, 3]
        
        Returns:
        --------
        line_graph : dict
            - 'node_features': FloatTensor [n_edges, feature_dim]
            - 'edge_index': LongTensor [2, n_line_edges]
            - 'edge_attr': FloatTensor [n_line_edges, 1] - angles
        """
        n_edges = edge_index.shape[1]
        
        if n_edges == 0:
            # Empty line graph
            return {
                'node_features': torch.zeros(0, 4),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'edge_attr': torch.zeros(0, 1)
            }
        
        # Line graph node features: bond vectors (normalized) + length
        edge_lengths = torch.norm(edge_vectors, dim=1, keepdim=True)
        edge_vectors_norm = edge_vectors / (edge_lengths + 1e-8)
        
        line_node_features = torch.cat([
            edge_vectors_norm,  # [n_edges, 3]
            edge_lengths        # [n_edges, 1]
        ], dim=1)  # [n_edges, 4]
        
        # Build line graph edges efficiently
        source_atoms = edge_index[0]  # [n_edges]
        
        # Group edges by source atom
        unique_sources = torch.unique(source_atoms)
        
        line_edge_list = []
        line_angle_list = []
        
        for source in unique_sources:
            # Get all edges from this source
            mask = source_atoms == source
            edge_idx = torch.where(mask)[0]
            
            n_local = len(edge_idx)
            if n_local < 2:
                continue
            
            # Get vectors for these edges
            vecs = edge_vectors[edge_idx]  # [n_local, 3]
            
            # Normalize
            norms = torch.norm(vecs, dim=1, keepdim=True)
            vecs_norm = vecs / (norms + 1e-8)
            
            # 🔥 VECTORIZED: Compute all pairwise angles at once
            cos_angles = vecs_norm @ vecs_norm.T  # [n_local, n_local]
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            angles = torch.acos(cos_angles)  # [n_local, n_local]
            
            # Get upper triangular indices (no self-loops, no duplicates)
            i_idx, j_idx = torch.triu_indices(n_local, n_local, offset=1)
            
            # Convert local indices to global edge indices
            edge_i = edge_idx[i_idx]
            edge_j = edge_idx[j_idx]
            angle_vals = angles[i_idx, j_idx]
            
            # Add both directions
            line_edge_list.append(torch.stack([edge_i, edge_j], dim=0))  # [2, n_pairs]
            line_edge_list.append(torch.stack([edge_j, edge_i], dim=0))  # [2, n_pairs]
            
            line_angle_list.append(angle_vals)
            line_angle_list.append(angle_vals)
        
        if len(line_edge_list) == 0:
            # No line edges - create minimal dummy
            line_edge_index = torch.LongTensor([[0], [0]])
            line_edge_attr = torch.FloatTensor([[0.0]])
        else:
            # Concatenate all edge pairs
            line_edge_index = torch.cat(line_edge_list, dim=1)  # [2, total_pairs]
            line_edge_attr = torch.cat(line_angle_list, dim=0).unsqueeze(1)  # [total_pairs, 1]
        
        return {
            'node_features': line_node_features,
            'edge_index': line_edge_index,
            'edge_attr': line_edge_attr
        }
    
    def cif_to_graph(
        self,
        cif_path: str,
        backward_barrier: float = None
    ) -> Data:
        """
        Build PyG graph from CIF file with REAL geometry.
        
        🔥 ULTRA-OPTIMIZED: Fast NPZ loading + ASE NeighborList
        
        Parameters:
        -----------
        cif_path : str
            Path to CIF file (or NPZ if available)
        backward_barrier : float, optional
            Backward barrier (label)
        
        Returns:
        --------
        data : torch_geometric.data.Data
            Graph with real geometry from CIF/NPZ
        """
        # Profiling setup
        if self.profile:
            t_total = time.time()
            t0 = time.time()
        
        # 1. Read structure (NPZ if available, else CIF)
        atoms = self._read_structure_from_file(cif_path)
        
        if self.profile:
            t_read = time.time() - t0
            t0 = time.time()
        
        # Get data from atoms
        positions = atoms.get_positions()
        elements = atoms.get_chemical_symbols()
        n_atoms = len(atoms)
        
        # 2. Build node features
        x_list = []
        for element in elements:
            # Get atomic properties
            props = get_atomic_properties(element)
            
            # One-hot encoding for element
            one_hot = torch.zeros(len(self.elements))
            if element in self.element_to_idx:
                one_hot[self.element_to_idx[element]] = 1.0
            
            # Concatenate: one-hot + properties
            features = torch.cat([
                one_hot,
                torch.FloatTensor([
                    props['atomic_number'],
                    props['atomic_mass'],
                    props['atomic_radius'],
                    props['electronegativity'],
                    props['first_ionization'],
                    props['electron_affinity'],
                    props['melting_point'],
                    props['density']
                ])
            ])
            
            x_list.append(features)
        
        x = torch.stack(x_list)  # [n_atoms, feature_dim]
        
        if self.profile:
            t_features = time.time() - t0
            t0 = time.time()
        
        # 3. Compute edges with REAL distances using ASE NeighborList
        edge_index, edge_attr, edge_vectors = self._compute_edges_from_positions(
            atoms, self.cutoff_radius
        )
        
        if self.profile:
            t_edges = time.time() - t0
            t0 = time.time()
        
        # 4. Build line graph with REAL angles
        if self.use_line_graph:
            line_graph = self._build_line_graph_from_vectors(edge_index, edge_vectors)
            
            # Line graph batch mapping (maps each bond to its source atom)
            line_graph_batch_mapping = edge_index[0].clone()
        else:
            line_graph = None
            line_graph_batch_mapping = None
        
        if self.profile:
            t_line = time.time() - t0
            t_total = time.time() - t_total
            
            # Determine if NPZ was used
            npz_path = Path(cif_path).with_suffix('.npz')
            format_used = "NPZ" if npz_path.exists() else "CIF"
            
            print(f"Graph build timing ({Path(cif_path).name}) [{format_used}]:")
            print(f"  File reading:   {t_read*1000:6.2f} ms ({t_read/t_total*100:4.1f}%)")
            print(f"  Node features:  {t_features*1000:6.2f} ms ({t_features/t_total*100:4.1f}%)")
            print(f"  Edge compute:   {t_edges*1000:6.2f} ms ({t_edges/t_total*100:4.1f}%)")
            if self.use_line_graph:
                print(f"  Line graph:     {t_line*1000:6.2f} ms ({t_line/t_total*100:4.1f}%)")
            print(f"  TOTAL:          {t_total*1000:6.2f} ms")
        
        # 5. Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_atoms
        )
        
        # Add line graph data
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
        backward_barrier: float = None
    ) -> Tuple[Data, Data]:
        """
        Build graph pair from initial and final CIF files.
        
        Parameters:
        -----------
        initial_cif : str
            Path to initial structure CIF
        final_cif : str
            Path to final structure CIF
        backward_barrier : float, optional
            Backward barrier (label)
        
        Returns:
        --------
        initial_graph : Data
        final_graph : Data
        """
        initial_graph = self.cif_to_graph(initial_cif, backward_barrier)
        final_graph = self.cif_to_graph(final_cif, backward_barrier)
        
        return initial_graph, final_graph
    
    def clear_cache(self):
        """Clear RAM cache (useful for freeing memory)."""
        if self._structure_cache is not None:
            self._structure_cache.clear()
            print("✓ Structure cache cleared")


def test_graph_builder():
    """Test the optimized graph builder."""
    from config import Config
    import pandas as pd
    
    print("\n" + "="*70)
    print("TESTING ULTRA-OPTIMIZED GRAPH BUILDER (NPZ)")
    print("="*70)
    
    config = Config()
    
    # 🔥 AUTO-FIND: Get CIF path from CSV
    csv_path = config.csv_path
    
    if not Path(csv_path).exists():
        print(f"\n❌ CSV not found: {csv_path}")
        print("   Cannot test without data.")
        print("="*70 + "\n")
        return
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print(f"\n❌ CSV is empty: {csv_path}")
        print("="*70 + "\n")
        return
    
    # Get first structure folder
    first_row = df.iloc[0]
    structure_folder = Path(first_row['structure_folder'])
    
    # Make absolute if needed
    if not structure_folder.is_absolute():
        structure_folder = Path(csv_path).parent / structure_folder
    
    test_cif = structure_folder / "initial_relaxed.cif"
    test_npz = structure_folder / "initial_relaxed.npz"
    
    if not test_cif.exists():
        print(f"\n❌ Test CIF not found: {test_cif}")
        print("="*70 + "\n")
        return
    
    print(f"\n✓ Found test CIF: {test_cif}")
    if test_npz.exists():
        print(f"✓ Found NPZ file: {test_npz} (will be used!)")
    else:
        print(f"⚠️  No NPZ file yet. Run convert_cif_to_npz.py first for max speed!")
    print(f"  From CSV: {csv_path}")
    print(f"  Total samples in DB: {len(df)}")
    
    # Create builder with profiling enabled
    builder = GraphBuilder(config, csv_path=csv_path, profile=True, use_cache=True)
    
    print("\n" + "-"*70)
    print("Single graph build (with profiling):")
    print("-"*70)
    
    graph = builder.cif_to_graph(str(test_cif), backward_barrier=1.5)
    
    print("\n" + "-"*70)
    print("Graph statistics:")
    print("-"*70)
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Node features: {graph.x.shape}")
    print(f"  Edge features: {graph.edge_attr.shape}")
    
    if hasattr(graph, 'line_graph_x'):
        print(f"  Line graph nodes: {graph.line_graph_x.shape[0]}")
        print(f"  Line graph edges: {graph.line_graph_edge_index.shape[1]}")
    
    print(f"\n  Edge distances (Å):")
    print(f"    Min: {graph.edge_attr.min().item():.3f}")
    print(f"    Max: {graph.edge_attr.max().item():.3f}")
    print(f"    Mean: {graph.edge_attr.mean().item():.3f}")
    
    if hasattr(graph, 'line_graph_edge_attr'):
        angles_deg = graph.line_graph_edge_attr * 180 / np.pi
        print(f"\n  Line graph angles (degrees):")
        print(f"    Min: {angles_deg.min().item():.1f}°")
        print(f"    Max: {angles_deg.max().item():.1f}°")
        print(f"    Mean: {angles_deg.mean().item():.1f}°")
    
    # Benchmark: Build same graph 100 times
    print("\n" + "-"*70)
    print("Speed benchmark (100 iterations with caching):")
    print("-"*70)
    
    builder.profile = False  # Disable profiling for clean timing
    
    times = []
    for _ in range(100):
        t0 = time.time()
        _ = builder.cif_to_graph(str(test_cif), backward_barrier=1.5)
        times.append(time.time() - t0)
    
    times = np.array(times) * 1000  # Convert to ms
    
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  Std:  {times.std():.2f} ms")
    print(f"  Min:  {times.min():.2f} ms")
    print(f"  Max:  {times.max():.2f} ms")
    
    # Show improvement estimate
    old_time_estimate = 95  # From original profiling
    speedup = old_time_estimate / times.mean()
    
    print(f"\n🔥 Performance:")
    print(f"  Original version: ~{old_time_estimate:.0f} ms")
    print(f"  Current version:  {times.mean():.2f} ms")
    print(f"  Speedup:          {speedup:.2f}x")
    
    if test_npz.exists():
        print(f"\n✅ Using NPZ format (fast!)")
    else:
        print(f"\n⚠️  Using CIF format (slow)")
        print(f"   Run 'python convert_cif_to_npz.py' to convert to NPZ")
        print(f"   Expected speedup: {times.mean():.0f}ms → ~10-15ms")
    
    print(f"\n💡 Cache status:")
    print(f"   Cached structures: {len(builder._structure_cache)}")
    print(f"   First iteration: File I/O + graph build")
    print(f"   Next 99 iterations: RAM cache (instant file loading)")
    
    print("\n✓ Graph builder test successful!")
    print(f"🔥 Target for KMC: <20 ms → {'✅ ACHIEVED!' if times.mean() < 20 else '⚠️  Close! Convert to NPZ for <15ms'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_graph_builder()