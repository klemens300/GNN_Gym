"""
Template-Based Graph Builder with ALIGNN-Style Line Graph

Architecture:
1. Atom Graph: Atoms connected by edges (bonds)
2. Line Graph: Edges connected by bond angles
   - Line graph nodes = atom graph edges
   - Line graph edges = angles between bonds sharing an atom

Features:
- Atom graph: Node features (elements + properties), Edge features (distances)
- Line graph: Node features (bond vectors), Edge features (angles)
- Line graph spatial cutoff: Only connect bonds if end-atoms are within cutoff

ROTATION INVARIANCE:
- Cartesian positions removed from atom node features
- Bond vectors in line graph are RELATIVE (rotation invariant)
- Angles are inherently rotation invariant
"""

import numpy as np
import torch
from ase.build import bulk
from torch_geometric.data import Data
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

from atomic_properties import ATOMIC_PROPERTIES, PROPERTY_KEYS


class TemplateGraphBuilder:
    """
    Builds atom graph + line graph from template.
    
    Atom Graph:
    - Nodes: Atoms with element + properties
    - Edges: Bonds with distances
    
    Line Graph:
    - Nodes: Bonds (atom graph edges) with bond vectors
    - Edges: Angles between bonds sharing an atom (with spatial cutoff)
    """
    
    def __init__(self, config, csv_path=None):
        """
        Initialize with automatic element detection.
        
        Parameters:
        -----------
        config : Config
            Configuration with graph parameters
        csv_path : str, optional
            Path to CSV to scan for elements
        """
        self.config = config
        self.cutoff_radius = config.cutoff_radius
        self.max_neighbors = config.max_neighbors
        self.use_line_graph = getattr(config, 'use_line_graph', True)
        
        # Auto-detect elements
        if csv_path is None:
            csv_path = config.csv_path
        
        self.elements = self._detect_elements_from_database(csv_path)
        print(f"✓ Detected elements: {self.elements}")
        
        self.element_to_idx = {elem: idx for idx, elem in enumerate(self.elements)}
        
        # Validate atomic properties
        self._validate_atomic_properties()
        
        # Build template
        self._build_template()
    
    def _detect_elements_from_database(self, csv_path: str) -> list:
        """Detect elements from CSV composition strings."""
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            raise ValueError(f"CSV is empty: {csv_path}")
        
        all_elements = set()
        for comp_string in df['composition_string'].unique():
            elements = re.findall(r'[A-Z][a-z]?', comp_string)
            all_elements.update(elements)
        
        return sorted(all_elements)
    
    def _validate_atomic_properties(self):
        """Check if all elements have atomic properties."""
        missing = [e for e in self.elements if e not in ATOMIC_PROPERTIES]
        if missing:
            available = ', '.join(sorted(ATOMIC_PROPERTIES.keys()))
            raise ValueError(
                f"Missing atomic properties for: {missing}\n"
                f"Available: {available}\n"
                f"Add to atomic_properties.py"
            )
    
    def _build_template(self):
        """
        Build atom graph + line graph template.
        
        Templates contain ONLY connectivity and geometric features.
        Element-specific features come from CIF files.
        """
        size = self.config.supercell_size
        lattice_param = self.config.lattice_parameter
        
        print(f"\n{'='*70}")
        print(f"BUILDING TEMPLATE GRAPH")
        print(f"{'='*70}")
        print(f"Supercell size: {size}×{size}×{size}")
        print(f"Lattice parameter: {lattice_param} Å")
        
        # Create BCC supercell
        crystal = bulk('Fe', crystalstructure='bcc', a=lattice_param, cubic=True)
        supercell = crystal.repeat([size, size, size])
        
        positions = torch.FloatTensor(supercell.positions)
        cell = supercell.cell.array
        
        # Find center atom (will be vacancy)
        supercell_center = np.diagonal(cell) / 2.0
        distances = np.linalg.norm(positions.numpy() - supercell_center, axis=1)
        center_idx = np.argmin(distances)
        
        # Remove center atom (create vacancy)
        mask = torch.ones(len(positions), dtype=torch.bool)
        mask[center_idx] = False
        positions_with_vacancy = positions[mask]
        
        print(f"\nAtom Graph Construction:")
        print(f"  Total atoms (with vacancy): {len(positions_with_vacancy)}")
        print(f"  Cutoff radius: {self.cutoff_radius} Å")
        print(f"  Max neighbors: {self.max_neighbors}")
        
        # Build atom graph
        edge_index, edge_attr, edge_vectors = self._build_atom_graph_edges(
            positions_with_vacancy.numpy(),
            cell
        )
        
        # Print atom graph statistics
        num_edges = edge_index.shape[1]
        print(f"  → Created {num_edges} edges")
        if num_edges > 0:
            edges_per_node = num_edges / len(positions_with_vacancy)
            print(f"  → Average neighbors per atom: {edges_per_node:.2f}")
            
            # Distance statistics
            distances = edge_attr.squeeze().numpy()
            print(f"  → Edge distances: [{distances.min():.3f}, {distances.max():.3f}] Å")
            print(f"  → Mean edge distance: {distances.mean():.3f} Å")
        
        # Store atom graph template
        self.template_edge_index = edge_index
        self.template_edge_attr = edge_attr  # [E, 1] distances
        self.template_num_nodes = len(positions_with_vacancy)
        self.template_cell = cell
        
        # Build line graph (if enabled)
        if self.use_line_graph:
            print(f"\nLine Graph Construction:")
            line_cutoff = getattr(self.config, 'line_graph_cutoff', None)
            if line_cutoff is None:
                print(f"  Cutoff: None (connecting ALL angles)")
            else:
                print(f"  Spatial cutoff: {line_cutoff} Å")
            
            line_graph_edge_index, line_graph_edge_attr, line_graph_node_attr = \
                self._build_line_graph(edge_index, edge_vectors)
            
            # Print line graph statistics
            num_line_nodes = num_edges  # Line graph nodes = atom graph edges
            num_line_edges = line_graph_edge_index.shape[1]
            
            print(f"  → Line graph nodes: {num_line_nodes} (= atom graph edges)")
            print(f"  → Line graph edges: {num_line_edges}")
            
            if num_line_edges > 0 and num_line_nodes > 0:
                angles_per_bond = num_line_edges / num_line_nodes
                print(f"  → Average angles per bond: {angles_per_bond:.2f}")
                
                # Angle statistics
                angles_rad = line_graph_edge_attr.squeeze().numpy()
                angles_deg = np.degrees(angles_rad)
                print(f"  → Angles: [{angles_deg.min():.1f}°, {angles_deg.max():.1f}°]")
                print(f"  → Mean angle: {angles_deg.mean():.1f}°")
            
            self.template_line_graph_edge_index = line_graph_edge_index
            self.template_line_graph_edge_attr = line_graph_edge_attr  # [LE, 1] angles
            self.template_line_graph_node_attr = line_graph_node_attr  # [E, 3] bond vectors
            self.template_line_graph_num_nodes = num_line_nodes
        else:
            print(f"\nLine Graph: DISABLED")
            self.template_line_graph_edge_index = None
            self.template_line_graph_edge_attr = None
            self.template_line_graph_node_attr = None
            self.template_line_graph_num_nodes = 0
        
        print(f"{'='*70}")
        print(f"TEMPLATE CONSTRUCTION COMPLETE")
        print(f"{'='*70}\n")
    
    def _build_atom_graph_edges(self, positions: np.ndarray, cell: np.ndarray):
        """
        Build atom graph edges with distances.
        
        Returns:
        --------
        edge_index : torch.Tensor [2, E]
        edge_attr : torch.Tensor [E, 1] - distances
        edge_vectors : torch.Tensor [E, 3] - bond vectors (for line graph)
        """
        src_list = []
        dst_list = []
        dist_list = []
        vec_list = []
        
        for i, pos_i in enumerate(positions):
            distances = []
            
            for j, pos_j in enumerate(positions):
                if i == j:
                    continue
                
                # Minimum image convention
                diff = pos_j - pos_i
                for k in range(3):
                    while diff[k] > cell[k, k] / 2:
                        diff[k] -= cell[k, k]
                    while diff[k] < -cell[k, k] / 2:
                        diff[k] += cell[k, k]
                
                dist = np.linalg.norm(diff)
                
                if dist < self.cutoff_radius:
                    distances.append((dist, j, diff))
            
            # Sort and take max_neighbors
            distances.sort()
            neighbors = distances[:self.max_neighbors]
            
            for dist, j, vec in neighbors:
                src_list.append(i)
                dst_list.append(j)
                dist_list.append(dist)
                vec_list.append(vec)
        
        if len(src_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
            edge_vectors = torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(dist_list, dtype=torch.float).unsqueeze(1)
            edge_vectors = torch.tensor(vec_list, dtype=torch.float)
        
        return edge_index, edge_attr, edge_vectors
    
    def _build_line_graph(self, edge_index: torch.Tensor, edge_vectors: torch.Tensor):
        """
        Build line graph from atom graph with spatial cutoff.
        
        Line Graph:
        - Nodes: Atom graph edges (bonds)
        - Edges: Pairs of bonds that share an atom AND whose end-atoms are within cutoff
        - Node features: Bond vectors (3D, rotation invariant)
        - Edge features: Angles between bonds
        
        NEW: Spatial cutoff - only connect bonds if their end-atoms are close enough.
        
        Parameters:
        -----------
        edge_index : torch.Tensor [2, E]
            Atom graph edges
        edge_vectors : torch.Tensor [E, 3]
            Bond vectors (from src to dst atom)
        
        Returns:
        --------
        line_edge_index : torch.Tensor [2, LE]
            Line graph edges
        line_edge_attr : torch.Tensor [LE, 1]
            Angles in radians
        line_node_attr : torch.Tensor [E, 3]
            Normalized bond vectors
        """
        num_edges = edge_index.shape[1]
        
        # Get line graph cutoff from config
        line_cutoff = getattr(self.config, 'line_graph_cutoff', None)
        
        # Group edges by source atom
        edges_by_src = defaultdict(list)
        for edge_idx in range(num_edges):
            src = edge_index[0, edge_idx].item()
            dst = edge_index[1, edge_idx].item()
            edges_by_src[src].append((edge_idx, dst))
        
        # Build line graph edges
        line_src = []
        line_dst = []
        angles = []
        
        for src_atom, edge_list in edges_by_src.items():
            # Connect all pairs of edges from same source atom
            for i, (edge_i, dst_i) in enumerate(edge_list):
                for j, (edge_j, dst_j) in enumerate(edge_list):
                    if i >= j:  # Avoid duplicates and self-loops
                        continue
                    
                    # ✅ NEW: Check spatial cutoff
                    if line_cutoff is not None:
                        # Distance between end-atoms (dst_i and dst_j)
                        vec_i = edge_vectors[edge_i]
                        vec_j = edge_vectors[edge_j]
                        
                        # Both vectors start at src_atom, so:
                        # Position of dst_i relative to src = vec_i
                        # Position of dst_j relative to src = vec_j
                        # Distance between dst_i and dst_j:
                        end_to_end_dist = torch.norm(vec_i - vec_j).item()
                        
                        if end_to_end_dist > line_cutoff:
                            continue  # Skip this line graph edge
                    
                    # Calculate angle between bond vectors
                    vec_i = edge_vectors[edge_i]
                    vec_j = edge_vectors[edge_j]
                    
                    # Normalize
                    norm_i = torch.norm(vec_i)
                    norm_j = torch.norm(vec_j)
                    
                    if norm_i > 1e-8 and norm_j > 1e-8:
                        vec_i_norm = vec_i / norm_i
                        vec_j_norm = vec_j / norm_j
                        
                        # Cosine of angle
                        cos_angle = torch.clamp(torch.dot(vec_i_norm, vec_j_norm), -1.0, 1.0)
                        angle = torch.acos(cos_angle)
                        
                        # Add both directions (undirected line graph)
                        line_src.extend([edge_i, edge_j])
                        line_dst.extend([edge_j, edge_i])
                        angles.extend([angle, angle])
        
        # Convert to tensors
        if len(line_src) == 0:
            line_edge_index = torch.zeros((2, 0), dtype=torch.long)
            line_edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            line_edge_index = torch.tensor([line_src, line_dst], dtype=torch.long)
            line_edge_attr = torch.tensor(angles, dtype=torch.float).unsqueeze(1)
        
        # Line graph node features: normalized bond vectors
        edge_norms = torch.norm(edge_vectors, dim=1, keepdim=True)
        line_node_attr = edge_vectors / (edge_norms + 1e-8)
        
        return line_edge_index, line_edge_attr, line_node_attr
    
    def _read_elements_and_positions_from_cif(self, cif_path: str):
        """Read elements and positions from CIF."""
        from pymatgen.io.cif import CifParser
        
        parser = CifParser(cif_path)
        structure = parser.parse_structures(primitive=False)[0]
        
        elements = [site.species_string for site in structure]
        positions = np.array([site.coords for site in structure])
        
        return elements, positions
    
    def _get_node_features(self, elements: list, positions: np.ndarray):
        """
        Create atom node features (rotation invariant).
        
        Features:
        - Element one-hot (N)
        - Atomic properties (4)
        
        NOTE: Positions NOT included for rotation invariance.
        """
        features = []
        
        for element in elements:
            # One-hot
            one_hot = torch.zeros(len(self.elements))
            if element in self.element_to_idx:
                one_hot[self.element_to_idx[element]] = 1
            else:
                raise ValueError(f"Unknown element: {element}")
            
            # Atomic properties
            props = ATOMIC_PROPERTIES[element]
            atomic_props = torch.tensor([props[key] for key in PROPERTY_KEYS])
            
            # Concatenate
            node_feat = torch.cat([one_hot, atomic_props])
            features.append(node_feat)
        
        return torch.stack(features)
    
    def cif_to_graph(self, cif_path: str, barrier: float):
        """
        Convert CIF to graph pair (atom graph + line graph).
        
        Returns:
        --------
        graph : Data
            PyG Data with:
            - x: atom node features [N, N_elements+4]
            - edge_index: atom edges [2, E]
            - edge_attr: distances [E, 1]
            - y: barrier [1]
            
            If use_line_graph:
            - line_graph_x: bond node features [E, 3]
            - line_graph_edge_index: angle edges [2, LE]
            - line_graph_edge_attr: angles [LE, 1]
        """
        elements, positions = self._read_elements_and_positions_from_cif(cif_path)
        
        if len(elements) != self.template_num_nodes:
            raise ValueError(
                f"Structure has {len(elements)} atoms, "
                f"expected {self.template_num_nodes}"
            )
        
        # Atom graph
        x = self._get_node_features(elements, positions)
        
        graph = Data(
            x=x,
            edge_index=self.template_edge_index.clone(),
            edge_attr=self.template_edge_attr.clone(),
            y=torch.tensor([barrier], dtype=torch.float32),
            num_nodes=self.template_num_nodes
        )
        
        # Add line graph
        if self.use_line_graph:
            graph.line_graph_x = self.template_line_graph_node_attr.clone()
            graph.line_graph_edge_index = self.template_line_graph_edge_index.clone()
            graph.line_graph_edge_attr = self.template_line_graph_edge_attr.clone()
            graph.line_graph_num_nodes = self.template_line_graph_num_nodes
        
        return graph
    
    def build_pair_graph(self, initial_cif: str, final_cif: str, backward_barrier: float):
        """Build graph pair for transition."""
        initial_graph = self.cif_to_graph(initial_cif, barrier=backward_barrier)
        final_graph = self.cif_to_graph(final_cif, barrier=backward_barrier)
        
        return initial_graph, final_graph


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from config import Config
    
    print("="*70)
    print("ALIGNN-STYLE LINE GRAPH BUILDER TEST")
    print("="*70)
    
    config = Config()
    builder = TemplateGraphBuilder(config)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Elements detected: {builder.elements}")
    print(f"Node features: {len(builder.elements) + 4} "
          f"(one-hot: {len(builder.elements)}, props: 4)")
    print(f"\nAtom Graph:")
    print(f"  Nodes: {builder.template_num_nodes}")
    print(f"  Edges: {builder.template_edge_index.shape[1]}")
    
    if builder.use_line_graph:
        print(f"\nLine Graph:")
        print(f"  Nodes: {builder.template_line_graph_num_nodes}")
        print(f"  Edges: {builder.template_line_graph_edge_index.shape[1]}")
    
    print(f"{'='*70}\n")
    
    # Test with actual data if available
    if Path(config.csv_path).exists():
        df = pd.read_csv(config.csv_path)
        
        if len(df) > 0:
            sample = df.iloc[0]
            structure_folder = Path(sample['structure_folder'])
            
            # Make absolute if relative
            if not structure_folder.is_absolute():
                csv_parent = Path(config.csv_path).parent.resolve()
                structure_folder = csv_parent / structure_folder
            
            initial_cif = structure_folder / "initial_relaxed.cif"
            final_cif = structure_folder / "final_relaxed.cif"
            
            if initial_cif.exists() and final_cif.exists():
                print(f"\nTesting on: {structure_folder.name}")
                
                backward_barrier = sample['backward_barrier_eV']
                print(f"Backward barrier: {backward_barrier:.4f} eV")
                
                # Build graph pair
                import time
                start = time.time()
                initial_graph, final_graph = builder.build_pair_graph(
                    str(initial_cif),
                    str(final_cif),
                    backward_barrier=backward_barrier
                )
                elapsed = (time.time() - start) * 1000
                
                print(f"\n✓ Graph pair built in {elapsed:.2f}ms!")
                print(f"\nInitial graph:")
                print(f"  Nodes: {initial_graph.num_nodes}")
                print(f"  Node features: {initial_graph.x.shape}")
                print(f"  Edges: {initial_graph.edge_index.shape[1]}")
                print(f"  Label: {initial_graph.y.item():.4f} eV")
                
                if builder.use_line_graph:
                    print(f"  Line graph nodes: {initial_graph.line_graph_num_nodes}")
                    print(f"  Line graph edges: {initial_graph.line_graph_edge_index.shape[1]}")
                
                print(f"\nFinal graph:")
                print(f"  Nodes: {final_graph.num_nodes}")
                print(f"  Node features: {final_graph.x.shape}")
                print(f"  Edges: {final_graph.edge_index.shape[1]}")
                print(f"  Label: {final_graph.y.item():.4f} eV")
                
                if builder.use_line_graph:
                    print(f"  Line graph nodes: {final_graph.line_graph_num_nodes}")
                    print(f"  Line graph edges: {final_graph.line_graph_edge_index.shape[1]}")
                
                # Benchmark
                print("\nBenchmark: Building 100 graph pairs...")
                start = time.time()
                for _ in range(100):
                    _ = builder.build_pair_graph(
                        str(initial_cif),
                        str(final_cif),
                        backward_barrier=backward_barrier
                    )
                elapsed = (time.time() - start) * 1000
                print(f"✓ Average time per pair: {elapsed/100:.2f}ms")
                
            else:
                print(f"\n✗ CIF files not found:")
                print(f"  {initial_cif}")
                print(f"  {final_cif}")
        else:
            print("\n✗ No data in CSV")
    else:
        print(f"\n✗ CSV not found: {config.csv_path}")
    
    print("\n" + "="*70)