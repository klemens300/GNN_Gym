"""
Template-Based Graph Builder for Identical BCC Structures

Key Insight: All structures have identical geometry (4x4x4 BCC with vacancy).
Only the element occupancy and atomic positions change between samples.

Strategy:
1. Auto-detect elements from database
2. Build template graph ONCE (connectivity is fixed)
3. For each sample: Update node features (positions + elements from CIF)
4. Graphs created on-the-fly with REQUIRED barrier labels

Performance: ~100x faster than building individual graphs
Storage: No .pt files needed
"""

import numpy as np
import torch
from ase.build import bulk
from torch_geometric.data import Data
from pathlib import Path
import pandas as pd
import re

# Import atomic properties from separate database
from atomic_properties import ATOMIC_PROPERTIES, PROPERTY_KEYS


class TemplateGraphBuilder:
    """
    Builds graphs from template for identical crystal structures.
    
    Automatically detects elements from database.
    Assumes all structures are 4x4x4 BCC supercells with one vacancy.
    
    Node Features (7 + N):
    - Cartesian positions (3) - FROM CIF
    - Element one-hot encoding (N): Automatically detected from database
    - Atomic properties (4): radius, mass, electronegativity, valence
    
    Edge Features (1):
    - Distance between atoms (from template - fixed geometry)
    
    Label:
    - y: Energy barrier in eV (REQUIRED)
    """
    
    def __init__(self, config, csv_path=None):
        """
        Initialize with automatic element detection from database.
        
        Parameters:
        -----------
        config : Config
            Configuration with supercell_size, lattice_parameter, cutoff_radius, etc.
        csv_path : str, optional
            Path to CSV to scan for elements
            If None, uses config.csv_path
        """
        self.config = config
        self.cutoff_radius = config.cutoff_radius
        self.max_neighbors = config.max_neighbors
        
        # Auto-detect elements from database
        if csv_path is None:
            csv_path = config.csv_path
        
        self.elements = self._detect_elements_from_database(csv_path)
        print(f"✓ Detected elements from database: {self.elements}")
        
        self.element_to_idx = {elem: idx for idx, elem in enumerate(self.elements)}
        
        # Validate: all elements must have atomic properties
        missing_elements = []
        for elem in self.elements:
            if elem not in ATOMIC_PROPERTIES:
                missing_elements.append(elem)
        
        if missing_elements:
            available = ', '.join(sorted(ATOMIC_PROPERTIES.keys()))
            raise ValueError(
                f"Elements found in database but no atomic properties: {missing_elements}\n"
                f"Available elements: {available}\n"
                f"Add properties for these elements to atomic_properties.py"
            )
        
        # Atomic properties
        self.atomic_properties = ATOMIC_PROPERTIES
        self.property_keys = PROPERTY_KEYS
        
        # Build template once (only connectivity!)
        print(f"Building template graph...")
        self._build_template()
        print(f"✓ Template created: {self.template_num_nodes} nodes, "
              f"{self.template_edge_index.shape[1]} edges")
        print(f"✓ Node features: {3 + len(self.elements) + 4} "
              f"(pos: 3, one-hot: {len(self.elements)}, props: 4)")
    
    def _detect_elements_from_database(self, csv_path: str) -> list:
        """
        Detect all unique elements from database composition strings.
        
        Parses composition strings like "Mo25Nb50Ta25" or "Mo0Nb3O17Ta42W38"
        to extract all element symbols.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV database
        
        Returns:
        --------
        elements : list of str
            Sorted list of unique element symbols
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                f"Generate data first with: python bulk_calculation.py"
            )
        
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            raise ValueError(
                f"CSV is empty: {csv_path}\n"
                f"Generate data first with: python bulk_calculation.py"
            )
        
        # Extract all unique elements from composition strings
        all_elements = set()
        
        for comp_string in df['composition_string'].unique():
            # Extract element symbols (e.g., "Mo25Nb50Ta25" → ["Mo", "Nb", "Ta"])
            # Regex: capital letter followed by optional lowercase letter
            elements = re.findall(r'[A-Z][a-z]?', comp_string)
            all_elements.update(elements)
        
        # Sort for consistency (alphabetical)
        elements_sorted = sorted(all_elements)
        
        return elements_sorted
    
    def _build_template(self):
        """
        Create template graph with fixed geometry.
        
        Template contains ONLY connectivity (edge_index and edge_attr).
        Node positions will come from CIF files!
        
        This NEVER changes between samples!
        """
        size = self.config.supercell_size
        lattice_param = self.config.lattice_parameter
        
        # Create ideal BCC supercell (without vacancy)
        crystal = bulk('Fe', crystalstructure='bcc', a=lattice_param, cubic=True)
        supercell = crystal.repeat([size, size, size])
        
        # Get positions (only needed for building template connectivity)
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
        
        # Build edges using periodic boundary conditions
        edge_index, edge_attr = self._build_edges(
            positions_with_vacancy.numpy(), 
            cell
        )
        
        # Store template (ONLY connectivity, not positions!)
        self.template_edge_index = edge_index  # [2, E]
        self.template_edge_attr = edge_attr  # [E, 1]
        self.template_num_nodes = len(positions_with_vacancy)
        self.template_cell = cell
    
    def _build_edges(self, positions: np.ndarray, cell: np.ndarray):
        """
        Build edge connectivity with periodic boundary conditions.
        
        Parameters:
        -----------
        positions : np.ndarray [N, 3]
            Atomic positions (only for template building)
        cell : np.ndarray [3, 3]
            Unit cell vectors
        
        Returns:
        --------
        edge_index : torch.Tensor [2, E]
            Source and target node indices
        edge_attr : torch.Tensor [E, 1]
            Edge distances
        """
        src_list = []
        dst_list = []
        dist_list = []
        
        # For each atom, find neighbors within cutoff
        for i, pos_i in enumerate(positions):
            distances = []
            
            for j, pos_j in enumerate(positions):
                if i == j:
                    continue
                
                # Minimum image convention (PBC)
                diff = pos_j - pos_i
                for k in range(3):
                    while diff[k] > cell[k, k] / 2:
                        diff[k] -= cell[k, k]
                    while diff[k] < -cell[k, k] / 2:
                        diff[k] += cell[k, k]
                
                dist = np.linalg.norm(diff)
                
                if dist < self.cutoff_radius:
                    distances.append((dist, j))
            
            # Sort by distance and take closest max_neighbors
            distances.sort()
            neighbors = distances[:self.max_neighbors]
            
            for dist, j in neighbors:
                src_list.append(i)
                dst_list.append(j)
                dist_list.append(dist)
        
        if len(src_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(dist_list, dtype=torch.float).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def _read_elements_and_positions_from_cif(self, cif_path: str) -> tuple:
        """
        Read element sequence AND positions from CIF file.
        
        Parameters:
        -----------
        cif_path : str
            Path to CIF file
        
        Returns:
        --------
        elements : list of str
            Element symbols in order
        positions : np.ndarray [N, 3]
            Atomic Cartesian positions
        """
        from pymatgen.io.cif import CifParser
        
        parser = CifParser(cif_path)
        structure = parser.parse_structures(primitive=False)[0]
        
        # Get elements and positions
        elements = [site.species_string for site in structure]
        positions = np.array([site.coords for site in structure])
        
        return elements, positions
    
    def _get_node_features(self, elements: list, positions: np.ndarray) -> torch.Tensor:
        """
        Create node features from element list and positions.
        
        Features (7 + N total):
        - Cartesian positions (3) - FROM CIF (changes between samples!)
        - Element one-hot encoding (N) - dynamically sized
        - Atomic properties (4) - radius, mass, electronegativity, valence
        
        Parameters:
        -----------
        elements : list of str
            Element symbols for each node
        positions : np.ndarray [N, 3]
            Atomic positions FROM CIF
        
        Returns:
        --------
        node_features : torch.Tensor [N, 7+N]
        """
        features = []
        
        for i, element in enumerate(elements):
            # Position from CIF (CHANGES between initial/final!)
            pos = torch.FloatTensor(positions[i])
            
            # One-hot encoding (dynamically sized!)
            one_hot = torch.zeros(len(self.elements))
            if element in self.element_to_idx:
                one_hot[self.element_to_idx[element]] = 1
            else:
                raise ValueError(
                    f"Unknown element: {element}\n"
                    f"Element not in database: {self.elements}\n"
                    f"This should not happen if database was scanned correctly!"
                )
            
            # Atomic properties (4 features)
            props = self.atomic_properties[element]
            atomic_props = torch.tensor([props[key] for key in self.property_keys])
            
            # Concatenate: [pos(3), one_hot(N), props(4)] = 7+N features
            node_feat = torch.cat([pos, one_hot, atomic_props])
            features.append(node_feat)
        
        return torch.stack(features)
    
    def cif_to_graph(self, cif_path: str, barrier: float) -> Data:
        """
        Convert CIF file to graph using template.
        
        Parameters:
        -----------
        cif_path : str
            Path to CIF file
        barrier : float
            Energy barrier in eV (REQUIRED)
        
        Returns:
        --------
        graph : torch_geometric.data.Data
            PyTorch Geometric graph with:
            - x: node features [N, 7+N_elements]
            - edge_index: connectivity [2, E] (from template)
            - edge_attr: distances [E, 1] (from template)
            - y: barrier label [1] (REQUIRED)
            - num_nodes: N
        """
        # Read element sequence AND positions from CIF
        elements, positions = self._read_elements_and_positions_from_cif(cif_path)
        
        # Validate
        if len(elements) != self.template_num_nodes:
            raise ValueError(
                f"Structure has {len(elements)} atoms, "
                f"but template expects {self.template_num_nodes}"
            )
        
        # Create node features (positions AND elements change between samples!)
        x = self._get_node_features(elements, positions)
        
        # Create graph with REQUIRED label
        graph = Data(
            x=x,
            edge_index=self.template_edge_index.clone(),
            edge_attr=self.template_edge_attr.clone(),
            y=torch.tensor([barrier], dtype=torch.float32),
            num_nodes=self.template_num_nodes
        )
        
        return graph
    
    def build_pair_graph(
        self, 
        initial_cif: str, 
        final_cif: str,
        backward_barrier: float
    ) -> tuple:
        """
        Build graph pair for transition barrier prediction.
        
        Both graphs get the same label (backward_barrier) because
        the GNN model learns to predict the barrier from BOTH structures.
        
        Parameters:
        -----------
        initial_cif : str
            Path to initial structure CIF
        final_cif : str
            Path to final structure CIF
        backward_barrier : float
            Backward barrier in eV (REQUIRED)
        
        Returns:
        --------
        initial_graph : Data
            Initial structure with y = backward_barrier
        final_graph : Data
            Final structure with y = backward_barrier
        
        Note:
        -----
        Both graphs share the SAME label because the model predicts
        the barrier for the transition (initial → final).
        """
        initial_graph = self.cif_to_graph(initial_cif, barrier=backward_barrier)
        final_graph = self.cif_to_graph(final_cif, barrier=backward_barrier)
        
        return initial_graph, final_graph


# Example usage and testing
if __name__ == "__main__":
    from config import Config
    
    print("="*70)
    print("TEMPLATE GRAPH BUILDER TEST")
    print("="*70)
    
    # Initialize
    config = Config()
    builder = TemplateGraphBuilder(config)
    
    # Test with actual data (if available)
    if Path(config.csv_path).exists():
        df = pd.read_csv(config.csv_path)
        
        if len(df) > 0:
            sample = df.iloc[0]
            structure_folder = Path(sample['structure_folder'])
            initial_cif = structure_folder / "initial_relaxed.cif"
            final_cif = structure_folder / "final_relaxed.cif"
            
            if initial_cif.exists() and final_cif.exists():
                print(f"\nTesting on: {initial_cif.parent.name}")
                
                # Get barriers from CSV
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
                
                print(f"\nFinal graph:")
                print(f"  Nodes: {final_graph.num_nodes}")
                print(f"  Node features: {final_graph.x.shape}")
                print(f"  Edges: {final_graph.edge_index.shape[1]}")
                print(f"  Label: {final_graph.y.item():.4f} eV")
                
                # Check position differences
                pos_diff = (initial_graph.x[:, :3] - final_graph.x[:, :3]).abs().sum().item()
                print(f"\n✓ Position difference: {pos_diff:.2f}")
                
                if pos_diff > 0:
                    print("✓ Initial and final structures have different positions!")
                
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
                print(f"\n✗ CIF files not found")
        else:
            print("\n✗ No data in CSV")
    else:
        print(f"\n✗ CSV not found: {config.csv_path}")
    
    print("\n" + "="*70)