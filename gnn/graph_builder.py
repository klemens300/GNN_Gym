"""
Graph Builder for Diffusion Barrier Prediction with Atom Embeddings

Builds PyG graphs from CIF/NPZ files with:
- Element indices for learned embeddings
- Atomic properties as separate features
- RBF-expanded edge features (replaces scalar distances)
- Relaxation progress for weighted loss training
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

from gnn.atomic_properties import get_atomic_properties


def rbf_expansion(distances: np.ndarray, n_gaussians: int, cutoff: float) -> np.ndarray:
    """
    Expand scalar distances into a Gaussian radial basis function (RBF) vector.

    Gaussian centers are distributed uniformly between 0 and cutoff.
    The width of each Gaussian is set to the spacing between centers,
    so neighboring Gaussians overlap smoothly.

    Parameters:
    -----------
    distances : np.ndarray, shape [n_edges]
        Interatomic distances in Angstrom.
    n_gaussians : int
        Number of Gaussian basis functions (= output feature dimension per edge).
    cutoff : float
        Maximum distance in Angstrom. Should match the graph cutoff radius.

    Returns:
    --------
    np.ndarray, shape [n_edges, n_gaussians]
        RBF-expanded edge features. Values are in [0, 1].
    """
    # Centers equally spaced from 0 to cutoff
    centers = np.linspace(0.0, cutoff, n_gaussians)  # [n_gaussians]

    # Width = spacing between centers
    width = centers[1] - centers[0] if n_gaussians > 1 else cutoff

    # Gaussian: exp(-(d - center)^2 / width^2)
    distances = distances[:, np.newaxis]  # [n_edges, 1]
    diff = distances - centers[np.newaxis, :]  # [n_edges, n_gaussians]
    return np.exp(-(diff ** 2) / (width ** 2))  # [n_edges, n_gaussians]


class GraphBuilder:
    """
    Build PyG graphs from CIF/NPZ files with learned atom embeddings
    and RBF-expanded edge features.
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

        # RBF parameters
        self.rbf_num_gaussians = getattr(config, 'rbf_num_gaussians', 64)
        self.rbf_cutoff = getattr(config, 'rbf_cutoff', config.cutoff_radius)
        
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
        print("GRAPH BUILDER INITIALIZED (ATOM EMBEDDINGS + RBF + PROGRESS)")
        print("="*70)
        print(f"Elements: {self.elements}")
        print(f"Element indices: {self.element_to_idx}")
        print(f"Cutoff radius: {self.cutoff_radius} Angstrom")
        print(f"Max neighbors: {self.max_neighbors}")
        print(f"RBF Gaussians: {self.rbf_num_gaussians}")
        print(f"RBF Cutoff: {self.rbf_cutoff} Angstrom")
        print(f"Line graph: {self.use_line_graph}")
        if self.use_line_graph:
            print(f"Line graph cutoff: {self.line_graph_cutoff} Angstrom")
        print(f"Atomic properties: {self.use_atomic_properties}")
        print(f"RAM caching: {self.use_cache}")
        print(f"Profiling: {self.profile}")
        print("="*70 + "\n")
    
    def _detect_elements_from_csv(self, csv_path: str) -> List[str]:
        """Detect unique elements from CSV database."""
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
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
        
        warnings.warn("Could not detect elements, using config")
        return getattr(self.config, 'elements', ['Mo', 'Nb', 'Ta', 'W'])
    
    def _read_structure_from_npz(self, npz_path: str) -> Atoms:
        """
        Read structure from NPZ file (fast).
        Also extracts 'progress' scalar if available.
        """
        data = np.load(npz_path)
        atoms = Atoms(
            numbers=data['numbers'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc']
        )
        
        if 'progress' in data:
            atoms.info['relax_progress'] = float(data['progress'][0])
        else:
            # Default to 1.0 (fully relaxed) for legacy files
            atoms.info['relax_progress'] = 1.0
            
        return atoms
    
    def _read_structure_from_file(self, file_path: str) -> Atoms:
        """Read structure with NPZ fallback and optional caching."""
        if self.use_cache and file_path in self._structure_cache:
            return self._structure_cache[file_path].copy()
        
        npz_path = Path(file_path).with_suffix('.npz')
        if npz_path.exists():
            try:
                atoms = self._read_structure_from_npz(str(npz_path))
                if self.use_cache:
                    self._structure_cache[file_path] = atoms
                return atoms.copy()
            except Exception as e:
                warnings.warn(f"Failed to read NPZ {npz_path}: {e}, falling back to CIF")
        
        atoms = ase_read(file_path)
        atoms.info['relax_progress'] = 1.0
        
        try:
            self._save_structure_as_npz(atoms, str(npz_path))
        except Exception as e:
            warnings.warn(f"Could not create NPZ file {npz_path}: {e}")
        
        if self.use_cache:
            self._structure_cache[file_path] = atoms
        
        return atoms.copy()

    def _save_structure_as_npz(self, atoms: Atoms, npz_path: str):
        """Save ASE atoms as NPZ for faster loading."""
        progress = atoms.info.get('relax_progress', 1.0)
        np.savez_compressed(
            npz_path,
            positions=atoms.positions,
            numbers=atoms.numbers,
            cell=atoms.cell.array,
            pbc=atoms.pbc,
            progress=np.array([progress], dtype=np.float32)
        )
    
    def _compute_edges_from_positions(
        self,
        atoms: Atoms,
        cutoff: float
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute edges with RBF-expanded distances using ASE NeighborList.

        Returns:
        --------
        edge_index : LongTensor [2, n_edges]
        edge_attr  : FloatTensor [n_edges, rbf_num_gaussians]  (RBF features)
        edge_vectors : FloatTensor [n_edges, 3]               (for line graph)
        """
        n_atoms = len(atoms)
        
        cutoffs = [cutoff / 2] * n_atoms
        nl = NeighborList(
            cutoffs,
            skin=0.0,
            self_interaction=False,
            bothways=True
        )
        nl.update(atoms)
        
        edge_list = []
        distance_list = []
        vector_list = []
        
        for i in range(n_atoms):
            indices, offsets = nl.get_neighbors(i)
            
            if len(indices) == 0:
                continue
            
            pos_i = atoms.positions[i]
            pos_j_all = atoms.positions[indices] + offsets @ atoms.cell
            vecs = pos_j_all - pos_i
            dists = np.linalg.norm(vecs, axis=1)
            
            mask = dists < cutoff
            indices_filtered = indices[mask]
            dists_filtered = dists[mask]
            vecs_filtered = vecs[mask]
            
            if len(indices_filtered) == 0:
                continue
            
            if len(indices_filtered) > self.max_neighbors:
                sorted_idx = np.argsort(dists_filtered)[:self.max_neighbors]
                indices_filtered = indices_filtered[sorted_idx]
                dists_filtered = dists_filtered[sorted_idx]
                vecs_filtered = vecs_filtered[sorted_idx]
            
            for j, dist, vec in zip(indices_filtered, dists_filtered, vecs_filtered):
                edge_list.append([i, j])
                distance_list.append(dist)
                vector_list.append(vec)
        
        if len(edge_list) == 0:
            warnings.warn(f"No edges found with cutoff {cutoff} Angstrom! Using dummy edge.")
            edge_index = torch.LongTensor([[0], [1]])
            # Dummy RBF features for a distance equal to cutoff
            dummy_rbf = rbf_expansion(np.array([cutoff]), self.rbf_num_gaussians, self.rbf_cutoff)
            edge_attr = torch.FloatTensor(dummy_rbf)
            edge_vectors = torch.FloatTensor([[0, 0, cutoff]])
        else:
            distances_np = np.array(distance_list, dtype=np.float32)

            # Expand scalar distances to RBF vectors
            rbf_features = rbf_expansion(distances_np, self.rbf_num_gaussians, self.rbf_cutoff)

            edge_index = torch.from_numpy(np.array(edge_list, dtype=np.int64)).T
            edge_attr = torch.from_numpy(rbf_features.astype(np.float32))
            edge_vectors = torch.from_numpy(np.array(vector_list, dtype=np.float32))
        
        return edge_index, edge_attr, edge_vectors
    
    def _build_line_graph_from_vectors(
        self,
        edge_index: torch.LongTensor,
        edge_vectors: torch.FloatTensor
    ) -> Dict:
        """
        Build line graph from edge vectors with vectorized computation.
        Line graph nodes = bonds; line graph edges = bond pairs sharing an atom.
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
            edge_vectors_norm,  # [n_edges, 3]
            edge_lengths        # [n_edges, 1]
        ], dim=1)  # [n_edges, 4]
        
        source_atoms = edge_index[0]
        unique_sources = torch.unique(source_atoms)
        
        line_edge_list = []
        line_angle_list = []
        
        for source in unique_sources:
            mask = source_atoms == source
            edge_idx = torch.where(mask)[0]
            
            n_local = len(edge_idx)
            if n_local < 2:
                continue
            
            vecs = edge_vectors[edge_idx]
            norms = torch.norm(vecs, dim=1, keepdim=True)
            vecs_norm = vecs / (norms + 1e-8)
            
            cos_angles = vecs_norm @ vecs_norm.T
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            angles = torch.acos(cos_angles)
            
            i_idx, j_idx = torch.triu_indices(n_local, n_local, offset=1)
            
            edge_i = edge_idx[i_idx]
            edge_j = edge_idx[j_idx]
            angle_vals = angles[i_idx, j_idx]
            
            line_edge_list.append(torch.stack([edge_i, edge_j], dim=0))
            line_edge_list.append(torch.stack([edge_j, edge_i], dim=0))
            line_angle_list.append(angle_vals)
            line_angle_list.append(angle_vals)
        
        if len(line_edge_list) == 0:
            line_edge_index = torch.LongTensor([[0], [0]])
            line_edge_attr = torch.FloatTensor([[0.0]])
        else:
            line_edge_index = torch.cat(line_edge_list, dim=1)
            line_edge_attr = torch.cat(line_angle_list, dim=0).unsqueeze(1)
        
        return {
            'node_features': line_node_features,
            'edge_index': line_edge_index,
            'edge_attr': line_edge_attr
        }
    
    def atoms_to_graph(
        self,
        atoms: Atoms,
        backward_barrier: float = None,
        progress_override: Optional[float] = None,
        timing_label: Optional[str] = None
    ) -> Data:
        """
        Build PyG graph from an in-memory ASE Atoms object.

        This is the core graph-building logic and contains no file I/O.
        Used directly by KMC (no disk roundtrip per move) and indirectly by
        cif_to_graph (which reads the structure first and then delegates here).

        Args:
            atoms: ASE Atoms object. Positions, chemical symbols, cell and pbc
                are read directly. atoms.info['relax_progress'] is consulted
                if progress_override is None.
            backward_barrier: Target barrier value (optional, attached as data.y).
            progress_override: Force the relax_progress scalar in [0, 1]. If
                None, falls back to atoms.info.get('relax_progress', 1.0).
            timing_label: Optional label printed in profiling output to help
                identify the source of the structure (e.g. file name).

        Returns:
            PyG Data object. edge_attr has shape [n_edges, rbf_num_gaussians].
        """
        if self.profile:
            t_total = time.time()
            t0 = time.time()

        # Resolve relaxation progress scalar
        if progress_override is not None:
            progress = float(progress_override)
        else:
            progress = atoms.info.get('relax_progress', 1.0)

        # Node features: element indices and (optional) atomic properties
        elements = atoms.get_chemical_symbols()
        n_atoms = len(atoms)

        element_indices = []
        atomic_props_list = []

        for element in elements:
            if element in self.element_to_idx:
                element_indices.append(self.element_to_idx[element])
            else:
                warnings.warn(f"Unknown element {element}, using index 0")
                element_indices.append(0)

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

        x_element = torch.LongTensor(element_indices)

        if self.use_atomic_properties:
            x_props = torch.FloatTensor(atomic_props_list)
        else:
            x_props = None

        if self.profile:
            t_features = time.time() - t0
            t0 = time.time()

        # edge_attr is [n_edges, rbf_num_gaussians] (RBF-expanded distances)
        edge_index, edge_attr, edge_vectors = self._compute_edges_from_positions(
            atoms, self.cutoff_radius
        )

        if self.profile:
            t_edges = time.time() - t0
            t0 = time.time()

        # Line graph (bond-bond angles) if enabled
        if self.use_line_graph:
            line_graph = self._build_line_graph_from_vectors(edge_index, edge_vectors)
            line_graph_batch_mapping = edge_index[0].clone()
        else:
            line_graph = None
            line_graph_batch_mapping = None

        if self.profile:
            t_line = time.time() - t0
            t_total = time.time() - t_total

            label = timing_label if timing_label else f"atoms({n_atoms})"
            print(f"Graph build timing ({label}):")
            print(f"  Node features:  {t_features*1000:6.2f} ms ({t_features/t_total*100:4.1f}%)")
            print(f"  Edge compute:   {t_edges*1000:6.2f} ms ({t_edges/t_total*100:4.1f}%)")
            if self.use_line_graph:
                print(f"  Line graph:     {t_line*1000:6.2f} ms ({t_line/t_total*100:4.1f}%)")
            print(f"  TOTAL:          {t_total*1000:6.2f} ms")
            print(f"  Edge feature shape: {edge_attr.shape}")

        # Assemble PyG Data object
        data = Data(
            x_element=x_element,
            edge_index=edge_index,
            edge_attr=edge_attr,       # [n_edges, rbf_num_gaussians]
            num_nodes=n_atoms,
            relax_progress=torch.tensor([[progress]], dtype=torch.float32)
        )

        if x_props is not None:
            data.x_props = x_props

        if line_graph is not None:
            data.line_graph_x = line_graph['node_features']
            data.line_graph_edge_index = line_graph['edge_index']
            data.line_graph_edge_attr = line_graph['edge_attr']
            data.line_graph_batch_mapping = line_graph_batch_mapping

        if backward_barrier is not None:
            data.y = torch.tensor([backward_barrier], dtype=torch.float32)

        return data

    def cif_to_graph(
        self,
        cif_path: str,
        backward_barrier: float = None,
        progress_override: Optional[float] = None
    ) -> Data:
        """
        Build PyG graph from CIF/NPZ file with atom embeddings and RBF edges.

        Thin wrapper around atoms_to_graph: reads the structure from disk
        (with optional NPZ caching) and delegates the graph construction.

        Args:
            cif_path: Path to CIF or NPZ file.
            backward_barrier: Target barrier value (optional).
            progress_override: Force progress value (0.0 to 1.0).

        Returns:
            PyG Data object. edge_attr has shape [n_edges, rbf_num_gaussians].
        """
        if self.profile:
            t_read_start = time.time()

        atoms = self._read_structure_from_file(cif_path)

        if self.profile:
            t_read = time.time() - t_read_start
            npz_path = Path(cif_path).with_suffix('.npz')
            format_used = "NPZ" if npz_path.exists() else "CIF"
            print(f"  File reading:   {t_read*1000:6.2f} ms [{format_used}] ({Path(cif_path).name})")

        return self.atoms_to_graph(
            atoms,
            backward_barrier=backward_barrier,
            progress_override=progress_override,
            timing_label=Path(cif_path).name
        )
    
    def build_pair_graph(
        self,
        initial_cif: str,
        final_cif: str,
        backward_barrier: float = None,
        progress_initial: float = None,
        progress_final: float = None
    ) -> Tuple[Data, Data]:
        """Build graph pair from initial and final structures."""
        initial_graph = self.cif_to_graph(initial_cif, backward_barrier, progress_override=progress_initial)
        final_graph = self.cif_to_graph(final_cif, backward_barrier, progress_override=progress_final)
        return initial_graph, final_graph
    
    def clear_cache(self):
        """Clear RAM cache to free memory."""
        if self._structure_cache is not None:
            self._structure_cache.clear()
            print("Structure cache cleared")