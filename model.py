"""
ALIGNN-Style GNN for Diffusion Barrier Prediction with Atom Embeddings

Architecture:
1. Atom Embeddings (learned per element type)
2. Atom Graph Encoder (GNN)
3. Line Graph Encoder (for bond angles)
4. Graph-to-Graph Message Passing
5. Barrier Predictor (MLP) - MODIFIED for Relaxation Progress

Key features:
- Learned embeddings instead of one-hot encoding
- Optional atomic property concatenation
- Residual connections for gradient flow
- Normalized bond-to-atom aggregation
- NEW: Weighted Loss support via Progress input
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class GraphConvLayer(MessagePassing):
    """
    Message passing layer for atom graph with residual connections.
    
    Aggregates information from neighboring atoms via edge features.
    Includes residual connection to prevent vanishing gradients.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='mean')
        
        # Message network: combines source node, target node, and edge features
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network: combines node with aggregated messages
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        # Projection for residual connection if dimensions don't match
        self.residual_projection = None
        if node_dim != hidden_dim:
            self.residual_projection = nn.Linear(node_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # Store input for residual connection
        residual = x
        
        # Message passing: aggregate neighbor information
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update: combine node with aggregated messages
        out = self.update_net(torch.cat([x, out], dim=1))
        
        # Apply residual connection
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        
        return out + residual
    
    def message(self, x_i, x_j, edge_attr):
        """
        Construct messages from source node (j) to target node (i).
        Combines source features, target features, and edge features.
        """
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(msg_input)


class LineGraphConvLayer(MessagePassing):
    """
    Message passing layer for line graph with residual connections.
    
    Line graph: nodes = bonds (edges of atom graph)
                edges = angles between bonds
    
    This allows the model to learn angular information between bonds.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='mean')
        
        # Message network for line graph
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        # Projection for residual if dimensions don't match
        self.residual_projection = None
        if node_dim != hidden_dim:
            self.residual_projection = nn.Linear(node_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # Store input for residual
        residual = x
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_net(torch.cat([x, out], dim=1))
        
        # Residual connection
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        
        return out + residual
    
    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(msg_input)


class ALIGNNEncoder(nn.Module):
    """
    ALIGNN-style encoder with atom embeddings and line graph.
    
    Process:
    1. Embed element types as learned vectors
    2. Optionally concatenate atomic properties
    3. Alternate message passing:
       - Atom graph (atoms communicate)
       - Line graph (bonds communicate)  
       - Bond-to-atom updates (bonds update atoms)
    4. Pool to graph-level representation
    """
    
    def __init__(
        self,
        num_elements: int,
        atom_embedding_dim: int = 32,
        atom_edge_dim: int = 1,
        line_node_dim: int = 4,
        line_edge_dim: int = 1,
        atom_hidden_dim: int = 64,
        line_hidden_dim: int = 64,
        atom_num_layers: int = 5,
        line_num_layers: int = 3,
        embedding_dim: int = 64,
        use_line_graph: bool = True,
        use_atomic_properties: bool = True
    ):
        super().__init__()
        
        self.use_line_graph = use_line_graph
        self.use_atomic_properties = use_atomic_properties
        self.atom_num_layers = atom_num_layers
        self.line_num_layers = line_num_layers
        
        # Learned atom embeddings (one per element type)
        self.atom_embedding = nn.Embedding(
            num_embeddings=num_elements,
            embedding_dim=atom_embedding_dim
        )
        
        # Determine input dimension for atom encoder
        # Embedding + optional atomic properties (8 features)
        if use_atomic_properties:
            atom_input_dim = atom_embedding_dim + 8
        else:
            atom_input_dim = atom_embedding_dim
        
        # Project combined features to hidden dimension
        self.atom_node_embedding = nn.Linear(atom_input_dim, atom_hidden_dim)
        
        # Atom graph message passing layers
        self.atom_conv_layers = nn.ModuleList([
            GraphConvLayer(atom_hidden_dim, atom_edge_dim, atom_hidden_dim)
            for _ in range(atom_num_layers)
        ])
        
        # Line graph components (if enabled)
        if use_line_graph:
            # Line graph node embedding
            self.line_node_embedding = nn.Linear(line_node_dim, line_hidden_dim)
            
            # Line graph message passing layers
            self.line_conv_layers = nn.ModuleList([
                LineGraphConvLayer(line_hidden_dim, line_edge_dim, line_hidden_dim)
                for _ in range(line_num_layers)
            ])
            
            # Bond-to-atom update layers
            # Each bond contributes information to its source atom
            self.bond_to_atom_update = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(atom_hidden_dim + line_hidden_dim, atom_hidden_dim),
                    nn.BatchNorm1d(atom_hidden_dim),
                    nn.SiLU()
                )
                for _ in range(min(atom_num_layers, line_num_layers))
            ])
        
        # Output projection to final embedding dimension
        final_dim = atom_hidden_dim
        if use_line_graph:
            final_dim += line_hidden_dim
        
        self.output_projection = nn.Sequential(
            nn.Linear(final_dim, embedding_dim),
            nn.SiLU()
        )
    
    def forward(self, data):
        """
        Forward pass through encoder.
        """
        # Get atom graph data
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else \
                torch.zeros(data.x_element.size(0), dtype=torch.long, device=data.x_element.device)
        
        # Get learned embeddings for each element
        x = self.atom_embedding(data.x_element)  # [n_atoms, embedding_dim]
        
        # Optionally concatenate atomic properties
        if self.use_atomic_properties and hasattr(data, 'x_props'):
            x = torch.cat([x, data.x_props], dim=1)  # [n_atoms, embedding_dim + 8]
        
        # Project to hidden dimension
        x = self.atom_node_embedding(x)
        
        if self.use_line_graph:
            # Get line graph data
            line_x = data.line_graph_x
            line_edge_index = data.line_graph_edge_index
            line_edge_attr = data.line_graph_edge_attr
            line_graph_batch_mapping = data.line_graph_batch_mapping
            
            # Embed line graph nodes (bonds)
            line_x = self.line_node_embedding(line_x)
            
            # Interleaved message passing between atom and line graphs
            num_joint_layers = min(self.atom_num_layers, self.line_num_layers)
            
            for i in range(num_joint_layers):
                # Update atoms via atom graph
                x = self.atom_conv_layers[i](x, edge_index, edge_attr)
                
                # Update bonds via line graph
                line_x = self.line_conv_layers[i](line_x, line_edge_index, line_edge_attr)
                
                # Bond-to-atom update with normalized aggregation
                bond_messages = line_x  # [num_bonds, line_hidden_dim]
                
                # Count bonds per atom for normalization
                bond_count = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                bond_count.index_add_(
                    0, 
                    edge_index[0], 
                    torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
                )
                bond_count = bond_count.clamp(min=1)  # Avoid division by zero
                
                # Aggregate bond messages to atoms
                atom_bond_features = torch.zeros(
                    x.size(0), line_x.size(1), 
                    device=x.device, dtype=x.dtype
                )
                atom_bond_features.index_add_(0, edge_index[0], bond_messages)
                
                # Normalize by bond count (mean instead of sum)
                atom_bond_features = atom_bond_features / bond_count.unsqueeze(1)
                
                # Update atoms with bond information
                x = self.bond_to_atom_update[i](torch.cat([x, atom_bond_features], dim=1))
            
            # Process remaining layers
            for i in range(num_joint_layers, self.atom_num_layers):
                x = self.atom_conv_layers[i](x, edge_index, edge_attr)
            
            for i in range(num_joint_layers, self.line_num_layers):
                line_x = self.line_conv_layers[i](line_x, line_edge_index, line_edge_attr)
            
            # Pool both graphs to get graph-level embeddings
            atom_embedding = global_mean_pool(x, batch)
            
            # Create batch assignment for line graph
            # Each bond belongs to the same graph as its source atom
            line_batch = batch[line_graph_batch_mapping]
            line_embedding = global_mean_pool(line_x, line_batch)
            
            # Concatenate atom and bond embeddings
            graph_embedding = torch.cat([atom_embedding, line_embedding], dim=1)
        
        else:
            # No line graph: standard GNN
            for conv in self.atom_conv_layers:
                x = conv(x, edge_index, edge_attr)
            
            graph_embedding = global_mean_pool(x, batch)
        
        # Final projection
        graph_embedding = self.output_projection(graph_embedding)
        
        return graph_embedding


class BarrierPredictor(nn.Module):
    """
    MLP for barrier prediction from graph embeddings.
    
    MODIFIED: Takes embeddings of initial and final structures AND RELAXATION PROGRESS
    to predict the energy barrier.
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Input: concatenation of initial, final, difference embeddings + 2 progress scalars
        input_dim = 3 * embedding_dim + 2
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer: output single barrier value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, emb_initial, emb_final, progress_initial, progress_final):
        """
        Predict barrier from structure embeddings and progress.
        
        Args:
            emb_initial: Initial structure embedding [batch_size, embedding_dim]
            emb_final: Final structure embedding [batch_size, embedding_dim]
            progress_initial: Relaxation progress of initial struct [batch_size, 1]
            progress_final: Relaxation progress of final struct [batch_size, 1]
        """
        # Compute difference embedding
        delta_emb = emb_final - emb_initial
        
        # Ensure progress shapes are correct [Batch, 1]
        if progress_initial.dim() == 1:
            progress_initial = progress_initial.unsqueeze(1)
        if progress_final.dim() == 1:
            progress_final = progress_final.unsqueeze(1)
            
        # Concatenate all information
        combined = torch.cat([
            emb_initial, 
            emb_final, 
            delta_emb,
            progress_initial,
            progress_final
        ], dim=1)
        
        return self.network(combined)


class DiffusionBarrierModel(nn.Module):
    """
    Complete model for diffusion barrier prediction.
    
    Architecture:
    1. Shared ALIGNN encoder for both structures
    2. MLP predictor for barrier from embeddings + Progress
    """
    
    def __init__(
        self,
        num_elements: int,
        atom_embedding_dim: int = 32,
        edge_input_dim: int = 1,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 5,
        gnn_embedding_dim: int = 64,
        mlp_hidden_dims: list = None,
        dropout: float = 0.1,
        use_line_graph: bool = True,
        line_graph_hidden_dim: int = 64,
        line_graph_num_layers: int = 3,
        use_atomic_properties: bool = True
    ):
        super().__init__()
        
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [512, 256, 128]
        
        self.use_line_graph = use_line_graph
        self.use_atomic_properties = use_atomic_properties
        
        # Store model configuration for saving/loading
        self.model_config = {
            'num_elements': num_elements,
            'atom_embedding_dim': atom_embedding_dim,
            'edge_input_dim': edge_input_dim,
            'gnn_hidden_dim': gnn_hidden_dim,
            'gnn_num_layers': gnn_num_layers,
            'gnn_embedding_dim': gnn_embedding_dim,
            'mlp_hidden_dims': mlp_hidden_dims,
            'dropout': dropout,
            'use_line_graph': use_line_graph,
            'line_graph_hidden_dim': line_graph_hidden_dim,
            'line_graph_num_layers': line_graph_num_layers,
            'use_atomic_properties': use_atomic_properties
        }
        
        # Shared encoder for both initial and final structures
        self.encoder = ALIGNNEncoder(
            num_elements=num_elements,
            atom_embedding_dim=atom_embedding_dim,
            atom_edge_dim=edge_input_dim,
            line_node_dim=4,  # Bond vectors: normalized direction (3D) + length (1D)
            line_edge_dim=1,  # Bond angles
            atom_hidden_dim=gnn_hidden_dim,
            line_hidden_dim=line_graph_hidden_dim,
            atom_num_layers=gnn_num_layers,
            line_num_layers=line_graph_num_layers,
            embedding_dim=gnn_embedding_dim,
            use_line_graph=use_line_graph,
            use_atomic_properties=use_atomic_properties
        )
        
        # Predictor MLP
        self.predictor = BarrierPredictor(
            embedding_dim=gnn_embedding_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )
    
    def forward(self, initial_graph, final_graph):
        """
        Predict barrier between initial and final structures.
        
        Automatically extracts relaxation progress from graphs.
        """
        # Encode both structures with shared encoder
        emb_initial = self.encoder(initial_graph)
        emb_final = self.encoder(final_graph)
        
        # Extract Progress (NEW FEATURE)
        # Note: GraphBuilder adds this attribute to the Data object
        progress_initial = initial_graph.relax_progress
        progress_final = final_graph.relax_progress
        
        # Predict barrier
        return self.predictor(emb_initial, emb_final, progress_initial, progress_final)


def create_model_from_config(config, num_elements: int):
    """
    Create model from configuration object.
    
    Args:
        config: Config object with model parameters
        num_elements: Number of element types in dataset
    
    Returns:
        DiffusionBarrierModel instance
    """
    return DiffusionBarrierModel(
        num_elements=num_elements,
        atom_embedding_dim=config.atom_embedding_dim,
        edge_input_dim=1,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_num_layers=config.gnn_num_layers,
        gnn_embedding_dim=config.gnn_embedding_dim,
        mlp_hidden_dims=config.mlp_hidden_dims,
        dropout=config.dropout,
        use_line_graph=getattr(config, 'use_line_graph', True),
        line_graph_hidden_dim=getattr(config, 'line_graph_hidden_dim', 64),
        line_graph_num_layers=getattr(config, 'line_graph_num_layers', 3),
        use_atomic_properties=getattr(config, 'use_atomic_properties', True)
    )


def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Returns:
        Dictionary with parameter counts for each component
    """
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    predictor_params = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'encoder': encoder_params,
        'predictor': predictor_params,
        'total': total_params
    }


if __name__ == "__main__":
    print("="*70)
    print("ALIGNN MODEL WITH ATOM EMBEDDINGS (UPDATED FOR PROGRESS)")
    print("="*70)
    
    # Test model creation
    num_elements = 4  # Example: Mo, Nb, Cr, V
    
    model = DiffusionBarrierModel(
        num_elements=num_elements,
        atom_embedding_dim=32,
        gnn_hidden_dim=64,
        gnn_num_layers=5,
        gnn_embedding_dim=64,
        mlp_hidden_dims=[512, 256, 128],
        dropout=0.1,
        use_line_graph=True,
        line_graph_hidden_dim=64,
        line_graph_num_layers=3,
        use_atomic_properties=True
    )
    
    params = count_parameters(model)
    
    print(f"\nModel created successfully!")
    print(f"\nParameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Predictor: {params['predictor']:,}")
    print(f"  Total: {params['total']:,}")
    
    print(f"\nAtom Embedding:")
    print(f"  Num elements: {num_elements}")
    print(f"  Embedding dim: 32")
    print(f"  Total embedding params: {num_elements * 32:,}")
    
    print("\n" + "="*70)