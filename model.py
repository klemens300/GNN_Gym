"""
ALIGNN-Style GNN for Diffusion Barrier Prediction

Architecture:
1. Atom Graph Encoder (GNN)
2. Line Graph Encoder (for bond angles)
3. Graph-to-Graph Message Passing
4. Barrier Predictor (MLP)

Key: Both atom and line graph use message passing, with information
     exchange between the two graphs.

FIXED:
- Added residual connections to prevent gradient vanishing
- Fixed line graph batching using batch mapping
- Added normalization to bond-to-atom aggregation
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class GraphConvLayer(MessagePassing):
    """
    Message passing for atom graph with residual connection.
    
    FIXED: Added residual connection for better gradient flow.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='mean')
        
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        # ✅ FIX: Projection for residual if dimensions don't match
        self.residual_projection = None
        if node_dim != hidden_dim:
            self.residual_projection = nn.Linear(node_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # Store input for residual
        residual = x
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_net(torch.cat([x, out], dim=1))
        
        # ✅ FIX: Residual connection
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        
        return out + residual
    
    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(msg_input)


class LineGraphConvLayer(MessagePassing):
    """
    Message passing for line graph with residual connection.
    
    Line graph nodes = atom graph edges (bonds)
    Line graph edges = angles between bonds
    
    FIXED: Added residual connection for better gradient flow.
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
        
        # ✅ FIX: Projection for residual if dimensions don't match
        self.residual_projection = None
        if node_dim != hidden_dim:
            self.residual_projection = nn.Linear(node_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # Store input for residual
        residual = x
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_net(torch.cat([x, out], dim=1))
        
        # ✅ FIX: Residual connection
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        
        return out + residual
    
    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(msg_input)


class ALIGNNEncoder(nn.Module):
    """
    ALIGNN-style encoder with atom graph + line graph.
    
    Architecture:
    1. Encode atom nodes and bond nodes separately
    2. Alternate message passing between:
       - Atom graph (atoms communicate)
       - Line graph (bonds communicate)
       - Graph-to-graph (bonds update atoms)
    3. Pool to graph-level embedding
    
    FIXED:
    - Correct line graph batching using batch mapping
    - Normalized bond-to-atom aggregation
    - Residual connections in conv layers
    """
    
    def __init__(
        self,
        atom_node_dim: int,
        atom_edge_dim: int = 1,
        line_node_dim: int = 4,
        line_edge_dim: int = 1,
        atom_hidden_dim: int = 64,
        line_hidden_dim: int = 64,
        atom_num_layers: int = 5,
        line_num_layers: int = 3,
        embedding_dim: int = 64,
        use_line_graph: bool = True
    ):
        super().__init__()
        
        self.use_line_graph = use_line_graph
        self.atom_num_layers = atom_num_layers
        self.line_num_layers = line_num_layers
        
        # Atom graph layers
        self.atom_node_embedding = nn.Linear(atom_node_dim, atom_hidden_dim)
        self.atom_conv_layers = nn.ModuleList([
            GraphConvLayer(atom_hidden_dim, atom_edge_dim, atom_hidden_dim)
            for _ in range(atom_num_layers)
        ])
        
        # Line graph layers (if enabled)
        if use_line_graph:
            self.line_node_embedding = nn.Linear(line_node_dim, line_hidden_dim)
            self.line_conv_layers = nn.ModuleList([
                LineGraphConvLayer(line_hidden_dim, line_edge_dim, line_hidden_dim)
                for _ in range(line_num_layers)
            ])
            
            # Graph-to-graph: Update atom features from line graph
            # Each bond (line node) contributes to its source and target atoms
            self.bond_to_atom_update = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(atom_hidden_dim + line_hidden_dim, atom_hidden_dim),
                    nn.BatchNorm1d(atom_hidden_dim),
                    nn.SiLU()
                )
                for _ in range(min(atom_num_layers, line_num_layers))
            ])
        
        # Output projection
        final_dim = atom_hidden_dim
        if use_line_graph:
            final_dim += line_hidden_dim
        
        self.output_projection = nn.Sequential(
            nn.Linear(final_dim, embedding_dim),
            nn.SiLU()
        )
    
    def forward(self, data):
        """
        Forward pass through ALIGNN encoder.
        
        Args:
            data: PyG Data with:
                - x, edge_index, edge_attr (atom graph)
                - line_graph_x, line_graph_edge_index, line_graph_edge_attr (line graph)
                - line_graph_batch_mapping (CRITICAL for batching)
                - batch (for batching)
        
        Returns:
            Graph embedding [batch_size, embedding_dim]
        """
        # Atom graph data
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Embed atom nodes
        x = self.atom_node_embedding(x)
        
        if self.use_line_graph:
            # Line graph data
            line_x = data.line_graph_x
            line_edge_index = data.line_graph_edge_index
            line_edge_attr = data.line_graph_edge_attr
            
            # ✅ FIX: Get batch mapping for line graph
            # This maps each bond (line node) to its source atom
            line_graph_batch_mapping = data.line_graph_batch_mapping
            
            # Embed line nodes (bonds)
            line_x = self.line_node_embedding(line_x)
            
            # Interleaved message passing
            num_joint_layers = min(self.atom_num_layers, self.line_num_layers)
            
            for i in range(num_joint_layers):
                # Update atoms
                x = self.atom_conv_layers[i](x, edge_index, edge_attr)
                
                # Update bonds
                line_x = self.line_conv_layers[i](line_x, line_edge_index, line_edge_attr)
                
                # ✅ FIX: Bond → Atom with normalized aggregation
                # Each bond (edge in atom graph) updates its source atom
                
                bond_messages = line_x  # [num_bonds, line_hidden_dim]
                
                # ✅ FIX: Count bonds per atom for normalization
                bond_count = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                bond_count.index_add_(
                    0, 
                    edge_index[0], 
                    torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
                )
                bond_count = bond_count.clamp(min=1)  # Avoid division by zero
                
                # Aggregate bond messages to atoms (sum)
                atom_bond_features = torch.zeros(
                    x.size(0), line_x.size(1), 
                    device=x.device, dtype=x.dtype
                )
                atom_bond_features.index_add_(0, edge_index[0], bond_messages)
                
                # ✅ FIX: Normalize by bond count (mean instead of sum)
                atom_bond_features = atom_bond_features / bond_count.unsqueeze(1)
                
                # Update atoms with normalized bond information
                x = self.bond_to_atom_update[i](torch.cat([x, atom_bond_features], dim=1))
            
            # Remaining atom layers (if any)
            for i in range(num_joint_layers, self.atom_num_layers):
                x = self.atom_conv_layers[i](x, edge_index, edge_attr)
            
            # Remaining line layers (if any)
            for i in range(num_joint_layers, self.line_num_layers):
                line_x = self.line_conv_layers[i](line_x, line_edge_index, line_edge_attr)
            
            # Pool both graphs
            atom_embedding = global_mean_pool(x, batch)
            
            # ✅ FIX: Create batch assignment for line graph using mapping
            # line_graph_batch_mapping[i] = atom index that bond i belongs to
            # batch[line_graph_batch_mapping[i]] = graph ID for that atom
            line_batch = batch[line_graph_batch_mapping]
            line_embedding = global_mean_pool(line_x, line_batch)
            
            # Concatenate embeddings
            graph_embedding = torch.cat([atom_embedding, line_embedding], dim=1)
        
        else:
            # No line graph: standard GNN
            for conv in self.atom_conv_layers:
                x = conv(x, edge_index, edge_attr)
            
            graph_embedding = global_mean_pool(x, batch)
        
        # Final projection
        graph_embedding = self.output_projection(graph_embedding)
        
        return graph_embedding


class DualALIGNNEncoder(nn.Module):
    """
    Dual ALIGNN encoder for initial and final structures.
    
    This is used in the model to encode both structures separately
    before combining them for barrier prediction.
    
    FIXED: Same fixes as ALIGNNEncoder
    """
    
    def __init__(
        self,
        atom_node_dim: int,
        atom_edge_dim: int = 1,
        line_node_dim: int = 4,
        line_edge_dim: int = 1,
        atom_hidden_dim: int = 64,
        line_hidden_dim: int = 64,
        atom_num_layers: int = 5,
        line_num_layers: int = 4,
        embedding_dim: int = 64,
        use_line_graph: bool = True
    ):
        super().__init__()
        
        self.use_line_graph = use_line_graph
        self.atom_num_layers = atom_num_layers
        self.line_num_layers = line_num_layers
        
        # Atom graph layers
        self.atom_node_embedding = nn.Linear(atom_node_dim, atom_hidden_dim)
        self.atom_conv_layers = nn.ModuleList([
            GraphConvLayer(atom_hidden_dim, atom_edge_dim, atom_hidden_dim)
            for _ in range(atom_num_layers)
        ])
        
        # Line graph layers (if enabled)
        if use_line_graph:
            self.line_node_embedding = nn.Linear(line_node_dim, line_hidden_dim)
            self.line_conv_layers = nn.ModuleList([
                LineGraphConvLayer(line_hidden_dim, line_edge_dim, line_hidden_dim)
                for _ in range(line_num_layers)
            ])
            
            # Graph-to-graph: Update atom features from line graph
            self.bond_to_atom_update = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(atom_hidden_dim + line_hidden_dim, atom_hidden_dim),
                    nn.BatchNorm1d(atom_hidden_dim),
                    nn.SiLU()
                )
                for _ in range(min(atom_num_layers, line_num_layers))
            ])
        
        # Output projection
        final_dim = atom_hidden_dim
        if use_line_graph:
            final_dim += line_hidden_dim
        
        self.output_projection = nn.Sequential(
            nn.Linear(final_dim, embedding_dim),
            nn.SiLU()
        )
    
    def forward(self, data):
        """
        Forward pass through ALIGNN encoder.
        
        FIXED: Correct line graph batching and normalized aggregation.
        """
        # Atom graph data
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Embed atom nodes
        x = self.atom_node_embedding(x)
        
        if self.use_line_graph:
            # Line graph data
            line_x = data.line_graph_x
            line_edge_index = data.line_graph_edge_index
            line_edge_attr = data.line_graph_edge_attr
            
            # ✅ FIX: Get batch mapping for line graph
            line_graph_batch_mapping = data.line_graph_batch_mapping
            
            # Embed line nodes (bonds)
            line_x = self.line_node_embedding(line_x)
            
            # Interleaved message passing
            num_joint_layers = min(self.atom_num_layers, self.line_num_layers)
            
            for i in range(num_joint_layers):
                # Update atoms
                x = self.atom_conv_layers[i](x, edge_index, edge_attr)
                
                # Update bonds
                line_x = self.line_conv_layers[i](line_x, line_edge_index, line_edge_attr)
                
                # ✅ FIX: Bond → Atom with normalized aggregation
                bond_messages = line_x  # [num_bonds, line_hidden_dim]
                
                # ✅ FIX: Count bonds per atom for normalization
                bond_count = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                bond_count.index_add_(
                    0, 
                    edge_index[0], 
                    torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
                )
                bond_count = bond_count.clamp(min=1)  # Avoid division by zero
                
                # Aggregate bond messages to atoms (sum)
                atom_bond_features = torch.zeros(
                    x.size(0), line_x.size(1), 
                    device=x.device, dtype=x.dtype
                )
                atom_bond_features.index_add_(0, edge_index[0], bond_messages)
                
                # ✅ FIX: Normalize by bond count (mean instead of sum)
                atom_bond_features = atom_bond_features / bond_count.unsqueeze(1)
                
                # Update atoms with normalized bond information
                x = self.bond_to_atom_update[i](torch.cat([x, atom_bond_features], dim=1))
            
            # Remaining atom layers (if any)
            for i in range(num_joint_layers, self.atom_num_layers):
                x = self.atom_conv_layers[i](x, edge_index, edge_attr)
            
            # Remaining line layers (if any)
            for i in range(num_joint_layers, self.line_num_layers):
                line_x = self.line_conv_layers[i](line_x, line_edge_index, line_edge_attr)
            
            # Pool both graphs
            atom_embedding = global_mean_pool(x, batch)
            
            # ✅ FIX: Create batch assignment for line graph using mapping
            line_batch = batch[line_graph_batch_mapping]
            line_embedding = global_mean_pool(line_x, line_batch)
            
            # Concatenate embeddings
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
    """MLP for barrier prediction from graph embeddings."""
    
    def __init__(self, embedding_dim: int = 64, hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        input_dim = 3 * embedding_dim
        
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
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, emb_initial, emb_final):
        delta_emb = emb_final - emb_initial
        combined = torch.cat([emb_initial, emb_final, delta_emb], dim=1)
        return self.network(combined)


class DiffusionBarrierModel(nn.Module):
    """
    Full ALIGNN-style model for diffusion barriers.
    
    Architecture:
    1. Shared ALIGNN encoder (atom + line graph)
    2. Barrier predictor (MLP)
    
    FIXED:
    - Residual connections in encoders
    - Correct line graph batching
    - Normalized bond-to-atom aggregation
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int = 1,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 5,
        gnn_embedding_dim: int = 64,
        mlp_hidden_dims: list = None,
        dropout: float = 0.1,
        use_line_graph: bool = True,
        line_graph_hidden_dim: int = 64,
        line_graph_num_layers: int = 3,
        line_graph_embedding_dim: int = 64
    ):
        super().__init__()
        
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [512, 256, 128]
        
        self.use_line_graph = use_line_graph
        
        self.model_config = {
            'node_input_dim': node_input_dim,
            'edge_input_dim': edge_input_dim,
            'gnn_hidden_dim': gnn_hidden_dim,
            'gnn_num_layers': gnn_num_layers,
            'gnn_embedding_dim': gnn_embedding_dim,
            'mlp_hidden_dims': mlp_hidden_dims,
            'dropout': dropout,
            'use_line_graph': use_line_graph,
            'line_graph_hidden_dim': line_graph_hidden_dim,
            'line_graph_num_layers': line_graph_num_layers,
            'line_graph_embedding_dim': line_graph_embedding_dim
        }
        
        # Shared encoder (uses DualALIGNNEncoder which has all fixes)
        self.encoder = DualALIGNNEncoder(
            atom_node_dim=node_input_dim,
            atom_edge_dim=edge_input_dim,
            line_node_dim=4,  # Bond vectors (3D)
            line_edge_dim=1,  # Angles
            atom_hidden_dim=gnn_hidden_dim,
            line_hidden_dim=line_graph_hidden_dim,
            atom_num_layers=gnn_num_layers,
            line_num_layers=line_graph_num_layers,
            embedding_dim=gnn_embedding_dim,
            use_line_graph=use_line_graph
        )
        
        # Predictor
        self.predictor = BarrierPredictor(
            embedding_dim=gnn_embedding_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )
    
    def forward(self, initial_graph, final_graph):
        emb_initial = self.encoder(initial_graph)
        emb_final = self.encoder(final_graph)
        return self.predictor(emb_initial, emb_final)


def create_model_from_config(config, node_input_dim: int):
    """Create model from config."""
    return DiffusionBarrierModel(
        node_input_dim=node_input_dim,
        edge_input_dim=1,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_num_layers=config.gnn_num_layers,
        gnn_embedding_dim=config.gnn_embedding_dim,
        mlp_hidden_dims=config.mlp_hidden_dims,
        dropout=config.dropout,
        use_line_graph=getattr(config, 'use_line_graph', True),
        line_graph_hidden_dim=getattr(config, 'line_graph_hidden_dim', 64),
        line_graph_num_layers=getattr(config, 'line_graph_num_layers', 3),
        line_graph_embedding_dim=getattr(config, 'line_graph_embedding_dim', 64)
    )


def count_parameters(model):
    """Count parameters."""
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
    print("ALIGNN MODEL ARCHITECTURE (FIXED)")
    print("="*70)
    
    node_input_dim = 12
    
    model = create_model_from_config(
        type('Config', (), {
            'gnn_hidden_dim': 64,
            'gnn_num_layers': 5,
            'gnn_embedding_dim': 64,
            'mlp_hidden_dims': [512, 256, 128],
            'dropout': 0.1,
            'use_line_graph': True,
            'line_graph_hidden_dim': 64,
            'line_graph_num_layers': 3,
            'line_graph_embedding_dim': 64
        })(),
        node_input_dim
    )
    
    params = count_parameters(model)
    print(f"\n✓ Model created!")
    print(f"\nFixes applied:")
    print(f"  ✓ Residual connections in all conv layers")
    print(f"  ✓ Correct line graph batching using batch mapping")
    print(f"  ✓ Normalized bond-to-atom aggregation (mean instead of sum)")
    print(f"\nParameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Predictor: {params['predictor']:,}")
    print(f"  Total: {params['total']:,}")
    print("\n" + "="*70)