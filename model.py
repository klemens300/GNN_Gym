"""
ALIGNN-Style GNN for Diffusion Barrier Prediction with Atom Embeddings

Architecture:
1. Atom Embeddings (learned per element type)
2. Atom Graph Encoder (GNN)
3. Line Graph Encoder (for bond angles)
4. Graph-to-Graph Message Passing
5. Barrier Predictor (MLP) - CLEANED (No progress input)

Key features:
- Learned embeddings instead of one-hot encoding
- Optional atomic property concatenation
- Residual connections for gradient flow
- Normalized bond-to-atom aggregation
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class GraphConvLayer(MessagePassing):
    """
    Message passing layer for atom graph with residual connections.
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
        
        self.residual_projection = None
        if node_dim != hidden_dim:
            self.residual_projection = nn.Linear(node_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        residual = x
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_net(torch.cat([x, out], dim=1))
        
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        
        return out + residual
    
    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(msg_input)


class LineGraphConvLayer(MessagePassing):
    """
    Message passing layer for line graph with residual connections.
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
        
        self.residual_projection = None
        if node_dim != hidden_dim:
            self.residual_projection = nn.Linear(node_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        residual = x
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_net(torch.cat([x, out], dim=1))
        
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        
        return out + residual
    
    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(msg_input)


class ALIGNNEncoder(nn.Module):
    """
    ALIGNN-style encoder with atom embeddings and line graph.
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
        
        self.atom_embedding = nn.Embedding(
            num_embeddings=num_elements,
            embedding_dim=atom_embedding_dim
        )
        
        if use_atomic_properties:
            atom_input_dim = atom_embedding_dim + 8
        else:
            atom_input_dim = atom_embedding_dim
        
        self.atom_node_embedding = nn.Linear(atom_input_dim, atom_hidden_dim)
        
        self.atom_conv_layers = nn.ModuleList([
            GraphConvLayer(atom_hidden_dim, atom_edge_dim, atom_hidden_dim)
            for _ in range(atom_num_layers)
        ])
        
        if use_line_graph:
            self.line_node_embedding = nn.Linear(line_node_dim, line_hidden_dim)
            
            self.line_conv_layers = nn.ModuleList([
                LineGraphConvLayer(line_hidden_dim, line_edge_dim, line_hidden_dim)
                for _ in range(line_num_layers)
            ])
            
            self.bond_to_atom_update = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(atom_hidden_dim + line_hidden_dim, atom_hidden_dim),
                    nn.BatchNorm1d(atom_hidden_dim),
                    nn.SiLU()
                )
                for _ in range(min(atom_num_layers, line_num_layers))
            ])
        
        final_dim = atom_hidden_dim
        if use_line_graph:
            final_dim += line_hidden_dim
        
        self.output_projection = nn.Sequential(
            nn.Linear(final_dim, embedding_dim),
            nn.SiLU()
        )
    
    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else \
                torch.zeros(data.x_element.size(0), dtype=torch.long, device=data.x_element.device)
        
        x = self.atom_embedding(data.x_element)
        
        if self.use_atomic_properties and hasattr(data, 'x_props'):
            x = torch.cat([x, data.x_props], dim=1)
        
        x = self.atom_node_embedding(x)
        
        if self.use_line_graph:
            line_x = data.line_graph_x
            line_edge_index = data.line_graph_edge_index
            line_edge_attr = data.line_graph_edge_attr
            line_graph_batch_mapping = data.line_graph_batch_mapping
            
            line_x = self.line_node_embedding(line_x)
            
            num_joint_layers = min(self.atom_num_layers, self.line_num_layers)
            
            for i in range(num_joint_layers):
                x = self.atom_conv_layers[i](x, edge_index, edge_attr)
                line_x = self.line_conv_layers[i](line_x, line_edge_index, line_edge_attr)
                
                bond_messages = line_x
                
                bond_count = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                bond_count.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype))
                bond_count = bond_count.clamp(min=1)
                
                atom_bond_features = torch.zeros(x.size(0), line_x.size(1), device=x.device, dtype=x.dtype)
                atom_bond_features.index_add_(0, edge_index[0], bond_messages)
                atom_bond_features = atom_bond_features / bond_count.unsqueeze(1)
                
                x = self.bond_to_atom_update[i](torch.cat([x, atom_bond_features], dim=1))
            
            for i in range(num_joint_layers, self.atom_num_layers):
                x = self.atom_conv_layers[i](x, edge_index, edge_attr)
            
            for i in range(num_joint_layers, self.line_num_layers):
                line_x = self.line_conv_layers[i](line_x, line_edge_index, line_edge_attr)
            
            atom_embedding = global_mean_pool(x, batch)
            line_batch = batch[line_graph_batch_mapping]
            line_embedding = global_mean_pool(line_x, line_batch)
            
            graph_embedding = torch.cat([atom_embedding, line_embedding], dim=1)
        
        else:
            for conv in self.atom_conv_layers:
                x = conv(x, edge_index, edge_attr)
            
            graph_embedding = global_mean_pool(x, batch)
        
        graph_embedding = self.output_projection(graph_embedding)
        
        return graph_embedding


class BarrierPredictor(nn.Module):
    """
    MLP for barrier prediction from graph embeddings.
    CLEANED: Removed relaxation progress inputs.
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Input: concatenation of initial, final, difference embeddings
        # REMOVED: + 2 for progress scalars
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
        """
        Predict barrier from structure embeddings.
        Args:
            emb_initial: Initial structure embedding [batch_size, embedding_dim]
            emb_final: Final structure embedding [batch_size, embedding_dim]
        """
        # Compute difference embedding
        delta_emb = emb_final - emb_initial
        
        # Concatenate structure information only
        combined = torch.cat([
            emb_initial, 
            emb_final, 
            delta_emb
        ], dim=1)
        
        return self.network(combined)


class DiffusionBarrierModel(nn.Module):
    """
    Complete model for diffusion barrier prediction.
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
        
        self.encoder = ALIGNNEncoder(
            num_elements=num_elements,
            atom_embedding_dim=atom_embedding_dim,
            atom_edge_dim=edge_input_dim,
            line_node_dim=4,
            line_edge_dim=1,
            atom_hidden_dim=gnn_hidden_dim,
            line_hidden_dim=line_graph_hidden_dim,
            atom_num_layers=gnn_num_layers,
            line_num_layers=line_graph_num_layers,
            embedding_dim=gnn_embedding_dim,
            use_line_graph=use_line_graph,
            use_atomic_properties=use_atomic_properties
        )
        
        self.predictor = BarrierPredictor(
            embedding_dim=gnn_embedding_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )
    
    def forward(self, initial_graph, final_graph):
        """
        Predict barrier between initial and final structures.
        """
        emb_initial = self.encoder(initial_graph)
        emb_final = self.encoder(final_graph)
        
        # Predict barrier (no progress passed)
        return self.predictor(emb_initial, emb_final)


def create_model_from_config(config, num_elements: int):
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
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    predictor_params = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'encoder': encoder_params,
        'predictor': predictor_params,
        'total': total_params
    }