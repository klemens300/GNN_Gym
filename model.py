"""
GNN Model for Diffusion Barrier Prediction

Architecture:
- GraphConvLayer: Custom message passing layer
- GNNEncoder: Stacks multiple GraphConvLayers for graph encoding
- BarrierPredictor: MLP that predicts barriers from embeddings
- DiffusionBarrierModel: Full model combining encoder + predictor

Key: node_input_dim is now DYNAMIC (depends on elements in database)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class GraphConvLayer(MessagePassing):
    """
    Message passing layer for crystal structures.
    
    Uses edge features (distances) to compute messages between connected atoms.
    Implements: message → aggregate → update
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        """
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features (typically 1 for distance)
            hidden_dim: Hidden dimension for message/update networks
        """
        super().__init__(aggr='mean')  # Mean aggregation over neighbors
        
        # Message network: computes messages from node pairs + edge
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network: combines node features with aggregated messages
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Start propagation (calls message → aggregate)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update node features
        out = self.update_net(torch.cat([x, out], dim=1))
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages for each edge.
        
        Args:
            x_i: Source node features [num_edges, node_dim]
            x_j: Target node features [num_edges, node_dim]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Messages [num_edges, hidden_dim]
        """
        # Concatenate source, target, and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(msg_input)


class GNNEncoder(nn.Module):
    """
    Graph encoder that stacks multiple GraphConvLayers.
    
    Processes crystal structure graphs into fixed-size embeddings.
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 5,
        embedding_dim: int = 64
    ):
        """
        Args:
            node_input_dim: Input node feature dimension (DYNAMIC!)
            edge_input_dim: Input edge feature dimension
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of message passing layers
            embedding_dim: Output graph embedding dimension
        """
        super().__init__()
        
        # Initial node embedding
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        
        # Stack of graph convolution layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, edge_input_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final projection to embedding space
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.SiLU()
        )
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with attributes:
                - x: Node features [num_nodes, node_input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_input_dim]
                - batch: Batch assignment [num_nodes] (optional)
        
        Returns:
            Graph embedding [batch_size, embedding_dim]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Handle batching
        batch = data.batch if hasattr(data, 'batch') else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Embed nodes
        x = self.node_embedding(x)
        
        # Message passing layers
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
        
        # Global pooling: aggregate node features to graph-level
        graph_embedding = global_mean_pool(x, batch)
        
        # Project to final embedding space
        graph_embedding = self.output_projection(graph_embedding)
        
        return graph_embedding


class BarrierPredictor(nn.Module):
    """
    MLP that predicts energy barriers from graph embeddings.
    
    Takes embeddings of initial and final structures, computes their
    difference, and predicts backward diffusion barrier.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: list = None,
        dropout: float = 0.15
    ):
        """
        Args:
            embedding_dim: Dimension of input graph embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        
        # Input: [emb_initial, emb_final, delta_emb]
        input_dim = 3 * embedding_dim
        
        # Build MLP layers
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
        
        # Output layer: predict single barrier value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, emb_initial, emb_final):
        """
        Args:
            emb_initial: Initial structure embeddings [batch_size, embedding_dim]
            emb_final: Final structure embeddings [batch_size, embedding_dim]
        
        Returns:
            Barrier predictions [batch_size, 1] (eV)
        """
        # Compute difference embedding
        delta_emb = emb_final - emb_initial
        
        # Concatenate all embeddings
        combined = torch.cat([emb_initial, emb_final, delta_emb], dim=1)
        
        # Predict barrier
        return self.network(combined)


class DiffusionBarrierModel(nn.Module):
    """
    Full model for diffusion barrier prediction.
    
    Architecture:
    1. Encode initial structure → embedding (shared encoder)
    2. Encode final structure → embedding (shared encoder)
    3. Predict barrier from embeddings
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int = 1,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 5,
        gnn_embedding_dim: int = 64,
        mlp_hidden_dims: list = None,
        dropout: float = 0.15
    ):
        """
        Args:
            node_input_dim: Number of node features (DYNAMIC!)
            edge_input_dim: Number of edge features (default: 1)
            gnn_hidden_dim: Hidden dimension for GNN layers
            gnn_num_layers: Number of GNN layers
            gnn_embedding_dim: Output dimension of GNN encoder
            mlp_hidden_dims: Hidden dimensions for MLP predictor
            dropout: Dropout rate
        """
        super().__init__()
        
        # Default MLP architecture
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [1024, 512, 256]
        
        # Store model configuration
        self.model_config = {
            'node_input_dim': node_input_dim,
            'edge_input_dim': edge_input_dim,
            'gnn_hidden_dim': gnn_hidden_dim,
            'gnn_num_layers': gnn_num_layers,
            'gnn_embedding_dim': gnn_embedding_dim,
            'mlp_hidden_dims': mlp_hidden_dims,
            'dropout': dropout
        }
        
        # Shared encoder for initial and final structures
        self.encoder = GNNEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            embedding_dim=gnn_embedding_dim
        )
        
        # Barrier predictor
        self.predictor = BarrierPredictor(
            embedding_dim=gnn_embedding_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )
    
    def forward(self, initial_graph, final_graph):
        """
        Args:
            initial_graph: PyG Batch of initial structures
            final_graph: PyG Batch of final structures
        
        Returns:
            Predictions [batch_size, 1] - barrier in eV
        """
        # Encode both structures using shared encoder
        emb_initial = self.encoder(initial_graph)
        emb_final = self.encoder(final_graph)
        
        # Predict barrier
        return self.predictor(emb_initial, emb_final)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_model_from_config(config, node_input_dim: int):
    """
    Create DiffusionBarrierModel from Config and dynamic node_input_dim.
    
    Args:
        config: Config object with model hyperparameters
        node_input_dim: Dynamic node feature dimension (from TemplateGraphBuilder)
    
    Returns:
        DiffusionBarrierModel instance
    
    Example:
        >>> config = Config()
        >>> builder = TemplateGraphBuilder(config)
        >>> node_input_dim = 3 + len(builder.elements) + 4
        >>> model = create_model_from_config(config, node_input_dim)
    """
    return DiffusionBarrierModel(
        node_input_dim=node_input_dim,
        edge_input_dim=1,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_num_layers=config.gnn_num_layers,
        gnn_embedding_dim=config.gnn_embedding_dim,
        mlp_hidden_dims=config.mlp_hidden_dims,
        dropout=config.dropout
    )


def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        dict with parameter counts
    
    Example:
        >>> model = DiffusionBarrierModel(node_input_dim=12)
        >>> params = count_parameters(model)
        >>> print(f"Total: {params['total']:,}")
    """
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    predictor_params = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'encoder': encoder_params,
        'predictor': predictor_params,
        'total': total_params
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    
    # Example with dynamic node_input_dim
    node_input_dim = 12  # 3 (pos) + 5 (elements) + 4 (props)
    
    print(f"\nCreating model with node_input_dim={node_input_dim}")
    
    model = DiffusionBarrierModel(
        node_input_dim=node_input_dim,
        edge_input_dim=1,
        gnn_hidden_dim=64,
        gnn_num_layers=5,
        gnn_embedding_dim=64,
        mlp_hidden_dims=[1024, 512, 256],
        dropout=0.15
    )
    
    # Count parameters
    params = count_parameters(model)
    
    print(f"\n✓ Model created successfully!")
    print(f"\nParameter counts:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Predictor: {params['predictor']:,}")
    print(f"  Total: {params['total']:,}")
    
    print(f"\nModel configuration:")
    for key, value in model.model_config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)