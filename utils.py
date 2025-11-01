"""
Utility Functions for Model I/O and Prediction

Functions:
- get_node_input_dim: Calculate node feature dimension from builder
- save_model_for_inference: Save model with metadata for later use
- load_model_for_inference: Load model with validation
- predict_single: Predict barrier for single structure pair
- predict_batch: Predict barriers for multiple structure pairs
- validate_elements: Check if elements match
- print_model_info: Display model information
"""

import torch
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from torch_geometric.data import Batch

from model import create_model_from_config, count_parameters


# ============================================================================
# NODE FEATURE DIMENSION
# ============================================================================

def get_node_input_dim(builder) -> int:
    """
    Calculate node input dimension from builder.
    
    Node features:
    - 3D coordinates (3)
    - One-hot element encoding (n_elements)
    - 4 additional features (degree, coordination, etc.)
    
    Args:
        builder: TemplateGraphBuilder instance
    
    Returns:
        node_input_dim: Total node feature dimension
    
    Example:
        >>> builder = TemplateGraphBuilder(config)
        >>> dim = get_node_input_dim(builder)
        >>> print(f"Node input dim: {dim}")
    """
    return 3 + len(builder.elements) + 4


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model_for_inference(
    model,
    filepath: str,
    config,
    builder,
    epoch: int = None,
    metrics: dict = None,
    optimizer_state: dict = None
):
    """
    Save model with full metadata for inference and resuming.
    
    Saves:
    - Model weights
    - Config parameters
    - Element information
    - Training metadata
    - Optional: optimizer state
    
    Args:
        model: Model to save
        filepath: Save path
        config: Config object
        builder: TemplateGraphBuilder (for element info)
        epoch: Current epoch (optional)
        metrics: Training metrics (optional)
        optimizer_state: Optimizer state dict (optional)
    
    Example:
        >>> save_model_for_inference(
        ...     model, 'checkpoints/best_model.pt',
        ...     config, builder, epoch=100,
        ...     metrics={'val_loss': 0.123}
        ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Build checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'elements': builder.elements,
        'node_input_dim': get_node_input_dim(builder),
        'config': {
            'gnn_hidden_dim': config.gnn_hidden_dim,
            'gnn_num_layers': config.gnn_num_layers,
            'gnn_embedding_dim': config.gnn_embedding_dim,
            'mlp_hidden_dims': config.mlp_hidden_dims,
            'dropout': config.dropout,
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # Add optional data
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    # Save
    torch.save(checkpoint, filepath)
    
    print(f"✓ Model saved: {filepath}")


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model_for_inference(filepath: str, config, validate: bool = False):
    """
    Load model for inference using Config parameters.
    
    Args:
        filepath: Path to checkpoint
        config: Config object (provides elements and architecture)
        validate: Whether to validate (default: False, not recommended)
    
    Returns:
        model: Loaded model in eval mode
        checkpoint: Full checkpoint dict (for metadata access)
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    
    Note:
        Model architecture is determined by Config, not checkpoint.
        This allows using checkpoints with different configs.
    
    Example:
        >>> model, checkpoint = load_model_for_inference(
        ...     'checkpoints/best_model.pt', config
        ... )
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')
    
    print(f"Loading model from: {filepath}")
    
    # Get elements and node_input_dim from Config (not checkpoint)
    from template_graph_builder import TemplateGraphBuilder
    builder = TemplateGraphBuilder(config)
    node_input_dim = 3 + len(builder.elements) + 4
    
    print(f"  Using Config:")
    print(f"    Elements: {builder.elements}")
    print(f"    Node input dim: {node_input_dim}")
    
    # Optional: Show checkpoint metadata if available
    if 'timestamp' in checkpoint:
        print(f"  Checkpoint saved: {checkpoint['timestamp']}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")
    
    # Create model with Config architecture
    model = create_model_from_config(config, node_input_dim)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load all weights: {e}")
        print(f"   This may happen if model architecture changed")
        # Try partial loading
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✓ Model loaded (partial)")
    
    # Set to eval mode
    model.eval()
    
    return model, checkpoint


# ============================================================================
# VALIDATION
# ============================================================================

def validate_elements(checkpoint_elements: List[str], current_elements: List[str]) -> bool:
    """
    Check if elements match between checkpoint and current database.
    
    Args:
        checkpoint_elements: Elements from checkpoint
        current_elements: Elements from current database
    
    Returns:
        match: True if elements match
    
    Example:
        >>> match = validate_elements(['Mo', 'W'], ['Mo', 'W'])
        >>> print(f"Elements match: {match}")
    """
    return checkpoint_elements == current_elements


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_single(
    model,
    initial_graph,
    final_graph,
    device: str = 'cpu'
) -> float:
    """
    Predict barrier for single structure pair.
    
    Args:
        model: Trained model
        initial_graph: Initial structure graph
        final_graph: Final structure graph
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        barrier: Predicted barrier (eV)
    
    Example:
        >>> barrier = predict_single(model, initial_graph, final_graph)
        >>> print(f"Predicted barrier: {barrier:.3f} eV")
    """
    model.eval()
    device = torch.device(device)
    
    # Batch single graphs
    initial_batch = Batch.from_data_list([initial_graph]).to(device)
    final_batch = Batch.from_data_list([final_graph]).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(initial_batch, final_batch)
    
    return prediction.item()


def predict_batch(
    model,
    initial_graphs: List,
    final_graphs: List,
    device: str = 'cpu',
    batch_size: int = 32
) -> List[float]:
    """
    Predict barriers for multiple structure pairs.
    
    Args:
        model: Trained model
        initial_graphs: List of initial structure graphs
        final_graphs: List of final structure graphs
        device: Device to use ('cpu' or 'cuda')
        batch_size: Batch size for prediction
    
    Returns:
        barriers: List of predicted barriers (eV)
    
    Example:
        >>> barriers = predict_batch(
        ...     model, initial_graphs, final_graphs, batch_size=64
        ... )
        >>> print(f"Predicted {len(barriers)} barriers")
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    
    predictions = []
    
    # Process in batches
    for i in range(0, len(initial_graphs), batch_size):
        batch_initial = initial_graphs[i:i+batch_size]
        batch_final = final_graphs[i:i+batch_size]
        
        # Create batches
        initial_batch = Batch.from_data_list(batch_initial).to(device)
        final_batch = Batch.from_data_list(batch_final).to(device)
        
        # Predict
        with torch.no_grad():
            batch_predictions = model(initial_batch, final_batch)
        
        predictions.extend(batch_predictions.cpu().numpy().tolist())
    
    return predictions


# ============================================================================
# MODEL INFO
# ============================================================================

def print_model_info(model, checkpoint: dict = None):
    """
    Print detailed model information.
    
    Args:
        model: Model to inspect
        checkpoint: Optional checkpoint dict
    
    Example:
        >>> model, checkpoint = load_model_for_inference(path, config)
        >>> print_model_info(model, checkpoint)
    """
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    
    # Model architecture
    print("\nArchitecture:")
    print(f"  Total parameters: {count_parameters(model):,}")
    
    # Checkpoint info
    if checkpoint is not None:
        print("\nCheckpoint:")
        if 'timestamp' in checkpoint:
            print(f"  Saved: {checkpoint['timestamp']}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'elements' in checkpoint:
            print(f"  Elements: {checkpoint['elements']}")
        if 'node_input_dim' in checkpoint:
            print(f"  Node input dim: {checkpoint['node_input_dim']}")
        if 'metrics' in checkpoint:
            print(f"  Metrics:")
            for key, value in checkpoint['metrics'].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
    
    # Config info
    if checkpoint is not None and 'config' in checkpoint:
        print("\nConfiguration:")
        for key, value in checkpoint['config'].items():
            print(f"  {key}: {value}")
    
    print("="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from config import Config
    from template_graph_builder import TemplateGraphBuilder
    
    print("="*70)
    print("UTILS MODULE")
    print("="*70)
    
    config = Config()
    builder = TemplateGraphBuilder(config)
    
    print("\nNode Input Dimension:")
    dim = get_node_input_dim(builder)
    print(f"  {dim} (3 coords + {len(builder.elements)} elements + 4 features)")
    
    print("\nAvailable Functions:")
    print("  - get_node_input_dim(builder)")
    print("  - save_model_for_inference(model, path, config, builder, ...)")
    print("  - load_model_for_inference(path, config, validate=False)")
    print("  - predict_single(model, initial_graph, final_graph)")
    print("  - predict_batch(model, initial_graphs, final_graphs)")
    print("  - validate_elements(checkpoint_elements, current_elements)")
    print("  - print_model_info(model, checkpoint)")
    
    print("\n" + "="*70)