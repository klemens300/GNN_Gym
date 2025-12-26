"""
Utility Functions for Model I/O and Prediction

Functions for:
- Reproducibility (set_seed)
- Model saving/loading with atom embeddings
- Prediction utilities
- Model information display
"""

import torch
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from torch_geometric.data import Batch

from model import create_model_from_config, count_parameters


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python, NumPy, and PyTorch (CPU and CUDA).
    
    Note: Full determinism is disabled for CUDA operations to maintain
    performance and compatibility with CuBLAS operations.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Disable strict determinism for better performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_node_input_dim(builder) -> int:
    """
    Calculate node input dimension for model.
    
    With atom embeddings, this is just the number of element types
    (used for embedding table size, not as input dimension).
    
    Args:
        builder: GraphBuilder instance
    
    Returns:
        Number of element types
    """
    return len(builder.elements)


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
    Save model with metadata for inference and resuming.
    
    Saves:
    - Model weights
    - Config parameters
    - Element information
    - Training metadata
    - Optimizer state (optional)
    
    Args:
        model: Model to save
        filepath: Save path
        config: Config object
        builder: GraphBuilder (for element info)
        epoch: Current epoch (optional)
        metrics: Training metrics (optional)
        optimizer_state: Optimizer state dict (optional)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Build checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'elements': builder.elements,
        'num_elements': len(builder.elements),
        'config': {
            'atom_embedding_dim': config.atom_embedding_dim,
            'gnn_hidden_dim': config.gnn_hidden_dim,
            'gnn_num_layers': config.gnn_num_layers,
            'gnn_embedding_dim': config.gnn_embedding_dim,
            'mlp_hidden_dims': config.mlp_hidden_dims,
            'dropout': config.dropout,
            'use_line_graph': config.use_line_graph,
            'line_graph_hidden_dim': config.line_graph_hidden_dim,
            'line_graph_num_layers': config.line_graph_num_layers,
            'use_atomic_properties': config.use_atomic_properties
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
    
    print(f"Model saved: {filepath}")


def load_model_for_inference(filepath: str, config, validate: bool = False):
    """
    Load model for inference using Config parameters.
    
    Model architecture is determined by Config, not checkpoint.
    This allows using checkpoints with different configs.
    
    Args:
        filepath: Path to checkpoint
        config: Config object (provides architecture)
        validate: Whether to validate elements match (optional, default: False)
    
    Returns:
        Tuple of (model, checkpoint)
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')
    
    print(f"Loading model from: {filepath}")
    
    # Get number of elements from Config
    from graph_builder import GraphBuilder
    builder = GraphBuilder(config)
    num_elements = len(builder.elements)
    
    print(f"  Using Config:")
    print(f"    Elements: {builder.elements}")
    print(f"    Num elements: {num_elements}")
    
    # Show checkpoint metadata if available
    if 'timestamp' in checkpoint:
        print(f"  Checkpoint saved: {checkpoint['timestamp']}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")
    if 'elements' in checkpoint:
        print(f"  Checkpoint elements: {checkpoint['elements']}")
    
    # Create model with Config architecture
    model = create_model_from_config(config, num_elements)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load all weights: {e}")
        print(f"  This may happen if model architecture changed")
        # Try partial loading
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Model loaded (partial)")
    
    # Set to eval mode
    model.eval()
    
    return model, checkpoint


def validate_elements(checkpoint_elements: List[str], current_elements: List[str]) -> bool:
    """
    Check if elements match between checkpoint and current database.
    
    Args:
        checkpoint_elements: Elements from checkpoint
        current_elements: Elements from current database
    
    Returns:
        True if elements match exactly
    """
    return checkpoint_elements == current_elements


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
        Predicted barrier in eV
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
    
    Processes in batches for efficiency.
    
    Args:
        model: Trained model
        initial_graphs: List of initial structure graphs
        final_graphs: List of final structure graphs
        device: Device to use ('cpu' or 'cuda')
        batch_size: Batch size for prediction
    
    Returns:
        List of predicted barriers in eV
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


def print_model_info(model, checkpoint: dict = None):
    """
    Print detailed model information.
    
    Displays:
    - Architecture details
    - Parameter counts
    - Checkpoint metadata
    - Configuration
    
    Args:
        model: Model to inspect
        checkpoint: Optional checkpoint dict with metadata
    """
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    
    # Model architecture
    print("\nArchitecture:")
    params = count_parameters(model)
    print(f"  Encoder parameters: {params['encoder']:,}")
    print(f"  Predictor parameters: {params['predictor']:,}")
    print(f"  Total parameters: {params['total']:,}")
    
    # Checkpoint info
    if checkpoint is not None:
        print("\nCheckpoint:")
        if 'timestamp' in checkpoint:
            print(f"  Saved: {checkpoint['timestamp']}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'elements' in checkpoint:
            print(f"  Elements: {checkpoint['elements']}")
        if 'num_elements' in checkpoint:
            print(f"  Num elements: {checkpoint['num_elements']}")
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


if __name__ == "__main__":
    from config import Config
    from graph_builder import GraphBuilder
    
    print("="*70)
    print("UTILS MODULE (ATOM EMBEDDINGS)")
    print("="*70)
    
    # Test set_seed
    print("\nTesting set_seed:")
    set_seed(42)
    print("  Random seeds set to 42")
    print("  Seeds set for Python, NumPy, and PyTorch")
    
    config = Config()
    builder = GraphBuilder(config)
    
    print("\nNum Elements:")
    num_elem = get_node_input_dim(builder)
    print(f"  {num_elem} element types")
    print(f"  Elements: {builder.elements}")
    print(f"  Element indices: {builder.element_to_idx}")
    
    print("\nAvailable Functions:")
    print("  - set_seed(seed)")
    print("  - get_node_input_dim(builder)")
    print("  - save_model_for_inference(model, path, config, builder, ...)")
    print("  - load_model_for_inference(path, config)")
    print("  - predict_single(model, initial_graph, final_graph)")
    print("  - predict_batch(model, initial_graphs, final_graphs)")
    print("  - validate_elements(checkpoint_elements, current_elements)")
    print("  - print_model_info(model, checkpoint)")
    
    print("\n" + "="*70)