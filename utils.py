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
from typing import List, Tuple
from torch_geometric.data import Batch

from model import create_model_from_config, count_parameters


def get_node_input_dim(builder) -> int:
    """
    Calculate node input dimension from TemplateGraphBuilder.
    
    Node features:
    - Positions: 3
    - Element one-hot: len(elements)
    - Atomic properties: 4
    
    Args:
        builder: TemplateGraphBuilder instance
    
    Returns:
        node_input_dim: Total node feature dimension
    
    Example:
        >>> builder = TemplateGraphBuilder(config)
        >>> node_input_dim = get_node_input_dim(builder)
        >>> print(node_input_dim)  # 12 (for 5 elements)
    """
    return 3 + len(builder.elements) + 4


def save_model_for_inference(
    model,
    node_input_dim: int,
    elements: List[str],
    config,
    filepath: str,
    metadata: dict = None
):
    """
    Save model with all metadata needed for inference.
    
    Saves:
    - Model weights
    - node_input_dim (critical for reconstruction)
    - Elements (for validation)
    - Config (for model architecture)
    - Optional metadata (training info, etc.)
    
    Args:
        model: Trained model
        node_input_dim: Node feature dimension
        elements: List of elements (from builder)
        config: Config object
        filepath: Path to save checkpoint
        metadata: Optional additional metadata
    
    Example:
        >>> save_model_for_inference(
        ...     model, 12, ['Mo', 'Nb', 'O', 'Ta', 'W'],
        ...     config, 'best_model.pt',
        ...     metadata={'val_mae': 0.023, 'epoch': 150}
        ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        # Model
        'model_state_dict': model.state_dict(),
        'model_config': model.model_config,
        'node_input_dim': node_input_dim,
        'elements': elements,
        
        # Config
        'config': {
            'gnn_hidden_dim': config.gnn_hidden_dim,
            'gnn_num_layers': config.gnn_num_layers,
            'gnn_embedding_dim': config.gnn_embedding_dim,
            'mlp_hidden_dims': config.mlp_hidden_dims,
            'dropout': config.dropout
        },
        
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }
    
    # Add optional metadata
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, filepath)
    
    print(f"✓ Model saved: {filepath}")
    print(f"  Elements: {elements}")
    print(f"  Node input dim: {node_input_dim}")


def load_model_for_inference(filepath: str, config, validate: bool = True):
    """
    Load model for inference with validation.
    
    Args:
        filepath: Path to checkpoint
        config: Config object (with current database)
        validate: Whether to validate elements (default: True)
    
    Returns:
        model: Loaded model in eval mode
        checkpoint: Full checkpoint dict (for metadata access)
    
    Raises:
        ValueError: If elements don't match (and validate=True)
        FileNotFoundError: If checkpoint doesn't exist
    
    Example:
        >>> model, checkpoint = load_model_for_inference('best_model.pt', config)
        >>> print(f"Model trained on: {checkpoint['elements']}")
        >>> print(f"Val MAE: {checkpoint['metadata']['val_mae']}")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')
    
    print(f"Loading model from: {filepath}")
    print(f"  Saved: {checkpoint['timestamp']}")
    print(f"  Elements: {checkpoint['elements']}")
    print(f"  Node input dim: {checkpoint['node_input_dim']}")
    
    # Validate elements if requested
    if validate:
        from template_graph_builder import TemplateGraphBuilder
        builder = TemplateGraphBuilder(config)
        
        if builder.elements != checkpoint['elements']:
            raise ValueError(
                f"Element mismatch!\n"
                f"  Model trained on: {checkpoint['elements']}\n"
                f"  Current database: {builder.elements}\n"
                f"Cannot use this model with different elements!\n"
                f"Set validate=False to load anyway (not recommended)."
            )
    
    # Create model with correct architecture
    model = create_model_from_config(config, checkpoint['node_input_dim'])
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to eval mode
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    return model, checkpoint


def validate_elements(model_elements: List[str], current_elements: List[str]) -> bool:
    """
    Validate that elements match.
    
    Args:
        model_elements: Elements model was trained on
        current_elements: Elements in current database
    
    Returns:
        match: True if elements match
    
    Example:
        >>> match = validate_elements(
        ...     ['Mo', 'Nb', 'Ta', 'W'],
        ...     ['Mo', 'Nb', 'Ta', 'W']
        ... )
        >>> print(match)  # True
    """
    return model_elements == current_elements


def predict_single(
    model,
    initial_cif: str,
    final_cif: str,
    builder,
    device: str = 'cpu'
) -> float:
    """
    Predict barrier for single structure pair.
    
    Args:
        model: Trained model
        initial_cif: Path to initial structure CIF
        final_cif: Path to final structure CIF
        builder: TemplateGraphBuilder instance
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        barrier: Predicted barrier (eV)
    
    Example:
        >>> barrier = predict_single(
        ...     model, 'initial.cif', 'final.cif', builder
        ... )
        >>> print(f"Predicted: {barrier:.4f} eV")
    """
    model.eval()
    model = model.to(device)
    
    # Build graphs (with dummy barrier)
    initial_graph, final_graph = builder.build_pair_graph(
        initial_cif,
        final_cif,
        backward_barrier=0.0  # Dummy value
    )
    
    # Convert to batch
    initial_batch = Batch.from_data_list([initial_graph]).to(device)
    final_batch = Batch.from_data_list([final_graph]).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(initial_batch, final_batch)
    
    return prediction.item()


def predict_batch(
    model,
    cif_pairs: List[Tuple[str, str]],
    builder,
    batch_size: int = 32,
    device: str = 'cpu'
) -> List[float]:
    """
    Predict barriers for multiple structure pairs.
    
    Args:
        model: Trained model
        cif_pairs: List of (initial_cif, final_cif) tuples
        builder: TemplateGraphBuilder instance
        batch_size: Batch size for prediction
        device: Device to run on
    
    Returns:
        barriers: List of predicted barriers (eV)
    
    Example:
        >>> cif_pairs = [
        ...     ('init1.cif', 'final1.cif'),
        ...     ('init2.cif', 'final2.cif')
        ... ]
        >>> barriers = predict_batch(model, cif_pairs, builder)
        >>> print(barriers)  # [0.589, 0.723]
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    
    # Process in batches
    for i in range(0, len(cif_pairs), batch_size):
        batch_pairs = cif_pairs[i:i+batch_size]
        
        # Build graphs for batch
        initial_graphs = []
        final_graphs = []
        
        for initial_cif, final_cif in batch_pairs:
            initial, final = builder.build_pair_graph(
                initial_cif, final_cif, backward_barrier=0.0
            )
            initial_graphs.append(initial)
            final_graphs.append(final)
        
        # Create batches
        initial_batch = Batch.from_data_list(initial_graphs).to(device)
        final_batch = Batch.from_data_list(final_graphs).to(device)
        
        # Predict
        with torch.no_grad():
            predictions = model(initial_batch, final_batch)
        
        # Collect predictions
        all_predictions.extend(predictions.squeeze().cpu().tolist())
    
    return all_predictions


def print_model_info(model, checkpoint: dict = None):
    """
    Print detailed model information.
    
    Args:
        model: Model instance
        checkpoint: Optional checkpoint dict with metadata
    
    Example:
        >>> model, checkpoint = load_model_for_inference('best.pt', config)
        >>> print_model_info(model, checkpoint)
    """
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    
    # Architecture
    print("\nArchitecture:")
    for key, value in model.model_config.items():
        print(f"  {key}: {value}")
    
    # Parameters
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Predictor: {params['predictor']:,}")
    print(f"  Total: {params['total']:,}")
    
    # Checkpoint info
    if checkpoint is not None:
        print(f"\nCheckpoint:")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
        print(f"  Elements: {checkpoint.get('elements', 'N/A')}")
        print(f"  Node input dim: {checkpoint.get('node_input_dim', 'N/A')}")
        
        if 'metadata' in checkpoint:
            print(f"\nTraining metadata:")
            for key, value in checkpoint['metadata'].items():
                print(f"  {key}: {value}")
    
    print("="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from config import Config
    from template_graph_builder import TemplateGraphBuilder
    from model import create_model_from_config
    
    print("\n" + "="*70)
    print("UTILS TEST")
    print("="*70)
    
    # Setup
    config = Config()
    builder = TemplateGraphBuilder(config)
    
    # Get node input dim
    node_input_dim = get_node_input_dim(builder)
    print(f"\n✓ Node input dim: {node_input_dim}")
    
    # Create model
    model = create_model_from_config(config, node_input_dim)
    print(f"✓ Model created")
    
    # Save model
    save_model_for_inference(
        model,
        node_input_dim,
        builder.elements,
        config,
        'test_model.pt',
        metadata={'test': True}
    )
    
    # Load model
    loaded_model, checkpoint = load_model_for_inference('test_model.pt', config)
    print(f"✓ Model loaded")
    
    # Print info
    print_model_info(loaded_model, checkpoint)
    
    print("="*70 + "\n")