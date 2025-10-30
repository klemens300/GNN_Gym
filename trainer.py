"""
Trainer for Diffusion Barrier Prediction

Handles training loop, validation, checkpointing, early stopping, and logging.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import Optional

# Wandb import (optional, graceful fallback)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class Trainer:
    """
    Trainer for DiffusionBarrierModel.
    
    Features:
    - Training and validation loops
    - Early stopping
    - Checkpoint saving/loading
    - Learning rate scheduling (flexible: plateau, cosine, step)
    - Training history tracking
    - Weights & Biases integration
    """
    
    def __init__(self, model, config, save_dir: str = "checkpoints"):
        """
        Args:
            model: DiffusionBarrierModel instance
            config: Config object
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler - flexible based on config
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rates': []
        }
        
        # Wandb initialization
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        # Prepare config dict for wandb
        wandb_config = {
            # Model architecture
            'gnn_hidden_dim': self.config.gnn_hidden_dim,
            'gnn_num_layers': self.config.gnn_num_layers,
            'gnn_embedding_dim': self.config.gnn_embedding_dim,
            'mlp_hidden_dims': self.config.mlp_hidden_dims,
            'dropout': self.config.dropout,
            
            # Training
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'patience': self.config.patience,
            'gradient_clip_norm': self.config.gradient_clip_norm,
            
            # Scheduler
            'scheduler_type': self.config.scheduler_type if self.config.use_scheduler else 'none',
            'scheduler_factor': self.config.scheduler_factor,
            'scheduler_patience': self.config.scheduler_patience,
            
            # Data
            'min_barrier': self.config.min_barrier,
            'max_barrier': self.config.max_barrier,
            'val_split': self.config.val_split,
            
            # Graph
            'cutoff_radius': self.config.cutoff_radius,
            'max_neighbors': self.config.max_neighbors,
            'supercell_size': self.config.supercell_size,
        }
        
        # Generate run name if not provided
        run_name = self.config.wandb_run_name
        if run_name is None:
            run_name = self.config.get_experiment_name()
        
        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            tags=self.config.wandb_tags,
            notes=self.config.wandb_notes,
            config=wandb_config,
            dir=str(self.save_dir)
        )
        
        # Watch model (gradients and parameters)
        if self.config.wandb_watch_model:
            wandb.watch(
                self.model,
                log='all',
                log_freq=self.config.wandb_watch_freq
            )
        
        print(f"✓ Wandb initialized: {wandb.run.name}")
        print(f"  Project: {self.config.wandb_project}")
        print(f"  URL: {wandb.run.url}")
    
    def _create_scheduler(self):
        """
        Create learning rate scheduler based on config.
        
        Returns:
            scheduler or None
        """
        if not self.config.use_scheduler:
            return None
        
        scheduler_type = self.config.scheduler_type.lower()
        
        if scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler_t_max,
                eta_min=self.config.scheduler_eta_min
            )
        
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_factor
            )
        
        elif scheduler_type == "none":
            return None
        
        else:
            print(f"Warning: Unknown scheduler type '{scheduler_type}', not using scheduler")
            return None
    
    def _step_scheduler(self, val_loss: Optional[float] = None):
        """
        Step the learning rate scheduler.
        
        Args:
            val_loss: Validation loss (required for ReduceLROnPlateau)
        """
        if self.scheduler is None:
            return
        
        # ReduceLROnPlateau needs validation loss
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            # Other schedulers don't need metrics
            self.scheduler.step()
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            avg_loss: Average training loss
            avg_mae: Average training MAE
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0
        
        for batch in train_loader:
            # Unpack tuple from dataloader
            initial_graphs, final_graphs, barriers = batch
            
            # Move data to device
            initial_graphs = initial_graphs.to(self.device)
            final_graphs = final_graphs.to(self.device)
            barriers = barriers.to(self.device)
            
            # Forward pass
            predictions = self.model(initial_graphs, final_graphs)
            
            # Compute loss
            loss = self.criterion(predictions.squeeze(), barriers)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions.squeeze() - barriers)).item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        
        return avg_loss, avg_mae
    
    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            avg_loss: Average validation loss
            avg_mae: Average validation MAE
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0
        
        for batch in val_loader:
            # Unpack tuple from dataloader
            initial_graphs, final_graphs, barriers = batch
            
            # Move data to device
            initial_graphs = initial_graphs.to(self.device)
            final_graphs = final_graphs.to(self.device)
            barriers = barriers.to(self.device)
            
            # Forward pass
            predictions = self.model(initial_graphs, final_graphs)
            
            # Compute loss
            loss = self.criterion(predictions.squeeze(), barriers)
            
            # Track metrics
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions.squeeze() - barriers)).item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        
        return avg_loss, avg_mae
    
    def train(self, train_loader, val_loader, verbose: bool = True):
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Print progress
        
        Returns:
            history: Training history dict
        """
        if verbose:
            print("\n" + "="*70)
            print("STARTING TRAINING")
            print("="*70)
            print(f"Device: {self.device}")
            print(f"Optimizer: AdamW (lr={self.config.learning_rate}, wd={self.config.weight_decay})")
            if self.scheduler is not None:
                print(f"Scheduler: {self.config.scheduler_type}")
            else:
                print("Scheduler: None")
            if self.use_wandb:
                print(f"Wandb: Enabled ({wandb.run.name})")
            else:
                print("Wandb: Disabled")
            print(f"Max epochs: {self.config.epochs}")
            print(f"Early stopping patience: {self.config.patience}")
            print("="*70 + "\n")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_mae = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_mae = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rates'].append(current_lr)
            
            # Log to wandb
            if self.use_wandb and (epoch % self.config.wandb_log_interval == 0):
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/mae': train_mae,
                    'val/loss': val_loss,
                    'val/mae': val_mae,
                    'learning_rate': current_lr,
                    'patience_counter': self.patience_counter
                })
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1:4d}/{self.config.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train MAE: {train_mae:.4f} | "
                      f"Val MAE: {val_mae:.4f} | "
                      f"LR: {current_lr:.2e}")
            
            # Learning rate scheduling
            self._step_scheduler(val_loss)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(
                    filepath=self.save_dir / "best_model.pt",
                    is_best=True
                )
                
                # Log best model to wandb
                if self.use_wandb:
                    wandb.run.summary['best_val_loss'] = val_loss
                    wandb.run.summary['best_val_mae'] = val_mae
                    wandb.run.summary['best_epoch'] = epoch + 1
                
                if verbose:
                    print(f"  → New best model! Val loss: {val_loss:.4f}")
            
            else:
                self.patience_counter += 1
                
                if verbose and self.patience_counter > 0:
                    print(f"  → No improvement for {self.patience_counter} epoch(s)")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(
                    filepath=self.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best validation loss: {self.best_val_loss:.4f}")
                break
        
        # Save final history
        self.save_history()
        
        # Finish wandb run
        if self.use_wandb:
            # Save best model artifact
            artifact = wandb.Artifact(
                name=f'model-{wandb.run.id}',
                type='model',
                description=f'Best model from run {wandb.run.name}'
            )
            artifact.add_file(str(self.save_dir / "best_model.pt"))
            wandb.log_artifact(artifact)
            
            wandb.finish()
        
        if verbose:
            print("\n" + "="*70)
            print("TRAINING COMPLETED")
            print("="*70)
            print(f"Total epochs: {self.current_epoch + 1}")
            print(f"Best val loss: {self.best_val_loss:.4f}")
            print(f"Final train loss: {train_loss:.4f}")
            print(f"Final val loss: {val_loss:.4f}")
            print("="*70 + "\n")
        
        return self.history
    
    def save_checkpoint(self, filepath: Path, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'history': self.history,
            'config': {
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'scheduler_type': self.config.scheduler_type if self.config.use_scheduler else 'none',
                'gnn_hidden_dim': self.config.gnn_hidden_dim,
                'gnn_num_layers': self.config.gnn_num_layers,
                'gnn_embedding_dim': self.config.gnn_embedding_dim,
                'mlp_hidden_dims': self.config.mlp_hidden_dims,
                'dropout': self.config.dropout
            }
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded from epoch {self.current_epoch + 1}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.save_dir / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✓ Training history saved: {history_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Trainer module - import and use with training script")