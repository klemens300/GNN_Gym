"""
Trainer for Diffusion Barrier Prediction

Handles training loop, validation, checkpointing, early stopping, and logging.
Includes Cosine Warm Restarts scheduler and comprehensive MAE tracking.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
import shutil
from typing import Optional
import math

# Wandb import (optional, graceful fallback)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logger(name: str, log_file: Path, level: str = "INFO", also_console: bool = True):
    """
    Setup a logger that writes to file and optionally console.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        also_console: Also log to console
    
    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, level.upper()))
    
    # Console handler (optional)
    if also_console:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    if also_console:
        ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    if also_console:
        logger.addHandler(ch)
    
    return logger


class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warm Restarts and configurable restart behavior.
    
    Features:
    - Cosine decay from initial LR to eta_min
    - Periodic restarts with configurable period multiplication
    - Configurable LR scaling after restart (decay, constant, or amplification)
    
    Args:
        optimizer: Wrapped optimizer
        T_0: Number of epochs for the first restart period
        T_mult: Factor to increase period length after each restart (default: 1)
        eta_min: Minimum learning rate
        restart_decay: LR multiplier after restart (< 1: decay, = 1: constant, > 1: amplification)
        last_epoch: The index of last epoch
    
    Example:
        >>> scheduler = CosineAnnealingWarmRestarts(
        ...     optimizer, T_0=100, T_mult=2, eta_min=1e-6, restart_decay=0.5
        ... )
        >>> # Epochs 0-100: max_lr -> eta_min (100 epochs)
        >>> # Epochs 101-300: 0.5*max_lr -> eta_min (200 epochs, T_mult=2)
        >>> # Epochs 301-700: 0.25*max_lr -> eta_min (400 epochs)
    """
    
    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        restart_decay: float = 1.0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.restart_decay = restart_decay
        self.T_cur = last_epoch
        self.T_i = T_0
        self.restart_count = 0
        
        # CRITICAL: Store initial LRs BEFORE calling super().__init__
        # because super().__init__ calls step() which calls get_lr()
        self.base_lrs_initial = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate current learning rate."""
        # Safety check
        if self.base_lrs_initial is None:
            return self.base_lrs
        
        # Calculate cosine position in current period
        if self.T_cur == -1:
            # First call, return base_lrs
            return self.base_lrs
        
        # Current position in period (0 to 1)
        progress = self.T_cur / self.T_i
        
        # Cosine annealing
        lrs = []
        for base_lr_initial, base_lr_current in zip(self.base_lrs_initial, self.base_lrs):
            # Apply decay based on restart count
            max_lr = base_lr_initial * (self.restart_decay ** self.restart_count)
            
            # Cosine decay from max_lr to eta_min
            lr = self.eta_min + (max_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            lrs.append(lr)
        
        return lrs
    
    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
        else:
            self.T_cur = epoch
        
        self.last_epoch = epoch
        
        # Check if we need to restart
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.restart_count += 1
            self.T_i = self.T_i * self.T_mult  # Increase period length
            
            # Update base_lrs for next period
            for i, base_lr_initial in enumerate(self.base_lrs_initial):
                self.base_lrs[i] = base_lr_initial * (self.restart_decay ** self.restart_count)
        
        # Update optimizer learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class Trainer:
    """
    Trainer for GNN diffusion barrier prediction.
    
    Features:
    - Training and validation loops
    - Early stopping with patience
    - Learning rate scheduling (including Cosine Warm Restarts)
    - Checkpoint management
    - Comprehensive logging (file + wandb)
    - MAE and relative MAE tracking
    - Bar charts for best metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        save_dir: str = "checkpoints",
        cycle: int = None,
        is_final_model: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Configuration object
            save_dir: Directory to save checkpoints
            cycle: Active learning cycle number (optional)
            is_final_model: Whether this is the final model training
        """
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cycle = cycle
        self.is_final_model = is_final_model
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = self._get_loss_function()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler() if config.use_scheduler else None
        
        # Early stopping
        # Use higher patience for final model
        patience = config.final_model_patience if is_final_model else config.patience
        self.patience = patience
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.best_val_rel_mae = float('inf')
        self.best_train_mae = float('inf')
        
        # Tracking
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rel_mae': [],
            'val_rel_mae': [],
            'lr': []
        }
        
        # Setup file logger
        log_file = self.save_dir / "training.log"
        self.logger = setup_logger(
            "trainer",
            log_file,
            level=config.log_level,
            also_console=config.log_to_console
        )
        
        # Wandb initialization
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
    
    def _get_loss_function(self):
        """Get loss function based on config."""
        loss_type = self.config.loss_function.lower()
        
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "mae" or loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "huber":
            return nn.HuberLoss()
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            self.logger.warning(f"Unknown loss function: {loss_type}, using MSE")
            return nn.MSELoss()
    
    def _get_scheduler(self):
        """Get learning rate scheduler based on config."""
        scheduler_type = self.config.scheduler_type.lower()
        
        if scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                verbose=True
            )
        
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler_t_max,
                eta_min=self.config.scheduler_eta_min
            )
        
        elif scheduler_type == "cosine_warm_restarts":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.scheduler_t_0,
                T_mult=self.config.scheduler_t_mult,
                eta_min=self.config.scheduler_eta_min,
                restart_decay=self.config.scheduler_restart_decay
            )
        
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_factor
            )
        
        else:
            self.logger.warning(f"Unknown scheduler type: {scheduler_type}, no scheduler used")
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if not WANDB_AVAILABLE:
            self.logger.warning("Wandb not available, logging disabled")
            self.use_wandb = False
            return
        
        # Get dataset size from config or default
        n_samples = getattr(self.config, '_current_dataset_size', None)
        
        # Generate run name
        if self.config.wandb_run_name:
            run_name = self.config.wandb_run_name
        else:
            run_name = self.config.get_experiment_name(n_samples=n_samples, cycle=self.cycle)
        
        # Add "final" prefix if this is final model
        if self.is_final_model:
            run_name = f"FINAL-{run_name}"
        
        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            config=vars(self.config),
            tags=self.config.wandb_tags,
            notes=self.config.wandb_notes,
            reinit=True
        )
        
        # Log additional metadata
        wandb.config.update({
            "dataset_size": n_samples,
            "cycle": self.cycle if self.cycle is not None else "N/A",
            "is_final_model": self.is_final_model,
            "device": str(self.device)
        })
        
        # Watch model gradients
        if self.config.wandb_watch_model:
            wandb.watch(
                self.model,
                log='all',
                log_freq=self.config.wandb_watch_freq
            )
        
        # Save config.py to wandb
        try:
            config_path = Path("config.py")
            if config_path.exists():
                wandb.save(str(config_path))
                self.logger.info("Config file uploaded to wandb")
        except Exception as e:
            self.logger.warning(f"Could not upload config.py to wandb: {e}")
        
        self.logger.info(f"Wandb initialized: {run_name}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_rel_mae = 0
        n_batches = 0
        
        for batch_idx, (initial_batch, final_batch, barriers) in enumerate(train_loader):
            # Move to device
            initial_batch = initial_batch.to(self.device)
            final_batch = final_batch.to(self.device)
            barriers = barriers.to(self.device).unsqueeze(1)
            
            # Forward pass
            predictions = self.model(initial_batch, final_batch)
            loss = self.criterion(predictions, barriers)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Calculate MAE and relative MAE
            with torch.no_grad():
                mae = torch.abs(predictions - barriers).mean()
                rel_mae = (torch.abs(predictions - barriers) / (barriers + 1e-8)).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_rel_mae += rel_mae.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        avg_rel_mae = total_rel_mae / n_batches
        
        return avg_loss, avg_mae, avg_rel_mae
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rel_mae = 0
        n_batches = 0
        
        with torch.no_grad():
            for initial_batch, final_batch, barriers in val_loader:
                # Move to device
                initial_batch = initial_batch.to(self.device)
                final_batch = final_batch.to(self.device)
                barriers = barriers.to(self.device).unsqueeze(1)
                
                # Forward pass
                predictions = self.model(initial_batch, final_batch)
                loss = self.criterion(predictions, barriers)
                
                # Calculate MAE and relative MAE
                mae = torch.abs(predictions - barriers).mean()
                rel_mae = (torch.abs(predictions - barriers) / (barriers + 1e-8)).mean()
                
                total_loss += loss.item()
                total_mae += mae.item()
                total_rel_mae += rel_mae.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        avg_rel_mae = total_rel_mae / n_batches
        
        return avg_loss, avg_mae, avg_rel_mae
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'best_val_rel_mae': self.best_val_rel_mae,
            'best_train_mae': self.best_train_mae,
            'history': self.history,
            'config': vars(self.config),
            'cycle': self.cycle,
            'is_final_model': self.is_final_model
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = Path(filepath).parent / "best_model.pt"
            shutil.copy(filepath, best_path)
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        self.best_val_rel_mae = checkpoint.get('best_val_rel_mae', float('inf'))
        self.best_train_mae = checkpoint.get('best_train_mae', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def train(self, train_loader, val_loader=None, verbose: bool = True):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            verbose: Print progress
        
        Returns:
            history: Training history dict
        """
        self.logger.info("="*70)
        self.logger.info("TRAINING START")
        self.logger.info("="*70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Patience: {self.patience}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        self.logger.info(f"Scheduler: {self.config.scheduler_type if self.config.use_scheduler else 'None'}")
        if self.is_final_model:
            self.logger.info("*** FINAL MODEL TRAINING ***")
        if self.cycle is not None:
            self.logger.info(f"Active Learning Cycle: {self.cycle}")
        self.logger.info("="*70)
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_mae, train_rel_mae = self.train_epoch(train_loader)
            
            # Validate
            if val_loader is not None:
                val_loss, val_mae, val_rel_mae = self.validate(val_loader)
            else:
                val_loss, val_mae, val_rel_mae = train_loss, train_mae, train_rel_mae
            
            # Update history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['train_rel_mae'].append(train_rel_mae)
            self.history['val_rel_mae'].append(val_rel_mae)
            self.history['lr'].append(current_lr)
            
            # Update best metrics
            if train_mae < self.best_train_mae:
                self.best_train_mae = train_mae
            
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
            
            if val_rel_mae < self.best_val_rel_mae:
                self.best_val_rel_mae = val_rel_mae
            
            # Log to console
            if verbose and epoch % self.config.wandb_log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch:4d} | "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                    f"MAE: {train_mae:.4f}/{val_mae:.4f} | "
                    f"RelMAE: {train_rel_mae:.4f}/{val_rel_mae:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Log to wandb
            if self.use_wandb and epoch % self.config.wandb_log_interval == 0:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'train/mae': train_mae,
                    'val/mae': val_mae,
                    'train/rel_mae': train_rel_mae,
                    'val/rel_mae': val_rel_mae,
                    'learning_rate': current_lr,
                    'best/train_mae': self.best_train_mae,
                    'best/val_mae': self.best_val_mae,
                    'best/val_rel_mae': self.best_val_rel_mae
                })
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(str(checkpoint_path), is_best=True)
                
                if verbose:
                    self.logger.info(f"  → New best model! Val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(str(checkpoint_path), is_best=False)
        
        # Log final bar chart to wandb
        if self.use_wandb:
            self._log_final_metrics_barchart()
        
        # Save final history
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info("="*70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best validation MAE: {self.best_val_mae:.4f} eV")
        self.logger.info(f"Best validation Rel MAE: {self.best_val_rel_mae:.4f}")
        self.logger.info(f"Best training MAE: {self.best_train_mae:.4f} eV")
        self.logger.info(f"Final epoch: {self.current_epoch}")
        self.logger.info("="*70)
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
        
        return self.history
    
    def _log_final_metrics_barchart(self):
        """Log final best metrics as bar chart to wandb."""
        try:
            import wandb
            
            # Create bar chart data
            data = [
                ["Train MAE", self.best_train_mae],
                ["Val MAE", self.best_val_mae]
            ]
            
            table = wandb.Table(data=data, columns=["Metric", "Value"])
            
            wandb.log({
                "best_metrics_barchart": wandb.plot.bar(
                    table,
                    "Metric",
                    "Value",
                    title="Best MAE Metrics"
                )
            })
        except Exception as e:
            self.logger.warning(f"Could not create bar chart: {e}")


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("TRAINER MODULE")
    print("="*70)
    
    print("\nFeatures:")
    print("  - Training and validation loops")
    print("  - Early stopping with patience")
    print("  - Learning rate scheduling:")
    print("    * ReduceLROnPlateau")
    print("    * CosineAnnealingLR")
    print("    * CosineAnnealingWarmRestarts (NEW!)")
    print("    * StepLR")
    print("  - Checkpoint management")
    print("  - Comprehensive logging (file + wandb)")
    print("  - MAE and Relative MAE tracking")
    print("  - Bar charts for best metrics")
    print("  - Config file upload to wandb")
    
    print("\n" + "="*70)