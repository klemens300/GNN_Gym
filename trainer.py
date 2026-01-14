"""
Trainer for Diffusion Barrier Prediction

ENHANCED WITH:
- Weighted Loss for KMC Strategy (Unrelaxed > Relaxed)
- R² Score tracking & Diagnostics
- Prediction variance monitoring
- Gradient norm tracking
- Comprehensive WandB logging with plots
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
import shutil
from typing import Optional
import math
import numpy as np

# Wandb import (optional, graceful fallback)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Scikit-learn for R² score
try:
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def setup_logger(name: str, log_file: Path, level: str = "INFO", also_console: bool = True):
    """
    Setup a logger that writes to file and optionally console.
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
        self.base_lrs_initial = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate current learning rate."""
        if self.base_lrs_initial is None:
            return self.base_lrs
        
        if self.T_cur == -1:
            return self.base_lrs
        
        progress = self.T_cur / self.T_i
        lrs = []
        for base_lr_initial, base_lr_current in zip(self.base_lrs_initial, self.base_lrs):
            max_lr = base_lr_initial * (self.restart_decay ** self.restart_count)
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
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.restart_count += 1
            self.T_i = self.T_i * self.T_mult
            
            for i, base_lr_initial in enumerate(self.base_lrs_initial):
                self.base_lrs[i] = base_lr_initial * (self.restart_decay ** self.restart_count)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class Trainer:
    """
    Trainer for GNN diffusion barrier prediction.
    
    Includes:
    - Weighted Loss (KMC Strategy)
    - Comprehensive diagnostics (R², Stuck Detection)
    - Gradient monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        save_dir: str = "checkpoints",
        cycle: int = None,
        is_final_model: bool = False
    ):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cycle = cycle
        self.is_final_model = is_final_model
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Early stopping
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
            'train_loss': [], 'val_loss': [], 
            'train_mae': [], 'val_mae': [], 
            'train_rel_mae': [], 'val_rel_mae': [], 
            'lr': []
        }
        
        # Logger
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if cycle is not None:
            log_file = log_dir / f"training_cycle_{cycle}.log"
        elif is_final_model:
            log_file = log_dir / "training_final.log"
        else:
            log_file = log_dir / "training.log"
        
        self.logger = setup_logger(
            name=f"trainer_{cycle if cycle is not None else 'final'}",
            log_file=log_file,
            level=config.log_level,
            also_console=config.log_to_console
        )
        
        # Optimizations
        if getattr(config, 'cudnn_benchmark', False) and torch.cuda.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            self.logger.info("? cuDNN benchmark enabled")
        
        if getattr(config, 'compile_model', False) and hasattr(torch, 'compile'):
            compile_mode = getattr(config, 'compile_mode', 'default')
            self.logger.info(f"? Compiling model (mode={compile_mode})...")
            self.model = torch.compile(self.model, mode=compile_mode)
        
        # Loss function
        self.criterion = self._get_loss_function()
        
        # Optimizer
        use_fused = getattr(config, 'use_fused_optimizer', False) and torch.cuda.is_available()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=use_fused
        )
        
        # Mixed Precision (AMP)
        self.use_amp = getattr(config, 'use_amp', False) and torch.cuda.is_available()
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            self.logger.info("? Mixed Precision (AMP) enabled")
        else:
            self.scaler = None
        
        # Scheduler
        self.scheduler = self._get_scheduler() if config.use_scheduler else None
        
        # WandB
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
    
    def _get_loss_function(self):
        """
        Get loss function from config.
        NOTE: Uses reduction='none' to allow for weighted loss calculation!
        """
        loss_type = self.config.loss_function.lower()
        
        if loss_type == "mse":
            return nn.MSELoss(reduction='none')
        elif loss_type == "mae" or loss_type == "l1":
            return nn.L1Loss(reduction='none')
        elif loss_type == "huber":
            return nn.HuberLoss(reduction='none')
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss(reduction='none')
        else:
            self.logger.warning(f"Unknown loss function: {loss_type}, using MSE")
            return nn.MSELoss(reduction='none')
    
    def _get_scheduler(self):
        """Get learning rate scheduler based on config."""
        scheduler_type = self.config.scheduler_type.lower()
        
        if scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
                verbose=True
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.cosine_t_max,
                eta_min=self.config.cosine_eta_min
            )
        elif scheduler_type == "cosine_warm_restarts":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.warm_restart_t_0,
                T_mult=int(self.config.warm_restart_t_mult),
                eta_min=self.config.warm_restart_eta_min,
                restart_decay=self.config.warm_restart_decay
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.step_gamma
            )
        else:
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if not WANDB_AVAILABLE: return
        
        n_samples = getattr(self.config, '_current_dataset_size', None)
        if self.config.wandb_run_name:
            run_name = self.config.wandb_run_name
        else:
            run_name = self.config.get_experiment_name(n_samples=n_samples, cycle=self.cycle)
        
        if self.is_final_model:
            run_name = f"FINAL-{run_name}"
        
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            config=vars(self.config),
            tags=self.config.wandb_tags,
            notes=self.config.wandb_notes,
            reinit=True
        )
        
        if self.config.wandb_watch_model:
            wandb.watch(self.model, log="all", log_freq=self.config.wandb_watch_freq)
        
        self.logger.info(f"Wandb initialized: {run_name}")

    def _calculate_weighted_loss(self, predictions, targets, progress):
        """
        Calculate loss weighted by relaxation progress.
        
        Strategy:
        - Unrelaxed (Progress ~ 0.0): Weight 1.0 (Critical for KMC)
        - Relaxed (Progress ~ 1.0): Weight 0.2 (Physics guidance)
        """
        # Ensure targets match prediction shape
        targets = targets.view_as(predictions)
        
        # Flatten progress to match batch dimension
        progress = progress.view(-1).to(predictions.device)
        
        # Calculate raw element-wise loss (possible because reduction='none')
        raw_loss = self.criterion(predictions, targets)
        
        # Calculate weights: 1.0 -> 0.2
        # Formula: Weight = 1.0 - (decay * progress)
        decay = 0.8
        weights = 1.0 - (decay * progress)
        
        # Ensure weights broadcast correctly if needed (usually 1D matches 1D)
        weights = weights.view_as(raw_loss)
        
        # Apply weighting
        weighted_loss = (raw_loss * weights).mean()
        
        return weighted_loss

    def train_epoch(self, train_loader):
        """Train for one epoch with weighted loss."""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_rel_mae = 0
        n_batches = 0
        
        # UNPACKING 4 VALUES NOW (Added progress_batch)
        for batch_idx, (initial_batch, final_batch, barriers, progress_batch) in enumerate(train_loader):
            # Move to device
            initial_batch = initial_batch.to(self.device)
            final_batch = final_batch.to(self.device)
            barriers = barriers.to(self.device).unsqueeze(1)
            progress_batch = progress_batch.to(self.device) # New!
            
            # Forward Pass
            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    predictions = self.model(initial_batch, final_batch)
                    # WEIGHTED LOSS CALCULATION
                    loss = self._calculate_weighted_loss(predictions, barriers, progress_batch)
            else:
                predictions = self.model(initial_batch, final_batch)
                # WEIGHTED LOSS CALCULATION
                loss = self._calculate_weighted_loss(predictions, barriers, progress_batch)
            
            # Backward Pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"? NaN/Inf in loss! Stopping.")
                    raise RuntimeError("Training diverged: NaN/Inf in loss")
            
            # Diagnostics: Gradient Norm (First batch only)
            if batch_idx == 0 and self.use_wandb:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                wandb.log({"diagnostics/gradient_norm_total": total_norm})
            
            # Gradient Clipping
            if self.config.gradient_clip_norm > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            # Optimizer Step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                mae = torch.abs(predictions - barriers).mean()
                rel_mae = (torch.abs(predictions - barriers) / (barriers + 1e-8)).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_rel_mae += rel_mae.item()
            n_batches += 1
        
        return total_loss / n_batches, total_mae / n_batches, total_rel_mae / n_batches
    
    def validate(self, val_loader):
        """
        Validate model with comprehensive diagnostics.
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rel_mae = 0
        n_batches = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            # UNPACKING 4 VALUES
            for initial_batch, final_batch, barriers, progress_batch in val_loader:
                initial_batch = initial_batch.to(self.device)
                final_batch = final_batch.to(self.device)
                barriers = barriers.to(self.device).unsqueeze(1)
                progress_batch = progress_batch.to(self.device)
                
                predictions = self.model(initial_batch, final_batch)
                
                # Use Weighted Loss for validation too (consistency)
                # Or use raw loss - usually weighted is better to see if objective matches
                loss = self._calculate_weighted_loss(predictions, barriers, progress_batch)
                
                mae = torch.abs(predictions - barriers).mean()
                rel_mae = (torch.abs(predictions - barriers) / (barriers + 1e-8)).mean()
                
                total_loss += loss.item()
                total_mae += mae.item()
                total_rel_mae += rel_mae.item()
                n_batches += 1
                
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(barriers.cpu().numpy().flatten())
        
        # Run Diagnostics (Stuck detection, R2, etc.) every 10 epochs
        if self.use_wandb and self.current_epoch % 10 == 0:
            self._log_diagnostics(all_preds, all_labels)
        
        return total_loss / n_batches, total_mae / n_batches, total_rel_mae / n_batches
    
    def _log_diagnostics(self, all_preds, all_labels):
        """Log comprehensive diagnostics to WandB."""
        pred_array = np.array(all_preds)
        label_array = np.array(all_labels)
        
        # R² Score
        if SKLEARN_AVAILABLE:
            r2 = r2_score(label_array, pred_array)
        else:
            ss_res = np.sum((label_array - pred_array) ** 2)
            ss_tot = np.sum((label_array - label_array.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Statistics
        pred_std = pred_array.std()
        pred_mean = pred_array.mean()
        mean_deviation = np.abs(pred_array - label_array.mean()).mean()
        
        # Stuck Detection
        is_stuck = False
        stuck_reasons = []
        if r2 < 0.05:
            is_stuck = True
            stuck_reasons.append(f"R²={r2:.4f}")
        if pred_std < 0.01:
            is_stuck = True
            stuck_reasons.append(f"Std={pred_std:.4f}")
        
        log_dict = {
            "diagnostics/r2_score": r2,
            "diagnostics/is_stuck": 1 if is_stuck else 0,
            "diagnostics/pred_std": pred_std,
            "diagnostics/pred_mean": pred_mean,
            "diagnostics/mean_deviation_avg": mean_deviation,
            "diagnostics/prediction_histogram": wandb.Histogram(pred_array),
            "diagnostics/residuals_histogram": wandb.Histogram(pred_array - label_array)
        }
        
        wandb.log(log_dict)
        
        if is_stuck:
            self.logger.warning(f"?? Model stuck! Reasons: {stuck_reasons}")
        else:
            self.logger.info(f"? Diagnostics OK. R²={r2:.4f}")

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
            'history': self.history
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
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.history = checkpoint.get('history', self.history)
        self.logger.info(f"Checkpoint loaded: {filepath}")
    
    def train(self, train_loader, val_loader=None, verbose: bool = True):
        """Main training loop."""
        self.logger.info("="*70)
        self.logger.info("TRAINING START")
        self.logger.info("="*70)
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_mae, train_rel_mae = self.train_epoch(train_loader)
            
            # Validate
            if val_loader:
                val_loss, val_mae, val_rel_mae = self.validate(val_loader)
            else:
                val_loss, val_mae, val_rel_mae = train_loss, train_mae, train_rel_mae
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Track Best
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_val_rel_mae = val_rel_mae
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save Best
                self.save_checkpoint(str(self.save_dir / f"checkpoint_epoch_{epoch}.pt"), is_best=True)
                if verbose: self.logger.info(f"  ? New best model! Val MAE: {val_mae:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if verbose and epoch % self.config.wandb_log_interval == 0:
                self.logger.info(f"Ep {epoch:3d} | Loss: {train_loss:.4f}/{val_loss:.4f} | MAE: {train_mae:.4f}/{val_mae:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'train/loss': train_loss, 'val/loss': val_loss,
                    'train/mae': train_mae, 'val/mae': val_mae,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'patience': self.patience_counter
                })
            
            # Scheduler Step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Periodic Save
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(str(self.save_dir / f"checkpoint_epoch_{epoch}.pt"))
        
        # Finish
        if self.use_wandb:
            self._log_final_metrics_barchart()
            wandb.finish()
            
        return self.history

    def _log_final_metrics_barchart(self):
        """Log final best metrics as bar chart to wandb."""
        try:
            data = [["Train MAE", self.best_train_mae], ["Val MAE", self.best_val_mae]]
            table = wandb.Table(data=data, columns=["Metric", "Value"])
            wandb.log({"best_metrics_barchart": wandb.plot.bar(table, "Metric", "Value", title="Best MAE")})
        except:
            pass