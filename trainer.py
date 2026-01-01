"""
Trainer for Diffusion Barrier Prediction

ENHANCED WITH DIAGNOSTICS:
- R² Score tracking
- Prediction variance monitoring
- Mean prediction detection
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
    print("⚠️  sklearn not available - R² score will not be computed")


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
        # Safety check
        if self.base_lrs_initial is None:
            return self.base_lrs
        
        # Calculate cosine position in current period
        if self.T_cur == -1:
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
            self.T_i = self.T_i * self.T_mult
            
            # Update base_lrs for next period
            for i, base_lr_initial in enumerate(self.base_lrs_initial):
                self.base_lrs[i] = base_lr_initial * (self.restart_decay ** self.restart_count)
        
        # Update optimizer learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class Trainer:
    """
    Trainer for GNN diffusion barrier prediction.
    
    ENHANCED WITH:
    - Comprehensive diagnostics
    - Mean prediction detection
    - R² score tracking
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
        """
        Initialize trainer.
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
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rel_mae': [],
            'val_rel_mae': [],
            'lr': []
        }
        
        # Logger (CREATE FIRST!)
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
        
        # 🔥 NOW we can use logger for optimizations!
        
        # 🔥 cuDNN Optimization
        if getattr(config, 'cudnn_benchmark', False) and torch.cuda.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            self.logger.info("✓ cuDNN benchmark enabled")
        
        # 🔥 Model Compilation (PyTorch 2.0+)
        if getattr(config, 'compile_model', False) and hasattr(torch, 'compile'):
            compile_mode = getattr(config, 'compile_mode', 'default')
            self.logger.info(f"🔥 Compiling model (mode={compile_mode})...")
            self.model = torch.compile(self.model, mode=compile_mode)
            self.logger.info("✓ Model compiled!")
        
        # Loss function
        self.criterion = self._get_loss_function()
        
        # 🔥 Optimizer (with optional fused mode for speed)
        use_fused = getattr(config, 'use_fused_optimizer', False) and torch.cuda.is_available()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=use_fused  # 🔥 GPU-optimized if available
        )
        if use_fused:
            self.logger.info("✓ Using fused AdamW optimizer")
        
        # 🔥 Mixed Precision (AMP)
        self.use_amp = getattr(config, 'use_amp', False) and torch.cuda.is_available()
        if self.use_amp:
            from torch.cuda.amp import autocast, GradScaler
            self.scaler = GradScaler()
            self.logger.info("✓ Mixed Precision (AMP) enabled")
        else:
            self.scaler = None
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler() if config.use_scheduler else None
        
        # Initialize wandb
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
    
    def _get_loss_function(self):
        """Get loss function from config."""
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
        
        # Upload config file if exists
        config_file = Path("config.py")
        if config_file.exists():
            wandb.save(str(config_file))
            self.logger.info("Config file uploaded to wandb")
        
        # Watch model
        if self.config.wandb_watch_model:
            wandb.watch(
                self.model,
                log="all",
                log_freq=self.config.wandb_watch_freq
            )
        
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
            
            # 🔥 Mixed Precision Forward Pass
            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    predictions = self.model(initial_batch, final_batch)
                    loss = self.criterion(predictions, barriers)
            else:
                # Standard forward pass
                predictions = self.model(initial_batch, final_batch)
                loss = self.criterion(predictions, barriers)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            # 🔥 Mixed Precision Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
                # ?? EMERGENCY: Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"? NaN/Inf in loss! Stopping.")
                    raise RuntimeError("Training diverged: NaN/Inf in loss")
                
            
            # 🔥 DIAGNOSTICS: Gradient monitoring (first batch only)
            if batch_idx == 0 and self.use_wandb:
                total_norm = 0
                max_norm = 0
                min_norm = float('inf')
                
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        max_norm = max(max_norm, param_norm)
                        min_norm = min(min_norm, param_norm)
                
                total_norm = total_norm ** 0.5
                
                wandb.log({
                    "diagnostics/gradient_norm_total": total_norm,
                    "diagnostics/gradient_norm_max": max_norm,
                    "diagnostics/gradient_norm_min": min_norm
                })
                
                # Warnings
                if total_norm < 1e-5:
                    self.logger.warning(f"⚠️  Gradients vanishing! Norm = {total_norm:.2e}")
                elif total_norm > 100:
                    self.logger.warning(f"⚠️  Gradients exploding! Norm = {total_norm:.2e}")
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            # 🔥 Mixed Precision Optimizer Step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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
        """
        Validate model with comprehensive diagnostics.
        
        🔥 ENHANCED: Includes R², prediction variance, scatter plots
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rel_mae = 0
        n_batches = 0
        
        # 🔥 Collect all predictions and labels for diagnostics
        all_preds = []
        all_labels = []
        
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
                
                # 🔥 Collect for diagnostics
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(barriers.cpu().numpy().flatten())
        
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        avg_rel_mae = total_rel_mae / n_batches
        
        # ====================================================================
        # 🔥 DIAGNOSTICS - Run every 10 epochs
        # ====================================================================
        if self.use_wandb and self.current_epoch % 10 == 0:
            self._log_diagnostics(all_preds, all_labels)
        
        return avg_loss, avg_mae, avg_rel_mae
    
    def _log_diagnostics(self, all_preds, all_labels):
        """
        Log comprehensive diagnostics to WandB.
        
        Detects if model is stuck predicting mean!
        """
        pred_array = np.array(all_preds)
        label_array = np.array(all_labels)
        
        # ----------------------------------------------------------------
        # 1. R² Score (MOST IMPORTANT!)
        # ----------------------------------------------------------------
        if SKLEARN_AVAILABLE:
            r2 = r2_score(label_array, pred_array)
        else:
            # Manual R² calculation
            ss_res = np.sum((label_array - pred_array) ** 2)
            ss_tot = np.sum((label_array - label_array.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # ----------------------------------------------------------------
        # 2. Prediction Statistics
        # ----------------------------------------------------------------
        pred_mean = pred_array.mean()
        pred_std = pred_array.std()
        pred_min = pred_array.min()
        pred_max = pred_array.max()
        
        # ----------------------------------------------------------------
        # 3. Dataset Statistics (for comparison)
        # ----------------------------------------------------------------
        dataset_mean = label_array.mean()
        dataset_std = label_array.std()
        
        # ----------------------------------------------------------------
        # 4. Deviation from Mean
        # ----------------------------------------------------------------
        mean_deviation = np.abs(pred_array - dataset_mean)
        mean_dev_avg = mean_deviation.mean()
        mean_dev_max = mean_deviation.max()
        
        # ----------------------------------------------------------------
        # 5. Residuals
        # ----------------------------------------------------------------
        residuals = pred_array - label_array
        residuals_mean = residuals.mean()
        residuals_std = residuals.std()
        
        # ----------------------------------------------------------------
        # 6. Check if stuck
        # ----------------------------------------------------------------
        is_stuck = False
        stuck_reasons = []
        
        if r2 < 0.05:
            is_stuck = True
            stuck_reasons.append(f"R²={r2:.4f} < 0.05")
        
        if pred_std < 0.01:
            is_stuck = True
            stuck_reasons.append(f"Pred_Std={pred_std:.4f} < 0.01")
        
        if mean_dev_avg < 0.02:
            is_stuck = True
            stuck_reasons.append(f"Mean_Dev={mean_dev_avg:.4f} < 0.02")
        
        # ----------------------------------------------------------------
        # 7. Log to WandB
        # ----------------------------------------------------------------
        log_dict = {
            # === Core Metrics ===
            "diagnostics/r2_score": r2,
            "diagnostics/is_stuck": 1 if is_stuck else 0,
            
            # === Prediction Stats ===
            "diagnostics/pred_mean": pred_mean,
            "diagnostics/pred_std": pred_std,
            "diagnostics/pred_min": pred_min,
            "diagnostics/pred_max": pred_max,
            "diagnostics/pred_range": pred_max - pred_min,
            
            # === Dataset Stats (reference) ===
            "diagnostics/dataset_mean": dataset_mean,
            "diagnostics/dataset_std": dataset_std,
            
            # === Deviation from Mean ===
            "diagnostics/mean_deviation_avg": mean_dev_avg,
            "diagnostics/mean_deviation_max": mean_dev_max,
            
            # === Residuals ===
            "diagnostics/residuals_mean": residuals_mean,
            "diagnostics/residuals_std": residuals_std,
            
            # === Histograms ===
            "diagnostics/prediction_histogram": wandb.Histogram(pred_array),
            "diagnostics/residuals_histogram": wandb.Histogram(residuals),
        }
        
        # Add scatter plot (sample 1000 points to avoid huge plots)
        sample_size = min(1000, len(pred_array))
        indices = np.random.choice(len(pred_array), sample_size, replace=False)
        
        log_dict["diagnostics/prediction_scatter"] = wandb.plot.scatter(
            wandb.Table(
                data=[[label_array[i], pred_array[i]] for i in indices],
                columns=["True", "Predicted"]
            ),
            "True", "Predicted",
            title=f"Predictions vs True (Epoch {self.current_epoch}, R²={r2:.3f})"
        )
        
        wandb.log(log_dict)
        
        # ----------------------------------------------------------------
        # 8. Log warnings
        # ----------------------------------------------------------------
        if is_stuck:
            warning_msg = f"⚠️  MODEL STUCK AT MEAN! Reasons: {', '.join(stuck_reasons)}"
            self.logger.warning(warning_msg)
            
            # Alert in wandb
            if hasattr(wandb, 'alert'):
                wandb.alert(
                    title="Model Stuck at Mean",
                    text=warning_msg,
                    level=wandb.AlertLevel.WARN
                )
        else:
            self.logger.info(
                f"✅ Model learning! R²={r2:.4f}, Pred_Std={pred_std:.4f}"
            )
        
        # ----------------------------------------------------------------
        # 9. Print diagnostic summary
        # ----------------------------------------------------------------
        print("\n" + "="*70)
        print(f"📊 DIAGNOSTICS (Epoch {self.current_epoch})")
        print("="*70)
        print(f"R² Score:           {r2:.4f}  {'❌ STUCK!' if r2 < 0.05 else '✅'}")
        print(f"Pred Mean:          {pred_mean:.4f}  (Dataset: {dataset_mean:.4f})")
        print(f"Pred Std Dev:       {pred_std:.4f}  {'❌ Too low!' if pred_std < 0.01 else '✅'}")
        print(f"Pred Range:         [{pred_min:.4f}, {pred_max:.4f}]")
        print(f"Mean Deviation:     {mean_dev_avg:.4f}  {'❌ Too low!' if mean_dev_avg < 0.02 else '✅'}")
        print(f"Residuals Std:      {residuals_std:.4f}")
        
        if is_stuck:
            print(f"\n⚠️  WARNING: Model appears stuck at mean!")
            print(f"   Consider: Increase learning rate or reduce batch size")
        
        print("="*70 + "\n")
    
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
        
        🔥 ENHANCED: With diagnostics every 10 epochs
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
            
            # Log to WandB
            if self.use_wandb and epoch % self.config.wandb_log_interval == 0:
                wandb.log({
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'train/mae': train_mae,
                    'val/mae': val_mae,
                    'learning_rate': current_lr,
                    'patience_counter': self.patience_counter,
                    'best/train_mae': self.best_train_mae,
                    'best/val_mae': self.best_val_mae
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
                    self.logger.info(f"  ✓ New best model! Val loss: {val_loss:.4f}")
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


if __name__ == "__main__":
    print("="*70)
    print("TRAINER MODULE (WITH DIAGNOSTICS)")
    print("="*70)
    
    print("\n🔥 ENHANCED FEATURES:")
    print("  ✅ R² Score tracking")
    print("  ✅ Prediction variance monitoring")
    print("  ✅ Mean prediction detection")
    print("  ✅ Gradient norm tracking")
    print("  ✅ Scatter plots & histograms")
    print("  ✅ Automatic stuck detection")
    
    print("\n📊 DIAGNOSTIC METRICS:")
    print("  - diagnostics/r2_score")
    print("  - diagnostics/pred_std")
    print("  - diagnostics/mean_deviation_avg")
    print("  - diagnostics/gradient_norm_total")
    print("  - diagnostics/prediction_scatter")
    print("  - diagnostics/is_stuck (0 or 1)")
    
    print("\n⚠️  ALERTS:")
    print("  - R² < 0.05 → Model stuck at mean!")
    print("  - Pred_Std < 0.01 → All predictions identical!")
    print("  - Gradient_Norm < 1e-5 → Gradients vanishing!")
    
    print("\n" + "="*70)