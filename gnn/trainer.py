"""
Trainer for Diffusion Barrier Prediction
FIXED: Restored sample count in WandB run name.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import shutil
import math
import numpy as np

# Wandb import
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
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, level.upper()))
    
    if also_console:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    if also_console: ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    if also_console: logger.addHandler(ch)
    return logger


class Trainer:
    """
    Trainer for GNN diffusion barrier prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        save_dir: str = "checkpoints",
        cycle: int = None,
        is_final_model: bool = False
    ):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cycle = cycle
        self.is_final_model = is_final_model
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.patience = config.final_model_patience if is_final_model else config.patience
        self.patience_counter = 0
        self.best_val_mae = float('inf')
        self.best_val_rel_mae = float('inf')
        
        self.current_epoch = 0
        self.history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
        
        # Normalization statistics (will be computed in train())
        self.target_mean = 0.0
        self.target_std = 1.0
        
        # Logger Setup
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        name = f"trainer_{cycle if cycle is not None else 'final'}"
        log_file = log_dir / (f"training_cycle_{cycle}.log" if cycle is not None else "training_final.log")
        self.logger = setup_logger(name, log_file, config.log_level, config.log_to_console)
        
        # Optimizations
        if getattr(config, 'cudnn_benchmark', False) and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        self.criterion = self._get_loss_function()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.use_amp = getattr(config, 'use_amp', False) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        self.scheduler = self._get_scheduler() if config.use_scheduler else None
        
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb: self._init_wandb()
    
    def _compute_target_stats(self, loader):
        """Compute mean and std of targets in training set for normalization."""
        all_barriers = []
        for _, _, barriers, _ in loader:
            all_barriers.append(barriers)
        
        all_barriers = torch.cat(all_barriers)
        self.target_mean = all_barriers.mean().item()
        self.target_std = all_barriers.std().item()
        
        if self.target_std < 1e-6: self.target_std = 1.0 # Prevent div by zero
        
        self.logger.info(f"Target Normalization stats: Mean={self.target_mean:.4f}, Std={self.target_std:.4f}")

    def _get_loss_function(self):
        loss_type = self.config.loss_function.lower()
        if loss_type == "mse": return nn.MSELoss(reduction='none')
        elif loss_type in ["mae", "l1"]: return nn.L1Loss(reduction='none')
        elif loss_type == "huber": return nn.HuberLoss(reduction='none')
        else: return nn.MSELoss(reduction='none')
    
    def _get_scheduler(self):
        scheduler_type = self.config.scheduler_type.lower()
        if scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.config.plateau_factor, 
                patience=self.config.plateau_patience
            )
        # Add other schedulers if needed
        return None
    
    def _init_wandb(self):
        if not WANDB_AVAILABLE: return
        
        # --- FIX: Retrieve n_samples explicitly ---
        n_samples = getattr(self.config, '_current_dataset_size', None)
        
        if self.config.wandb_run_name:
            run_name = self.config.wandb_run_name
        else:
            # Pass n_samples to get_experiment_name so it appears in WandB
            run_name = self.config.get_experiment_name(n_samples=n_samples, cycle=self.cycle)
            
        if self.is_final_model: run_name = f"FINAL-{run_name}"
        
        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config=vars(self.config),
            reinit=True
        )
        if self.config.wandb_watch_model:
            wandb.watch(self.model, log="all", log_freq=self.config.wandb_watch_freq)

    def _calculate_weighted_loss(self, predictions, targets, progress):
        """
        Calculate loss weighted by relaxation progress.
        Unrelaxed (Progress 0.0) -> Weight 1.0
        Relaxed (Progress 1.0) -> Weight 0.2 (or as configured)
        """
        raw_loss = self.criterion(predictions, targets)
        
        # Weight formula: 1.0 - (0.8 * progress) -> Prioritize unrelaxed
        weights = 1.0 - (0.8 * progress)
        weights = weights.view_as(raw_loss)
        
        return (raw_loss * weights).mean()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_mae = 0
        n_batches = 0
        
        for initial_batch, final_batch, barriers, progress_batch in train_loader:
            initial_batch = initial_batch.to(self.device)
            final_batch = final_batch.to(self.device)
            
            # --- FIX: Use view(-1, 1) instead of unsqueeze(1) to avoid broadcasting errors ---
            barriers = barriers.to(self.device).view(-1, 1)
            progress_batch = progress_batch.to(self.device).view(-1, 1)
            
            # Normalize Targets for Loss Calculation
            targets_norm = (barriers - self.target_mean) / self.target_std
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions_norm = self.model(initial_batch, final_batch)
                    loss = self._calculate_weighted_loss(predictions_norm, targets_norm, progress_batch)
                self.scaler.scale(loss).backward()
                
                # Gradient Clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions_norm = self.model(initial_batch, final_batch)
                loss = self._calculate_weighted_loss(predictions_norm, targets_norm, progress_batch)
                loss.backward()
                
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.optimizer.step()
            
            # Metrics (De-normalize for readable MAE)
            with torch.no_grad():
                preds_denorm = predictions_norm * self.target_std + self.target_mean
                mae = torch.abs(preds_denorm - barriers).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            n_batches += 1
        
        return total_loss / n_batches, total_mae / n_batches, 0.0 # RelMAE not computed in train
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rel_mae = 0
        n_batches = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for initial_batch, final_batch, barriers, progress_batch in val_loader:
                initial_batch = initial_batch.to(self.device)
                final_batch = final_batch.to(self.device)
                
                # --- FIX: Use view(-1, 1) here too ---
                barriers = barriers.to(self.device).view(-1, 1)
                progress_batch = progress_batch.to(self.device).view(-1, 1)
                
                # Normalize Targets for Loss Consistency
                targets_norm = (barriers - self.target_mean) / self.target_std
                
                predictions_norm = self.model(initial_batch, final_batch)
                loss = self._calculate_weighted_loss(predictions_norm, targets_norm, progress_batch)
                
                # De-normalize for Real Metrics (eV)
                preds_denorm = predictions_norm * self.target_std + self.target_mean
                
                mae = torch.abs(preds_denorm - barriers).mean()
                rel_mae = (torch.abs(preds_denorm - barriers) / (barriers + 1e-8)).mean()
                
                total_loss += loss.item()
                total_mae += mae.item()
                total_rel_mae += rel_mae.item()
                n_batches += 1
                
                all_preds.extend(preds_denorm.cpu().numpy().flatten())
                all_labels.extend(barriers.cpu().numpy().flatten())
        
        if self.use_wandb and self.current_epoch % 10 == 0:
            self._log_diagnostics(all_preds, all_labels)
        
        return total_loss / n_batches, total_mae / n_batches, total_rel_mae / n_batches
    
    def _log_diagnostics(self, all_preds, all_labels):
        pred_array, label_array = np.array(all_preds), np.array(all_labels)
        if SKLEARN_AVAILABLE:
            r2 = r2_score(label_array, pred_array)
        else:
            ss_res = np.sum((label_array - pred_array) ** 2)
            ss_tot = np.sum((label_array - label_array.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
        wandb.log({
            "diagnostics/r2_score": r2,
            "diagnostics/pred_mean": pred_array.mean(),
            "diagnostics/label_mean": label_array.mean()
        })
        self.logger.info(f"Diagnostics: R²={r2:.4f}, Pred Mean={pred_array.mean():.2f}")

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_mae': self.best_val_mae,
            'target_mean': self.target_mean, # Save stats
            'target_std': self.target_std
        }
        if self.scheduler: checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        if is_best: shutil.copy(filepath, Path(filepath).parent / "best_model.pt")
    
    def train(self, train_loader, val_loader=None, verbose: bool = True):
        # 1. Compute Normalization Stats
        self._compute_target_stats(train_loader)
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            train_loss, train_mae, _ = self.train_epoch(train_loader)
            
            if val_loader:
                val_loss, val_mae, val_rel_mae = self.validate(val_loader)
            else:
                val_loss, val_mae, val_rel_mae = train_loss, train_mae, 0.0
            
            # Use MAE for scheduling/saving (real eV values)
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_val_rel_mae = val_rel_mae
                self.patience_counter = 0
                self.save_checkpoint(str(self.save_dir / f"checkpoint_epoch_{epoch}.pt"), is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and epoch % self.config.wandb_log_interval == 0:
                self.logger.info(f"Ep {epoch:3d} | Loss: {train_loss:.4f}/{val_loss:.4f} | MAE: {train_mae:.4f}/{val_mae:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'train/loss': train_loss, 'val/loss': val_loss,
                    'train/mae': train_mae, 'val/mae': val_mae,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            if self.scheduler:
                # Scheduler on VALIDATION LOSS (which is weighted/normalized) or MAE?
                # Usually Plateau works best on the objective function (Loss)
                self.scheduler.step(val_loss)
                
        if self.use_wandb: wandb.finish()