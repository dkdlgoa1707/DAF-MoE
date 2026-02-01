"""
Model Trainer Module
====================
Manages the training loop, validation, checkpointing, and testing processes.
Handles both standard baselines and DAF-MoE specific loss calculations.
"""

import os
import json
import time 
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from src.utils.metrics import Evaluator

class Trainer:
    """
    Encapsulates the training pipeline.
    
    Attributes:
        model: PyTorch model instance.
        criterion: Loss function.
        optimizer: Optimizer (e.g., AdamW).
        config: Configuration object.
        device: 'cuda' or 'cpu'.
        evaluator: Metrics calculator.
    """
    def __init__(self, model, criterion, optimizer, config, device, logger, verbose=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = logger
        self.verbose = verbose
        self.evaluator = Evaluator(task_type=config.task_type)
        
        self.lr = float(config.learning_rate)
        
        # Determine optimization metric (Default: ACC or RMSE)
        default_metric = 'acc' if config.task_type == 'classification' else 'rmse'
        self.target_metric = getattr(config, 'optimize_metric', default_metric)
        
        self.minimize_metrics = ['loss', 'rmse', 'mse', 'mae']
        if self.target_metric in self.minimize_metrics:
            self.best_metric = float('inf')
        else:
            self.best_metric = -float('inf')
            
        if self.verbose:
            os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]", disable=not self.verbose)
        
        for inputs, targets in pbar:
            for k in inputs: inputs[k] = inputs[k].to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(**inputs)
            logits = outputs['logits']

            # 1. DAF-MoE: Custom Loss Handling (Task + Aux Losses)
            if self.config.model_name.lower().startswith('daf_moe'):
                loss_dict = self.criterion(logits, targets, outputs.get('history'), outputs.get('psi_x')) # Updated key: 'meta' -> 'psi_x'
                final_loss = loss_dict['total']
            
            # 2. Baselines: Standard Loss
            else:
                if self.config.out_dim == 1:
                     base_loss = self.criterion(logits, targets.float().view_as(logits))
                else:
                     base_loss = self.criterion(logits, targets.long().view(-1))
                final_loss = base_loss + (outputs.get('aux_loss', 0) or 0)
            
            self.optimizer.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += final_loss.item()
            if self.verbose:
                pbar.set_postfix({'Loss': f"{final_loss.item():.4f}"})
            
        return total_loss / len(loader)

    def validate(self, loader, epoch):
        self.model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for inputs, targets in loader:
                for k in inputs: inputs[k] = inputs[k].to(self.device)
                outputs = self.model(**inputs)
                val_preds.append(outputs['logits'])
                val_targets.append(targets)
        
        if not val_preds: return {}
        
        # Calculate metrics on full validation set
        metrics = self.evaluator(torch.cat(val_targets), torch.cat(val_preds))
        self._check_best_and_save(metrics, epoch)
        
        return metrics

    def _check_best_and_save(self, metrics, epoch):
        """Checks if current epoch is the best and saves checkpoint."""
        current_score = metrics.get(self.target_metric)
        if current_score is None:
            current_score = metrics.get('loss', float('inf')) 

        if self.verbose:
            log_msg = f"Epoch {epoch+1} [Val] "
            for k, v in metrics.items():
                log_msg += f"{k.upper()}: {v:.4f} | "
            self.logger.info(log_msg)

        is_best = False
        if self.target_metric in self.minimize_metrics:
            if current_score < self.best_metric:
                is_best = True
                self.best_metric = current_score
        else:
            if current_score > self.best_metric:
                is_best = True
                self.best_metric = current_score

        if is_best and self.verbose:
            # Save checkpoint with seed to prevent overwriting
            save_path = f"checkpoints/{self.config.dataset_name}_{self.config.model_name}_seed{self.config.seed}_best.pth"
            
            state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
            torch.save(state_dict, save_path)
            
            self.logger.info(f"    🏆 New Best Model Saved (Seed {self.config.seed})! ({self.target_metric.upper()}: {self.best_metric:.4f})")

    def fit(self, train_loader, val_loader):
        if self.verbose:
            self.logger.info(f"\n🔥 Start Training Loop (Seed: {self.config.seed})...")
            
        start_time = time.time()
        for epoch in range(self.config.epochs):
            self.train_epoch(train_loader, epoch)
            self.validate(val_loader, epoch)
        total_train_time = time.time() - start_time
            
        if self.verbose:
            self.logger.info(f"\n✅ Training Finished. Best {self.target_metric.upper()}: {self.best_metric:.4f} (Time: {total_train_time:.2f}s)")
        
        return self.best_metric

    def test(self, test_loader):
        """Loads the best checkpoint and performs final evaluation."""
        if not self.verbose: return {}

        load_path = f"checkpoints/{self.config.dataset_name}_{self.config.model_name}_seed{self.config.seed}_best.pth"
        self.logger.info(f"\n🔍 Loading Best Model from: {load_path}")
        
        if not os.path.exists(load_path):
            self.logger.error(f"🚨 Checkpoint not found at {load_path}! Test aborted.")
            return {}

        checkpoint = torch.load(load_path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        test_preds, test_targets = [], []
        
        # --- Inference Time Measurement ---
        # 1. Warmup
        if torch.cuda.is_available():
            dummy_input, _ = next(iter(test_loader))
            for k in dummy_input: dummy_input[k] = dummy_input[k].to(self.device)
            with torch.no_grad(): _ = self.model(**dummy_input)
            torch.cuda.synchronize()

        # 2. Measurement
        start_time = time.time()
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="[Test] Final Evaluation"):
                for k in inputs: inputs[k] = inputs[k].to(self.device)
                outputs = self.model(**inputs)
                test_preds.append(outputs['logits'])
                test_targets.append(targets)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        inference_time = end_time - start_time
        total_samples = len(test_loader.dataset)
        throughput = total_samples / inference_time if inference_time > 0 else 0
        
        # Metrics
        final_metrics = self.evaluator(torch.cat(test_targets), torch.cat(test_preds))
        final_metrics['inference_time_sec'] = inference_time
        final_metrics['throughput_samples_per_sec'] = throughput

        # Logging
        print("\n" + "="*50)
        print(f"📊 Final Test Results (Seed {self.config.seed})")
        print("="*50)
        for k, v in final_metrics.items():
            print(f"  🔹 {k.upper()}: {v:.4f}")
        print("="*50)
        
        self.logger.info(f"FINAL TEST (Seed {self.config.seed}) | " + " | ".join([f"{k}: {v:.4f}" for k, v in final_metrics.items()]))
        
        # Save JSON Log
        result_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": self.config.dataset_name,
            "model": self.config.model_name,
            "seed": self.config.seed,
            "metrics": final_metrics,
            "config": {k: str(v) for k, v in self.config.__dict__.items() if not k.startswith('__')}
        }
        
        save_dir = self.config.result_dir 
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.config.dataset_name}_{self.config.model_name}_seed{self.config.seed}.json")
        
        try:
            with open(save_path, 'w') as f:
                json.dump(result_data, f, indent=4)
            self.logger.info(f"💾 Results saved to: {save_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save results JSON: {str(e)}")
            
        return final_metrics