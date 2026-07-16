"""
Model Trainer Module
====================
Manages the training loop, validation, checkpointing, and testing processes.
Handles both standard baselines and DAF-MoE specific loss calculations.
"""

import copy
import os
import json
import time 
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from src.utils.metrics import Evaluator
from src.data.provenance import stable_hash

def compute_baseline_loss(outputs, targets, criterion, out_dim):
    """Use mean member loss for ensembles and ordinary task loss otherwise."""
    logits = outputs['logits']
    logits_k = outputs.get('logits_k')
    if logits_k is None:
        if out_dim == 1:
            return criterion(logits, targets.float().view_as(logits))
        return criterion(logits, targets.long().view(-1))

    batch_size, n_members, member_out_dim = logits_k.shape
    if out_dim == 1:
        expanded_targets = targets.float().view(batch_size, 1, 1).expand(
            -1, n_members, -1
        )
        return criterion(logits_k, expanded_targets)
    expanded_targets = targets.long().view(batch_size, 1).expand(-1, n_members)
    return criterion(
        logits_k.reshape(-1, member_out_dim), expanded_targets.reshape(-1)
    )

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

        self._retrieval_train_loader = None
        self.best_epoch = -1
        self.epochs_completed = 0
        self.stopped_early = False
        self._epochs_without_improvement = 0
        self._best_state_dict = None
        self._last_validation_improved = False

    def _get_retrieval_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _wire_retrieval_context(self, train_loader):
        """Load the fixed training split into retrieval-based baselines once."""
        model = self._get_retrieval_model()
        has_candidates = hasattr(model, 'set_candidates')
        has_train_context = hasattr(model, 'set_train_context')
        if not (has_candidates or has_train_context):
            return

        input_parts = {}
        target_parts = []
        for inputs, targets in train_loader:
            for key, value in inputs.items():
                input_parts.setdefault(key, []).append(value)
            target_parts.append(targets)

        if not target_parts:
            raise ValueError("Cannot wire retrieval context from an empty train loader.")

        context_inputs = {
            key: torch.cat(parts, dim=0).detach().cpu()
            for key, parts in input_parts.items()
        }
        context_targets = torch.cat(target_parts, dim=0).detach().cpu()

        if has_candidates:
            model.set_candidates(context_inputs, context_targets)
            self.logger.info(
                f"TabR retrieval candidates wired: {len(context_targets)} rows"
            )
        if has_train_context:
            model.set_train_context(context_inputs, context_targets)
            self.logger.info(
                f"ModernNCA train context wired: {len(context_targets)} rows"
            )

        provenance = (
            model.candidate_provenance()
            if hasattr(model, 'candidate_provenance')
            else None
        )
        if provenance is not None and hasattr(self.config, 'phase2_manifest'):
            manifest = self.config.phase2_manifest
            manifest['retrieval_candidates'] = provenance
            payload = {key: value for key, value in manifest.items() if key != 'manifest_hash'}
            manifest['manifest_hash'] = stable_hash(payload)
        self.config.retrieval_candidate_provenance = provenance

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
                base_loss = compute_baseline_loss(
                    outputs,
                    targets,
                    self.criterion,
                    self.config.out_dim,
                )
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

        if is_best:
            model = (
                self.model.module
                if isinstance(self.model, nn.DataParallel)
                else self.model
            )
            self._best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        self._last_validation_improved = is_best

        checkpoint_path = getattr(self.config, 'checkpoint_path', None)
        if is_best and (self.verbose or checkpoint_path):
            save_path = checkpoint_path or (
                f"checkpoints/{self.config.dataset_name}_{self.config.model_name}_"
                f"seed{self.config.seed}_best.pth"
            )
            checkpoint_dir = os.path.dirname(save_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(self._best_state_dict, save_path)

        if is_best and self.verbose:
            self.logger.info(f"    🏆 New Best Model Saved (Seed {self.config.seed})! ({self.target_metric.upper()}: {self.best_metric:.4f})")
        return is_best

    def fit(self, train_loader, val_loader):
        self._retrieval_train_loader = train_loader
        self._wire_retrieval_context(train_loader)

        if self.verbose:
            self.logger.info(f"\n🔥 Start Training Loop (Seed: {self.config.seed})...")
            
        start_time = time.time()
        for epoch in range(self.config.epochs):
            self.train_epoch(train_loader, epoch)
            metrics = self.validate(val_loader, epoch)
            self.epochs_completed = epoch + 1
            patience = int(getattr(self.config, 'patience', 0))
            if (
                metrics
                and patience > 0
                and self._epochs_without_improvement >= patience
            ):
                self.stopped_early = True
                if self.verbose:
                    self.logger.info(
                        f"Early stopping at epoch {epoch + 1}; "
                        f"best epoch was {self.best_epoch + 1}."
                    )
                break
        total_train_time = time.time() - start_time

        if self._best_state_dict is not None:
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            model.load_state_dict(self._best_state_dict)
            
        if self.verbose:
            self.logger.info(f"\n✅ Training Finished. Best {self.target_metric.upper()}: {self.best_metric:.4f} (Time: {total_train_time:.2f}s)")
        
        return self.best_metric

    def test(self, test_loader, train_loader=None):
        """Loads the best checkpoint and performs final evaluation."""
        context_loader = train_loader or self._retrieval_train_loader
        if context_loader is not None:
            self._wire_retrieval_context(context_loader)

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
