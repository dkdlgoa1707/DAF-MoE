import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.metrics import Evaluator

class Trainer:
    def __init__(self, model, criterion, optimizer, config, device, logger):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = logger
        self.evaluator = Evaluator(task_type=config.task_type)
        
        self.best_metric = -float('inf') if config.task_type == 'classification' else float('inf')
        os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]")
        
        for inputs, targets in pbar:
            for k in inputs: inputs[k] = inputs[k].to(self.device)
            targets = targets.to(self.device)

            # -----------------------------------------------------------
            # 1. 모델 실행 (무조건 Dictionary 반환)
            # -----------------------------------------------------------
            outputs = self.model(**inputs)
            logits = outputs['logits']

            # -----------------------------------------------------------
            # 2. Loss 계산 (모델별 분기)
            # -----------------------------------------------------------
            if self.config.model_name == 'daf_moe':
                # DAF-MoE는 hist, meta 정보가 필요함
                losses = self.criterion(
                    logits, targets, 
                    outputs.get('history'), outputs.get('meta')
                )
                final_loss = losses['total']
            else:
                # 일반 모델: logits와 targets만 있으면 됨
                # BCE/MSE는 스칼라 값을 바로 반환
                base_loss = self.criterion(logits, targets.float().view_as(logits))
                
                # 모델 자체 보조 Loss (예: TabNet sparsity) 합산
                aux_loss = outputs.get('aux_loss', 0)
                if aux_loss is None: aux_loss = 0
                
                final_loss = base_loss + aux_loss
            
            # -----------------------------------------------------------
            # 3. 역전파
            # -----------------------------------------------------------
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

            total_loss += final_loss.item()
            pbar.set_postfix({'Loss': f"{final_loss.item():.4f}"})
            
        return total_loss / len(loader)

    def validate(self, loader, epoch):
        self.model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for inputs, targets in loader:
                for k in inputs: inputs[k] = inputs[k].to(self.device)
                targets = targets.to(self.device)
                
                # Validation에서도 Dictionary에서 logits만 꺼냄
                outputs = self.model(**inputs)
                val_preds.append(outputs['logits'])
                val_targets.append(targets)
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        
        metrics = self.evaluator(val_targets, val_preds)
        self._log_and_save(metrics, epoch)

    def _log_and_save(self, metrics, epoch):
        log_msg = f"Epoch {epoch+1} [Val] "
        for k, v in metrics.items():
            log_msg += f"{k.upper()}: {v:.4f} | "
        
        self.logger.info(log_msg)

        # Best Model 저장 로직
        score = metrics.get('auroc', -metrics.get('mse', 0))
        if score > self.best_metric:
            self.best_metric = score
            save_path = f"checkpoints/{self.config.dataset_name}_{self.config.model_name}_best.pth"
            
            if isinstance(self.model, nn.DataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
                
            torch.save(state_dict, save_path)
            self.logger.info(f"   🏆 New Best Saved! ({self.best_metric:.4f})")

    def fit(self, train_loader, val_loader):
        self.logger.info("\n🔥 Start Training...")
        for epoch in range(self.config.epochs):
            self.train_epoch(train_loader, epoch)
            self.validate(val_loader, epoch)
        self.logger.info(f"\n✅ Finished. Best Score: {self.best_metric:.4f}")