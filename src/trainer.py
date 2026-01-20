import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.metrics import Evaluator

class Trainer:
    def __init__(self, model, criterion, optimizer, config, device, logger): # [연결] logger 받음
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = logger # [저장]
        self.evaluator = Evaluator(task_type=config.task_type)
        
        self.best_metric = -float('inf') if config.task_type == 'classification' else float('inf')
        os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        
        # Tqdm은 터미널에만 보여주고 싶으므로 file=sys.stdout 등 처리가 필요하지만
        # 여기선 간단히 pbar 유지. (로그 파일엔 pbar 안 찍힘)
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]")
        
        for inputs, targets in pbar:
            for k in inputs: inputs[k] = inputs[k].to(self.device)
            targets = targets.to(self.device)

            logits, hist, meta = self.model(**inputs)
            losses = self.criterion(logits, targets, hist, meta)

            self.optimizer.zero_grad()
            losses['total'].backward()
            self.optimizer.step()

            total_loss += losses['total'].item()
            pbar.set_postfix({'Loss': f"{losses['total'].item():.4f}"})
            
        return total_loss / len(loader)

    def validate(self, loader, epoch):
        self.model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for inputs, targets in loader:
                for k in inputs: inputs[k] = inputs[k].to(self.device)
                targets = targets.to(self.device)
                
                logits, _, _ = self.model(**inputs)
                val_preds.append(logits)
                val_targets.append(targets)
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        metrics = self.evaluator(val_targets, val_preds)
        
        self._log_and_save(metrics, epoch)

    def _log_and_save(self, metrics, epoch):
        # [수정] print -> logger.info
        log_msg = f"Epoch {epoch+1} [Val] "
        for k, v in metrics.items():
            log_msg += f"{k.upper()}: {v:.4f} | "
        
        self.logger.info(log_msg) # 파일과 화면 동시에 출력됨

        score = metrics.get('auroc', -metrics.get('mse', 0))
        if score > self.best_metric:
            self.best_metric = score
            save_path = f"checkpoints/{self.config.dataset_name}_best.pth"
            
            if isinstance(self.model, nn.DataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
                
            torch.save(state_dict, save_path)
            self.logger.info(f"   🏆 New Best Model Saved! ({self.best_metric:.4f}) at {save_path}")

    def fit(self, train_loader, val_loader):
        self.logger.info("\n🔥 Start Training Loop...")
        for epoch in range(self.config.epochs):
            self.train_epoch(train_loader, epoch)
            self.validate(val_loader, epoch)
        self.logger.info(f"\n✅ Training Finished. Best Score: {self.best_metric:.4f}")