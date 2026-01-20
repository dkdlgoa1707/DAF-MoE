import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

class Evaluator:
    def __init__(self, task_type='classification'):
        self.task_type = task_type

    def __call__(self, y_true, y_logits):
        """
        y_true: (N,) 정답 레이블 (numpy/list)
        y_logits: (N, 1) or (N, C) 모델 출력 로짓 (numpy/list)
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_logits, torch.Tensor):
            y_logits = y_logits.detach().cpu().numpy()

        results = {}
        
        if self.task_type == 'classification':
            # Binary Classification 가정 (Sigmoid 적용)
            y_probs = 1 / (1 + np.exp(-y_logits))
            y_pred = (y_probs > 0.5).astype(int)
            
            # 1. Basic Metrics
            results['acc'] = accuracy_score(y_true, y_pred)
            results['f1'] = f1_score(y_true, y_pred, average='binary')
            results['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
            results['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
            
            # 2. Advanced Metrics (AUROC) - 클래스가 2개 이상 존재해야 계산 가능
            try:
                results['auroc'] = roc_auc_score(y_true, y_probs)
            except ValueError:
                results['auroc'] = 0.0
                
        else: # Regression
            # MSE, MAE 등
            results['mse'] = np.mean((y_true - y_logits)**2)
            
        return results