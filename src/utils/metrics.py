"""
Evaluation Metrics Calculator
=============================
Handles calculation of standard metrics for classification and regression tasks.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, 
    mean_squared_error, r2_score, mean_absolute_error, average_precision_score
)
import torch.nn.functional as F

class Evaluator:
    """
    Computes performance metrics based on task type.
    
    Args:
        task_type (str): 'classification' or 'regression'.
    """
    def __init__(self, task_type):
        self.task_type = task_type

    def __call__(self, targets, preds):
        """
        Args:
            targets (Tensor): Ground truth labels.
            preds (Tensor): Model raw outputs (logits).
            
        Returns:
            dict: Dictionary containing calculated metrics.
        """
        y_true = targets.detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy()
        results = {}

        if self.task_type == 'classification':
            # 1. Binary Classification
            if y_pred.shape[1] == 1:
                # Logits -> Probabilities (Sigmoid)
                y_prob = 1 / (1 + np.exp(-y_pred.flatten()))
                
                y_true_eval = y_true.flatten()
                y_pred_label = (y_prob > 0.5).astype(int)
                
                # Basic Metrics
                results['acc'] = accuracy_score(y_true_eval, y_pred_label)
                results['f1'] = f1_score(y_true_eval, y_pred_label, average='macro')
                
                # AUROC / AUPRC
                try: 
                    results['auroc'] = roc_auc_score(y_true_eval, y_prob)
                except ValueError: 
                    results['auroc'] = 0.5 # Fail-safe for single-class batches
                
                try:
                    results['auprc'] = average_precision_score(y_true_eval, y_prob)
                except ValueError:
                    results['auprc'] = 0.0
                
            # 2. Multi-Class Classification
            else:
                y_prob = F.softmax(torch.tensor(y_pred), dim=1).numpy()
                y_true_eval = y_true.flatten().astype(int)
                y_pred_label = np.argmax(y_prob, axis=1)
                
                results['acc'] = accuracy_score(y_true_eval, y_pred_label)
                results['f1'] = f1_score(y_true_eval, y_pred_label, average='macro')
                
                try: 
                    results['auroc'] = roc_auc_score(y_true_eval, y_prob, multi_class='ovr')
                except ValueError: 
                    results['auroc'] = 0.5
                
                # AUPRC is generally omitted for multi-class tasks in tabular benchmarks
                results['auprc'] = 0.0
        
        # Regression
        else:
            y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
            results['mse'] = mean_squared_error(y_true_flat, y_pred_flat)
            results['rmse'] = np.sqrt(results['mse'])
            results['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)
            results['r2'] = r2_score(y_true_flat, y_pred_flat) 
            
        return results