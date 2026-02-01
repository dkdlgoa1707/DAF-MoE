"""
Default Configuration for DAF-MoE
=================================
This module defines the DAFConfig dataclass, which serves as a superset 
of all hyperparameters for DAF-MoE and baseline models.
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DAFConfig:
    """
    Superset Configuration containing parameters for all models:
    DAF-MoE, FT-Transformer, ResNet, and MLP.
    """

    # ==========================================
    # 1. Experiment Meta & Optimization
    # ==========================================
    model_name: str = "daf_moe"
    data_config_path: str = "" 
    dataset_name: str = "default"
    gpu_ids: str = "0"
    seed: int = 42
    
    # Target metric for optimization (e.g., 'acc', 'auroc', 'rmse')
    optimize_metric: str = "acc" 

    # ==========================================
    # 2. Data Constraints (Auto-filled by DataLoader)
    # ==========================================
    n_numerical: int = 0
    n_categorical: int = 0
    n_features: int = 0
    total_cats: int = 0
    
    task_type: str = 'classification'  # 'classification' or 'regression'
    out_dim: int = 1

    # ==========================================
    # 3. Training Hyperparameters
    # ==========================================
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 16

    # ==========================================
    # 4. Shared Model Architecture
    # ==========================================
    n_layers: int = 3       # Number of transformer blocks or residual layers
    dropout: float = 0.1
    
    # Embedding / Token Dimension (D)
    d_emb: int = 128                
    d_token: Optional[int] = None   # Alias for d_emb used in some HPO scripts
    
    # Internal Dimension Factors
    d_ff_factor: float = 1.33       # FFN expansion factor
    d_hidden_factor: float = 2.0    # Hidden layer factor for ResNet
    
    # Transformer Specific
    n_heads: int = 8
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    
    # ==========================================
    # 5. DAF-MoE Specific Parameters
    # ==========================================
    n_experts: int = 4              # Total number of experts (N_E)
    top_k: int = 2                  # Number of selected experts (K)
    d_ff: Optional[int] = None      # Internal FFN dimension (auto-calculated if None)
    
    router_noise_std: float = 0.1   # Noise epsilon for exploration
    mu_init_strategy: str = "linspace" # Initialization for centroids mu_k
    
    # Auxiliary Loss Weights (Equation 15)
    lambda_spec: float = 0.1        # Specialization loss weight
    lambda_repel: float = 0.1       # Centroid repulsion loss weight
    lambda_bal: float = 0.001       # Load balancing loss weight

    # ==========================================
    # 6. Feature Preprocessing
    # ==========================================
    n_quantiles: int = 1000         # For quantile transformation (Phi calculation)
    subsample: int = 100000         # Max samples for computing global stats

    # ==========================================
    # 7. Ablation Control
    # ==========================================
    use_raw_path: bool = True       # Preservation Path (Raw Signal)
    use_deep_path: bool = True      # Transformation Path (Non-linear FFN)
    use_dist_token: bool = True     # Distributional Metadata Injection (m_j)
    
    # Result storage directory
    result_dir: str = "results/scores"