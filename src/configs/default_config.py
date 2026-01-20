from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DAFConfig:
    """
    [Superset Configuration]
    DAF-MoE 뿐만 아니라, 비교 실험할 Baseline 모델들의 설정까지 모두 포함합니다.
    사용하지 않는 모델의 파라미터는 무시됩니다.
    """

    # ==========================================
    # 1. Experiment Meta & Model Selection
    # ==========================================
    model_name: str = "daf_moe"  # [핵심] 사용할 모델 이름 ('daf_moe', 'tabnet', 'ft_transformer')
    
    data_config_path: str = "" 
    dataset_name: str = "default"
    gpu_ids: str = "0"
    
    # [Data Constraints] (Auto-filled)
    n_numerical: int = 0
    n_categorical: int = 0
    n_features: int = 0
    total_cats: int = 0

    # ==========================================
    # 2. Common Hyperparameters (모든 모델 공통)
    # ==========================================
    task_type: str = 'classification'
    out_dim: int = 1
    
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Backbone 공통
    d_emb: int = 64            # Hidden Dimension
    n_layers: int = 4          # Depth
    n_heads: int = 4           # Attention Heads
    dropout: float = 0.1

    # ==========================================
    # 3. [Model Specific] DAF-MoE
    # ==========================================
    n_experts: int = 4
    top_k: int = 2
    d_ff: int = 128
    router_noise_std: float = 0.1
    mu_init_strategy: str = 'uniform'
    
    lambda_spec: float = 0.1
    lambda_repel: float = 0.01
    lambda_bal: float = 0.001

    # ==========================================
    # 4. [Model Specific] TabNet (Example)
    # ==========================================
    # TabNet을 쓸 때만 이 값들이 사용됨
    n_d: int = 8               # Decision prediction layer dim
    n_a: int = 8               # Attention layer dim
    n_steps: int = 3           # Architecture steps
    gamma: float = 1.3         # Coefficient for feature reusage
    
    # ==========================================
    # 5. Preprocessing Params
    # ==========================================
    n_quantiles: int = 1000
    subsample: int = 100000