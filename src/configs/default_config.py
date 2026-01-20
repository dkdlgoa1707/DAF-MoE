from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DAFConfig:
    # [Data Constraints] (데이터 로드 후 동적 설정됨)
    n_numerical: int = 0
    n_categorical: int = 0
    total_cats: int = 0       # 임베딩 Vocab Size
    n_features: int = 0       # n_num + n_cat
    
    # [Model Architecture]
    d_emb: int = 64           # 임베딩 차원
    n_layers: int = 4         # 트랜스포머 레이어 수
    n_heads: int = 4          # 어텐션 헤드 수
    dropout: float = 0.1
    
    # [MoE Specifics]
    n_experts: int = 4
    top_k: int = 2
    d_ff: int = 128           # Expert 내부 FFN 차원
    router_noise_std: float = 0.1
    mu_init_strategy: str = 'uniform' # 'uniform' or 'custom'
    initial_mu: Optional[List[float]] = None
    
    # [Preprocessing]
    n_quantiles: int = 1000   # QuantileTransformer 해상도
    subsample: int = 100000   # 전처리 샘플링 개수
    
    # [Training]
    task_type: str = 'classification' # 'classification' or 'regression'
    out_dim: int = 1          # 이진분류=1, 다중분류=N
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # [Loss Weights]
    lambda_spec: float = 0.1   # 전문가 유도 (Induction)
    lambda_repel: float = 0.01 # 전문가 다양성 (Diversity)
    lambda_bal: float = 0.001  # 붕괴 방지 (Safety Net)