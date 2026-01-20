from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DAFConfig:
    """
    [DAF-MoE Configuration Schema]
    모든 하이퍼파라미터의 기본값을 정의합니다.
    실험 시 YAML 파일(experiments/*.yaml)을 통해 이 값들을 Override 합니다.
    """

    # ==========================================
    # 1. Experiment Meta (실험 연결 정보)
    # ==========================================
    # 사용할 데이터셋 명세 파일 경로 (예: configs/datasets/mimic.yaml)
    data_config_path: str = "" 
    dataset_name: str = "default"

    # ==========================================
    # 2. Data Constraints (데이터 로드 후 자동 계산됨)
    # ==========================================
    n_numerical: int = 0       # 수치형 변수 개수
    n_categorical: int = 0     # 범주형 변수 개수
    n_features: int = 0        # 전체 입력 피처 개수 (Num + Cat)
    total_cats: int = 0        # 임베딩할 전체 고유 범주 수 (Vocab Size)

    # ==========================================
    # 3. Model Architecture (Transformer Bone)
    # ==========================================
    d_emb: int = 64            # 임베딩 차원 (Hidden Dim)
    n_layers: int = 4          # Transformer 블록 수
    n_heads: int = 4           # Multi-head Attention 헤드 수
    dropout: float = 0.1       # Dropout 비율

    # ==========================================
    # 4. MoE Specifics (DAF-MoE Core)
    # ==========================================
    n_experts: int = 4         # 전문가 수
    top_k: int = 2             # 라우팅 시 선택할 전문가 수
    d_ff: int = 128            # Expert 내부 FFN 차원 (보통 d_emb * 2 or 4)
    router_noise_std: float = 0.1  # 라우팅 노이즈 (Exploration 유도)
    
    # 분포 중심점(mu) 초기화 전략 ('uniform' 권장)
    mu_init_strategy: str = 'uniform' 

    # ==========================================
    # 5. Preprocessing Params
    # ==========================================
    n_quantiles: int = 1000    # QuantileTransformer 해상도
    subsample: int = 100000    # 전처리 시 통계 계산에 사용할 샘플 수

    # ==========================================
    # 6. Training Hyperparameters
    # ==========================================
    task_type: str = 'classification' # 'classification' or 'regression'
    out_dim: int = 1           # 이진분류=1, 다중분류=N
    
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # GPU 설정 (YAML에서 지정 가능하도록 추가)
    gpu_ids: str = "0"

    # ==========================================
    # 7. Loss Weights (중요!)
    # ==========================================
    lambda_spec: float = 0.1   # L_spec: 전문가가 특정 분포(mu)를 맡도록 유도
    lambda_repel: float = 0.01 # L_repel: 전문가들이 서로 다른 분포를 맡도록 밀어냄
    lambda_bal: float = 0.001  # L_bal: 한 전문가에게만 쏠리는 것 방지 (Safety Net)