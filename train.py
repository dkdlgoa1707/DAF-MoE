import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.configs.default_config import DAFConfig
from src.utils.common import seed_everything, get_logger  
from src.data.loader import get_dataloaders
from src.trainer import Trainer
from src.models.factory import create_model
from src.losses.daf_moe_loss import DAFLoss

def load_config(yaml_path):
    config = DAFConfig()
    with open(yaml_path, 'r') as f:
        exp_args = yaml.safe_load(f)
    for k, v in exp_args.items():
        if hasattr(config, k): setattr(config, k, v)
    
    if config.data_config_path:
        with open(config.data_config_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
            return config, data_cfg
    raise ValueError("Experiment YAML must contain 'data_config_path'")

def main(args):
    # [연결 2] 시드 고정 (가장 먼저 실행)
    seed_everything(args.seed)
    
    # [연결 3] 로거 생성
    # logs/SMC_Mortality/ 폴더 아래에 로그 저장
    config, data_cfg = load_config(args.config)
    log_dir = os.path.join("logs", config.dataset_name)
    logger = get_logger(log_dir)
    
    logger.info(f"🚀 Experiment Start: {config.dataset_name}")
    logger.info(f"📄 Config: {args.config}")

    # 1. GPU Setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"⚙️  Device: {device} (IDs: {args.gpu_ids})")

    # 2. Data Loading
    logger.info("📂 Loading Data...")
    train_loader, val_loader, _ = get_dataloaders(config, data_cfg)

    # 3. Model Build
    model = create_model(config)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    logger.info(f"🧠 Model Built: {config.n_layers} Layers, {config.n_experts} Experts")

    # 4. Optimizer & Loss
    criterion = DAFLoss(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # 5. Trainer Init (Logger 전달!)
    # [연결 4] 만들어둔 logger를 Trainer에게 넘겨줌
    trainer = Trainer(model, criterion, optimizer, config, device, logger)
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed') # 시드 인자 추가
    args = parser.parse_args()
    main(args)