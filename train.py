import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

# [Import] 
from src.configs.default_config import DAFConfig
from src.utils.common import seed_everything, get_logger
from src.data.loader import get_dataloaders
from src.trainer import Trainer

# [New] 공장(Factory) Import
from src.models.factory import create_model
from src.losses.factory import create_criterion

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
    raise ValueError("YAML must contain 'data_config_path'")

def main(args):
    # 1. Setup
    seed_everything(args.seed)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config, data_cfg = load_config(args.config)
    
    # Logger 설정
    log_dir = os.path.join("logs", config.dataset_name)
    logger = get_logger(log_dir)
    logger.info(f"🚀 Experiment: {config.model_name} on {config.dataset_name}")

    # 2. Data Load
    train_loader, val_loader, _ = get_dataloaders(config, data_cfg)

    # 3. Model Build (Factory 사용!)
    model = create_model(config)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # 4. Loss & Optimizer (Factory 사용!)
    criterion = create_criterion(config, device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 5. Run Trainer
    trainer = Trainer(model, criterion, optimizer, config, device, logger)
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)