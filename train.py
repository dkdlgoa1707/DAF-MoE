"""
Main Training Script
====================

Description:
    The central entry point for training and evaluating DAF-MoE and baseline models.
    It handles configuration loading, data preparation, model initialization, 
    and the training loop via the `Trainer` class.

    It also supports on-the-fly configuration overrides for ablation studies
    (e.g., disabling specific paths or tokens via command line arguments).

Usage:
    python train.py --config configs/experiments/adult_daf_moe_best.yaml --gpu_ids 0
"""

import argparse
import os
import yaml
import torch
import torch.optim as optim

from src.configs.default_config import DAFConfig
from src.utils.common import seed_everything, get_logger
from src.data.loader import get_dataloaders
from src.trainer import Trainer
from src.models.factory import create_model
from src.losses.factory import create_criterion

class DummyLogger:
    """A no-op logger for silent execution (used during tuning)."""
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

def load_config(yaml_path):
    """Loads configuration from a YAML file and its linked data config."""
    config = DAFConfig()
    with open(yaml_path, 'r') as f:
        exp_args = yaml.safe_load(f)
    for k, v in exp_args.items():
        if hasattr(config, k): setattr(config, k, v)
    
    # Load associated data configuration
    if config.data_config_path:
        with open(config.data_config_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
            if 'dataset_name' in data_cfg:
                config.dataset_name = data_cfg['dataset_name']
            return config, data_cfg
            
    raise ValueError("The experiment YAML must contain 'data_config_path'.")

def str2bool(v):
    """Converts string representation of booleans to actual boolean values."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    # 1. Reproducibility
    seed_everything(args.seed)
    is_verbose = getattr(args, 'verbose', True)

    # 2. Device Setup
    if args.gpu_ids is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Load Config
    config, data_cfg = load_config(args.config)
    config.seed = args.seed
    config.result_dir = args.result_dir 

    # -----------------------------------------------------------
    # [Ablation Support] Dynamic Config Override
    # Appends suffixes to model_name based on disabled components.
    # -----------------------------------------------------------
    suffix = ""
    if args.use_raw_path is not None: 
        config.use_raw_path = str2bool(args.use_raw_path)
        if not config.use_raw_path: suffix += "_wo_raw"
            
    if args.use_deep_path is not None: 
        config.use_deep_path = str2bool(args.use_deep_path)
        if not config.use_deep_path: suffix += "_wo_deep"
            
    if args.use_dist_token is not None: 
        config.use_dist_token = str2bool(args.use_dist_token)
        if not config.use_dist_token: suffix += "_wo_token"
            
    if args.lambda_spec is not None: config.lambda_spec = float(args.lambda_spec)
    if args.lambda_repel is not None: config.lambda_repel = float(args.lambda_repel)

    # Loss Ablation Suffixes
    if (config.lambda_spec == 0.0) and (config.lambda_repel == 0.0):
        suffix += "_wo_aux"
    elif config.lambda_spec == 0.0:
        suffix += "_wo_spec"
    elif config.lambda_repel == 0.0:
        suffix += "_wo_repel"

    config.model_name = config.model_name + suffix
    # -----------------------------------------------------------

    # Data Path Override (Optional)
    if args.data_root:
        filename = os.path.basename(data_cfg.get('csv_path', ''))
        data_cfg['csv_path'] = os.path.join(args.data_root, filename)
    
    # 4. Logger Setup
    if is_verbose:
        log_dir = os.path.join("logs", config.dataset_name)
        logger = get_logger(log_dir, log_name=config.model_name)
        logger.info(f"🚀 Experiment: {config.model_name} on {config.dataset_name} (Seed {config.seed})")
    else:
        logger = DummyLogger()

    # 5. Pipeline Initialization
    train_loader, val_loader, test_loader = get_dataloaders(config, data_cfg)
    model = create_model(config).to(device)
    criterion = create_criterion(config, device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 6. Run Training
    trainer = Trainer(model, criterion, optimizer, config, device, logger, verbose=is_verbose)
    best_val_score = trainer.fit(train_loader, val_loader)
    
    # 7. Final Test & Return
    if is_verbose:
        metrics = trainer.test(test_loader)
        return metrics.get(getattr(config, 'optimize_metric', 'acc'), 0.0)
    else:
        return best_val_score
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to experiment config yaml")
    parser.add_argument('--gpu_ids', type=str, default="0", help="Visible GPU IDs (e.g., '0,1')")
    parser.add_argument('--data_root', type=str, default=None, help="Optional override for data directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--verbose', action='store_true', default=True, help="Enable logging and progress bars")
    
    # Result Directory
    parser.add_argument('--result_dir', type=str, default="results/scores", help="Directory to save JSON results")

    # Ablation Arguments (Optional overrides)
    parser.add_argument('--use_raw_path', type=str, default=None)
    parser.add_argument('--use_deep_path', type=str, default=None)
    parser.add_argument('--use_dist_token', type=str, default=None)
    parser.add_argument('--lambda_spec', type=float, default=None)
    parser.add_argument('--lambda_repel', type=float, default=None)

    args = parser.parse_args()
    main(args)