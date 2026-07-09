"""
Efficiency Measurement Script
==============================
Measures parameter count, training time, and inference throughput
for all DL models (DAF-MoE, FT-Transformer, TabM, ResNet, MLP)
on two representative datasets (Adult, MIMIC-IV).

Usage:
    python measure_efficiency.py --gpu_ids 0

Requirements:
    - Best HPO config yamls must exist under configs/experiments/
    - Named as: {dataset}_{model}_best.yaml
      e.g., adult_daf_moe_best.yaml, adult_tabm_best.yaml

Output:
    - Prints efficiency table to stdout
    - Saves CSV to results/efficiency/efficiency_report.csv
"""

import os
import sys
import time
import csv
import yaml
import argparse
import torch
import torch.optim as optim

# ── CUDA device must be set before importing torch (mirrors train.py) ─────────
def _set_cuda_device_early():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu_ids' and i + 1 < len(sys.argv):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
_set_cuda_device_early()

from src.configs.default_config import DAFConfig
from src.utils.common import seed_everything
from src.data.loader import get_dataloaders
from src.models.factory import create_model
from src.losses.factory import create_criterion
from src.trainer import Trainer

# ── Configuration ──────────────────────────────────────────────────────────────

# Datasets to measure (small + medium-large)
# Covertype excluded for TabM (not yet complete); MIMIC-IV used instead
DATASETS = ["mimic3", "adult"]

# Models to measure (DL only — GBDT excluded)
MODELS = ["daf_moe", "ft_transformer", "resnet", "mlp", "tabm"] #  "tabm"

# Config yaml naming convention: configs/experiments/{dataset}_{model}_best.yaml
CONFIG_DIR = "configs/experiments"

# Output directory
OUTPUT_DIR = "results/efficiency"

# Number of inference repetitions for stable throughput measurement
N_INFERENCE_REPS = 3

# Fixed seed for reproducibility
SEED = 42


# ── Utilities ──────────────────────────────────────────────────────────────────

class SilentLogger:
    """Suppresses all logging output during efficiency measurement."""
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def load_config(yaml_path: str):
    """Replicates train.py load_config logic exactly."""
    config = DAFConfig()
    with open(yaml_path, 'r') as f:
        exp_args = yaml.safe_load(f)
    for k, v in exp_args.items():
        if hasattr(config, k):
            setattr(config, k, v)

    if config.data_config_path:
        with open(config.data_config_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
            if 'dataset_name' in data_cfg:
                config.dataset_name = data_cfg['dataset_name']
            return config, data_cfg

    raise ValueError(f"data_config_path missing in {yaml_path}")


def count_params(model: torch.nn.Module):
    """Returns (total_params, trainable_params) in millions."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def measure_training_time(
    model, criterion, config, train_loader, val_loader, device
) -> float:
    """
    Trains the model with early stopping and returns total wall-clock
    training time in seconds.

    Uses a fresh optimizer to avoid any state from prior runs.
    Verbose is disabled to exclude logging overhead.
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    trainer = Trainer(
        model, criterion, optimizer, config,
        device, SilentLogger(), verbose=False
    )

    # Re-initialise model weights to ensure a fair starting point
    model.apply(_reset_weights)

    start = time.perf_counter()
    trainer.fit(train_loader, val_loader)
    elapsed = time.perf_counter() - start

    return elapsed


def _reset_weights(m):
    """Resets layer weights to random initialisation where possible."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


@torch.no_grad()
def measure_throughput(model, test_loader, device) -> float:
    """
    Measures inference throughput (samples/sec) as the average over
    N_INFERENCE_REPS repetitions, preceded by one warmup pass.

    Returns throughput in thousands of samples per second (K/s).
    """
    model.eval()

    def _run_inference():
        for inputs, _ in test_loader:
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warmup — ensures CUDA kernels are compiled and cached
    _run_inference()

    total_samples = len(test_loader.dataset)
    times = []

    for _ in range(N_INFERENCE_REPS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_inference()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    throughput_ks = (total_samples / avg_time) / 1e3  # K samples/sec
    return throughput_ks


# ── Main Measurement Loop ──────────────────────────────────────────────────────

def run_measurement(gpu_ids: str):
    seed_everything(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = []  # list of dicts for CSV output

    for dataset in DATASETS:
        print(f"{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        dataset_results = []

        for model_name in MODELS:
            config_path = os.path.join(
                CONFIG_DIR, f"{dataset}_{model_name}_best.yaml"
            )

            if not os.path.exists(config_path):
                print(f"  [{model_name}] Config not found: {config_path} — SKIP")
                continue

            print(f"  [{model_name}] Loading config from {config_path}")

            try:
                config, data_cfg = load_config(config_path)
                config.seed = SEED

                # ── Data ──────────────────────────────────────────────────────
                train_loader, val_loader, test_loader = get_dataloaders(
                    config, data_cfg
                )

                # ── Model ─────────────────────────────────────────────────────
                model = create_model(config).to(device)
                criterion = create_criterion(config, device)

                # ── ① Parameter Count ────────────────────────────────────────
                total_m, trainable_m = count_params(model)
                print(f"         Params: {total_m:.2f}M total, "
                      f"{trainable_m:.2f}M trainable")

                # ── ② Training Time ──────────────────────────────────────────
                print(f"         Measuring training time ...")
                train_sec = measure_training_time(
                    model, criterion, config,
                    train_loader, val_loader, device
                )
                print(f"         Train time: {train_sec:.1f}s")

                # ── ③ Inference Throughput ───────────────────────────────────
                # Reload best checkpoint if available; otherwise use trained state
                ckpt_path = (
                    f"checkpoints/{config.dataset_name}_"
                    f"{config.model_name}_seed{SEED}_best.pth"
                )
                if os.path.exists(ckpt_path):
                    state = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(state)
                    print(f"         Loaded checkpoint: {ckpt_path}")
                else:
                    print(f"         No checkpoint found — using end-of-training weights")

                print(f"         Measuring inference throughput "
                      f"({N_INFERENCE_REPS} reps) ...")
                throughput_ks = measure_throughput(model, test_loader, device)
                print(f"         Throughput: {throughput_ks:.1f} K samples/s\n")

                row = {
                    "dataset":        dataset,
                    "model":          model_name,
                    "params_total_M": round(total_m, 2),
                    "params_train_M": round(trainable_m, 2),
                    "train_time_s":   round(train_sec, 1),
                    "throughput_Ks":  round(throughput_ks, 1),
                }
                results.append(row)
                dataset_results.append(row)

            except Exception as e:
                print(f"  [{model_name}] ERROR: {e}\n")
                continue

        # ── Per-dataset summary table ──────────────────────────────────────────
        if dataset_results:
            _print_table(dataset_results, dataset)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if results:
        csv_path = os.path.join(OUTPUT_DIR, "efficiency_report.csv")
        fieldnames = [
            "dataset", "model",
            "params_total_M", "params_train_M",
            "train_time_s", "throughput_Ks"
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ CSV saved: {csv_path}")


def _print_table(rows: list, dataset: str):
    """Prints a formatted efficiency summary table."""
    header = (
        f"\n{'─'*65}\n"
        f"  Efficiency Summary — {dataset.upper()}\n"
        f"{'─'*65}\n"
        f"  {'Model':<20} {'Params(M)':>10} {'Train(s)':>10} "
        f"{'Throughput(K/s)':>16}\n"
        f"{'─'*65}"
    )
    print(header)
    for r in rows:
        print(
            f"  {r['model']:<20} "
            f"{r['params_total_M']:>10.2f} "
            f"{r['train_time_s']:>10.1f} "
            f"{r['throughput_Ks']:>16.1f}"
        )
    print(f"{'─'*65}\n")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure efficiency (params, train time, throughput) "
                    "for all DL models."
    )
    parser.add_argument(
        '--gpu_ids', type=str, default="0",
        help="CUDA_VISIBLE_DEVICES (e.g., '0')"
    )
    args = parser.parse_args()
    run_measurement(args.gpu_ids)