"""
DAF-MoE v1.5 Phase 1 runner.

Builds the 90-run command matrix. M5 runs receive seed-specific PLE boundaries
through generated YAML files under results/phase1_v15/generated_configs/.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.default_config import DAFConfig
from src.data.ple_utils import compute_ple_boundaries


DATASETS = ['california', 'adult', 'nhanes']
SEEDS = [42, 43, 44, 45, 46]
VARIANTS = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
GPU_ID = "0"
RESULT_DIR = "results/phase1_v15"
GENERATED_CONFIG_DIR = Path(RESULT_DIR) / "generated_configs"


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_config(config_path, seed, variant):
    if variant != 'M5':
        return config_path

    exp_cfg = load_yaml(config_path)
    config = DAFConfig()
    for key, value in exp_cfg.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.seed = seed

    data_cfg = load_yaml(config.data_config_path)
    exp_cfg['ple_boundaries'] = compute_ple_boundaries(config, data_cfg)

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    generated_path = GENERATED_CONFIG_DIR / f"{exp_cfg['dataset_name']}_{variant}_seed{seed}.yaml"
    with open(generated_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(exp_cfg, f, sort_keys=False)
    return str(generated_path)


def config_for(dataset, variant):
    if variant == 'M0':
        return f"configs/experiments/{dataset}_daf_moe_best.yaml"
    return f"configs/experiments/phase1_v15/{dataset}_{variant}.yaml"


def run_experiment(dataset, variant, seed):
    config_path = config_for(dataset, variant)
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    runtime_config = build_config(config_path, seed, variant)
    cmd = [
        "python", "train.py",
        "--config", runtime_config,
        "--gpu_ids", GPU_ID,
        "--seed", str(seed),
        "--verbose",
        "--result_dir", RESULT_DIR,
    ]

    print(f"\n[Start] {dataset} | {variant} | seed {seed}")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[Fail] {dataset} | {variant} | seed {seed}")
    else:
        print(f"[Done] {dataset} | {variant} | seed {seed}")
    time.sleep(1)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    total_runs = len(DATASETS) * len(VARIANTS) * len(SEEDS)
    print("Starting DAF-MoE v1.5 Phase 1")
    print(f"Datasets: {DATASETS}")
    print(f"Variants: {VARIANTS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {total_runs}")
    print("Estimated time: depends on dataset/GPU; expect roughly the prior DAF-MoE runtime times 90.")
    print(f"Save directory: {RESULT_DIR}")

    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                run_experiment(dataset, variant, seed)


if __name__ == "__main__":
    main()
