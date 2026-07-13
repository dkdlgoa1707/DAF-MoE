"""Run the 105 DAF-MoE v1.5 Phase 1 experiments."""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.default_config import DAFConfig
from src.data.loader import get_dataloaders
from src.data.ple_utils import compute_ple_boundaries, inject_ple_boundaries_into_yaml


DATASETS = ['california', 'adult', 'mimic4']
SEEDS = [42, 43, 44, 45, 46]
VARIANTS = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
GPU_ID = "0"
RESULT_DIR = Path("results/phase1_v15")
TMP_CONFIG_DIR = RESULT_DIR / "tmp_configs"

VARIANT_SUMMARIES = {
    "M0": "v1 baseline",
    "M1": "lightweight preservation only",
    "M2": "loss-free routing only",
    "M3": "loss-free routing + FiLM",
    "M4": "M3 + lightweight preservation",
    "M5": "M4 + PLE",
    "M6": "PLE only",
}


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as source:
        return yaml.safe_load(source)


def load_experiment_config(path, seed):
    experiment = load_yaml(path)
    config = DAFConfig()
    for key, value in experiment.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.seed = seed
    return config


def config_for(dataset, variant):
    if variant == 'M0':
        return Path(f"configs/experiments/{dataset}_daf_moe_best.yaml")
    return Path(f"configs/experiments/phase1_v15/{dataset}_{variant}.yaml")


def build_runtime_config(config_path, dataset, variant, seed):
    if variant == 'M0':
        return config_path

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    runtime_path = TMP_CONFIG_DIR / f"{dataset}_{variant}_seed{seed}.yaml"
    if variant in {'M5', 'M6'}:
        config = load_experiment_config(config_path, seed)
        data_cfg = load_yaml(config.data_config_path)
        train_loader, _, _ = get_dataloaders(config, data_cfg)
        x_num_scaled = train_loader.dataset.X_num[:, :, 0].numpy()
        boundaries = compute_ple_boundaries(x_num_scaled, config.ple_n_bins)
        inject_ple_boundaries_into_yaml(str(config_path), boundaries, str(runtime_path))
    else:
        shutil.copy2(config_path, runtime_path)

    runtime_config = load_yaml(runtime_path)
    runtime_config['model_name'] = f"daf_moe_v15_{variant}"
    with open(runtime_path, 'w', encoding='utf-8') as target:
        yaml.safe_dump(runtime_config, target, sort_keys=False)
    return runtime_path


def archive_checkpoint(dataset, variant, seed, runtime_config_path):
    config = load_experiment_config(runtime_config_path, seed)
    data_cfg = load_yaml(config.data_config_path)
    dataset_name = data_cfg.get('dataset_name', dataset)
    source = Path("checkpoints") / f"{dataset_name}_{config.model_name}_seed{seed}_best.pth"
    if not source.exists():
        return None

    archive_dir = RESULT_DIR / variant / "checkpoints"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{dataset}_seed{seed}_best.pth"
    shutil.copy2(source, target)
    return target


def run_experiment(dataset, variant, seed, gpu_id):
    config_path = config_for(dataset, variant)
    if not config_path.exists():
        return f"{dataset} | {variant} | seed {seed} | missing config: {config_path}"

    print(f"\n[Start] {dataset} | {variant} | seed {seed}")
    try:
        runtime_config = build_runtime_config(config_path, dataset, variant, seed)
    except Exception as exc:
        return f"{dataset} | {variant} | seed {seed} | preflight failed: {exc}"

    variant_result_dir = RESULT_DIR / variant
    variant_result_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "train.py",
        "--config", str(runtime_config),
        "--gpu_ids", gpu_id,
        "--seed", str(seed),
        "--verbose",
        "--result_dir", str(variant_result_dir),
    ]

    print("Command:", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        return f"{dataset} | {variant} | seed {seed} | exit code {result.returncode}"

    archived = archive_checkpoint(dataset, variant, seed, runtime_config)
    if archived is None:
        print(f"[Warn] checkpoint not found for {dataset} | {variant} | seed {seed}")
    else:
        print(f"[Checkpoint] {archived}")
    print(f"[Done] {dataset} | {variant} | seed {seed}")
    time.sleep(1)
    return None


def write_failures(failures, datasets, variants, seeds):
    tag = f"{'_'.join(datasets)}__{'_'.join(variants)}__{'_'.join(str(seed) for seed in seeds)}"
    failure_path = RESULT_DIR / f"failures__{tag}.log"
    contents = "\n".join(failures) if failures else "No failed runs."
    failure_path.write_text(contents + "\n", encoding='utf-8')
    return failure_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", default=GPU_ID)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    datasets_to_run = args.datasets
    variants_to_run = args.variants
    seeds_to_run = args.seeds

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    total_runs = len(datasets_to_run) * len(variants_to_run) * len(seeds_to_run)
    print(
        f"[Runner] datasets={datasets_to_run} variants={variants_to_run} "
        f"seeds={seeds_to_run}"
    )
    print(f"[Runner] GPU={args.gpu_id} total_runs={total_runs}")
    for variant in variants_to_run:
        print(f"  {variant}: {VARIANT_SUMMARIES[variant]}")

    failures = []
    for dataset in datasets_to_run:
        for variant in variants_to_run:
            for seed in seeds_to_run:
                failure = run_experiment(dataset, variant, seed, args.gpu_id)
                if failure:
                    failures.append(failure)
                    print(f"[Fail] {failure}")

    failure_path = write_failures(
        failures, datasets_to_run, variants_to_run, seeds_to_run
    )
    print(f"[Runner] failures log: {failure_path}")
    print(f"[Runner] {total_runs - len(failures)}/{total_runs} runs succeeded")


if __name__ == "__main__":
    main()
