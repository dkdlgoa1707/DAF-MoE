#!/usr/bin/env python
"""Isolated Adult/Covertype scale smoke for TabR and ModernNCA."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import platform
import queue
import resource
import subprocess
import sys
import time
import traceback

import torch
import torch.optim as optim
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.provenance import current_git_sha, stable_hash  # noqa: E402
from src.data.phase2_loader import get_phase2_hpo_dataloaders  # noqa: E402
from src.data.splits import load_raw_dataset  # noqa: E402
from src.hpo.engine import (  # noqa: E402
    TrialArtifactWriter,
    create_phase2_study,
    make_guarded_objective,
    run_until_valid_complete,
    valid_complete_count,
)
from src.hpo.identity import StudyIdentity, build_study_identity  # noqa: E402
from src.hpo.schema import load_search_space  # noqa: E402
from src.losses.factory import create_criterion  # noqa: E402
from src.models.baselines.faiss_search import require_faiss  # noqa: E402
from src.models.factory import create_model  # noqa: E402
from src.phase2_execution import execute_hpo_trial, materialize_neural_config  # noqa: E402
from src.phase2_protocol import PROTOCOL_VERSION  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src.utils.common import seed_everything  # noqa: E402


MATRIX = (
    ("adult", "tabr"),
    ("adult", "modernnca"),
    ("covertype", "tabr"),
    ("covertype", "modernnca"),
)
DEFAULT_ROOT = ROOT / "results/phase2/scale_smoke"


class SilentLogger:
    def info(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def _read_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _inputs(dataset, model):
    base_path = ROOT / f"configs/experiments/phase2/base/{dataset}.yaml"
    space_path = ROOT / f"configs/hpo/phase2/{model}.yaml"
    base = _read_yaml(base_path)
    space = load_search_space(space_path)
    data_config = _read_yaml(ROOT / base["data_config_path"])
    raw = load_raw_dataset(data_config)
    return base, space, raw


def _scale_identity(raw, base, space):
    production = build_study_identity(raw, base, space)
    components = dict(production.components)
    components["dataset_name"] = f"scale-smoke-{raw.dataset_name}"
    components["base_experiment_config_hash"] = stable_hash(
        {"scope": "scale-smoke", "base": dict(base)}
    )
    return StudyIdentity.from_components(components)


def _memory_total_bytes():
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _environment(device, identity, model):
    faiss = require_faiss()
    cuda_device = torch.device(device)
    return {
        "git_sha": current_git_sha(),
        "protocol_version": PROTOCOL_VERSION,
        "study_signature": identity.signature,
        "study_name": identity.study_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _memory_total_bytes(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "faiss_version": str(faiss.__version__),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": (
            torch.cuda.get_device_name(cuda_device)
            if cuda_device.type == "cuda"
            else None
        ),
        "upstream_reference": identity.components["model_implementation_version"],
        "model": model,
    }


def _sample_config(space, n_rows, task_type):
    import optuna
    from optuna.samplers import TPESampler

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=42)
    )
    trial = study.ask()
    resolved = space.sample(trial, n_rows=n_rows, task_type=task_type)
    return resolved, dict(trial.params)


def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _refresh_manifest(config, model):
    if not hasattr(model, "candidate_provenance"):
        return None
    provenance = model.candidate_provenance()
    if provenance is not None:
        config.phase2_manifest["retrieval_candidates"] = provenance
        payload = {
            key: value
            for key, value in config.phase2_manifest.items()
            if key != "manifest_hash"
        }
        config.phase2_manifest["manifest_hash"] = stable_hash(payload)
    return provenance


def run_stage1(dataset, model_name, device_name, output_root):
    started = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
    base, space, raw = _inputs(dataset, model_name)
    identity = _scale_identity(raw, base, space)
    resolved, sampled_params = _sample_config(
        space, len(raw.features), base["task_type"]
    )
    stage_base = dict(base)
    stage_base.update({"epochs": 1, "patience": 0})
    checkpoint = (
        output_root / "artifacts" / identity.study_name / "stage1_best.pth"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = materialize_neural_config(
        stage_base,
        space,
        resolved,
        base["task_type"],
        base["optimize_metric"],
        42,
        checkpoint_path=checkpoint,
    )
    config.retrieval_measure_performance = True

    seed_everything(42)
    preprocess_started = time.perf_counter()
    train_loader, validation_loader = get_phase2_hpo_dataloaders(config, raw)
    preprocessing_seconds = time.perf_counter() - preprocess_started
    model = create_model(config).to(device)
    criterion = create_criterion(config, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        config,
        device,
        SilentLogger(),
        verbose=False,
    )
    context_started = time.perf_counter()
    trainer._wire_retrieval_context(train_loader)
    _cuda_sync(device)
    context_seconds = time.perf_counter() - context_started

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_started = time.perf_counter()
    loss = float(trainer.train_epoch(train_loader, 0))
    _cuda_sync(device)
    epoch_seconds = time.perf_counter() - train_started
    validation_started = time.perf_counter()
    metrics = trainer.validate(validation_loader, 0)
    _cuda_sync(device)
    validation_seconds = time.perf_counter() - validation_started
    metric = float(metrics[base["optimize_metric"]])
    if not math.isfinite(loss) or not math.isfinite(metric):
        raise FloatingPointError(f"Nonfinite Stage 1 result: loss={loss}, metric={metric}")

    torch.save(model.state_dict(), checkpoint)
    provenance = _refresh_manifest(config, model)
    if provenance["candidate_count"] != len(train_loader.dataset):
        raise RuntimeError("Train candidate count does not match the train split.")
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    total_seconds = time.perf_counter() - started
    measured_query_count = len(train_loader.dataset) + len(validation_loader.dataset)
    measured_query_seconds = epoch_seconds + validation_seconds
    retrieval_measurements = {
        "query_count": provenance.get("total_query_count", measured_query_count),
        "effective_candidate_count": provenance.get(
            "last_effective_candidate_count", provenance["candidate_count"]
        ),
        "index_refresh_count": provenance.get("index_refresh_count"),
        "index_refresh_seconds": provenance.get("index_refresh_seconds"),
        "search_or_prediction_seconds": provenance.get(
            "search_seconds", provenance.get("prediction_seconds")
        ),
        "queries_per_second": provenance.get("queries_per_second"),
        "candidate_comparisons_per_second": provenance.get(
            "candidate_comparisons_per_second"
        ),
        "end_to_end_queries_per_second": (
            measured_query_count / measured_query_seconds
            if measured_query_seconds > 0
            else None
        ),
    }
    return {
        "stage": 1,
        "status": "PASS",
        "dataset": dataset,
        "dataset_name": raw.dataset_name,
        "model": model_name,
        "environment": _environment(device_name, identity, model_name),
        "data": {
            "total_rows": len(raw.features),
            "train_rows": len(train_loader.dataset),
            "validation_rows": len(validation_loader.dataset),
            "candidate_count": provenance["candidate_count"],
            "candidate_row_id_hash": provenance["row_id_hash"],
            "batch_size": config.batch_size,
            "sample_rate": (
                getattr(config, "nca_sample_rate", None)
                if model_name == "modernnca"
                else None
            ),
            "context_size": (
                getattr(config, "tabr_n_candidates", None)
                if model_name == "tabr"
                else None
            ),
            "retrieval_backend": provenance.get("retrieval_backend"),
            "index_type": provenance.get("index_type"),
            "test_partition_accessed": False,
            "subsample": None,
        },
        "sampled_params": sampled_params,
        "resolved_config": resolved,
        "timing": {
            "preprocessing_seconds": preprocessing_seconds,
            "context_or_index_setup_seconds": context_seconds,
            "epoch_training_seconds": epoch_seconds,
            "validation_seconds": validation_seconds,
            "total_seconds": total_seconds,
        },
        "memory": {
            "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "max_cuda_memory_allocated": peak_allocated,
            "max_cuda_memory_reserved": peak_reserved,
            "oom": False,
        },
        "training": {
            "epochs_completed": 1,
            "loss": loss,
            "validation_metric_name": base["optimize_metric"],
            "validation_metric": metric,
            "finite": True,
        },
        "retrieval": provenance,
        "retrieval_measurements": retrieval_measurements,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
        },
    }


def run_stage2(dataset, model_name, device_name, output_root):
    started = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
    base, space, raw = _inputs(dataset, model_name)
    identity = _scale_identity(raw, base, space)
    storage = identity.default_storage_url(output_root / "databases")
    study = create_phase2_study(identity, "maximize", storage)
    valid_before_run = valid_complete_count(study, identity)
    writer = TrialArtifactWriter(output_root / "artifacts")
    checkpoint = (
        output_root / "artifacts" / identity.study_name / "stage2_trial_best.pth"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def evaluator(resolved, _trial):
        outcome = execute_hpo_trial(
            raw,
            base,
            space,
            resolved,
            base["task_type"],
            base["optimize_metric"],
            checkpoint_path=checkpoint,
            device=device,
            study_identity=identity,
        )
        return {
            "metric_value": outcome.metric_value,
            "metrics": outcome.metrics,
            "split_hash": outcome.split_hash,
            "preprocessing_hash": outcome.preprocessing_hash,
            "execution_manifest": outcome.manifest,
            "best_epoch_or_iteration": outcome.best_epoch_or_iteration,
            "epochs_completed": outcome.epochs_completed,
        }

    objective = make_guarded_objective(
        space,
        evaluator,
        n_rows=len(raw.features),
        artifact_writer=writer,
        study_identity=identity,
        task_type=base["task_type"],
    )
    run_until_valid_complete(study, objective, target=1, identity=identity)
    trial = study.best_trial
    _cuda_sync(device)
    artifact = (
        output_root
        / "artifacts"
        / identity.study_name
        / f"trial_{trial.number:05d}.json"
    )
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    manifest = artifact_payload.get("execution_manifest", {})
    provenance = manifest.get("retrieval_candidates", {})
    return {
        "stage": 2,
        "status": "PASS",
        "dataset": dataset,
        "dataset_name": raw.dataset_name,
        "model": model_name,
        "environment": _environment(device_name, identity, model_name),
        "storage": storage,
        "valid_complete_trials": valid_complete_count(study, identity),
        "trial_number": trial.number,
        "sampled_params": dict(trial.params),
        "resolved_config": trial.user_attrs["resolved_config"],
        "validation_metric": float(trial.value),
        "best_epoch": artifact_payload.get("best_epoch_or_iteration"),
        "epochs_completed": artifact_payload.get("epochs_completed"),
        "timing": {
            "total_trial_seconds": artifact_payload.get("elapsed_seconds"),
            "orchestration_seconds": time.perf_counter() - started,
            "reused_existing_complete_trial": valid_before_run >= 1,
        },
        "memory": {
            "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "max_cuda_memory_allocated": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            ),
            "max_cuda_memory_reserved": (
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else 0
            ),
            "oom": False,
        },
        "retrieval": provenance,
        "artifact": {
            "path": str(artifact.relative_to(ROOT)),
            "bytes": artifact.stat().st_size,
            "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
        },
        "test_partition_accessed": False,
        "subsample": None,
    }


def _fragment_path(output_root, dataset, model, stage):
    return output_root / "reports/fragments" / f"{dataset}_{model}_stage{stage}.json"


def worker(args):
    output_root = Path(args.output_root).resolve()
    fragment = _fragment_path(output_root, args.dataset, args.model, args.stage)
    fragment.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = (
            run_stage1(args.dataset, args.model, args.device, output_root)
            if args.stage == 1
            else run_stage2(args.dataset, args.model, args.device, output_root)
        )
    except Exception as exc:
        result = {
            "stage": args.stage,
            "status": "FAIL",
            "dataset": args.dataset,
            "model": args.model,
            "failure_type": exc.__class__.__name__,
            "failure_reason": str(exc),
            "oom": "out of memory" in str(exc).lower(),
            "traceback": traceback.format_exc(),
            "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "test_partition_accessed": False,
            "subsample": None,
        }
    fragment.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"fragment": str(fragment), "status": result["status"]}))
    return 0 if result["status"] == "PASS" else 1


def _run_worker(output_root, dataset, model, stage, device, timeout):
    fragment = _fragment_path(output_root, dataset, model, stage)
    log = output_root / "logs" / f"{dataset}_{model}_stage{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        dataset,
        "--model",
        model,
        "--stage",
        str(stage),
        "--device",
        device,
        "--output-root",
        str(output_root),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        log.write_text(completed.stdout, encoding="utf-8")
        if fragment.exists():
            result = json.loads(fragment.read_text(encoding="utf-8"))
        else:
            result = {
                "stage": stage,
                "status": "FAIL",
                "dataset": dataset,
                "model": model,
                "failure_type": "MissingFragment",
                "failure_reason": f"worker returncode={completed.returncode}",
            }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        log.write_text(output, encoding="utf-8")
        result = {
            "stage": stage,
            "status": "TIMEOUT",
            "dataset": dataset,
            "model": model,
            "failure_type": "TimeoutExpired",
            "failure_reason": f"Exceeded {timeout} seconds",
            "elapsed_seconds": time.perf_counter() - started,
            "test_partition_accessed": False,
            "subsample": None,
        }
        fragment.parent.mkdir(parents=True, exist_ok=True)
        fragment.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _parallel_stage(output_root, combinations, stage, devices, timeout):
    available = queue.Queue()
    for device in devices:
        available.put(device)

    def run(item):
        device = available.get()
        try:
            return _run_worker(output_root, *item, stage, device, timeout)
        finally:
            available.put(device)

    results = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(run, item): item for item in combinations}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _trial_duration(stage2):
    if stage2.get("status") == "PASS":
        timing = stage2.get("timing", {})
        return (
            timing.get("artifact_elapsed_seconds")
            or timing.get("total_trial_seconds")
        ), False
    if stage2.get("status") == "TIMEOUT":
        return stage2.get("elapsed_seconds"), True
    return None, True


def _estimate_cost(combinations):
    observations = []
    per_combination = []
    for item in combinations:
        seconds, lower_bound = _trial_duration(item.get("stage2", {}))
        if seconds is None:
            continue
        observations.append((item["model"], float(seconds), lower_bound))
        per_combination.append({
            "dataset": item["dataset"],
            "model": item["model"],
            "observed_trial_seconds": float(seconds),
            "is_lower_bound": bool(lower_bound),
            "fifty_trial_gpu_hours": float(seconds) * 50 / 3600,
        })
    if not observations:
        return {"basis": "no Stage 2 observation", "per_combination": []}

    model_means = {}
    for model in ("tabr", "modernnca"):
        values = [seconds for name, seconds, _ in observations if name == model]
        if values:
            model_means[model] = sum(values) / len(values)
    total_gpu_hours = sum(model_means.values()) * 9 * 50 / 3600
    gpu_count = max(1, torch.cuda.device_count())
    complete = [
        item["stage2"] for item in combinations
        if item.get("stage2", {}).get("status") == "PASS"
    ]
    mean_artifact = (
        sum(item["artifact"]["bytes"] for item in complete) / len(complete)
        if complete else None
    )
    mean_checkpoint = (
        sum(item["artifact"]["checkpoint_bytes"] for item in complete)
        / len(complete)
        if complete else None
    )
    disk_bytes = (
        mean_artifact * 50 * 18 + mean_checkpoint * 18
        if mean_artifact is not None and mean_checkpoint is not None
        else None
    )
    return {
        "basis": (
            "completed trial wall time; timeout durations are lower bounds; "
            "9-dataset estimate uses the Adult/Covertype mean per model"
        ),
        "per_combination": per_combination,
        "observed_matrix_fifty_trial_gpu_hours_lower_bound": (
            sum(seconds for _, seconds, _ in observations) * 50 / 3600
        ),
        "nine_dataset_two_model_gpu_hours_lower_bound": total_gpu_hours,
        "available_gpu_count": gpu_count,
        "ideal_wall_clock_hours_at_available_gpu_count_lower_bound": (
            total_gpu_hours / gpu_count
        ),
        "estimated_hpo_disk_bytes": disk_bytes,
        "disk_estimate_basis": (
            "50 JSON artifacts plus one retained checkpoint per dataset/model; "
            "excludes SQLite and log overhead"
        ),
    }


def _write_markdown(path, report):
    lines = [
        "# Phase 2 Retrieval Scale Report",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "| Dataset | Model | Stage 1 | Total (s) | Train (s) | Val (s) | Metric | Peak CUDA GiB | Candidates | Query throughput (q/s) | Stage 2 | Trial time (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for item in report["combinations"]:
        one = item.get("stage1", {})
        two = item.get("stage2", {})
        memory = one.get("memory", {}).get("max_cuda_memory_allocated")
        trial_seconds, trial_is_lower_bound = _trial_duration(two)
        trial_text = _format(trial_seconds)
        if trial_is_lower_bound and trial_seconds is not None:
            trial_text = ">=" + trial_text
        query_rate = one.get("retrieval_measurements", {}).get(
            "queries_per_second"
        )
        lines.append(
            "| {dataset} | {model} | {s1} | {total} | {train} | {val} | {metric} | {memory} | {candidates} | {query_rate} | {s2} | {trial} |".format(
                dataset=item["dataset"],
                model=item["model"],
                s1=one.get("status", "NOT_RUN"),
                total=_format(one.get("timing", {}).get("total_seconds")),
                train=_format(one.get("timing", {}).get("epoch_training_seconds")),
                val=_format(one.get("timing", {}).get("validation_seconds")),
                metric=_format(one.get("training", {}).get("validation_metric")),
                memory=_format(memory / (1024 ** 3) if memory is not None else None),
                candidates=one.get("data", {}).get("candidate_count", "N/A"),
                query_rate=_format(query_rate),
                s2=two.get("status", "NOT_RUN"),
                trial=trial_text,
            )
        )
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Cost estimate",
            "",
            "```json",
            json.dumps(report["cost_estimate"], indent=2),
            "```",
            "",
            "Timeout-derived costs are lower bounds, not completion estimates.",
            "",
            "No production 50-trial HPO or 15-seed final evaluation was launched.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format(value):
    return "N/A" if value is None else f"{float(value):.4f}"


def orchestrate(args):
    output_root = Path(args.output_root).resolve()
    for name in ("databases", "artifacts", "logs", "reports"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    devices = args.devices or ["cuda:0"]
    if args.report_only:
        stage1 = []
        stage2 = []
        for dataset, model in MATRIX:
            for stage, destination in ((1, stage1), (2, stage2)):
                fragment = _fragment_path(output_root, dataset, model, stage)
                if fragment.exists():
                    destination.append(json.loads(fragment.read_text(encoding="utf-8")))
                else:
                    destination.append({
                        "stage": stage,
                        "status": "NOT_RUN",
                        "dataset": dataset,
                        "model": model,
                        "failure_reason": f"Missing fragment: {fragment}",
                    })
    else:
        stage1 = _parallel_stage(
            output_root, MATRIX, 1, devices, args.stage1_timeout
        )
        passed = [
            (item["dataset"], item["model"])
            for item in stage1
            if item.get("status") == "PASS"
        ]
        stage2 = (
            _parallel_stage(output_root, passed, 2, devices, args.stage2_timeout)
            if passed
            else []
        )
    by_key = {(item["dataset"], item["model"]): item for item in stage1}
    stage2_by_key = {(item["dataset"], item["model"]): item for item in stage2}
    combinations = []
    for dataset, model in MATRIX:
        combinations.append(
            {
                "dataset": dataset,
                "model": model,
                "stage1": by_key[(dataset, model)],
                "stage2": stage2_by_key.get(
                    (dataset, model),
                    {"status": "NOT_RUN", "reason": "Stage 1 did not pass"},
                ),
            }
        )
    ready = all(
        item["stage1"].get("status") == "PASS"
        and item["stage2"].get("status") == "PASS"
        and item["stage2"].get("valid_complete_trials") == 1
        for item in combinations
    )
    blockers = []
    for item in combinations:
        for stage_name in ("stage1", "stage2"):
            stage = item[stage_name]
            if stage.get("status") != "PASS":
                reason = stage.get("failure_reason", stage.get("reason", "unknown"))
                blockers.append(
                    f"{item['dataset']}/{item['model']} {stage_name} "
                    f"{stage.get('status')}: {reason}"
                )
        if (
            item["stage2"].get("status") == "PASS"
            and item["stage2"].get("valid_complete_trials") != 1
        ):
            blockers.append(
                f"{item['dataset']}/{item['model']} Stage 2 did not have "
                "exactly one valid finite COMPLETE trial."
            )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "decision": "READY_FOR_HPO" if ready else "NOT_READY_FOR_HPO",
        "production_experiments_started": False,
        "matrix": [list(item) for item in MATRIX],
        "combinations": combinations,
        "blockers": blockers,
        "cost_estimate": _estimate_cost(combinations),
    }
    report_path = output_root / "reports/retrieval_scale_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_markdown(ROOT / "docs/phase2_retrieval_scale_report.md", report)
    print(report["decision"])
    print(report_path)
    return 0 if ready else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate reports from existing worker fragments without execution.",
    )
    parser.add_argument("--dataset", choices=("adult", "covertype"))
    parser.add_argument("--model", choices=("tabr", "modernnca"))
    parser.add_argument("--stage", type=int, choices=(1, 2))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--stage1-timeout", type=int, default=900)
    parser.add_argument("--stage2-timeout", type=int, default=1800)
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        if not (args.dataset and args.model and args.stage):
            raise SystemExit("--worker requires --dataset, --model, and --stage")
        return worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
