import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.phase2_preflight import (
    _check_retrieval_scale,
    _retrieval_timeout_only,
    ALL_METHODS,
    TUNABLE_METHODS,
    final_commands,
    hpo_commands,
)
from analysis.summarize_phase2 import (
    build_availability,
    build_common_subset_ranking,
)
from src.data.adapters import ADAPTER_REGISTRY, DAFV2Adapter, NativeRawAdapter
from src.data.provenance import stable_hash
from src.data.splits import RawDataset, RawSplitRegistry
from src.hpo.engine import TrialArtifactWriter, make_guarded_objective
from src.hpo.identity import build_study_identity
from src.hpo.schema import load_search_space
from src.phase2_protocol import DATASETS, MAIN_RANK_INCLUDED, PROTOCOL_VERSION
from src.phase2_results import build_execution_manifest, reusable_result, write_result


SPACE_ROOT = Path("configs/hpo/phase2")


class DeterministicTrial:
    def __init__(self, number=0):
        self.number = number
        self.params = {}
        self.user_attrs = {}

    def suggest_float(self, name, low, high, log=False):
        value = float(np.sqrt(low * high) if log else (low + high) / 2.0)
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high, step=1, log=False):
        value = int(low)
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


def _raw(task_kind):
    n_rows = 90
    index = np.arange(n_rows)
    frame = pd.DataFrame(
        {
            "num": np.sin(index / 5.0),
            "num_missing": np.where(index % 13 == 0, np.nan, index / 10.0),
            "cat": np.resize(np.array(["a", "b", None], dtype=object), n_rows),
        }
    )
    if task_kind == "regression":
        target = pd.Series(np.cos(index / 7.0), name="target")
    elif task_kind == "binary":
        target = pd.Series(index % 2, name="target")
    else:
        target = pd.Series(index % 3, name="target")
    return RawDataset(
        features=frame,
        target=target,
        numerical_columns=("num", "num_missing"),
        categorical_columns=("cat",),
        target_column="target",
        dataset_name=f"tiny-{task_kind}",
        schema_version="integration-v1",
    )


class ProtocolIntegrationMatrixTests(unittest.TestCase):
    def test_all_tunable_schemas_guard_best_and_two_seed_manifests(self):
        for task_kind in ("regression", "binary", "multiclass"):
            task_type = "regression" if task_kind == "regression" else "classification"
            raw = _raw(task_kind)
            split = RawSplitRegistry(raw, task_type, 42).for_hpo()
            self.assertFalse(hasattr(split, "test"))
            for model in TUNABLE_METHODS:
                with self.subTest(task=task_kind, model=model):
                    space = load_search_space(SPACE_ROOT / f"{model}.yaml")
                    trial = DeterministicTrial()
                    base = {
                        "model_name": model,
                        "task_type": task_type,
                        "optimize_metric": "rmse"
                        if task_type == "regression"
                        else "acc",
                    }
                    identity = build_study_identity(raw, base, space)
                    with tempfile.TemporaryDirectory() as directory:
                        writer = TrialArtifactWriter(directory)

                        def evaluator(resolved, current_trial):
                            space.validate_resolved(
                                resolved,
                                n_rows=len(raw.features),
                                task_type=task_type,
                            )
                            return {
                                "metric_value": 0.5,
                                "split_hash": split.split_hash,
                                "preprocessing_hash": stable_hash(
                                    {"model": model, "task": task_kind}
                                ),
                            }

                        objective = make_guarded_objective(
                            space,
                            evaluator,
                            n_rows=len(raw.features),
                            artifact_writer=writer,
                            study_identity=identity,
                            task_type=task_type,
                        )
                        self.assertEqual(objective(trial), 0.5)
                        resolved = trial.user_attrs["resolved_config"]
                        artifact = next(Path(directory).rglob("trial_00000.json"))
                        self.assertEqual(
                            json.loads(artifact.read_text(encoding="utf-8"))["status"],
                            "COMPLETE",
                        )

                        manifests = []
                        for seed in (43, 44):
                            data_manifest = {
                                "protocol_version": PROTOCOL_VERSION,
                                "split_index_hash": RawSplitRegistry(
                                    raw, task_type, seed
                                ).indices.split_hash,
                                "fitted_state_hash": stable_hash(
                                    {"model": model, "task": task_kind, "seed": seed}
                                ),
                            }
                            manifests.append(
                                build_execution_manifest(
                                    data_manifest,
                                    model,
                                    resolved,
                                    space.schema_hash,
                                    seed,
                                    study_identity=identity,
                                )
                            )
                        self.assertNotEqual(
                            manifests[0]["split_index_hash"],
                            manifests[1]["split_index_hash"],
                        )
                        result = Path(directory) / "seed43.json"
                        write_result(result, {"resume_manifest": manifests[0]})
                        self.assertTrue(reusable_result(result, manifests[0]))
                        self.assertFalse(reusable_result(result, manifests[1]))

    def test_adapter_and_rank_registry_cover_every_method(self):
        self.assertEqual(set(ADAPTER_REGISTRY), set(ALL_METHODS))
        self.assertIs(ADAPTER_REGISTRY["daf_moe_v2"], DAFV2Adapter)
        for model in ("xgboost", "catboost", "realmlp", "tabicl"):
            self.assertIs(ADAPTER_REGISTRY[model], NativeRawAdapter)
        self.assertTrue(MAIN_RANK_INCLUDED["tabm"])
        self.assertFalse(MAIN_RANK_INCLUDED["tabm_ple"])

    def test_launch_plans_separate_hpo_and_final_jobs(self):
        hpo = hpo_commands()
        final = final_commands()
        self.assertEqual(len(hpo), 99)
        self.assertEqual(len(final), 108)
        self.assertTrue(all(" hpo " in command for command in hpo))
        self.assertTrue(all(" final " not in command for command in hpo))
        self.assertTrue(all(" final " in command for command in final))
        self.assertTrue(all(" hpo " not in command for command in final))
        self.assertFalse(any("phase2/tabicl.yaml" in command for command in hpo))
        self.assertTrue(any("phase2/tabicl.yaml" in command for command in final))
        for dataset in DATASETS:
            self.assertEqual(
                sum(f"base/{dataset}.yaml" in command for command in hpo),
                len(TUNABLE_METHODS),
            )
            self.assertEqual(
                sum(f"base/{dataset}.yaml" in command for command in final),
                len(ALL_METHODS),
            )


class Phase2HardeningTests(unittest.TestCase):
    def _retrieval_report(self, stage2_status="TIMEOUT", oom=False):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "decision": "PHASE2_NOT_READY",
            "blockers": ["Covertype retrieval exceeded 900 seconds."],
            "production_experiments_started": False,
            "cost_estimate": {"lower_bound": True},
            "combinations": [
                {
                    "stage1": {
                        "status": "PASS",
                        "training": {"finite": True},
                        "memory": {"oom": False},
                    },
                    "stage2": {"status": "PASS"},
                },
                {
                    "stage1": {
                        "status": "PASS",
                        "training": {"finite": True},
                        "memory": {"oom": oom},
                    },
                    "stage2": {"status": stage2_status},
                },
            ],
        }

    def test_retrieval_timeout_only_is_warning_but_oom_is_blocker(self):
        warning_report = self._retrieval_report()
        self.assertTrue(_retrieval_timeout_only(warning_report))
        with tempfile.TemporaryDirectory(dir=".") as directory:
            report_path = (Path(directory) / "retrieval.json").resolve()
            report_path.write_text(json.dumps(warning_report), encoding="utf-8")
            errors, warnings, checks = [], [], {}
            with patch("analysis.phase2_preflight.RETRIEVAL_SCALE_REPORT", report_path):
                _check_retrieval_scale(errors, warnings, checks)
            self.assertEqual(errors, [])
            self.assertEqual(len(warnings), 1)

            report_path.write_text(
                json.dumps(self._retrieval_report(oom=True)), encoding="utf-8"
            )
            errors, warnings, checks = [], [], {}
            with patch("analysis.phase2_preflight.RETRIEVAL_SCALE_REPORT", report_path):
                _check_retrieval_scale(errors, warnings, checks)
            self.assertEqual(warnings, [])
            self.assertEqual(len(errors), 1)

    def test_compute_limit_never_becomes_metric_or_rank(self):
        hpo_statuses = {
            ("covertype", "tabr"): {
                "status": "COMPUTE_LIMIT",
                "reason": "ceiling exhausted",
                "valid_complete": 17,
                "gpu_hours_lower_bound": 12.5,
            }
        }
        availability = build_availability({}, hpo_statuses)
        row = next(
            item
            for item in availability
            if item["dataset"] == "covertype" and item["model"] == "tabr"
        )
        self.assertEqual(row["status"], "COMPUTE_LIMIT")
        self.assertIsNone(row["metric"])
        self.assertIsNone(row["rank"])

        metadata = {
            dataset: {"metric": "rmse", "direction": "minimize"} for dataset in DATASETS
        }
        ranking = build_common_subset_ranking({}, metadata)
        self.assertEqual(ranking["common_dataset_subset"], [])
        self.assertEqual(ranking["rows"], [])
        self.assertEqual(ranking["average_ranks"], {})


if __name__ == "__main__":
    unittest.main()
