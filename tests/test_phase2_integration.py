import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.phase2_preflight import ALL_METHODS, TUNABLE_METHODS, launch_commands
from src.data.adapters import ADAPTER_REGISTRY, DAFV2Adapter, NativeRawAdapter
from src.data.provenance import stable_hash
from src.data.splits import RawDataset, RawSplitRegistry
from src.hpo.engine import TrialArtifactWriter, make_guarded_objective
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
                            study_name=f"{raw.dataset_name}__{model}__{PROTOCOL_VERSION}",
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

    def test_launch_plan_has_hpo_and_final_for_every_dataset_method(self):
        commands = launch_commands()
        self.assertEqual(len(commands), len(DATASETS) * (2 * len(TUNABLE_METHODS) + 1))
        for dataset in DATASETS:
            for model in ALL_METHODS:
                matches = [
                    command for command in commands
                    if f"base/{dataset}.yaml" in command
                    and f"phase2/{model}.yaml" in command
                ]
                self.assertEqual(len(matches), 1 if model == "tabicl" else 2)
                self.assertTrue(any(" final " in command for command in matches))
                self.assertEqual(
                    any(" hpo " in command for command in matches), model != "tabicl"
                )


if __name__ == "__main__":
    unittest.main()
