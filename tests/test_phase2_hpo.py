import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data.splits import RawDataset
from src.hpo.engine import (
    TrialArtifactWriter,
    make_guarded_objective,
    run_until_valid_complete,
    valid_complete_count,
)
from src.hpo.schema import load_search_space, parse_search_space
from src.phase2_execution import execute_final_seed, execute_hpo_trial
from src.phase2_results import reusable_result, write_result


SPACE_ROOT = Path("configs/hpo/phase2")


class FakeTrial:
    def __init__(self, number=0, fraction=0.5, choose_high=False):
        self.number = number
        self.fraction = fraction
        self.choose_high = choose_high
        self.params = {}
        self.user_attrs = {}

    def suggest_float(self, name, low, high, log=False):
        if log:
            value = float(np.exp(np.log(low) + self.fraction * (np.log(high) - np.log(low))))
        else:
            value = float(low + self.fraction * (high - low))
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high, step=1, log=False):
        if self.choose_high:
            value = high
        elif log:
            raw = int(round(np.exp(np.log(low) + self.fraction * (np.log(high) - np.log(low)))))
            value = min(high, max(low, raw))
        else:
            value = low
        if step != 1:
            value = low + ((value - low) // step) * step
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = choices[-1] if self.choose_high else choices[0]
        self.params[name] = value
        return value

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class _State:
    def __init__(self, name):
        self.name = name


class _FrozenTrial:
    def __init__(self, number, state, value):
        self.number = number
        self.state = _State(state)
        self.value = value


class FakeStudy:
    def __init__(self):
        self.trials = []

    def optimize(self, objective, n_trials, catch):
        self.assert_one(n_trials)
        number = len(self.trials)
        trial = FakeTrial(number=number)
        try:
            value = objective(trial)
        except catch:
            self.trials.append(_FrozenTrial(number, "FAIL", None))
        else:
            self.trials.append(_FrozenTrial(number, "COMPLETE", value))

    @staticmethod
    def assert_one(n_trials):
        if n_trials != 1:
            raise AssertionError("Engine must advance one trial at a time.")


def _tiny_raw(task_type="classification"):
    n_rows = 48
    frame = pd.DataFrame(
        {
            "num": np.linspace(-2.0, 2.0, n_rows),
            "num_missing": np.where(np.arange(n_rows) % 11 == 0, np.nan, np.arange(n_rows)),
            "cat": np.resize(np.array(["a", "b", None], dtype=object), n_rows),
        }
    )
    if task_type == "classification":
        target = pd.Series(np.arange(n_rows) % 2, name="target")
    else:
        target = pd.Series(np.linspace(0.0, 1.0, n_rows), name="target")
    return RawDataset(
        features=frame,
        target=target,
        numerical_columns=("num", "num_missing"),
        categorical_columns=("cat",),
        target_column="target",
        dataset_name="tiny",
        schema_version="test-v1",
    )


class SearchSpaceTests(unittest.TestCase):
    def test_all_model_yamls_validate_and_sample_within_bounds(self):
        paths = sorted(SPACE_ROOT.glob("*.yaml"))
        self.assertEqual(len(paths), 12)
        for path in paths:
            space = load_search_space(path)
            task_type = "classification"
            resolved_small = space.sample(FakeTrial(), n_rows=100_000, task_type=task_type)
            resolved_large = space.sample(
                FakeTrial(choose_high=True), n_rows=100_001, task_type=task_type
            )
            space.validate_resolved(resolved_small, n_rows=100_000, task_type=task_type)
            space.validate_resolved(resolved_large, n_rows=100_001, task_type=task_type)
            self.assertFalse(set(resolved_small).intersection(space.forbidden))
            for key, value in space.fixed.items():
                self.assertEqual(resolved_small[key], value)

    def test_exact_fixed_and_conditional_contracts(self):
        daf = load_search_space(SPACE_ROOT / "daf_moe_v2.yaml")
        self.assertEqual(daf.fixed["n_heads"], 8)
        self.assertEqual(daf.fixed["top_k"], 2)
        self.assertEqual(daf.search["d_emb"]["step"], 16)

        mlp = load_search_space(SPACE_ROOT / "mlp.yaml")
        small = mlp.sample(FakeTrial(choose_high=True), 100_000, "classification")
        large = mlp.sample(FakeTrial(choose_high=True), 100_001, "classification")
        self.assertEqual(small["n_layers"], 8)
        self.assertEqual(large["n_layers"], 16)
        self.assertEqual(small["first_width"], 512)
        self.assertEqual(large["first_width"], 1024)

        tabm = load_search_space(SPACE_ROOT / "tabm.yaml")
        tabm_ple = load_search_space(SPACE_ROOT / "tabm_ple.yaml")
        self.assertTrue(tabm.rank_included)
        self.assertFalse(tabm_ple.rank_included)
        self.assertNotIn("ple_n_bins", tabm.search)
        self.assertIn("ple_n_bins", tabm_ple.search)

    def test_realmlp_weighting_and_regression_label_smoothing(self):
        space = load_search_space(SPACE_ROOT / "realmlp.yaml")
        classification = space.sample(FakeTrial(fraction=0.5), 1000, "classification")
        regression_trial = FakeTrial(fraction=0.5)
        regression = space.sample(regression_trial, 1000, "regression")
        self.assertEqual(classification["label_smoothing"], 0.1)
        self.assertEqual(regression["label_smoothing"], 0.0)
        self.assertNotIn("label_smoothing__weighted_selector", regression_trial.params)
        space.validate_resolved(regression, n_rows=1000, task_type="regression")

    def test_out_of_range_fixed_collision_and_forbidden_fail(self):
        space = load_search_space(SPACE_ROOT / "daf_moe_v2.yaml")
        resolved = space.sample(FakeTrial(), 100, "classification")
        resolved["d_emb"] = 17
        with self.assertRaisesRegex(ValueError, "d_emb"):
            space.validate_resolved(resolved, n_rows=100, task_type="classification")

        payload = space.as_dict()
        payload["fixed"]["d_emb"] = 32
        with self.assertRaisesRegex(ValueError, "collision"):
            parse_search_space(payload)


class EngineSemanticsTests(unittest.TestCase):
    def test_fail_and_oom_do_not_count_toward_50_complete(self):
        study = FakeStudy()

        def objective(trial):
            if trial.number == 0:
                raise ValueError("invalid")
            if trial.number == 1:
                raise RuntimeError("CUDA out of memory")
            return float(trial.number)

        run_until_valid_complete(study, objective, target=50, max_attempts=60)
        self.assertEqual(valid_complete_count(study), 50)
        self.assertEqual(len(study.trials), 52)
        self.assertEqual([trial.state.name for trial in study.trials[:2]], ["FAIL", "FAIL"])

    def test_nonfinite_metric_raises_and_writes_fail_artifact(self):
        space = load_search_space(SPACE_ROOT / "mlp.yaml")
        with tempfile.TemporaryDirectory() as directory:
            writer = TrialArtifactWriter(directory)
            objective = make_guarded_objective(
                space,
                lambda resolved, trial: {"metric_value": float("nan")},
                n_rows=10,
                artifact_writer=writer,
                study_name="tiny__mlp__phase2-v1",
                task_type="classification",
            )
            with self.assertRaises(FloatingPointError):
                objective(FakeTrial())
            artifact = Path(directory) / "tiny__mlp__phase2-v1" / "trial_00000.json"
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "FAIL")
            self.assertEqual(payload["failure_type"], "FloatingPointError")


class ExecutionSmokeTests(unittest.TestCase):
    def setUp(self):
        self.space = load_search_space(SPACE_ROOT / "mlp.yaml")
        self.raw = _tiny_raw()
        self.base = {
            "model_name": "mlp",
            "task_type": "classification",
            "optimize_metric": "acc",
            "batch_size": 16,
        }
        self.resolved = self.space.sample(
            FakeTrial(fraction=0.0), len(self.raw.features), "classification"
        )

    def test_one_trial_uses_sealed_hpo_and_restores_disk_best(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trial_best.pth"
            with patch(
                "src.phase2_execution.get_phase2_dataloaders",
                side_effect=AssertionError("HPO requested test data"),
            ):
                outcome = execute_hpo_trial(
                    self.raw,
                    self.base,
                    self.space,
                    self.resolved,
                    "classification",
                    "acc",
                    checkpoint_path=checkpoint,
                    device=torch.device("cpu"),
                )
            self.assertTrue(np.isfinite(outcome.metric_value))
            self.assertTrue(checkpoint.exists())
            self.assertIsNotNone(outcome.best_epoch_or_iteration)
            self.assertLessEqual(outcome.epochs_completed, 400)

    def test_two_final_seeds_and_hash_gated_resume(self):
        outcomes = []
        with tempfile.TemporaryDirectory() as directory:
            for seed in (43, 44):
                outcomes.append(
                    execute_final_seed(
                        self.raw,
                        self.base,
                        self.space,
                        self.resolved,
                        "classification",
                        "acc",
                        seed,
                        checkpoint_path=Path(directory) / f"seed{seed}.pth",
                        device=torch.device("cpu"),
                    )
                )
            self.assertNotEqual(outcomes[0].split_hash, outcomes[1].split_hash)
            result_path = Path(directory) / "seed43.json"
            payload = {
                "manifest": outcomes[0].manifest,
                "resume_manifest": outcomes[0].manifest,
            }
            write_result(result_path, payload)
            self.assertTrue(reusable_result(result_path, outcomes[0].manifest))
            changed = dict(outcomes[0].manifest)
            changed["resolved_config_hash"] = "mismatch"
            self.assertFalse(reusable_result(result_path, changed))

    def test_hpo_seed_cannot_enter_final(self):
        with self.assertRaisesRegex(ValueError, "seed 42"):
            execute_final_seed(
                self.raw,
                self.base,
                self.space,
                self.resolved,
                "classification",
                "acc",
                42,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
