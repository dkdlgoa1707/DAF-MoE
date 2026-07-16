from pathlib import Path
import tempfile
import unittest

from src.hpo.engine import (
    TrialArtifactWriter,
    create_phase2_study,
    make_guarded_objective,
    run_until_valid_complete,
    valid_complete_count,
)
from src.hpo.identity import StudyIdentity
from src.hpo.schema import load_search_space
from src.phase2_protocol import PROTOCOL_VERSION, model_implementation_version


SPACE_ROOT = Path("configs/hpo/phase2")


def _identity(space, target_policy="class_mapping"):
    return StudyIdentity.from_components(
        {
            "protocol_version": PROTOCOL_VERSION,
            "dataset_name": "sqlite-fixture",
            "dataset_schema_hash": "schema-001",
            "dataset_schema_version": "fixture-v1",
            "model_name": space.model_name,
            "model_implementation_version": model_implementation_version(
                space.model_name
            ),
            "task_type": "classification",
            "optimize_metric": "acc",
            "search_space_hash": space.schema_hash,
            "base_experiment_config_hash": "base-001",
            "effective_regression_target_policy": target_policy,
        }
    )


class StudyIdentitySQLiteTests(unittest.TestCase):
    def test_real_sqlite_resume_isolation_and_strict_counting(self):
        space = load_search_space(SPACE_ROOT / "mlp.yaml")
        identity = _identity(space)
        with tempfile.TemporaryDirectory() as directory:
            storage = identity.default_storage_url(directory)
            self.assertTrue(storage.endswith(f"{identity.study_name}.db"))
            self.assertIn(identity.prefix, identity.study_name)

            writer = TrialArtifactWriter(Path(directory) / "artifacts")
            objective = make_guarded_objective(
                space,
                lambda resolved, trial: {"metric_value": 0.5 + trial.number * 0.01},
                n_rows=100,
                artifact_writer=writer,
                study_identity=identity,
                task_type="classification",
            )
            study = create_phase2_study(identity, "maximize", storage)
            run_until_valid_complete(study, objective, target=2, identity=identity)
            self.assertEqual(valid_complete_count(study, identity), 2)

            resumed = create_phase2_study(identity, "maximize", storage)
            self.assertEqual(valid_complete_count(resumed, identity), 2)
            run_until_valid_complete(resumed, objective, target=3, identity=identity)
            self.assertEqual(valid_complete_count(resumed, identity), 3)

            resumed.ask()

            def fail(_trial):
                raise RuntimeError("intentional failure")

            resumed.optimize(fail, n_trials=1, catch=(RuntimeError,))

            def mismatched(trial):
                trial.set_user_attr("study_signature", "wrong")
                trial.set_user_attr("search_space_hash", space.schema_hash)
                return 0.99

            resumed.optimize(mismatched, n_trials=1)
            self.assertEqual(valid_complete_count(resumed, identity), 3)

            changed_components = dict(identity.components)
            changed_components["search_space_hash"] = "different-search"
            changed_search = StudyIdentity.from_components(changed_components)
            self.assertNotEqual(identity.study_name, changed_search.study_name)
            self.assertNotEqual(
                identity.default_storage_url(directory),
                changed_search.default_storage_url(directory),
            )
            changed_search_study = create_phase2_study(
                changed_search,
                "maximize",
                changed_search.default_storage_url(directory),
            )
            self.assertEqual(
                valid_complete_count(changed_search_study, changed_search), 0
            )

            changed_policy = _identity(space, target_policy="standardize")
            self.assertNotEqual(identity.signature, changed_policy.signature)
            self.assertNotEqual(identity.study_name, changed_policy.study_name)
            changed_policy_study = create_phase2_study(
                changed_policy,
                "maximize",
                changed_policy.default_storage_url(directory),
            )
            self.assertEqual(
                valid_complete_count(changed_policy_study, changed_policy), 0
            )

    def test_custom_sqlite_path_must_include_signature_identity(self):
        space = load_search_space(SPACE_ROOT / "mlp.yaml")
        identity = _identity(space)
        with tempfile.TemporaryDirectory() as directory:
            wrong = f"sqlite:///{Path(directory) / 'shared.db'}"
            with self.assertRaisesRegex(ValueError, "signature-isolated"):
                create_phase2_study(identity, "maximize", wrong)


if __name__ == "__main__":
    unittest.main()
