"""Verify Phase 2 baseline construction and forward contracts."""

import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.default_config import DAFConfig
from src.models.factory import create_model
from src.trainer import Trainer


BASELINES = [
    "ft_transformer",
    "mlp",
    "resnet",
    "tabm",
    "tabr",
    "modernnca",
    "realmlp",
]


def make_synthetic_config(task):
    config = DAFConfig()
    config.n_numerical = 8
    config.n_categorical = 3
    config.n_features = 11
    config.total_cats = 20
    config.d_emb = 96
    config.d_ff = int(config.d_emb * config.d_ff_factor)
    config.task_type = "regression" if task == "regression" else "classification"
    config.out_dim = 1
    return config


def synthetic_inputs(config, batch_size):
    numerical = torch.randn(batch_size, config.n_numerical, 3)
    numerical[:, :, 1] = torch.rand(batch_size, config.n_numerical)
    categorical = torch.randint(
        0, config.total_cats, (batch_size, config.n_categorical)
    )
    categorical_meta = torch.rand(batch_size, config.n_categorical, 2)
    return {
        "x_numerical": numerical,
        "x_categorical_idx": categorical,
        "x_categorical_meta": categorical_meta,
    }


def attach_retrieval_context(model, config, task):
    context = synthetic_inputs(config, batch_size=12)
    labels = (
        torch.randn(12)
        if task == "regression"
        else torch.randint(0, 2, (12,)).float()
    )
    if hasattr(model, "set_candidates"):
        model.set_candidates(context, labels)
    if hasattr(model, "set_train_context"):
        model.set_train_context(context, labels)


def verify(model_name, task):
    config = make_synthetic_config(task)
    config.model_name = model_name
    model = create_model(config).eval()
    attach_retrieval_context(model, config, task)

    with torch.no_grad():
        output = model(**synthetic_inputs(config, batch_size=4))
    assert isinstance(output, dict), "forward output must be a dict"
    assert "logits" in output, "forward output is missing logits"
    assert output["logits"].shape == (4, config.out_dim), output["logits"].shape
    assert torch.isfinite(output["logits"]).all(), "logits contain non-finite values"
    return sum(parameter.numel() for parameter in model.parameters())


class _ContextDataset(Dataset):
    def __init__(self, size=8):
        self.numerical = torch.randn(size, 2, 3)
        self.categorical = torch.randint(0, 4, (size, 1))
        self.categorical_meta = torch.rand(size, 1, 2)
        self.targets = torch.randn(size)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "x_numerical": self.numerical[index],
            "x_categorical_idx": self.categorical[index],
            "x_categorical_meta": self.categorical_meta[index],
        }, self.targets[index]


class _ContextSpyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1)
        self.candidate_calls = 0
        self.context_calls = 0
        self.context_rows = 0

    def set_candidates(self, inputs, targets):
        self.candidate_calls += 1
        self.context_rows = len(targets)

    def set_train_context(self, inputs, targets):
        self.context_calls += 1
        self.context_rows = len(targets)

    def forward(self, x_numerical, **kwargs):
        return {"logits": self.head(x_numerical[:, :1, 0]), "aux_loss": None}


def verify_trainer_context_wiring():
    config = DAFConfig(
        model_name="context_spy",
        task_type="regression",
        out_dim=1,
        epochs=1,
        optimize_metric="rmse",
    )
    loader = DataLoader(_ContextDataset(), batch_size=4, shuffle=False)
    model = _ContextSpyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        nn.MSELoss(),
        optimizer,
        config,
        torch.device("cpu"),
        logging.getLogger("verify_baselines"),
        verbose=False,
    )
    trainer.fit(loader, loader)
    assert model.candidate_calls == 1
    assert model.context_calls == 1
    assert model.context_rows == len(loader.dataset)

    trainer.test(loader)
    assert model.candidate_calls == 2
    assert model.context_calls == 2
    print(
        "\n[trainer context wiring]\n"
        "  fit/test set_candidates + set_train_context calls PASS"
    )


def main():
    print("=" * 68)
    print("Phase 2 Baseline Verification")
    print("=" * 68)
    failures = []

    for model_name in BASELINES:
        print(f"\n[{model_name}]")
        for task in ("regression", "binary"):
            try:
                parameters = verify(model_name, task)
                print(
                    f"  task={task:<10} params={parameters:>10,} "
                    "logits=(4, 1) PASS"
                )
            except Exception as exc:
                failures.append((model_name, task, exc))
                print(f"  task={task:<10} FAIL: {exc}")

    try:
        verify_trainer_context_wiring()
    except Exception as exc:
        failures.append(("trainer", "context_wiring", exc))
        print(f"\n[trainer context wiring]\n  FAIL: {exc}")

    print("\n" + "=" * 68)
    if failures:
        print(f"Baseline verification failed: {len(failures)} case(s).")
        for model_name, task, exc in failures:
            print(f"  - {model_name}/{task}: {exc}")
        raise SystemExit(1)
    print(f"Baseline verification passed: {len(BASELINES) * 2} case(s).")
    print("=" * 68)


if __name__ == "__main__":
    main()
