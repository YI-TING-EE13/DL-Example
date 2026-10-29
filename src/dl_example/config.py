"""Typed experiment settings loaded from TOML and command-line overrides."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "mnist"
    epochs: int = 5
    batch_size: int = 64
    lr: float = 0.001
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    validation_fraction: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    early_stop_patience: int = 0
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    data_dir: str = "data"
    output_dir: str = "runs"
    tracking: bool = False

    def validate(self) -> None:
        if self.dataset not in {"mnist", "cifar10"}:
            raise ValueError("dataset must be mnist or cifar10")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        for name in ("epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("num_workers", "early_stop_patience", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed >= 2**32:
            raise ValueError("seed must be smaller than 2**32")
        for name in ("max_train_samples", "max_eval_samples"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer when provided")
        for name in ("lr", "validation_fraction", "momentum", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.lr <= 0 or self.weight_decay < 0 or not 0 <= self.momentum < 1:
            raise ValueError("lr must be positive; weight_decay >= 0; 0 <= momentum < 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if type(self.tracking) is not bool:
            raise ValueError("tracking must be a boolean")
        for name in ("data_dir", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path string")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | None = None, dataset: str | None = None) -> ExperimentConfig:
    values = {} if path is None else tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(values) - {field.name for field in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    if dataset is not None and "dataset" in values and values["dataset"] != dataset:
        raise ValueError("--dataset does not match the configuration file")
    selected = dataset or values.get("dataset", "mnist")
    defaults = {}
    if selected == "cifar10":
        defaults = dict(
            epochs=100, batch_size=128, lr=0.1, weight_decay=0.0005, early_stop_patience=10
        )
    return ExperimentConfig(**(defaults | {"dataset": selected} | values))
