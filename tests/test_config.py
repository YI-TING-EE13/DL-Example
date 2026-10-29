"""Validate settings before downloads or output creation."""

from dataclasses import replace

import pytest

from dl_example.cli import main
from dl_example.config import ExperimentConfig, load_config
from dl_example.reproducibility import select_device


@pytest.mark.parametrize(
    "settings",
    [
        {"lr": 0},
        {"lr": float("nan")},
        {"epochs": 0},
        {"batch_size": True},
        {"validation_fraction": 1},
        {"validation_fraction": 0},
        {"seed": -1},
        {"seed": 2**32},
        {"num_workers": -1},
        {"max_eval_samples": 0},
        {"device": "missing"},
        {"output_dir": ""},
        {"tracking": "yes"},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        replace(ExperimentConfig(), **settings).validate()


def test_unknown_config_key(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text("learning_rate = 0.01", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(str(path))


def test_config_and_dataset_agree(tmp_path):
    path = tmp_path / "cifar.toml"
    path.write_text('dataset = "cifar10"', encoding="utf-8")
    assert load_config(str(path)).epochs == 100
    with pytest.raises(ValueError, match="does not match"):
        load_config(str(path), "mnist")


def test_explicit_unavailable_cuda_fails(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda")


def test_invalid_cli_fails_before_training(monkeypatch, capsys):
    monkeypatch.setattr("dl_example.cli.prepare_data", lambda *_: pytest.fail("Downloaded data"))
    with pytest.raises(SystemExit) as result:
        main(["train", "--epochs", "0"])
    assert result.value.code == 1
    assert "positive integer" in capsys.readouterr().err
