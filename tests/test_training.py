"""Verify data isolation, real parameter updates, and evaluation artifacts."""

import csv

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_example.cli import check_environment
from dl_example.config import ExperimentConfig
from dl_example.data import split_indices
from dl_example.models import MnistCnn, ShakeShakeResNet
from dl_example.models.cifar_shake import ShakeShakeFunction
from dl_example.trainer import Trainer, load_checkpoint


def test_split_is_disjoint_and_repeatable():
    training, validation = split_indices(100, 0.1, 42)
    assert len(training) == 90 and len(validation) == 10
    assert not set(training) & set(validation)
    assert set(training) | set(validation) == set(range(100))
    assert (training, validation) == split_indices(100, 0.1, 42)
    assert (training, validation) != split_indices(100, 0.1, 43)


def test_loss_is_weighted_by_sample_count(tmp_path):
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    labels = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    trainer = Trainer(torch.nn.Identity(), None, torch.device("cpu"), tmp_path)
    actual = trainer.evaluate(loader)
    assert actual["loss"] == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())
    assert actual["accuracy"] == pytest.approx(2 / 3)
    assert actual["samples"] == 3


def test_empty_loader_fails(tmp_path):
    loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    trainer = Trainer(torch.nn.Identity(), None, torch.device("cpu"), tmp_path)
    with pytest.raises(ValueError, match="empty"):
        trainer.evaluate(loader)


def test_training_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loader = DataLoader(TensorDataset(torch.randn(7, 2), torch.arange(7) % 2), batch_size=3)
    before = model.weight.detach().clone()
    trainer = Trainer(model, optimizer, torch.device("cpu"), tmp_path)
    result = trainer.fit(loader, loader, ExperimentConfig(epochs=2), scheduler)
    assert not torch.equal(before, model.weight)
    restored = torch.nn.Linear(2, 2)
    payload = load_checkpoint(restored, tmp_path / "best.pt", torch.device("cpu"))
    restored_trainer = Trainer(restored, None, torch.device("cpu"), tmp_path)
    assert restored_trainer.evaluate(loader)["accuracy"] == result["best_validation_accuracy"]
    assert payload["scheduler_state_dict"] is not None
    assert (tmp_path / "last.pt").is_file()
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_synthetic_cpu_check_updates_parameters():
    result = check_environment("cpu")
    assert result["status"] == "pass" and result["dataset_downloaded"] is False


def test_model_shapes_and_evaluation_repeatability():
    mnist = MnistCnn().eval()
    cifar = ShakeShakeResNet(depth=8, w_base=4).eval()
    with torch.no_grad():
        assert mnist(torch.randn(2, 1, 28, 28)).shape == (2, 10)
        inputs = torch.randn(2, 3, 32, 32)
        first = cifar(inputs)
        assert first.shape == (2, 10)
        assert torch.equal(first, cifar(inputs))


def test_shake_uses_independent_backward_mixing():
    first = torch.ones(2, 3, requires_grad=True)
    second = torch.zeros(2, 3, requires_grad=True)
    alpha, beta = torch.tensor(0.2), torch.tensor(0.8)
    result = ShakeShakeFunction.apply(first, second, alpha, beta)
    assert torch.allclose(result, torch.full_like(result, 0.2))
    result.sum().backward()
    assert torch.allclose(first.grad, torch.full_like(first, 0.8))
    assert torch.allclose(second.grad, torch.full_like(second, 0.2))


def test_invalid_shake_depth():
    with pytest.raises(ValueError, match="depth"):
        ShakeShakeResNet(depth=9)
