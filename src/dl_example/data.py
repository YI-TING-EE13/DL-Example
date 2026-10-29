"""Independent training, validation, and held-out test data loaders."""

import hashlib
import json

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dl_example.config import ExperimentConfig
from dl_example.reproducibility import seed_worker


def split_indices(size: int, validation_fraction: float, seed: int) -> tuple[list, list]:
    n_validation = max(1, int(size * validation_fraction))
    if size < 2 or n_validation >= size:
        raise ValueError("The split must contain at least one training and validation example")
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[n_validation:], indices[:n_validation]


def _index_hash(indices: list) -> str:
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()


def prepare_data(config: ExperimentConfig, device: torch.device):
    if config.dataset == "mnist":
        evaluation_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        training_transform = evaluation_transform
        dataset_class = datasets.MNIST
    else:
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        training_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalization,
            ]
        )
        evaluation_transform = transforms.Compose([transforms.ToTensor(), normalization])
        dataset_class = datasets.CIFAR10
    train_source = dataset_class(
        config.data_dir, train=True, download=True, transform=training_transform
    )
    validation_source = dataset_class(
        config.data_dir, train=True, download=True, transform=evaluation_transform
    )
    test_source = dataset_class(
        config.data_dir, train=False, download=True, transform=evaluation_transform
    )
    train_indices, validation_indices = split_indices(
        len(train_source), config.validation_fraction, config.seed
    )
    train_indices = train_indices[: config.max_train_samples]
    validation_indices = validation_indices[: config.max_eval_samples]
    test_indices = list(range(len(test_source)))[: config.max_eval_samples]
    subsets = [
        Subset(train_source, train_indices),
        Subset(validation_source, validation_indices),
        Subset(test_source, test_indices),
    ]
    loaders = []
    for position, subset in enumerate(subsets):
        loaders.append(
            DataLoader(
                subset,
                batch_size=config.batch_size,
                shuffle=position == 0,
                num_workers=config.num_workers,
                pin_memory=device.type == "cuda",
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(config.seed + position),
            )
        )
    manifest = {
        "dataset": config.dataset,
        "split_seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test_indices),
        "train_index_sha256": _index_hash(train_indices),
        "validation_index_sha256": _index_hash(validation_indices),
        "test_index_sha256": _index_hash(test_indices),
        "training_transform": repr(training_transform),
        "evaluation_transform": repr(evaluation_transform),
        "test_policy": "Official test split; excluded from model selection and early stopping",
    }
    return (*loaders, manifest)
