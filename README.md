# DL-Example

[![CPU checks](https://github.com/YI-TING-EE13/DL-Example/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YI-TING-EE13/DL-Example/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://docs.python.org/3.12/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C)](https://pytorch.org/get-started/previous-versions/#v260)
[![License MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Project overview

Small PyTorch image-classification experiments for learning, running, modifying and studying deep learning algorithms. The project focuses on model behavior, training/evaluation and inspectable experiment records. The Git course borrows it for a small documentation PR; that activity is separate from the ML setup.

## Supported models

| Dataset | Model | Intended use |
|---|---|---|
| MNIST | Small CNN | Five-epoch reference configuration and short pipeline checks |
| CIFAR-10 | Original Shake-Shake classifier | Advanced educational architecture; full paper reproduction is not claimed |

## Requirements

Python 3.12, Git and [uv](https://docs.astral.sh/uv/getting-started/installation/). Choose one PyTorch CPU or CUDA extra. Windows is the primary local environment; CPU CI runs Windows and Ubuntu. See [verification](docs/verification.md) for measured scope. CUDA requires a compatible NVIDIA driver and is not certified by CPU CI.

## Quick start

Clone the project, install the locked CPU environment, and run a synthetic check without a dataset:

```powershell
git clone https://github.com/YI-TING-EE13/DL-Example.git
cd DL-Example
uv sync --locked --extra cpu
uv run --no-sync python -m dl_example check --device cpu
```

This verifies a small forward/backward/optimizer step, not full-training accuracy. Follow the sections below for GPU setup, real datasets and evaluation.

## Installation — CPU / CUDA

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/). Open PowerShell, clone the repository, and enter its directory:

```powershell
git clone https://github.com/YI-TING-EE13/DL-Example.git
cd DL-Example
```

For a classroom documentation contribution, use the minimal [classroom entry](docs/classroom-notes/README.md) and the instructor-provided course handout; clone your own fork.

Choose **one** PyTorch build. A CUDA build supports both GPU execution and explicit CPU execution:

```powershell
# NVIDIA GPU with a compatible driver: PyTorch 2.6 / CUDA 12.4 wheels
uv sync --locked --extra cuda
```

For a smaller installation on a machine without an NVIDIA GPU:

```powershell
uv sync --locked --extra cpu
```

uv creates the project environment and uses the committed `uv.lock`. Do not select both extras. The driver must support the wheel's CUDA runtime; installing a separate CUDA toolkit is not required for these prebuilt wheels. The commands below use `--no-sync` to preserve the build selected during installation. Rerun the appropriate `uv sync` command after changing dependencies.

## Check the environment

```powershell
uv run --no-sync python -m dl_example check
uv run --no-sync python -m dl_example check --device cpu
```

The check performs a synthetic forward pass, backward pass, and optimizer update without downloading a dataset. `--device auto` is the default. Use `--device cuda` to require CUDA; that request fails clearly if CUDA is unavailable.

## Training and evaluation

Run the complete MNIST configuration:

```powershell
uv run --no-sync python run_mnist.py --config configs/mnist.toml
```

For a short real-data test, use one epoch and subsets of 1,024 training, 256 validation, and 256 test examples:

```powershell
uv run --no-sync python run_mnist.py --smoke
uv run --no-sync python run_cifar.py --smoke
```

Datasets are downloaded on first use, including the complete source archives for smoke runs. Prepare data before class. CIFAR-10 training is substantially heavier than MNIST; a smoke run is a pipeline check, not an accuracy benchmark.

Settings live in `configs/mnist.toml` and `configs/cifar10.toml`. CLI overrides include `--device`, `--epochs`, `--batch-size`, `--lr`, `--seed`, `--num-workers`, `--data-dir`, and `--output-dir`. Use `python -m dl_example train --help` through uv for the full list. `--smoke` fixes its epoch and subset limits after other overrides.

## Experiment records and reproducibility

Each run prints its unique directory under `runs/`:

```text
runs/<dataset>-<UTC timestamp>-<id>/
  config.json     Resolved settings
  run.json        Environment, source revision, status, and command
  data.json       Split sizes, index hashes, and transforms
  metrics.csv     Training and validation metrics per epoch
  best.pt         Checkpoint selected using validation accuracy
  last.pt         Checkpoint from the last completed epoch
  summary.json    Selected epoch, held-out test metrics, and elapsed time
```

Evaluate a trusted checkpoint again, replacing the example path with the directory printed by training:

```powershell
uv run --no-sync python -m dl_example evaluate runs/YOUR-RUN/best.pt --device cpu
```

Validation examples come from a seeded split of the official training set. The official test set is used after model selection. Checkpoints support evaluation loading; exact interrupted-training resumption is not implemented. Fixed seeds reduce variability within a controlled environment and do not guarantee equality across hardware or library versions.

## Project structure

| Location | Responsibility |
|---|---|
| `run_mnist.py`, `run_cifar.py` | Small entry scripts retained for familiarity |
| `configs/` | Versioned experiment settings |
| `src/dl_example/` | CLI, data preparation, model definitions, training, and provenance |
| `tests/` | CPU tests without external dataset downloads |
| `docs/` | ML architecture, reproducibility, verification and a minimal classroom entry |

## Testing / CI

The default development group includes pytest and Ruff:

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

An optional `tracking` extra enables local W&B logging. Install it together with the chosen compute extra and pass `--tracking`. Cloud upload is not part of the default workflow.

## Documentation and contributing

- [Architecture and migration](docs/architecture.md)
- [Reproducibility protocol](docs/reproducibility.md)
- [Verification record and limitations](docs/verification.md)
- [Contribution guide](CONTRIBUTING.md)

For the Git course, fork/clone your own copy and use the minimal [classroom-note contribution entry](docs/classroom-notes/README.md). The detailed lesson, collection template and teacher preparation are maintained in the separate instructor-provided Git course materials. Classroom PRs may be closed without merging after assessment; participation does not imply integration into ML main.

## Versioning

The package version is 0.2.0. Stable code versions use annotated `vX.Y.Z` tags; historical `archive/` tags preserve earlier PR heads and are not software releases. A Git tag does not automatically create a GitHub Release or preserve data/environment. Read the [version policy](docs/versioning.md) for scope and compatibility.

## License

Distributed under the [MIT License](LICENSE). Dataset and dependency terms remain those of their providers. Datasets, weights, environments and generated runs are ignored; keep actual code/configuration/data/environment/result provenance together.
