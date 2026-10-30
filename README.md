# DL-Example

Small PyTorch image-classification experiments for learning reproducible development with Git and GitHub.

- **MNIST:** a small CNN and a five-epoch reference configuration.
- **CIFAR-10:** the original Shake-Shake classifier, retained as an advanced example.
- **Workflow:** configure an experiment, inspect a change, run a check, commit it, and open a pull request.

Windows with Python 3.12 and uv is the primary teaching environment. CUDA is used when available; CPU execution is supported. See [verification](docs/verification.md) for what was actually tested. Full CIFAR-10 accuracy reproduction is outside the current validation scope.

## Git classroom task

The [classroom contribution guide](docs/teaching.md) requires only Git, a GitHub account, a text editor, and a terminal. Fork and clone the project, write a short explanation, and open a pull request. Python, uv, PyTorch, and model checks are optional for this Git task. The installation and training sections below apply when you run the ML examples.

Create your own file in [docs/classroom-notes](docs/classroom-notes/README.md). After the two individual questions, save branch, contribution SHA, PR URL, Content answer, and Location answer together using the [submission template](docs/classroom-submission.md). Blocked publication uses truthful pending fields and private collection. See [course preparation and readiness](docs/course-ready.md) for instructor checks.

Use the latest `main` as the classroom starting point. [Branch authority and the current audit scope](docs/course-ready.md#branch-authority) distinguish delivery from preserved implementation and PR history.

## Install

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/). Open PowerShell, clone the repository, and enter its directory:

```powershell
git clone https://github.com/YI-TING-EE13/DL-Example.git
cd DL-Example
```

Students contributing through a fork should follow [the classroom workflow](docs/teaching.md) and clone their own fork instead.

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

## Train

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

## Read the results

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

## Develop

The default development group includes pytest and Ruff:

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

An optional `tracking` extra enables local W&B logging. Install it together with the chosen compute extra and pass `--tracking`. Cloud upload is not part of the default workflow.

## Project guide

| Location | Responsibility |
|---|---|
| `run_mnist.py`, `run_cifar.py` | Small entry scripts retained for familiarity |
| `configs/` | Versioned experiment settings |
| `src/dl_example/` | CLI, data preparation, model definitions, training, and provenance |
| `tests/` | CPU tests without external dataset downloads |
| `docs/` | Architecture, reproducibility, classroom contribution, and verification |

- [Architecture and migration](docs/architecture.md)
- [Reproducibility protocol](docs/reproducibility.md)
- [Fork, clone, and pull request practice](docs/teaching.md)
- [Verification record and limitations](docs/verification.md)
- [Contribution guide](CONTRIBUTING.md)

Datasets, environments, weights, and generated run artifacts are ignored by Git. Record results using their actual code revision, configuration, data, and environment. This project is distributed under the [MIT License](LICENSE). Dataset and dependency terms remain those of their respective providers.
