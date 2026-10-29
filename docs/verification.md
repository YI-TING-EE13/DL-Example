# Verification record

Verification completed on 2026-10-30 (Asia/Taipei). Full reference runs began on 2026-10-30 local time; run IDs use UTC. This is local execution and CI evidence, not student classroom acceptance.

## Scope and source

- Original executable baseline: `cafedbb5c23bbd9585e4e079c50b47b57a08b0c2`.
- Verified executable revision: `d3eac41e9f5c62639ab4370f6f0a422a07cc772a`. Later documentation commits preserve this result provenance.
- Environment: Windows, Python 3.12.11, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 2.5.4, CUDA runtime 12.4, cuDNN 9.1, NVIDIA RTX 3060 12 GB, driver 621.01, uv 0.9.18.
- Separate CPU-only installation: PyTorch 2.6.0+cpu. Default auto selection used CPU; explicitly requesting unavailable CUDA returned the expected nonzero exit with an explanation.
- Both final full runs recorded a clean working tree. Resolved settings, dataset-content hashes, split hashes, environment, and exact metrics are in [reference-results.json](validation/reference-results.json).

## Original baseline

`uv sync --python 3.12` and `uv run --frozen python run_mnist.py` completed successfully before restructuring. The original run reported 99.08% best accuracy after five epochs. It selected the best epoch using the official test split as validation. That value is retained as an original-execution observation, not a comparable independent held-out benchmark.

## Measured results

All times below are application elapsed time with datasets already downloaded. They include data preparation, training, evaluation, and checkpoint writes, but not uv installation. They are observations on this machine, not classroom timing guarantees.

| Run | Epochs | Train / validation / test samples | Test accuracy | Seconds | Result |
|---|---:|---:|---:|---:|---|
| MNIST full 1 | 5 | 54,000 / 6,000 / 10,000 | 99.0200% | 47.20 | PASS |
| MNIST full 2 | 5 | 54,000 / 6,000 / 10,000 | 99.0200% | 46.18 | PASS |
| MNIST CPU-only smoke | 1 | 1,024 / 256 / 256 | 83.9844% | 8.78 | PASS |
| CIFAR-10 GPU smoke | 1 | 1,024 / 256 / 256 | 16.7969% | 24.70 | PASS |

The two full MNIST runs achieved **99.02%** held-out test accuracy. Configuration, data manifest, per-epoch metric CSV, held-out metrics, and every tensor in the selected best-model state were exactly equal between the two runs. Their independent CSV records are [run 1](validation/mnist-repeat-1.csv) and [run 2](validation/mnist-repeat-2.csv). Elapsed times and run IDs intentionally differ.

The CPU and CIFAR-10 rows are subset pipeline checks. Their accuracy is reported for transparency and is not a full-training result. CIFAR-10 used two Windows DataLoader workers. Initial CIFAR-10 archive download was slow on this connection; prepare the complete dataset before a live demonstration.

## Commands and checks

Commands run from the repository root after selecting the indicated installation:

```powershell
uv sync --locked --extra cuda
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
uv run --no-sync python -m dl_example check
uv run --no-sync python run_mnist.py --config configs/mnist.toml
uv run --no-sync python run_mnist.py --config configs/mnist.toml
uv run --no-sync python run_cifar.py --smoke --num-workers 2
```

CPU-only installation was tested in a separate environment using `UV_PROJECT_ENVIRONMENT=.validation/cpu-env`, followed by `uv sync --locked --extra cpu --no-dev`. The default synthetic check and `python run_mnist.py --smoke --device cpu` passed there. `check --device cuda` failed as expected in that CPU-only environment. The environment variable was restored after this check.

Checkpoint evaluation was also run using `python -m dl_example evaluate <actual-run-directory>/best.pt --device cuda`. Its held-out result matched the stored summary. Only trusted local checkpoints were loaded.

| Check | Result | Evidence and coverage |
|---|---|---|
| Focused tests | PASS | 28 tests: settings, disjoint splits, sample-weighted loss, real updates, checkpoint round-trip, model shapes, surrogate backward coefficients, dataset hashes, and failure/interruption records |
| Ruff lint and format | PASS | Project checks completed without remaining diagnostics |
| Locked clean CPU-only installation | PASS | Separate local environment and real-data smoke execution |
| Explicit CUDA selection and CPU fallback | PASS | Real GPU execution, CPU-only auto selection, and expected explicit-CUDA rejection |
| Full MNIST training and repeatability | PASS | Two clean-source five-epoch runs; exact metric and model-state comparison |
| Separate checkpoint evaluation | PASS | Held-out metrics match the selected full-run checkpoint summary |
| CIFAR-10 pipeline | PASS | Real dataset; one epoch; model update, validation, checkpoint, held-out subset, and Windows workers |
| Windows and Linux CPU CI | PASS | [Implementation CI](https://github.com/YI-TING-EE13/DL-Example/actions/runs/40912354203), revision `d3eac41e9f5c62639ab4370f6f0a422a07cc772a` |
| Full CIFAR-10 accuracy reproduction | SKIPPED | Outside the agreed short-run scope |
| Exact interrupted-training resumption | SKIPPED | Not implemented or claimed |
| Optional W&B tracking | SKIPPED | Optional integration not included in this validation; core artifacts require no tracking service |
| Human classroom acceptance | SKIPPED | No student pilot has been conducted |

## Files and compatibility

The update adds the installable `dl_example` package, TOML settings, lockfile, tests, CI, and English project guides. It retains the two root script names and both model computations. Old `src.*` imports and unversioned checkpoints require migration; see [architecture](architecture.md). The validation split and sample-weighted loss establish a new protocol.

Datasets, weights, raw run logs, local environments, and earlier exploratory runs remain ignored local artifacts. Public evidence contains configuration, hashes, metrics, and environment summaries rather than those large files. No original Git history was rewritten. Ordinary commits preserve the baseline and make recovery possible.

Exact agreement here applies to this measured environment. It does not establish equality across different GPUs, CPU execution, package versions, or future changes.
