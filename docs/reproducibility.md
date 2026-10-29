# Reproducibility protocol

## What is being verified

The agreed scope is full five-epoch MNIST training, repeated in one controlled GPU environment, plus CPU execution and a short CIFAR-10 pipeline check. Installation, pipeline execution, predictive metrics, and repeatability are separate acceptance claims. A one-epoch subset test cannot establish the original CIFAR-10 accuracy target.

## Procedure

1. Start from a recorded commit and a clean working tree. Install Python 3.12 dependencies from `uv.lock` with either the `cpu` or `cuda` extra.
2. Run the synthetic environment check before downloading data.
3. Train with `configs/mnist.toml`, preserving the printed run directory. Train again with the same settings in the same environment.
4. Compare the resolved configuration, dataset index hashes, per-epoch metrics, selected epoch, and held-out test metrics. Exclude timestamps and elapsed time from numerical equality checks.
5. Reload `best.pt` with the evaluation command and compare its test metrics with the run summary.
6. Exercise explicit CPU execution and the CIFAR-10 smoke configuration. Label these as smoke checks when subset limits are enabled.

Example commands from the repository root after installation:

```powershell
uv run --no-sync python -m dl_example check
uv run --no-sync python run_mnist.py --config configs/mnist.toml
uv run --no-sync python run_mnist.py --config configs/mnist.toml
uv run --no-sync python run_mnist.py --smoke --device cpu
uv run --no-sync python run_cifar.py --smoke
```

## Evidence

Retain `run.json`, `config.json`, `data.json`, `metrics.csv`, `summary.json`, and the selected checkpoint together. `run.json` records the code commit and whether the working tree was dirty. Record the resolved package lock, GPU/driver information, and validation commands in the verification report. A later documentation commit does not change which code revision produced a result.

The Git SHA alone does not describe ignored data, weights, environments, or uncommitted edits. Index hashes identify the selected example indices; SHA-256 hashes separately identify the actual uncompressed input files used by the dataset loaders. Transforms and source split policies are recorded in `data.json`.

## Numerical expectations

The initial MNIST acceptance target is held-out accuracy of at least 98% after five epochs in the declared environment; report the measured value even if the target is missed. Repeatability is assessed independently by comparing both runs. Do not tune using the held-out test set. Any adjustment after observing test performance requires a separately labeled experiment.

The original README listed approximately 99% MNIST and 93% CIFAR-10 without committed raw experimental evidence. Those figures are context, not verified historical baselines. The new validation split changes the protocol.

Deterministic algorithms are requested explicitly. Python, NumPy, PyTorch, CUDA, and DataLoader worker random generators are seeded. Exact matching is assessed only in the tested environment. PyTorch does not guarantee identical results across versions, platforms, or CPU/GPU execution. See [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) and [uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Limits

- Full CIFAR-10 accuracy reproduction is not part of this verification.
- Checkpoint loading is supported for evaluation; exact training resumption is not implemented.
- CPU and GPU results may differ even with the same seed.
- Dataset download and package installation require network access on first use.
- A configured W&B offline run can be enabled separately; the core workflow does not upload results.
