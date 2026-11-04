# Versioning and compatibility

The installed package is version 0.2.0 (`pyproject.toml`). Annotated `vX.Y.Z` tags mark reviewed stable code snapshots. `archive/` tags retain historical PR heads; they are not releases. Never force-move a published tag. No GitHub Release is created automatically by this policy.

Version 0.2 uses `dl_example` package imports, version-1 configured checkpoints and held-out test evaluation after validation-based model selection. Root `run_mnist.py` and `run_cifar.py` entry scripts remain. Legacy unversioned checkpoints are not accepted by the current evaluate command. Exact interrupted-training resumption is not implemented.

The stable documentation revision preserves both model definitions, training/evaluation code, dependency lock and experiment semantics. Windows Python 3.12 locked CPU verification and Windows/Ubuntu CPU CI cover lint, formatting, unit tests and a synthetic forward/backward/update. Full MNIST/CIFAR-10 accuracy reproduction, CUDA repeatability and real student GitHub acceptance are separate, unclaimed checks. See [verification](verification.md) for the historical measured runs and their limits.

A tag identifies code. Record configuration, actual data/splits, software/hardware and result scope separately for each experiment. A GitHub Release, if later authorized, is a separate publication with notes/assets based on a tag.
