# Architecture and migration

`run_mnist.py` and `run_cifar.py` delegate to `dl_example.cli`. The CLI resolves and validates settings, selects a device, prepares independent data splits, constructs a model on that device, and then creates its optimizer.

## Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Typed settings, TOML loading, and boundary validation |
| `data.py` | Dataset transforms, deterministic splitting, and DataLoaders |
| `models/` | MNIST CNN and Shake-Shake classifier; raw class logits |
| `trainer.py` | Classification loops, sample-weighted metrics, model selection, and checkpoints |
| `reproducibility.py` | RNG setup, explicit device behavior, and environment metadata |
| `cli.py` | User commands, orchestration, artifacts, and optional offline tracking |

The trainer uses mean cross-entropy for classification. It multiplies each batch mean by its sample count before computing the epoch mean, including a short final batch. Training enables gradients; evaluation uses both evaluation mode and disabled gradients. The scheduler steps once after each epoch. Model placement occurs before optimizer construction.

## Data and model selection

MNIST uses 54,000 training and 6,000 validation examples by default. CIFAR-10 uses 45,000 and 5,000. Validation is selected from the official training source using a seeded permutation. Validation and test transforms exclude random augmentation. The official 10,000-example test set is excluded from checkpoint selection and early stopping. Smoke mode limits subsets after splitting.

The two model definitions retain their original computations. Shake-Shake deliberately uses different forward and surrogate backward mixing coefficients; a conventional numerical gradcheck is not the acceptance test for that deliberate mismatch. A focused test checks each coefficient's actual role. The model is an educational implementation, not a claim to reproduce the published paper's complete protocol.

## Artifacts and compatibility

Each run uses a unique subdirectory. The checkpoint format is version 1 and includes the configuration, model, optimizer, optional scheduler, epoch, and selection state. The CLI supports evaluation loading, not exact training resumption. RNG state and loader position are not restored. Only load trusted checkpoints produced by this project.

Version 0.2 replaces imports such as `src.models.mnist_cnn` with `dl_example.models.mnist_cnn`. Install the project with uv before running scripts. The two root script filenames are retained. Old checkpoints have no versioned configuration record and are not accepted by the new evaluation command. Original history and any local original checkpoints remain available separately.

Changes to validation splitting and sample-weighted loss establish a new experimental baseline. Do not compare old README numbers as if the protocols were identical. W&B is optional; the main run artifacts do not require an account or tracking service.
