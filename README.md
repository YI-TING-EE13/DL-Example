# 🧠 Deep Learning Training Framework — MNIST & CIFAR-10

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![WandB](https://img.shields.io/badge/Logging-WandB-orange.svg)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A professional, modular deep learning training framework for image classification experiments. Implements a **shared training engine** (Facade Pattern) with two benchmark experiments:

| Experiment | Model | Dataset | Target Accuracy |
|:---|:---|:---|:---|
| **MNIST** | Simple CNN | 28×28 Grayscale Digits | ~99% |
| **CIFAR-10** | Shake-Shake ResNet-26 | 32×32 RGB Natural Images | ~93% |

Designed as both a **teaching tool** and a **research-ready** baseline, with comprehensive bilingual comments (English + 繁體中文) throughout the codebase.

---

## ✨ Key Features

- **Facade Pattern Training Engine**: A single, shared `Trainer` class handles the entire training lifecycle — ensuring fair comparison between models under identical conditions.
- **Shake-Shake Regularisation**: Advanced custom `torch.autograd.Function` implementation with stochastic forward/backward gradient mixing for robust feature learning.
- **Automatic Best-Model Checkpointing**: Saves the best model (by validation accuracy) with full optimizer and scheduler state for seamless training resumption.
- **Early Stopping**: Configurable patience parameter to prevent wasted compute on overfitting models.
- **GPU Acceleration**: Native CUDA support with `pin_memory` and multi-worker data loading for optimal GPU utilisation.
- **Experiment Tracking**: Integrated WandB logging (offline mode by default) for loss curves, accuracy, and learning rate monitoring.
- **Reproducibility**: Deterministic seeding across Python, NumPy, PyTorch, and CuDNN to guarantee identical results on repeated runs.

---

## 📁 Project Structure

```
├── pyproject.toml              # Project config & dependencies (uv)
├── README.md                   # This file
├── run_mnist.py                # MNIST experiment entry-point
├── run_cifar.py                # CIFAR-10 experiment entry-point
│
├── src/                        # Core framework package
│   ├── __init__.py
│   ├── trainer.py              # Shared training engine (Facade Pattern)
│   ├── utils.py                # Seed setting, device detection
│   └── models/                 # Neural network architectures
│       ├── __init__.py
│       ├── mnist_cnn.py        # Simple CNN for MNIST
│       └── cifar_shake.py      # Shake-Shake ResNet-26 for CIFAR-10
│
├── data/                       # Auto-downloaded datasets (gitignored)
├── checkpoints/                # Saved model weights (gitignored)
└── wandb/                      # WandB offline logs (gitignored)
```

---

## 🛠️ Installation

### Prerequisites

- **Python 3.12+**
- **CUDA-enabled GPU** (recommended; CPU fallback supported)
- **NVIDIA Drivers** compatible with CUDA 12.4
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YI-TING-EE13/DL-Example.git
   cd DL-Example
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```
   This will automatically:
   - Create a virtual environment (`.venv/`)
   - Install PyTorch with CUDA 12.4 support
   - Install `torchvision`, `wandb`, `tqdm`, `numpy`

> [!NOTE]
> PyTorch is installed from the official CUDA 12.4 wheel index (`https://download.pytorch.org/whl/cu124`), configured in `pyproject.toml`. If your GPU uses a different CUDA version, update the `[tool.uv.index]` URL accordingly.

---

## 🚀 Usage

### Experiment 1: MNIST (Quick Start)

```bash
uv run python run_mnist.py
```

This trains a lightweight CNN on the MNIST handwritten-digit dataset for **5 epochs**. Expected output:

```
[GPU] Using: NVIDIA GeForce RTX 3060 (12.0 GB)

============================================================
  Starting training: MNIST
  Epochs: 5  |  Device: cuda
============================================================

[MNIST] Epoch 1/5: Train Loss=0.1946  |  Train Acc=0.9416  |  Val Acc=0.9852  |  LR=0.001000
  [SAVE] Checkpoint saved -> checkpoints/mnist/model_best.pt
...
[MNIST] Epoch 5/5: Train Loss=0.0440  |  Train Acc=0.9865  |  Val Acc=0.9908  |  LR=0.001000

[DONE] Training complete -- Best Val Accuracy: 0.9908
```

### Experiment 2: CIFAR-10 (Advanced)

```bash
uv run python run_cifar.py
```

This trains a **Shake-Shake ResNet-26** on CIFAR-10 for up to **100 epochs** with:
- SGD + Momentum optimiser
- Cosine annealing learning-rate schedule
- Early stopping (patience = 10 epochs)
- Data augmentation (RandomCrop + HorizontalFlip)

---

## ⚙️ Configuration

All hyperparameters are defined as a `CONFIG` dictionary at the top of each run script.

### MNIST (`run_mnist.py`)

| Parameter | Value | Description |
|:---|:---|:---|
| `epochs` | 5 | Number of training epochs |
| `batch_size` | 64 | Samples per mini-batch |
| `lr` | 0.001 | Adam learning rate |
| `seed` | 42 | Random seed for reproducibility |

### CIFAR-10 (`run_cifar.py`)

| Parameter | Value | Description |
|:---|:---|:---|
| `epochs` | 100 | Maximum training epochs |
| `batch_size` | 128 | Samples per mini-batch |
| `lr` | 0.1 | SGD initial learning rate |
| `momentum` | 0.9 | SGD momentum |
| `weight_decay` | 5e-4 | L2 regularisation strength |
| `early_stop_patience` | 10 | Epochs without improvement before stopping |

> [!TIP]
> To customise, simply edit the `CONFIG` dict in the corresponding run script. All parameters are automatically logged to WandB for experiment tracking.

---

## 🧠 Architecture Details

### MNIST: Simple CNN

```
Input (1, 28, 28) → Conv(1→32, k=3) → ReLU → Conv(32→64, k=3) → ReLU
    → MaxPool(2) → Dropout(0.25)
    → Flatten(9216) → FC(128) → ReLU → Dropout(0.5) → FC(10)
```

A minimal yet effective architecture. The spatial dimension derivation:

| Layer | Input | Kernel | Stride | Output |
|:---|:---|:---|:---|:---|
| Conv1 | 28×28 | 3 | 1 | 26×26 |
| Conv2 | 26×26 | 3 | 1 | 24×24 |
| MaxPool | 24×24 | 2 | 2 | 12×12 |
| Flatten | — | — | — | 64×12×12 = **9216** |

### CIFAR-10: Shake-Shake ResNet-26

```
Input (3, 32, 32) → Conv(3→16) → BN
    → Stage 1: 4× ShakeShakeBlock (16→32)      [32×32]
    → Stage 2: 4× ShakeShakeBlock (32→64)       [16×16]
    → Stage 3: 4× ShakeShakeBlock (64→128)      [8×8]
    → ReLU → AdaptiveAvgPool(1×1) → FC(128→10)
```

**Shake-Shake Regularisation**: Each residual block has two parallel branches. During training, their outputs are mixed with random coefficients α (forward) and β (backward), where α ≠ β. This gradient noise injection prevents co-adaptation and forces robust feature learning.

> [!IMPORTANT]
> The `ShakeShakeFunction` uses a custom `torch.autograd.Function` to deliberately use **different** mixing coefficients in the forward and backward passes. This is an advanced PyTorch technique that breaks the standard autograd chain rule for regularisation purposes.

---

## 📊 Experiment Tracking (WandB)

WandB runs in **offline mode** by default — no cloud account required. All logs are saved locally in the `wandb/` directory.

### View Offline Logs

```bash
# To sync a run to the WandB cloud later:
wandb sync wandb/offline-run-XXXXXXXX-XXXXXXXX
```

### Tracked Metrics

| Metric | Description |
|:---|:---|
| `train_loss` | Average training loss per epoch |
| `train_accuracy` | Training accuracy per epoch |
| `val_loss` | Validation loss per epoch |
| `val_accuracy` | Validation accuracy per epoch |
| `learning_rate` | Current learning rate |

---

## 💾 Checkpointing & Resume

The framework automatically saves the **best model** (by validation accuracy) to the `checkpoints/` directory.

Each checkpoint contains:
- `model_state_dict` — All model weights
- `optimizer_state_dict` — Optimiser internal state (e.g., Adam momentum buffers)
- `scheduler_state_dict` — LR scheduler progress (if applicable)
- `epoch` — Epoch number
- `val_accuracy` — Validation accuracy at save time

### Loading a Checkpoint

```python
import torch
from src.models.mnist_cnn import MnistCnn

model = MnistCnn()
checkpoint = torch.load("checkpoints/mnist/model_best.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['val_accuracy']:.4f}")
```

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|:---|:---|:---|
| **GPU** | Any CUDA-capable GPU | NVIDIA RTX 3060+ (12GB VRAM) |
| **VRAM** | 4 GB (MNIST) | 8 GB+ (CIFAR-10) |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 500 MB (datasets + checkpoints) | 1 GB |

> [!NOTE]
> MNIST can run on CPU in a few minutes. CIFAR-10 with Shake-Shake ResNet-26 is significantly more compute-intensive and strongly benefits from GPU acceleration.

---

## 🗂️ Design Patterns & Teaching Notes

### Facade Pattern (外觀模式)

The `Trainer` class in `src/trainer.py` implements the **Facade Pattern**: it wraps the complex training loop (forward/backward pass, validation, checkpointing, early stopping, logging) behind a single `trainer.fit()` method.

**Why it matters for research**: When comparing Model A vs Model B, they must share the exact same training procedure. The Facade ensures identical training conditions, making performance differences attributable solely to the model architecture.

### Custom Autograd Function

`ShakeShakeFunction` inherits from `torch.autograd.Function` to define custom forward/backward logic. This is the only way to make the backward pass use different coefficients (β) than the forward pass (α) — standard autograd would automatically derive gradients consistent with the forward computation.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Weights & Biases](https://wandb.ai/) — Experiment tracking
- [Shake-Shake Regularization](https://arxiv.org/abs/1705.07485) — Xavier Gastaldi, ICLR 2017 Workshop
- [He et al., 2015](https://arxiv.org/abs/1512.03385) — Deep Residual Learning (ResNet)