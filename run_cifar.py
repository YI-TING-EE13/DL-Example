"""
CIFAR-10 training entry-point.
CIFAR-10 彩色圖像分類的訓練入口腳本。

Uses a Shake-Shake ResNet-26 model with:
    - SGD + Momentum optimiser (ResNet 的標準選擇)
    - Cosine-annealing learning-rate schedule (平滑學習率衰減)
    - Standard CIFAR-10 data augmentation (Random Crop + Horizontal Flip)
    - Early stopping (patience = 10 epochs)

WandB runs in **offline** mode so nothing is synced to the cloud.

Usage:
    uv run python run_cifar.py
"""

import os

# ═══════════════════════════════════════════════════════════════════════
# WandB Offline Mode — 離線模式配置
# ═══════════════════════════════════════════════════════════════════════
# ⚠️ 必須在 import wandb 之前設定！
os.environ["WANDB_MODE"] = "offline"

import torch
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

from src.models.cifar_shake import ShakeShakeResNet
from src.trainer import Trainer
from src.utils import get_device, set_seed


# ═══════════════════════════════════════════════════════════════════════
# Experiment Configuration — 實驗超參數設定
# ═══════════════════════════════════════════════════════════════════════
CONFIG = {
    "epochs": 100,              # 深層模型需要更多 epoch 才能收斂
    "batch_size": 128,          # CIFAR-10 圖像較小(32×32)，可用較大 batch
    "lr": 0.1,                  # SGD 通常需要比 Adam 更高的初始學習率
    "momentum": 0.9,            # SGD momentum: 加速收斂、幫助跨越局部極小值
    "weight_decay": 5e-4,       # L2 正則化: 防止權重過大，λ=5e-4 是 ResNet 的標準值
    "architecture": "ShakeShake-ResNet26",
    "dataset": "CIFAR-10",
    "seed": 42,
    "early_stop_patience": 10,  # 連續 10 個 epoch 沒改善就停止
}


def main() -> None:
    """Main training pipeline for CIFAR-10. CIFAR-10 訓練主流程。"""

    # ── 1. Reproducibility ──────────────────────────────────────────────
    set_seed(CONFIG["seed"])

    # ── 2. WandB Initialization ─────────────────────────────────────────
    wandb.init(project="cifar10-demo", config=CONFIG)

    # ── 3. Device Detection ─────────────────────────────────────────────
    device = get_device()

    # ── 4. Data Pipeline with Augmentation (資料管線 + 資料增強) ────────
    # CIFAR-10 vs MNIST 的關鍵差異：
    #   - 彩色 (RGB 3 channels) vs 灰階 (1 channel)
    #   - 自然場景圖像 vs 簡單手寫數字
    #   - 更高的類間相似度 (e.g. 汽車 vs 卡車) → 更容易過擬合
    # 因此，Data Augmentation 在 CIFAR-10 上至關重要！
    transform_train = transforms.Compose(
        [
            # RandomCrop: 先在四周各 pad 4 pixels (pad → 40×40)，再隨機裁切回 32×32
            # 效果：模擬圖像的小幅平移，增加位移不變性
            transforms.RandomCrop(32, padding=4),

            # RandomHorizontalFlip: 50% 機率水平翻轉
            # 效果：增加鏡像不變性 (大多數自然場景左右翻轉後語義不變)
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            # Normalize: CIFAR-10 訓練集的三通道 (R, G, B) 統計量
            # mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    # 測試集：不做 augmentation，只做 normalize (確保評估的確定性)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    dataset_train = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    dataset_val = datasets.CIFAR10(
        "./data", train=False, transform=transform_test
    )

    use_cuda = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,   # GPU 訓練時加速 Host→Device 資料傳輸
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1000,
        num_workers=2,
        pin_memory=use_cuda,
    )

    # ── 5. Model / Optimiser / Scheduler / Loss ─────────────────────────
    model = ShakeShakeResNet()

    # SGD + Momentum vs Adam:
    #   - 在 ResNet / deeper models 上，SGD + Momentum 通常能達到更好的泛化效能
    #   - Adam 雖然收斂更快，但最終 test accuracy 往往略低於 SGD
    #   - weight_decay (L2 regularisation): ||w||² 的懲罰項，防止權重過大
    optimizer = optim.SGD(
        model.parameters(),
        lr=CONFIG["lr"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"],
    )

    # CosineAnnealingLR (餘弦退火學習率排程):
    #   LR(t) = η_min + 0.5 × (η_max - η_min) × (1 + cos(π × t / T_max))
    #
    #   - 從初始 LR 平滑地衰減到接近 0
    #   - 比 StepLR (階梯式衰減) 更平穩，是目前 SOTA 模型的標準選擇
    #   - T_max = total epochs (一個完整的餘弦半周期)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )

    criterion = torch.nn.CrossEntropyLoss()

    # ── 6. Training ─────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir="checkpoints/cifar10",
        early_stop_patience=CONFIG["early_stop_patience"],
    )
    wandb.watch(model)

    trainer.fit(
        epochs=CONFIG["epochs"],
        scheduler=scheduler,
        experiment_tag="CIFAR-10",
    )

    wandb.finish()


if __name__ == "__main__":
    main()
