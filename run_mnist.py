"""
MNIST training entry-point.
MNIST 手寫數字分類的訓練入口腳本。

Demonstrates a basic CNN on the MNIST handwritten-digit dataset using the
shared ``Trainer`` engine.  WandB runs in **offline** mode so nothing is
synced to the cloud.

Usage:
    uv run python run_mnist.py
"""

import os

# ═══════════════════════════════════════════════════════════════════════
# WandB Offline Mode — 離線模式配置
# ═══════════════════════════════════════════════════════════════════════
# ⚠️ 此環境變數必須在 import wandb 之前設定！
# wandb 在 import 時就會讀取此變數來決定運行模式。
# 「offline」模式：所有 log 存在本機 ./wandb/ 目錄，不會同步到雲端。
# 如需上傳，事後可執行: wandb sync <run_dir>
os.environ["WANDB_MODE"] = "offline"

import torch
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

from src.models.mnist_cnn import MnistCnn
from src.trainer import Trainer
from src.utils import get_device, set_seed


# ═══════════════════════════════════════════════════════════════════════
# Experiment Configuration — 實驗超參數設定
# ═══════════════════════════════════════════════════════════════════════
# 將所有超參數集中在一個 dict 中，方便：
#   1. 一目瞭然地看到所有可調整的旋鈕
#   2. 直接傳給 wandb.init(config=...) 做實驗追蹤
#   3. 未來可擴展為從 YAML / CLI 讀取
CONFIG = {
    "epochs": 5,            # MNIST 很簡單，5 個 epoch 就能達到 ~99%
    "batch_size": 64,       # 每個 mini-batch 包含 64 張圖像
    "lr": 1e-3,             # Adam 的預設建議學習率
    "architecture": "SimpleCNN",
    "dataset": "MNIST",
    "seed": 42,             # 隨機種子，確保實驗可重現
}


def main() -> None:
    """Main training pipeline for MNIST. MNIST 訓練主流程。"""

    # ── 1. Reproducibility (可重現性) ───────────────────────────────────
    set_seed(CONFIG["seed"])

    # ── 2. WandB Initialization ─────────────────────────────────────────
    # project: WandB 中的專案名稱 (同一個 project 下的 run 可互相比較)
    # config:  自動記錄超參數，方便事後查詢和篩選
    wandb.init(project="mnist-demo", config=CONFIG)

    # ── 3. Device Detection ─────────────────────────────────────────────
    device = get_device()

    # ── 4. Data Pipeline (資料管線) ─────────────────────────────────────
    # MNIST 的前處理很簡單：轉 Tensor + 標準化
    # Normalize 的參數 (0.1307, 0.3081) 是 MNIST 訓練集的全局 mean / std
    # 標準化的目的：讓輸入分佈接近 N(0,1)，幫助模型更快收斂
    transform = transforms.Compose(
        [
            transforms.ToTensor(),                        # PIL → Tensor, 範圍 [0, 1]
            transforms.Normalize((0.1307,), (0.3081,)),   # 標準化 → 近似 N(0, 1)
        ]
    )

    dataset_train = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    dataset_val = datasets.MNIST("./data", train=False, transform=transform)

    # DataLoader 參數說明:
    #   - batch_size: 每個 mini-batch 的樣本數
    #   - shuffle=True: 每個 epoch 打亂訓練資料順序 (防止模型學到順序偏差)
    #   - num_workers: 使用多個子進程做資料預處理 (I/O pipeline 並行化)
    #   - pin_memory=True: 將資料鉅陣存放在 page-locked (pinned) 記憶體中，
    #     accelerates Host→GPU data transfer (DMA can be used directly)
    #     加速 CPU→GPU 的資料搬運 (可直接使用 DMA 傳輸)
    use_cuda = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1000,
        num_workers=2,
        pin_memory=use_cuda,
    )

    # ── 5. Model / Optimiser / Loss ─────────────────────────────────────
    model = MnistCnn()

    # Adam optimiser:
    #   - 自適應學習率 (每個參數有獨立的學習率)
    #   - 結合 Momentum (一階矩) 和 RMSProp (二階矩)
    #   - 對於簡單任務和小模型，Adam 通常是最安全的選擇
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # CrossEntropyLoss = Softmax + NLLLoss
    # 它在內部自動對 raw logits 做 softmax，所以模型最後一層不需要加 softmax
    criterion = torch.nn.CrossEntropyLoss()

    # ── 6. Training ─────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir="checkpoints/mnist",
    )

    # wandb.watch(): 自動追蹤模型的梯度和參數分佈 (可在 WandB dashboard 查看)
    wandb.watch(model)

    # fit() 一鍵啟動完整訓練流程
    trainer.fit(epochs=CONFIG["epochs"], experiment_tag="MNIST")

    # 結束 WandB run，確保所有 log 被寫入磁碟
    wandb.finish()


if __name__ == "__main__":
    main()
