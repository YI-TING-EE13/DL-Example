"""
Generic training engine — Facade Pattern (外觀模式).
通用訓練引擎：將訓練迴圈的所有細節封裝成一個統一介面。

This module decouples the training / validation loop from the model
architecture and dataset details.  Both the MNIST and CIFAR-10 experiments
share this exact same ``Trainer``, which guarantees that any performance
difference is attributable solely to the model — not to the training
procedure.

Design Pattern — Facade (外觀模式):
    在軟體工程中，Facade 模式的目標是「為一組複雜的子系統提供一個統一的高階介面」。
    這裡的子系統包括：
        1. Training loop (前向/反向/更新)
        2. Validation loop (推論/評估)
        3. Checkpoint management (模型存檔)
        4. Early stopping (提前停止)
        5. WandB logging (實驗追蹤)
        6. LR scheduler stepping (學習率排程)
    使用者只需呼叫 ``trainer.fit()`` 就能啟動一切。

    這對研究尤其重要：當你要比較 Model A 和 Model B 時，共用同一個 Trainer
    可以保證「控制變因」——訓練流程、評估標準、日誌格式完全一致，
    實驗結果才具備科學上的公平性和可重現性。

Enhanced features beyond the basic loop:
    - Automatic best-model checkpointing (自動儲存最佳模型)
    - Early stopping with configurable patience (可設定耐心值的提前停止)
    - Full ``fit()`` method that orchestrates everything (一鍵啟動訓練)
"""

from __future__ import annotations

from pathlib import Path

import torch
import wandb
from tqdm import tqdm


class Trainer:
    """
    Standard training-loop wrapper.
    標準訓練迴圈封裝器。

    This class manages the entire lifecycle of a training experiment:
    forward pass → loss computation → backward pass → weight update →
    validation → checkpointing → early stopping.

    Args:
        model:        The ``nn.Module`` to train.
                      要訓練的模型，必須繼承自 ``nn.Module``。
        train_loader: DataLoader for the training split.
                      訓練資料的 DataLoader。
        val_loader:   DataLoader for the validation / test split.
                      驗證 / 測試資料的 DataLoader。
        optimizer:    Any ``torch.optim.Optimizer`` (e.g. Adam, SGD).
                      任何 PyTorch 優化器。
        criterion:    Loss function (e.g. ``CrossEntropyLoss``).
                      損失函數。
        device:       Target device (``cuda`` or ``cpu``).
                      目標計算設備。
        checkpoint_dir: Directory for saving model checkpoints.
                        模型 checkpoint 儲存目錄。
        early_stop_patience: Number of epochs without improvement
                             before stopping.  Set to 0 to disable.
                             連續多少個 epoch 沒有改善就停止訓練。設為 0 則關閉。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        early_stop_patience: int = 0,
    ) -> None:
        # 將模型移至指定設備 (GPU or CPU)
        # .to(device) 會遞迴地將模型中的所有 parameters 和 buffers 搬到目標設備
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # ── Checkpoint management ───────────────────────────────────────
        # 使用 pathlib.Path 而非 os.path，這是 Python 3 的現代慣例
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── Early-stopping state ────────────────────────────────────────
        # patience: 允許連續幾個 epoch 沒改善 (0 = 關閉 early stopping)
        # best_acc: 歷史最佳驗證準確率，用於判斷是否改善
        # _epochs_no_improve: 連續沒改善的 epoch 計數器
        self.patience = early_stop_patience
        self.best_acc = 0.0
        self._epochs_no_improve = 0

    # ==================================================================
    # Core Loops — 核心訓練 / 驗證迴圈
    # ==================================================================

    def train_epoch(self) -> tuple[float, float]:
        """
        Execute one full epoch of training.
        執行一個完整的訓練 epoch。

        The standard training step consists of five phases:
        標準訓練步驟由五個階段組成：

        1. ``optimizer.zero_grad()``  — 清除上一步的梯度
           (PyTorch 梯度是「累加」的，不清除會導致梯度越來越大)
        2. ``model(x)``              — 前向傳播 (Forward Pass)
        3. ``criterion(output, y)``  — 計算損失值
        4. ``loss.backward()``       — 反向傳播 (Backward Pass)，計算所有參數的梯度
        5. ``optimizer.step()``      — 根據梯度更新權重

        Returns:
            A tuple of ``(average_loss, accuracy)`` over the full epoch.
            回傳整個 epoch 的 ``(平均 loss, 準確率)``。
        """
        # model.train() 啟用訓練模式：
        #   - Dropout 層會隨機丟棄神經元
        #   - BatchNorm 層會計算並追蹤 running statistics
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for x, y in tqdm(self.train_loader, desc="  Training", leave=False):
            # 將資料搬到與模型相同的設備上 (GPU or CPU)
            x, y = x.to(self.device), y.to(self.device)

            # --- 標準五步優化 (Standard 5-Step Optimisation) ---
            self.optimizer.zero_grad()           # 1. 清除梯度
            output = self.model(x)               # 2. 前向傳播
            loss = self.criterion(output, y)     # 3. 計算 loss
            loss.backward()                      # 4. 反向傳播 (計算 ∂L/∂w)
            self.optimizer.step()                # 5. 更新權重 (w ← w - lr × ∂L/∂w)

            # 累加 batch 統計量
            total_loss += loss.item()
            pred = output.argmax(dim=1)          # 取最大 logit 的 index 作為預測類別
            correct += pred.eq(y).sum().item()   # 統計預測正確的數量
            total_samples += y.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total_samples
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """
        Evaluate the model on the validation set.
        在驗證集上評估模型表現。

        Key differences from training:
        與訓練的關鍵差異：

        - ``model.eval()``: Switches **off** Dropout and fixes BatchNorm
          statistics to use running mean/var instead of batch statistics.
          關閉 Dropout 並讓 BatchNorm 使用累積統計量。
        - ``@torch.no_grad()``: Disables gradient computation entirely,
          saving ~50% GPU memory and accelerating inference.
          完全停止梯度計算，節省約 50% GPU 記憶體並加速推論。

        Returns:
            A tuple of ``(average_loss, accuracy)``.
            回傳 ``(平均 loss, 準確率)``。
        """
        self.model.eval()
        correct = 0
        total_loss = 0.0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            total_loss += loss.item()

            # argmax(dim=1): 在 class dimension 上取最大值的索引
            # 例如 output shape = (N, 10)，取 dim=1 上最大的那個類別
            pred = output.argmax(dim=1)
            correct += pred.eq(y).sum().item()

        accuracy = correct / len(self.val_loader.dataset)
        return total_loss / len(self.val_loader), accuracy

    # ==================================================================
    # Checkpoint Helpers — 模型存檔管理
    # ==================================================================

    def _save_checkpoint(
        self,
        epoch: int,
        val_acc: float,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        tag: str = "best",
    ) -> None:
        """
        Save a model checkpoint to disk.
        將模型 checkpoint 儲存到磁碟。

        A complete checkpoint includes:
        一個完整的 checkpoint 包含：
            - model_state_dict:     所有模型權重
            - optimizer_state_dict: 優化器狀態 (e.g. Adam 的 momentum buffers)
            - scheduler_state_dict: 學習率排程器狀態 (可選)
            - epoch:                當前 epoch 編號
            - val_accuracy:         當前驗證準確率

        Why save optimizer & scheduler state?
        為什麼要存優化器和排程器狀態？
            - 如果訓練中斷需要恢復 (resume)，僅載入模型權重是不夠的。
            - Adam 等優化器有內部的 momentum / adaptive learning rate 累積值。
            - CosineAnnealingLR 等排程器需要知道當前的 step 進度。
            - 不存這些狀態，恢復訓練後 LR 會從頭開始、momentum 會歸零。

        Args:
            epoch:     Current epoch number.
            val_acc:   Current validation accuracy.
            scheduler: Optional LR scheduler to save state from.
            tag:       Filename tag (e.g. ``"best"``, ``"last"``).
        """
        path = self.checkpoint_dir / f"model_{tag}.pt"
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": val_acc,
        }
        # 只在有 scheduler 時才存其狀態，保持向後兼容
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(payload, path)
        print(f"  [SAVE] Checkpoint saved -> {path}")

    # ==================================================================
    # High-level fit — 完整訓練流程
    # ==================================================================

    def fit(
        self,
        epochs: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        experiment_tag: str = "experiment",
    ) -> None:
        """
        Full training loop with validation, checkpointing, and optional
        early stopping.
        完整訓練迴圈：包含驗證、模型存檔、可選的提前停止。

        This is the main entry point for running an experiment. It
        orchestrates the following per-epoch workflow:
        這是啟動實驗的主要入口，每個 epoch 的流程如下：

        1. ``train_epoch()``       → 訓練一個 epoch
        2. ``validate()``          → 在驗證集上評估
        3. ``scheduler.step()``    → 更新學習率 (如果有 scheduler)
        4. ``wandb.log()``         → 記錄 epoch 級別的 metrics
        5. Checkpoint best model   → 儲存歷史最佳模型
        6. Check early stopping    → 判斷是否提前停止

        Args:
            epochs:         Total number of epochs to run.
                            總共要訓練幾個 epoch。
            scheduler:      Optional learning-rate scheduler (stepped
                            once per epoch **after** validation).
                            可選的學習率排程器，每個 epoch 結束後呼叫一次。
            experiment_tag: Prefix printed in console logs (e.g. ``MNIST``
                            or ``CIFAR-10``).
                            Console log 的前綴標籤。
        """
        print(f"\n{'=' * 60}")
        print(f"  Starting training: {experiment_tag}")
        print(f"  Epochs: {epochs}  |  Device: {self.device}")
        if self.patience > 0:
            print(f"  Early stopping patience: {self.patience}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, epochs + 1):
            # --- 1. Train ---
            train_loss, train_acc = self.train_epoch()

            # --- 2. Validate ---
            val_loss, val_acc = self.validate()

            # --- 3. Scheduler step ---
            # 讀取當前 LR (在 scheduler step 之前讀取，因為這個 epoch 是用這個 LR 訓練的)
            # scheduler.step() 會更新下一個 epoch 的 LR
            curr_lr = self.optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            # --- 4. Console output ---
            print(
                f"[{experiment_tag}] Epoch {epoch}/{epochs}: "
                f"Train Loss={train_loss:.4f}  |  "
                f"Train Acc={train_acc:.4f}  |  "
                f"Val Loss={val_loss:.4f}  |  "
                f"Val Acc={val_acc:.4f}  |  "
                f"LR={curr_lr:.6f}"
            )

            # --- 5. WandB epoch-level logging ---
            # 使用 step=epoch 確保 WandB 圖表的 x 軸是 epoch 編號，
            # 避免 batch-level 和 epoch-level metrics 的 step 混亂問題
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": curr_lr,
                },
                step=epoch,
            )

            # --- 6. Checkpointing: save best model ---
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint(epoch, val_acc, scheduler=scheduler, tag="best")
                self._epochs_no_improve = 0
            else:
                self._epochs_no_improve += 1

            # --- 7. Early stopping ---
            if self.patience > 0 and self._epochs_no_improve >= self.patience:
                print(
                    f"\n[STOP] Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        print(f"\n[DONE] Training complete -- Best Val Accuracy: {self.best_acc:.4f}\n")
