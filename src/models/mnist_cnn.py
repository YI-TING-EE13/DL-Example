"""
Simple CNN for MNIST handwritten-digit classification.
MNIST 手寫數字分類用的簡單卷積神經網路。

Architecture (架構圖)
---------------------
Input (N, 1, 28, 28)                   ← 灰階圖像, 28×28 pixels
  │
  ├─ Conv2d(1→32, k=3, s=1, p=0)      ← 輸出: (N, 32, 26, 26)
  ├─ ReLU                               ← 非線性激活
  ├─ Conv2d(32→64, k=3, s=1, p=0)     ← 輸出: (N, 64, 24, 24)
  ├─ ReLU
  ├─ MaxPool2d(k=2, s=2)              ← 輸出: (N, 64, 12, 12)  ← 降低解析度
  ├─ Dropout(p=0.25)                   ← 訓練時隨機丟棄 25% 神經元
  │
  ├─ Flatten                           ← (N, 64×12×12) = (N, 9216)
  ├─ Linear(9216→128) + ReLU
  ├─ Dropout(p=0.5)                    ← FC 層之間的 Dropout (防止過擬合)
  └─ Linear(128→10)                    ← 輸出: (N, 10) 類別 logits

Spatial Dimension Derivation (空間維度推導)
------------------------------------------
卷積輸出尺寸公式:  out = floor((in + 2×padding - kernel) / stride) + 1

  Layer       | Input Size | Kernel | Stride | Padding | Output Size
  ------------|------------|--------|--------|---------|------------
  Conv1       |   28×28    |   3    |   1    |    0    |   26×26
  Conv2       |   26×26    |   3    |   1    |    0    |   24×24
  MaxPool2d   |   24×24    |   2    |   2    |    0    |   12×12

  Flatten: 64 channels × 12 × 12 = 9216 features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCnn(nn.Module):
    """
    A lightweight CNN designed for the MNIST dataset (28×28 grayscale).
    專為 MNIST 資料集設計的輕量級卷積神經網路。

    Design notes (設計說明):
        - 僅使用兩個 Conv 層即可在 MNIST 上達到 ~99% 準確率。
        - MNIST 是相對簡單的任務，過度複雜的架構反而可能過擬合。
        - ``kernel_size=3`` 是最常用的卷積核大小，3×3 的感受野 (receptive field)
          能有效捕捉局部特徵 (邊緣、筆畫轉角等)。

    Args:
        num_classes: Number of output categories (default 10 for digits 0-9).
                     輸出類別數 (預設 10，對應數字 0~9)。
    """

    def __init__(self, num_classes: int = 10) -> None:
        # super().__init__() 呼叫 nn.Module 的初始化方法。
        # nn.Module.__init__() 會建立管理 parameters 的內部字典。
        # 如果不呼叫，model.parameters() 會是空的 → 無法訓練。
        super().__init__()

        # ═══════════════════════════════════════════════════════════════
        # Feature Extractor (特徵提取器)
        # ═══════════════════════════════════════════════════════════════
        # Conv2d(in_channels, out_channels, kernel_size, stride)
        #   - in_channels=1:  MNIST 是灰階圖像 (RGB 的話是 3)
        #   - out_channels=32: 學習 32 種不同的濾波器 (filter)
        #   - kernel_size=3:  每個 filter 的大小是 3×3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        # Dropout: 訓練時隨機將 25% 的 feature map 元素設為 0
        # 目的：防止 feature extractor 過度依賴某些特定的特徵通道
        self.dropout1 = nn.Dropout(0.25)

        # ═══════════════════════════════════════════════════════════════
        # Classifier Head (分類頭)
        # ═══════════════════════════════════════════════════════════════
        # 9216 = 64 channels × 12×12 spatial size
        # (推導過程見 module docstring 的 Spatial Dimension Derivation 表格)
        self.fc1 = nn.Linear(9216, 128)

        # FC 層之間的 Dropout：隨機丟棄 50% 神經元
        # 比 feature extractor 的 0.25 更激進，因為 FC 層的參數量更大，更容易過擬合
        self.dropout2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        前向傳播。

        Args:
            x: Input tensor of shape ``(N, 1, 28, 28)``.

        Returns:
            Raw logits of shape ``(N, num_classes)``.
            原始 logits (未經 softmax)。CrossEntropyLoss 會在內部自動做 softmax。
        """
        # ── Feature Extraction ──────────────────────────────────────────
        x = F.relu(self.conv1(x))              # (N, 1, 28, 28) → (N, 32, 26, 26)
        x = F.relu(self.conv2(x))              # (N, 32, 26, 26) → (N, 64, 24, 24)
        x = F.max_pool2d(x, 2)                 # (N, 64, 24, 24) → (N, 64, 12, 12)
        x = self.dropout1(x)                   # Dropout (p=0.25)

        # ── Classifier ─────────────────────────────────────────────────
        x = torch.flatten(x, 1)                # (N, 64, 12, 12) → (N, 9216)
        x = F.relu(self.fc1(x))                # (N, 9216) → (N, 128)
        x = self.dropout2(x)                   # Dropout (p=0.5)
        x = self.fc2(x)                        # (N, 128) → (N, 10)
        return x
