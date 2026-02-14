"""
Shake-Shake ResNet-26 for CIFAR-10 classification.
CIFAR-10 分類用的 Shake-Shake ResNet-26 模型。

Key Idea (核心概念)
--------------------
Shake-Shake regularisation introduces **stochastic** forward/backward
coefficients so that the two residual branches receive *different* gradient
signals than the mixture used in the forward pass.  This noise injection
acts as a powerful regulariser that forces learning of robust, redundant
features.

一般的神經網路，Forward 和 Backward 的數學邏輯必須一致 (導數鏈式法則)。
但 Shake-Shake 故意打破這個規則：
    - Forward:  y = α · x₁ + (1 − α) · x₂   (α 是隨機的)
    - Backward: ∂L/∂x₁ = β · ∂L/∂y           (β 是另一個獨立的隨機數)
這相當於在梯度的流動中注入雜訊 (Gradient Noise Injection)，
強迫網路不能依賴單一路徑的特定特徵，必須學會更強健的特徵表示。

Reference:
    Xavier Gastaldi, "Shake-Shake regularization", ICLR 2017 Workshop.

Architecture (架構圖)
---------------------
Input (N, 3, 32, 32)                            ← RGB 彩色圖像
  │
  ├─ Conv2d(3→16, k=3) + BN                     ← 初始特徵提取
  │
  ├─ Stage 1: ShakeShakeBlock × 4 (16→w_base)    ← 32×32, 空間不變
  ├─ Stage 2: ShakeShakeBlock × 4 (w_base→2w)    ← 32→16, stride=2 降解析度
  ├─ Stage 3: ShakeShakeBlock × 4 (2w→4w)        ← 16→8,  stride=2 降解析度
  │
  ├─ ReLU → AdaptiveAvgPool(1×1)                 ← 全局平均池化
  └─ Linear(4w → 10)                             ← 分類輸出

Input : (N, 3, 32, 32)  RGB images
Output: (N, 10)          class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Custom Autograd Function — 自定義反向傳播邏輯
# ======================================================================

class ShakeShakeFunction(torch.autograd.Function):
    """
    Implements the Shake-Shake forward/backward mismatch.
    實作 Shake-Shake 的前向/反向不一致機制。

    Mathematical formulation (數學公式):
        Forward:  y = α · x₁ + (1 − α) · x₂
        Backward: ∂L/∂x₁ = β · ∂L/∂y ,  ∂L/∂x₂ = (1 − β) · ∂L/∂y

    The deliberate α ≠ β discrepancy injects noise into the gradient
    updates, preventing co-adaptation of the two residual branches.

    Why use ``torch.autograd.Function`` instead of normal ``nn.Module``?
    為什麼要用 ``torch.autograd.Function`` 而不是普通的 ``nn.Module``？

        一般的 PyTorch module，backward 的梯度計算是由 autograd 引擎根據
        forward 的計算圖自動推導的。但 Shake-Shake 需要在 backward 中「欺騙」
        autograd，使用不同於 forward 的系數 (β≠α)。
        這只能透過自定義的 autograd.Function 來實現。

    Memory Note (記憶體注意事項):
        We only save ``alpha`` and ``beta`` for the backward pass (not
        ``x1`` and ``x2``), because the backward computation only needs
        the mixing coefficients to scale ``grad_output``.
        只儲存 α 和 β (不儲存 x₁、x₂)，因為 backward 只需要混合係數。
    """

    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        # ctx (context object): PyTorch 提供的特殊物件，用於在
        # forward 和 backward 之間傳遞資訊
        # 只存混合係數，不存 x1/x2，節省 GPU 記憶體
        ctx.save_for_backward(alpha, beta)

        # 加權平均: y = α·x₁ + (1-α)·x₂
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        # 取回 forward 中儲存的張量
        alpha, beta = ctx.saved_tensors

        # 用 β (而非 α) 來分配梯度 → 梯度雜訊注入
        # 這使得 branch1 和 branch2 收到與 forward 不同比例的梯度信號
        grad_x1 = beta * grad_output
        grad_x2 = (1 - beta) * grad_output

        # 回傳 4 個梯度，對應 forward() 的 4 個輸入
        # alpha, beta 是超參數，不需要梯度 → 回傳 None
        return grad_x1, grad_x2, None, None


# ======================================================================
# Shake-Shake Residual Block — Shake-Shake 殘差塊
# ======================================================================

class ShakeShakeBlock(nn.Module):
    """
    A single residual block with two parallel branches.
    具有兩條平行路徑的殘差塊。

    Structure (結構):
                         ┌─── Branch 1 ───┐
        Input ─┬─────── │ ReLU→Conv→BN→   │ ─┐
               │        │ ReLU→Conv→BN    │  │
               │        └─────────────────┘  ├─ ShakeShake Mix ─→ + ─→ Output
               │        ┌─── Branch 2 ───┐  │
               │        │ ReLU→Conv→BN→   │ ─┘
               │        │ ReLU→Conv→BN    │
               │        └─────────────────┘
               │
               └──────── Shortcut (Skip) ──────────────────────→ +

    During training: branches are mixed with random α (forward) and β (backward).
    訓練時：兩條路徑使用隨機 α (前向) 和 β (反向) 混合。

    During evaluation: deterministic average (E[α] = 0.5).
    推論時：使用確定性的平均值 (0.5·x₁ + 0.5·x₂)。

    Args:
        in_channels:  Number of input feature-map channels.
        out_channels: Number of output feature-map channels.
        stride:       Spatial down-sampling stride (1 = keep size).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # 判斷是否需要 shortcut projection
        # 當 input/output channels 相同且 stride=1 時，可直接用 identity shortcut
        self.equal_io = (in_channels == out_channels) and (stride == 1)

        # ── Branch 1 — Residual Path A ──────────────────────────────────
        # Pre-activation style: ReLU 在 Conv 之前 (He et al., 2016)
        # bias=False: 因為後面接 BatchNorm，BN 已經有 bias 參數，
        #             Conv 再加 bias 是冗餘的
        self.branch1 = nn.Sequential(
            nn.ReLU(inplace=False),  # inplace=False: 保護原始 input 不被修改
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ── Branch 2 — Residual Path B ──────────────────────────────────
        # 與 Branch 1 結構完全相同，但權重是獨立隨機初始化的
        # 這兩條路徑會各自學到不同的特徵表示
        self.branch2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ── Shortcut (Skip Connection / 跳躍連接) ──────────────────────
        # 當維度改變 (channels 或 spatial size) 時，需要 1×1 Conv + BN
        # 來對齊 shortcut 和 residual 的維度，才能相加
        self.shortcut = nn.Sequential()  # identity by default
        if not self.equal_io:
            self.shortcut = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Shake-Shake mixing.

        Args:
            x: Input tensor of shape ``(N, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(N, C_out, H', W')``.
        """
        out1 = self.branch1(x)
        out2 = self.branch2(x)

        if self.training:
            # ── 訓練模式：隨機混合 ──────────────────────────────────────
            # 為每個 sample 獨立產生隨機係數 (per-sample stochasticity)
            # shape (N, 1, 1, 1) 可透過 broadcasting 與 (N, C, H, W) 相乘
            alpha = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            beta = torch.rand(x.size(0), 1, 1, 1, device=x.device)

            # 使用自定義 autograd function 進行前向/反向不一致的混合
            mixed = ShakeShakeFunction.apply(out1, out2, alpha, beta)
        else:
            # ── 推論模式：確定性平均 ────────────────────────────────────
            # E[α] = 0.5 (Uniform(0,1) 的期望值)
            mixed = 0.5 * out1 + 0.5 * out2

        # Shortcut: 如果維度不匹配，透過 1×1 Conv 對齊
        if not self.equal_io:
            x = self.shortcut(x)

        # Residual connection: output = shortcut + mixed_branches
        return x + mixed


# ======================================================================
# Full Network — 完整的 Shake-Shake ResNet
# ======================================================================

class ShakeShakeResNet(nn.Module):
    """
    ResNet-26 with Shake-Shake regularisation blocks.
    使用 Shake-Shake 正則化的 ResNet-26 模型。

    Structure: Input → Conv → BN → Stage1 → Stage2 → Stage3
               → ReLU → AdaptiveAvgPool → FC

    Feature Map Dimensions (特徵圖尺寸變化):
        Stage 1: (N,  w,  32, 32)  — stride=1, spatial 不變
        Stage 2: (N, 2w,  16, 16)  — stride=2, spatial 減半
        Stage 3: (N, 4w,   8,  8)  — stride=2, spatial 再減半
        AvgPool: (N, 4w,   1,  1)  — 全局平均池化

    Args:
        depth:       Total network depth (default 26 → 4 blocks per stage).
                     網路總深度 (預設 26，意味著每個 stage 有 4 個 block)。
                     計算方式: n_blocks_per_stage = (depth - 2) / 6
        w_base:      Base width (channels in Stage 1). Default 32.
                     基礎寬度 (Stage 1 的 channel 數)。
        num_classes: Number of output categories.
                     輸出類別數。
    """

    def __init__(
        self,
        depth: int = 26,
        w_base: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        # blocks per stage: (26 - 2) / 6 = 4
        # 每個 block 有 2 個 Conv 層 (branch1 + branch2 各一組但共享同一個 input)
        # 再加上初始的 Conv1，總共 depth = 2 + 3 stages × 4 blocks × 2 layers = 26
        n_units = (depth - 2) // 6

        # ── Initial convolution ─────────────────────────────────────────
        # 從 RGB 3 channels 提取 16 個初始特徵
        # bias=False 因為後面接 BatchNorm
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # ── Three stages with increasing channel width ──────────────────
        # 每個 stage 的第一個 block 負責降解析度 (stride>1) 和增加通道數
        # 後續的 block 則保持維度不變，專注於特徵精煉
        self.stage1 = self._make_stage(w_base, n_units, stride=1)       #  32×32
        self.stage2 = self._make_stage(w_base * 2, n_units, stride=2)   #  16×16
        self.stage3 = self._make_stage(w_base * 4, n_units, stride=2)   #   8×8

        # ── Final classifier ────────────────────────────────────────────
        self.fc = nn.Linear(w_base * 4, num_classes)

        # ── Weight initialisation (權重初始化) ──────────────────────────
        # Kaiming He init (He et al., 2015):
        # 針對 ReLU 激活函數設計的初始化方法，確保各層的方差穩定，
        # 避免訓練初期的梯度爆炸或消失
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # fan_out mode: 根據輸出通道數計算方差
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # BN 的 weight (γ) 初始化為 1, bias (β) 初始化為 0
                # 這讓 BN 在訓練初期接近 identity function
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(
        self, out_channels: int, n_units: int, stride: int
    ) -> nn.Sequential:
        """
        Build one stage consisting of ``n_units`` ShakeShakeBlocks.
        建立一個 stage，包含 ``n_units`` 個 ShakeShakeBlock。

        第一個 block 使用指定的 stride (可能 > 1 用於降解析度)，
        後續 block 的 stride 全部為 1 (維持空間尺寸)。
        """
        layers = [ShakeShakeBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(n_units - 1):
            layers.append(ShakeShakeBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full network.

        Args:
            x: Input tensor of shape ``(N, 3, 32, 32)``.

        Returns:
            Raw logits of shape ``(N, num_classes)``.
        """
        out = self.bn1(self.conv1(x))             # (N, 3, 32, 32) → (N, 16, 32, 32)
        out = self.stage1(out)                     # (N, 16, 32, 32) → (N, w, 32, 32)
        out = self.stage2(out)                     # → (N, 2w, 16, 16)
        out = self.stage3(out)                     # → (N, 4w, 8, 8)
        out = F.relu(out)                          # Final activation
        out = F.adaptive_avg_pool2d(out, (1, 1))   # → (N, 4w, 1, 1) 全局平均池化
        out = out.view(out.size(0), -1)            # → (N, 4w) Flatten
        out = self.fc(out)                         # → (N, num_classes)
        return out
