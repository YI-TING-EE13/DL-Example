"""Shake-Shake classifier retained from the original example, not a paper-reproduction claim."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShakeShakeFunction(torch.autograd.Function):
    """Use independent forward and surrogate backward mixing coefficients."""

    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        ctx.save_for_backward(alpha, beta)
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        alpha, beta = ctx.saved_tensors
        grad_x1 = beta * grad_output
        grad_x2 = (1 - beta) * grad_output
        return (grad_x1, grad_x2, None, None)


class ShakeShakeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.equal_io = in_channels == out_channels and stride == 1
        self.branch1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if not self.equal_io:
            self.shortcut = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        if self.training:
            alpha = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            beta = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            mixed = ShakeShakeFunction.apply(out1, out2, alpha, beta)
        else:
            mixed = 0.5 * out1 + 0.5 * out2
        if not self.equal_io:
            x = self.shortcut(x)
        return x + mixed


class ShakeShakeResNet(nn.Module):
    def __init__(self, depth: int = 26, w_base: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        if depth < 8 or (depth - 2) % 6 != 0 or w_base <= 0:
            raise ValueError("depth must be 6n+2 (n >= 1), and w_base must be positive")
        n_units = (depth - 2) // 6
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(w_base, n_units, stride=1)
        self.stage2 = self._make_stage(w_base * 2, n_units, stride=2)
        self.stage3 = self._make_stage(w_base * 4, n_units, stride=2)
        self.fc = nn.Linear(w_base * 4, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, out_channels: int, n_units: int, stride: int) -> nn.Sequential:
        layers = [ShakeShakeBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(n_units - 1):
            layers.append(ShakeShakeBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(x))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
