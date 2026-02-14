"""
Shared utilities for reproducibility and device management.
共用工具模組：負責實驗可重現性 (Reproducibility) 與硬體設備管理。

This module centralizes common setup routines so that every experiment
script uses the exact same seeding and hardware-detection logic —
a critical requirement for fair model comparison in research.

Design Rationale (設計理由):
    將 seed 設定和設備偵測集中在一個模組中，而非散布在各個 run script，
    可確保所有實驗的「控制變因」完全一致。這是科學實驗設計的基本要求。
"""

import logging
import random

import numpy as np
import torch

# ── Module-level logger ─────────────────────────────────────────────────
# 使用 logging 模組取代 print，讓日誌等級可控且格式統一。
# 在正式產品中，可透過 logging.config 統一管理所有模組的日誌格式。
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Fix **all** sources of randomness to guarantee reproducible results.
    鎖定所有隨機源，確保實驗結果可完整重現。

    This function covers four independent random number generators (RNGs)
    that PyTorch-based training pipelines typically rely on:

    1. ``random``        — Python 標準庫 (用於 data augmentation 等)
    2. ``numpy.random``  — NumPy RNG (用於部分 dataset / transform 實作)
    3. ``torch.manual_seed``  — PyTorch CPU RNG (權重初始化、Dropout mask 等)
    4. ``torch.cuda.manual_seed_all`` — PyTorch GPU RNG (所有 GPU 卡)

    Additionally, CuDNN is configured for deterministic mode:
    - ``cudnn.deterministic = True``  — 強制使用確定性演算法
    - ``cudnn.benchmark = False``     — 停用 auto-tuner，犧牲少量效能換取可重現性

    .. warning::
        Deterministic mode may slightly reduce training speed (~5-10%)
        because CuDNN cannot use its fastest non-deterministic kernels.

    Args:
        seed: The seed value to use across all RNGs. Default is 42
              (the answer to life, the universe, and everything).
    """
    # ── Python built-in ────────────────────────────────────────────────
    # 影響範圍：random.choice(), random.shuffle() 等
    random.seed(seed)

    # ── NumPy ──────────────────────────────────────────────────────────
    # 影響範圍：np.random.randint(), dataset shuffle 等
    np.random.seed(seed)

    # ── PyTorch CPU ────────────────────────────────────────────────────
    # 影響範圍：權重初始化 (kaiming_normal_, xavier_ 等)、Dropout mask、
    #          torch.rand() / torch.randn() 等所有 CPU 上的隨機操作
    torch.manual_seed(seed)

    # ── PyTorch CUDA (all GPUs) ────────────────────────────────────────
    if torch.cuda.is_available():
        # manual_seed_all() 同時設定所有 GPU 卡的 seed
        # (即使只有一張卡也建議用 _all 版本，避免多卡時遺漏)
        torch.cuda.manual_seed_all(seed)

        # CuDNN deterministic mode:
        # 確保同樣的輸入 → 同樣的輸出，但可能降低效能
        torch.backends.cudnn.deterministic = True
        # benchmark mode 會在第一次執行時測試不同的 kernel 選擇最快的，
        # 但這會導致每次執行時選到不同的 kernel → 結果不可重現
        torch.backends.cudnn.benchmark = False

    logger.info("Random seed set to %d (deterministic mode enabled)", seed)


def get_device() -> torch.device:
    """
    Detect and return the best available compute device.
    偵測並回傳最佳可用的計算設備。

    Priority: CUDA GPU > CPU.
    優先順序：CUDA GPU > CPU。

    When a CUDA device is available, this function also logs the GPU name
    and total memory to help diagnose hardware-related performance issues.

    Returns:
        A ``torch.device`` pointing to the selected hardware.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)

        # 顯示 GPU 顯存大小，幫助判斷 batch_size 是否合理
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU] Using: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA not available -- falling back to CPU")

    return device
