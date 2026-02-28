# time_delay.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence
import torch


@dataclass
class DelayConfig:
    # 대표 시간 간격 Δ (days)
    delta: float
    # lag list (e.g., [0,1,2])
    lags: List[int]

    def tau_by_lag(self) -> Dict[int, float]:
        """
        lag k -> τ_k = k * Δ
        lag 0 -> τ_0 = 0
        """
        return {int(k): float(k) * float(self.delta) for k in self.lags}

    def tau_max(self) -> float:
        return max(self.tau_by_lag().values())


def choose_delta_from_samples(t_list: Sequence[torch.Tensor], mode: str = "median") -> float:
    """
    Choose a global Δ from multiple samples with irregular observation times.
    Common choice: median of all adjacent time gaps across train samples.

    t_list: list of (N,) ascending tensors (in days).
    """
    gaps = []
    for t in t_list:
        if t.numel() < 2:
            continue
        dt = (t[1:] - t[:-1]).detach().cpu()
        gaps.append(dt)
    if not gaps:
        raise ValueError("Cannot choose delta: not enough time points.")
    gaps = torch.cat(gaps)
    if mode == "median":
        return float(gaps.median().item())
    elif mode == "mean":
        return float(gaps.mean().item())
    else:
        raise ValueError(f"Unknown mode: {mode}")
