# metrics.py
from __future__ import annotations
import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.sqrt(((y_true - y_pred) ** 2).mean()).item())


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true_mean = y_true.mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true_mean) ** 2).sum() + 1e-12
    return float((1.0 - ss_res / ss_tot).item())


def per_gene_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    y_true, y_pred: (N,G)
    returns dict with per-gene rmse/r2 arrays (G,)
    """
    resid2 = (y_true - y_pred) ** 2
    rmse_g = torch.sqrt(resid2.mean(dim=0))  # (G,)

    y_mean = y_true.mean(dim=0, keepdim=True)
    ss_res = resid2.sum(dim=0)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0) + 1e-12
    r2_g = 1.0 - ss_res / ss_tot
    return rmse_g, r2_g
