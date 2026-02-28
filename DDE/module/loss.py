# loss.py
from __future__ import annotations
import torch

def gaussian_nll_from_predictions(
    X_obs: torch.Tensor,   # (N,G)
    Z_pred: torch.Tensor,  # (N,G)
    sigma: float | torch.Tensor = 1.0,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Negative log-likelihood up to constant under iid Gaussian noise.
    If sigma is scalar: same noise for all genes.
    If sigma is (G,): gene-wise noise.
    """
    resid2 = (X_obs - Z_pred) ** 2
    denom = (2.0 * (sigma ** 2))
    nll = resid2 / denom
    if reduce == "mean":
        return nll.mean()
    if reduce == "sum":
        return nll.sum()
    raise ValueError("reduce must be 'mean' or 'sum'")
