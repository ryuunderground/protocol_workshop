# interp.py
from __future__ import annotations
import torch


def linear_interp_1d(T: torch.Tensor, Y: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 1D linear interpolation.

    T: (B,) strictly increasing
    Y: (B, G)
    t_query: (Q,)
    returns: (Q, G)

    Notes:
      - Extrapolation is clamped to endpoints (safe for DDE history usage).
    """
    if T.ndim != 1:
        raise ValueError("T must be 1D")
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (B,G)")
    if t_query.ndim != 1:
        raise ValueError("t_query must be 1D")

    B = T.shape[0]
    G = Y.shape[1]
    Q = t_query.shape[0]

    # clamp query within [T[0], T[-1]]
    tq = torch.clamp(t_query, T[0], T[-1])

    # find right indices: T[idx-1] <= tq < T[idx]
    # torch.searchsorted returns idx in [0..B]
    idx = torch.searchsorted(T, tq, right=False)
    idx = torch.clamp(idx, 1, B - 1)

    t0 = T[idx - 1]          # (Q,)
    t1 = T[idx]              # (Q,)
    y0 = Y[idx - 1, :]       # (Q,G)
    y1 = Y[idx, :]           # (Q,G)

    w = (tq - t0) / (t1 - t0 + 1e-12)  # (Q,)
    w = w.unsqueeze(1)                 # (Q,1)

    return (1.0 - w) * y0 + w * y1     # (Q,G)
