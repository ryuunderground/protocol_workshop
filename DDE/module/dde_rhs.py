# dde_rhs.py
from __future__ import annotations
from typing import Dict, Callable

import torch


def make_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "identity":
        return lambda x: x
    if name == "tanh":
        return torch.tanh
    if name == "softsign":
        return torch.nn.functional.softsign
    raise ValueError(f"Unknown activation: {name}")


class SharedParams(torch.nn.Module):
    """
    Shared parameters:
      - lam_i (decay) for each gene
      - w_e per edge per lag
      - optional bias b_i
    Uses box constraints via sigmoid parametrization.
    """
    def __init__(
        self,
        G: int,
        edge_idx_by_lag: Dict[int, Dict[str, torch.Tensor]],
        w_box=(-2.0, 2.0),
        lam_box=(1e-3, 0.5),
        use_bias: bool = True,
    ):
        super().__init__()
        self.G = G
        self.edge_idx_by_lag = edge_idx_by_lag
        self.lags = sorted(edge_idx_by_lag.keys())

        self.w_lo, self.w_hi = float(w_box[0]), float(w_box[1])
        self.lam_lo, self.lam_hi = float(lam_box[0]), float(lam_box[1])

        # unconstrained
        self.lam_u = torch.nn.Parameter(torch.zeros(G))
        self.b = torch.nn.Parameter(torch.zeros(G)) if use_bias else None

        # one learnable vector per lag (aligned with edge ordering)
        self.w_u = torch.nn.ParameterDict()
        for k in self.lags:
            E = int(edge_idx_by_lag[k]["src"].numel())
            self.w_u[str(k)] = torch.nn.Parameter(
                0.5 * torch.randn(E)   # or torch.empty(E).uniform_(-1, 1)
            )
        
        # 자기회귀 독점 방지
        self.use_bias = use_bias


    def lam(self) -> torch.Tensor:
        # lam in [lam_lo, lam_hi]
        return self.lam_lo + (self.lam_hi - self.lam_lo) * torch.sigmoid(self.lam_u)

    def w(self, lag: int) -> torch.Tensor:
        u = self.w_u[str(int(lag))]
        return self.w_lo + (self.w_hi - self.w_lo) * torch.sigmoid(u)

    def bias(self) -> torch.Tensor:
        if self.b is None:
            return torch.zeros(self.G, device=self.lam_u.device)
        return self.b


def dde_rhs_single(
    t: torch.Tensor,                 # scalar tensor
    x_t: torch.Tensor,               # (G,)
    x_delay_getter,                  # function: (t_query: (Q,)) -> (Q,G)
    shared: SharedParams,
    tau_by_lag: Dict[int, float],
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Compute dx/dt at time t for a single sample trajectory.

    x_delay_getter(tq): returns x(tq) for tq shape (Q,).
    """
    G = shared.G
    lam = shared.lam()               # (G,)
    b = shared.bias()                # (G,)

    # base: -lam * x(t) + b
    dx = -lam * x_t + b              # (G,)

    # add delayed regulatory terms per lag
    for k in shared.lags:
        tau = float(tau_by_lag[int(k)])
        if tau == 0.0:
            # lag 0 uses current x(t) (not delayed)
            x_k = x_t.view(1, G)     # (1,G)
        else:
            tq = (t - tau).view(1)   # (1,)
            x_k = x_delay_getter(tq) # (1,G)

        src = shared.edge_idx_by_lag[k]["src"].to(x_t.device)
        dst = shared.edge_idx_by_lag[k]["dst"].to(x_t.device)

        # gather source values
        src_vals = x_k[0, src]                    # (E,)
        src_vals = activation(src_vals)           # (E,)

        w = shared.w(k).to(x_t.device)            # (E,)
        contrib = w * src_vals                    # (E,)

        # scatter-add into dx at dst indices
        dx = dx.index_add(0, dst, contrib)

    return dx
