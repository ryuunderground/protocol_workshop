# history.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from module.interp import linear_interp_1d

# 히스토리 구간을 M개의 knot로 나눔
@dataclass
class HistoryGrid:
    M: int  # number of knots

    # 수학적으로 t_m^{(h)} = t_0 - \tau_{max} + \frac{m}{M-1}\tau_{max}
    # 완벽한 히스토리 복원이 아니라 과적합 방지지
    def knot_times(self, t0: torch.Tensor, tau_max: float, device=None) -> torch.Tensor:
        """
        Uniform knots on [t0 - tau_max, t0]
        """
        if device is None:
            device = t0.device
        return torch.linspace(float(t0.item()) - tau_max, float(t0.item()), self.M, device=device)

# 핵심 클래스
# 히스토리 값을 모델 파라미터로 두고, nn로 학습
class HistoryParam(torch.nn.Module):
    """
    h^{(s)}(t) on [t0 - tau_max, t0] parameterized by M knots.
    Learnable tensor H: (M,G).
    """
    def __init__(self, G: int, grid: HistoryGrid, init_from_first_obs: Optional[torch.Tensor] = None):
        super().__init__()
        self.G = G
        self.grid = grid

        H0 = torch.zeros(grid.M, G, dtype=torch.float32)
        if init_from_first_obs is not None:
            # init to constant history equal to first observation
            # 히스토리 초깃값 = 히스토리 초깃값
            H0[:] = init_from_first_obs.detach().float().view(1, G)

        self.H = torch.nn.Parameter(H0)

    def eval(self, t_query: torch.Tensor, t0: torch.Tensor, tau_max: float) -> torch.Tensor:
        """
        t_query: (Q,) expected in [t0-tau_max, t0] but safe-clamped by interp
        returns: (Q,G)
        """
        T = self.grid.knot_times(t0, tau_max, device=t_query.device)  # (M,)
        return linear_interp_1d(T, self.H, t_query)
