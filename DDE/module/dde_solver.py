# dde_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable

import torch

from module.interp import linear_interp_1d
from module.history import HistoryParam, HistoryGrid
from module.dde_rhs import SharedParams, dde_rhs_single, make_activation
from module.ode_rk import rk4_step


@dataclass
class SolverConfig:
    # 내부 적분 dt를 관측 간격의 몇 %로 둘지
    dt_frac_of_median_gap: float = 0.10
    # dt 상한(너무 촘촘하면 느려짐)
    dt_max: float = 1.0
    # dt 하한(너무 작으면 느려짐)
    dt_min: float = 1e-3
    # activation used in RHS
    activation: str = "tanh"
    # buffer에 매 step 저장할지 (True면 메모리↑ / 보간 안정↑)
    store_every_step: bool = True


class TrajectoryBuffer:
    """
    Stores (T, Z) where:
      T: (B,) strictly increasing
      Z: (B,G)
    Provides differentiable interpolation z(t_query).
    """
    def __init__(self, T_init: torch.Tensor, Z_init: torch.Tensor):
        # both on device
        if T_init.ndim != 1:
            raise ValueError("T_init must be 1D")
        if Z_init.ndim != 2:
            raise ValueError("Z_init must be 2D (B,G)")
        if T_init.shape[0] != Z_init.shape[0]:
            raise ValueError("T_init and Z_init length mismatch")

        self.T = T_init.contiguous()
        self.Z = Z_init.contiguous()

    def append(self, t: torch.Tensor, z: torch.Tensor):
        """
        t: scalar tensor
        z: (G,)
        """
        t1 = t.view(1)
        z1 = z.view(1, -1)
        # concatenate (simple and reliable; later 최적화 가능)
        self.T = torch.cat([self.T, t1], dim=0)
        self.Z = torch.cat([self.Z, z1], dim=0)

    def eval(self, t_query: torch.Tensor) -> torch.Tensor:
        """
        t_query: (Q,)
        returns: (Q,G)
        """
        return linear_interp_1d(self.T, self.Z, t_query)


def choose_internal_dt(t_obs: torch.Tensor, cfg: SolverConfig) -> float:
    """
    dt = min(dt_max, max(dt_min, frac * median_gap))
    """
    if t_obs.numel() < 2:
        return float(cfg.dt_min)
    gaps = (t_obs[1:] - t_obs[:-1]).detach()
    med = gaps.median().item()
    dt = cfg.dt_frac_of_median_gap * float(med)
    dt = min(float(cfg.dt_max), max(float(cfg.dt_min), float(dt)))
    return float(dt)


def check_dt_vs_delays(dt: float, tau_by_lag: Dict[int, float]):
    """
    RK4 stage evaluation assumes delays are not tiny relative to dt.
    Practical rule: dt should be much smaller than smallest positive delay.
    """
    pos = [tau for tau in tau_by_lag.values() if tau > 0]
    if not pos:
        return
    tau_min = min(pos)
    # 보수적으로 dt <= tau_min / 4 권장
    if dt > tau_min / 4.0:
        raise ValueError(
            f"Internal dt={dt:.6g} is too large vs smallest positive delay={tau_min:.6g}. "
            f"Reduce dt (e.g., dt_frac_of_median_gap) or choose larger Δ."
        )


@torch.no_grad()
def init_history_grid_from_param(history: HistoryParam, t0: torch.Tensor, tau_max: float) -> torch.Tensor:
    """
    Get knot times used by history (for initializing buffer).
    """
    return history.grid.knot_times(t0, tau_max, device=t0.device)


def solve_dde_at_observation_times(
    t_obs: torch.Tensor,                 # (N,)
    shared: SharedParams,
    history: HistoryParam,
    tau_by_lag: Dict[int, float],
    tau_max: float,
    cfg: SolverConfig,
    z0_override: Optional[torch.Tensor] = None,   # (G,) optional
) -> torch.Tensor:
    """
    Core solver: returns Z_pred at observation times.
    - method-of-steps + RK4
    - history provides z(t) for t in [t0 - tau_max, t0]
    - buffer stores integrated trajectory (and history knots)
    """
    device = t_obs.device
    act = make_activation(cfg.activation)

    # ---- basic checks ----
    if t_obs.ndim != 1:
        raise ValueError("t_obs must be 1D")
    if not torch.all(t_obs[1:] >= t_obs[:-1]):
        raise ValueError("t_obs must be non-decreasing")
    if t_obs.numel() < 1:
        raise ValueError("t_obs empty")

    t0 = t_obs[0].detach()  # scalar
    if hasattr(cfg, "fixed_dt") and cfg.fixed_dt is not None:
        dt = cfg.fixed_dt
    else:
        dt = choose_internal_dt(t_obs, cfg)
    check_dt_vs_delays(dt, tau_by_lag)

    # ---- initialize buffer with history knots (differentiable values) ----
    # knot times (M,)
    T_hist = history.grid.knot_times(t0, tau_max, device=device)  # (M,)
    # history values (M,G)
    Z_hist = history.eval(T_hist, t0=t0, tau_max=tau_max)         # (M,G)

    buf = TrajectoryBuffer(T_hist, Z_hist)

    # initial state at t0
    if z0_override is not None:
        z_curr = z0_override
    else:
        # history at t0 (exactly last knot)
        z_curr = Z_hist[-1, :].contiguous()

    # ensure buffer includes (t0, z0) — already should, but safe
    if buf.T[-1].item() != float(t0.item()):
        buf.append(t0, z_curr)

    # ---- delay getter: routes to history or buffer ----
    def z_delay_getter(t_query: torch.Tensor) -> torch.Tensor:
        """
        t_query: (Q,)
        returns: (Q,G)
        For t <= t0 : history
        For t >  t0 : buffer interpolation
        """
        # split mask
        mask_hist = (t_query <= t0)
        if mask_hist.all():
            return history.eval(t_query, t0=t0, tau_max=tau_max)

        if (~mask_hist).all():
            return buf.eval(t_query)

        # mixed: compute separately then merge
        out = torch.empty((t_query.numel(), shared.G), device=device, dtype=torch.float32)
        if mask_hist.any():
            out[mask_hist] = history.eval(t_query[mask_hist], t0=t0, tau_max=tau_max)
        if (~mask_hist).any():
            out[~mask_hist] = buf.eval(t_query[~mask_hist])
        return out

    # ---- RHS wrapper (needs current buffer/history) ----
    def rhs(tt: torch.Tensor, zz: torch.Tensor) -> torch.Tensor:
        return dde_rhs_single(
            t=tt, z_t=zz, z_delay_getter=z_delay_getter,
            shared=shared, tau_by_lag=tau_by_lag,
            activation=act
        )

    # ---- integrate and record at obs times ----
    N = t_obs.numel()
    G = shared.G
    Z_pred = torch.empty((N, G), device=device, dtype=torch.float32)

    t_curr = t0
    # first obs is at t0
    Z_pred[0, :] = z_curr

    # internal dt tensor
    dt_t = torch.tensor(dt, device=device, dtype=torch.float32)

    for n in range(1, N):
        t_target = t_obs[n]
        # integrate up to t_target
        while (t_curr + dt_t) < t_target:
            z_next = rk4_step(t_curr, z_curr, dt_t, f=rhs)
            t_curr = t_curr + dt_t
            z_curr = z_next
            if cfg.store_every_step:
                buf.append(t_curr, z_curr)

        # final step to land exactly on t_target
        h = (t_target - t_curr)
        if h > 0:
            z_next = rk4_step(t_curr, z_curr, h, f=rhs)
            t_curr = t_target
            z_curr = z_next
            buf.append(t_curr, z_curr)

        Z_pred[n, :] = z_curr

    return Z_pred
