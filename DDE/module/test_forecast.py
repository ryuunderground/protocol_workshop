# test_forecast.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from module.history import HistoryParam, HistoryGrid
from module.dde_rhs import SharedParams
from module.dde_solver import solve_dde_at_observation_times, SolverConfig
from module.loss import gaussian_nll_from_predictions
from module.metrics import rmse, r2_score, per_gene_metrics


@dataclass
class TestConfig:
    K_fit: int = 2                 # 초기 K개 관측으로 history만 맞춤
    iters_hist: int = 800
    lr_hist: float = 1e-2
    sigma: float = 1.0
    M_hist: int = 12
    solver_cfg: SolverConfig = field(
        default_factory=SolverConfig
    )
    grad_clip: Optional[float] = 1.0


def fit_history_and_forecast(
    test_sample,
    shared: SharedParams,
    tau_by_lag: Dict[int, float],
    tau_max: float,
    device: torch.device,
    cfg: TestConfig,
) -> Dict:
    """
    shared fixed
    optimize only test history using first K points,
    then predict full horizon and report metrics (overall + after K).

    Returns dict with:
      Z_pred, loss_fit, metrics_all, metrics_future, per_gene...
    """
    G = test_sample.X.shape[1]
    assert test_sample.X.device == device

    # freeze shared
    shared.eval()
    for p in shared.parameters():
        p.requires_grad_(False)

    grid = HistoryGrid(M=cfg.M_hist)
    hist = HistoryParam(G=G, grid=grid, init_from_first_obs=test_sample.X[0]).to(device)

    opt = torch.optim.Adam(hist.parameters(), lr=cfg.lr_hist)

    K = min(cfg.K_fit, test_sample.X.shape[0])

    best_loss = float("inf")
    best_state = None

    for it in range(cfg.iters_hist):
        opt.zero_grad()

        Z_pred = solve_dde_at_observation_times(
            t_obs=test_sample.t, shared=shared, history=hist,
            tau_by_lag=tau_by_lag, tau_max=tau_max, cfg=cfg.solver_cfg
        )

        loss_fit = gaussian_nll_from_predictions(
            test_sample.X[:K], Z_pred[:K], sigma=cfg.sigma, reduce="mean"
        )
        loss_fit.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(hist.parameters(), cfg.grad_clip)

        opt.step()

        lv = float(loss_fit.item())
        if lv < best_loss:
            best_loss = lv
            best_state = {k: v.detach().clone() for k, v in hist.state_dict().items()}

    if best_state is not None:
        hist.load_state_dict(best_state)

    with torch.no_grad():
        Z_pred = solve_dde_at_observation_times(
            t_obs=test_sample.t, shared=shared, history=hist,
            tau_by_lag=tau_by_lag, tau_max=tau_max, cfg=cfg.solver_cfg
        )

    # metrics
    metrics_all = {
        "rmse": rmse(test_sample.X, Z_pred),
        "r2": r2_score(test_sample.X, Z_pred),
    }
    if K < test_sample.X.shape[0]:
        metrics_future = {
            "rmse": rmse(test_sample.X[K:], Z_pred[K:]),
            "r2": r2_score(test_sample.X[K:], Z_pred[K:]),
        }
    else:
        metrics_future = {"rmse": float("nan"), "r2": float("nan")}

    rmse_g, r2_g = per_gene_metrics(test_sample.X, Z_pred)

    return {
        "history": hist,
        "Z_pred": Z_pred,
        "loss_fit_best": best_loss,
        "K_used": K,
        "metrics_all": metrics_all,
        "metrics_future": metrics_future,
        "rmse_per_gene": rmse_g,
        "r2_per_gene": r2_g,
    }
