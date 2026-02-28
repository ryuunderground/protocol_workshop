# train_joint.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import copy
import torch

from module.history import HistoryParam, HistoryGrid
from module.dde_rhs import SharedParams
from module.dde_solver import solve_dde_at_observation_times, SolverConfig
from module.loss import gaussian_nll_from_predictions


@dataclass
class TrainConfig:
    # optimizer
    lr_shared: float = 1e-2
    lr_hist: float = 1e-2
    epochs: int = 2000

    # noise (fixed)
    sigma: float = 1.0

    # history
    M_hist: int = 12

    # solver
    solver_cfg: SolverConfig = field(
        default_factory=SolverConfig
    )

    # optional: gradient clipping
    grad_clip: Optional[float] = 1.0

    # early stopping
    patience: int = 200
    min_delta: float = 1e-6


def _state_dict_deepcopy(module: torch.nn.Module) -> Dict:
    return copy.deepcopy(module.state_dict())


def train_joint_two_samples(
    sample_a, sample_b,
    edge_idx_by_lag: Dict[int, Dict[str, torch.Tensor]],
    tau_by_lag: Dict[int, float],
    tau_max: float,
    device: torch.device,
    cfg: TrainConfig,
    seed: Optional[int] = None,
) -> Tuple[SharedParams, Dict[str, HistoryParam], Dict]:
    """
    Joint MLE:
      minimize NLL(sample_a) + NLL(sample_b)
    Variables:
      shared: {w, lam, (b)}
      history_a, history_b: sample-specific knot values

    Returns:
      shared, histories, info(best_loss, history, etc.)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    G = sample_a.X.shape[1]
    assert sample_b.X.shape[1] == G, "Gene dimension mismatch"
    assert sample_a.X.device == device and sample_b.X.device == device

    # shared params
    shared = SharedParams(G=G, edge_idx_by_lag=edge_idx_by_lag).to(device)

    # histories (sample-specific)
    grid = HistoryGrid(M=cfg.M_hist)
    hist_a = HistoryParam(G=G, grid=grid, init_from_first_obs=sample_a.X[0]).to(device)
    hist_b = HistoryParam(G=G, grid=grid, init_from_first_obs=sample_b.X[0]).to(device)
    histories = {sample_a.name: hist_a, sample_b.name: hist_b}

    # two optimizers (different LRs)
    opt = torch.optim.Adam([
        {"params": shared.parameters(), "lr": cfg.lr_shared},
        {"params": hist_a.parameters(), "lr": cfg.lr_hist},
        {"params": hist_b.parameters(), "lr": cfg.lr_hist},
    ])

    best_loss = float("inf")
    best_state = {
        "shared": None,
        "hist_a": None,
        "hist_b": None,
        "epoch": -1,
    }

    no_improve = 0
    bias_warmup_epochs = int(0.2 * cfg.epochs)
    # bias_warmup_epochs = 300

    for epoch in range(cfg.epochs):
        opt.zero_grad()

        # --- bias warm-up: early epochs, disable bias ---
        if shared.b is not None and epoch < bias_warmup_epochs:
            with torch.no_grad():
                shared.b.zero_()

        # forward: solve DDE at observation times for each sample
        Z_a = solve_dde_at_observation_times(
            t_obs=sample_a.t, shared=shared, history=hist_a,
            tau_by_lag=tau_by_lag, tau_max=tau_max, cfg=cfg.solver_cfg
        )
        Z_b = solve_dde_at_observation_times(
            t_obs=sample_b.t, shared=shared, history=hist_b,
            tau_by_lag=tau_by_lag, tau_max=tau_max, cfg=cfg.solver_cfg
        )

        # NLL (Gaussian) sum
        loss_a = gaussian_nll_from_predictions(sample_a.X, Z_a, sigma=cfg.sigma, reduce="mean")
        loss_b = gaussian_nll_from_predictions(sample_b.X, Z_b, sigma=cfg.sigma, reduce="mean")
        loss = loss_a + loss_b

        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(shared.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(hist_a.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(hist_b.parameters(), cfg.grad_clip)

        opt.step()

        # track best
        loss_val = float(loss.item())
        if loss_val + cfg.min_delta < best_loss:
            best_loss = loss_val
            best_state["shared"] = _state_dict_deepcopy(shared)
            best_state["hist_a"] = _state_dict_deepcopy(hist_a)
            best_state["hist_b"] = _state_dict_deepcopy(hist_b)
            best_state["epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            break

    # restore best
    if best_state["shared"] is not None:
        shared.load_state_dict(best_state["shared"])
        hist_a.load_state_dict(best_state["hist_a"])
        hist_b.load_state_dict(best_state["hist_b"])

    info = {
        "best_loss": best_loss,
        "best_epoch": best_state["epoch"],
        "stopped_epoch": epoch,
        "epochs_run": epoch + 1,
    }
    return shared, histories, info
