# module/multistart.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import os
import multiprocessing as mp

import numpy as np
import torch

from module.train_joint import train_joint_two_samples, TrainConfig
from module.test_forecast import fit_history_and_forecast, TestConfig


@dataclass
class MultiStartConfig:
    seeds: List[int]
    gpu_ids: Optional[List[int]] = None  # (CUDA multi-gpu only; not used on M1)
    ms_workers: int = 1                  # CPU parallel workers (safe only when device=cpu)


def _force_cpu_single_thread():
    # Avoid CPU oversubscription: (processes) x (threads) explosion
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _worker_train_eval_cpu(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe CPU worker:
      - receives only numpy / python primitives
      - rebuilds tensors on CPU inside worker
      - limits torch threads to 1
    """
    _force_cpu_single_thread()

    seed: int = int(args["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    # rebuild samples
    def rebuild_sample(sample_pack):
        # sample_pack: {"name": str, "t": np.ndarray, "X": np.ndarray, "genes": List[str]}
        # NOTE: 여기서는 load_expression_csv에서 만든 Sample 구조를 모르므로,
        #       동일 attribute를 가진 "간단 객체"를 만들어 준다.
        class _S:  # minimal duck-typed sample
            pass

        s = _S()
        s.name = sample_pack["name"]
        s.genes = sample_pack["genes"]
        s.gene_to_idx = {g: i for i, g in enumerate(s.genes)}
        s.t = torch.tensor(sample_pack["t"], dtype=torch.float32, device=device)
        s.X = torch.tensor(sample_pack["X"], dtype=torch.float32, device=device)
        return s

    train_a = rebuild_sample(args["train_a"])
    train_b = rebuild_sample(args["train_b"])
    test_s  = rebuild_sample(args["test_sample"])

    # rebuild edges
    edge_idx_by_lag_np = args["edge_idx_by_lag"]
    edge_idx_local = {}
    for k_str, d in edge_idx_by_lag_np.items():
        k = int(k_str)
        edge_idx_local[k] = {
            "src": torch.tensor(d["src"], dtype=torch.long, device=device),
            "dst": torch.tensor(d["dst"], dtype=torch.long, device=device),
        }

    tau_by_lag = {int(k): float(v) for k, v in args["tau_by_lag"].items()}
    tau_max = float(args["tau_max"])

    train_cfg: TrainConfig = args["train_cfg"]
    test_cfg: TestConfig   = args["test_cfg"]

    shared, histories, info = train_joint_two_samples(
        train_a, train_b,
        edge_idx_by_lag=edge_idx_local,
        tau_by_lag=tau_by_lag,
        tau_max=tau_max,
        device=device,
        cfg=train_cfg,
        seed=seed
    )

    test_out = fit_history_and_forecast(
        test_sample=test_s,
        shared=shared,
        tau_by_lag=tau_by_lag,
        tau_max=tau_max,
        device=device,
        cfg=test_cfg
    )

    # state_dict는 CPU 텐서로 확실히 반환(메인에서 저장/로딩 안정)
    shared_sd = {k: v.detach().cpu() for k, v in shared.state_dict().items()}
    hist_train_sd = {
        name: {k: v.detach().cpu() for k, v in h.state_dict().items()}
        for name, h in histories.items()
    }
    hist_test_sd = {k: v.detach().cpu() for k, v in test_out["history"].state_dict().items()}

    return {
        "seed": seed,
        "train_info": info,
        "test_metrics_all": test_out["metrics_all"],
        "test_metrics_future": test_out["metrics_future"],
        "loss_fit_best": test_out["loss_fit_best"],
        "shared_state": shared_sd,
        "hist_train_state": hist_train_sd,
        "hist_test_state": hist_test_sd,
    }


def run_multistart(
    train_a, train_b, test_sample,
    edge_idx_by_lag, tau_by_lag, tau_max,
    train_cfg: TrainConfig,
    test_cfg: TestConfig,
    ms_cfg: MultiStartConfig,
    device: torch.device,
) -> List[Dict]:
    """
    SAFE policy:
      - device in {mps,cuda} -> always sequential on that device
      - device == cpu:
          * if ms_workers <= 1 -> sequential
          * else -> spawn multiprocessing, CPU workers only
    """
    results: List[Dict] = []

    # ------------------------------------------------------------
    # 1) GPU/MPS: sequential only (safest)
    # ------------------------------------------------------------
    if device.type in ("mps", "cuda"):
        # move once
        train_a.t, train_a.X = train_a.t.to(device), train_a.X.to(device)
        train_b.t, train_b.X = train_b.t.to(device), train_b.X.to(device)
        test_sample.t, test_sample.X = test_sample.t.to(device), test_sample.X.to(device)

        edge_idx_local = {
            k: {"src": d["src"].to(device), "dst": d["dst"].to(device)}
            for k, d in edge_idx_by_lag.items()
        }

        for seed in ms_cfg.seeds:
            shared, histories, info = train_joint_two_samples(
                train_a, train_b,
                edge_idx_by_lag=edge_idx_local,
                tau_by_lag=tau_by_lag,
                tau_max=tau_max,
                device=device,
                cfg=train_cfg,
                seed=seed
            )
            test_out = fit_history_and_forecast(
                test_sample=test_sample,
                shared=shared,
                tau_by_lag=tau_by_lag,
                tau_max=tau_max,
                device=device,
                cfg=test_cfg
            )
            results.append({
                "seed": seed,
                "train_info": info,
                "test_metrics_all": test_out["metrics_all"],
                "test_metrics_future": test_out["metrics_future"],
                "loss_fit_best": test_out["loss_fit_best"],
                "shared_state": shared.state_dict(),
                "hist_train_state": {k: v.state_dict() for k, v in histories.items()},
                "hist_test_state": test_out["history"].state_dict(),
            })
        return results

    # ------------------------------------------------------------
    # 2) CPU: sequential or spawn-parallel
    # ------------------------------------------------------------
    assert device.type == "cpu"

    # CPU sequential
    if ms_cfg.ms_workers <= 1:
        _force_cpu_single_thread()  # still helps reproducibility / stable runtime

        # move once
        train_a.t, train_a.X = train_a.t.to(device), train_a.X.to(device)
        train_b.t, train_b.X = train_b.t.to(device), train_b.X.to(device)
        test_sample.t, test_sample.X = test_sample.t.to(device), test_sample.X.to(device)

        edge_idx_local = {
            k: {"src": d["src"].to(device), "dst": d["dst"].to(device)}
            for k, d in edge_idx_by_lag.items()
        }

        for seed in ms_cfg.seeds:
            shared, histories, info = train_joint_two_samples(
                train_a, train_b,
                edge_idx_by_lag=edge_idx_local,
                tau_by_lag=tau_by_lag,
                tau_max=tau_max,
                device=device,
                cfg=train_cfg,
                seed=seed
            )
            test_out = fit_history_and_forecast(
                test_sample=test_sample,
                shared=shared,
                tau_by_lag=tau_by_lag,
                tau_max=tau_max,
                device=device,
                cfg=test_cfg
            )
            results.append({
                "seed": seed,
                "train_info": info,
                "test_metrics_all": test_out["metrics_all"],
                "test_metrics_future": test_out["metrics_future"],
                "loss_fit_best": test_out["loss_fit_best"],
                "shared_state": shared.state_dict(),
                "hist_train_state": {k: v.state_dict() for k, v in histories.items()},
                "hist_test_state": test_out["history"].state_dict(),
            })
        return results

    # CPU spawn parallel
    ctx = mp.get_context("spawn")

    # pack samples as numpy (pickle-safe)
    def pack_sample(s):
        return {
            "name": s.name,
            "genes": list(s.genes),
            "t": s.t.detach().cpu().numpy().astype(np.float32),
            "X": s.X.detach().cpu().numpy().astype(np.float32),
        }

    # pack edges as numpy
    edge_pack = {}
    for k, d in edge_idx_by_lag.items():
        edge_pack[str(int(k))] = {
            "src": d["src"].detach().cpu().numpy().astype(np.int64),
            "dst": d["dst"].detach().cpu().numpy().astype(np.int64),
        }

    base_args = {
        "train_a": pack_sample(train_a),
        "train_b": pack_sample(train_b),
        "test_sample": pack_sample(test_sample),
        "edge_idx_by_lag": edge_pack,
        "tau_by_lag": {int(k): float(v) for k, v in tau_by_lag.items()},
        "tau_max": float(tau_max),
        "train_cfg": train_cfg,
        "test_cfg": test_cfg,
    }

    # IMPORTANT: do not create too many processes on M1
    n_workers = min(int(ms_cfg.ms_workers), len(ms_cfg.seeds))
    with ctx.Pool(processes=n_workers) as pool:
        job_args = []
        for seed in ms_cfg.seeds:
            a = dict(base_args)
            a["seed"] = int(seed)
            job_args.append(a)
        results = pool.map(_worker_train_eval_cpu, job_args)

    # stable ordering
    results = sorted(results, key=lambda r: r["seed"])
    return results