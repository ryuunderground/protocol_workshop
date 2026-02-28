# module/dde_validation.py
from __future__ import annotations

import os
import numpy as np
import torch
import multiprocessing as mp
from typing import Dict, Any, Optional, Tuple


# =========================
# Metric
# =========================
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =========================
# Worker safety
# =========================
# module/dde_validation.py

def _force_cpu_single_thread():
    # Environment: subprocess 시작 초기에 세팅되는 게 가장 확실
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Torch thread control
    # set_num_threads는 대부분 안전하지만, interop은 "이미 병렬이 시작"되면 예외가 날 수 있음.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # interop threads는 RuntimeError가 날 수 있으니 "시도하되 실패하면 무시"
    try:
        # 이미 1이면 굳이 건드리지 않음
        if hasattr(torch, "get_num_interop_threads"):
            cur = torch.get_num_interop_threads()
            if cur != 1:
                torch.set_num_interop_threads(1)
        else:
            # get이 없으면 그냥 시도
            torch.set_num_interop_threads(1)
    except RuntimeError:
        # "cannot set number of interop threads after parallel work has started" 방지
        pass


# =========================================================
# 1) TIME PERMUTATION (inter only, no retrain) - SAFE
# =========================================================
def _time_perm_worker(args: Dict[str, Any]) -> Tuple[float, float]:
    """
    Worker computes one permutation metric pair (r2, rmse).
    Inputs are numpy only (pickle-safe).
    """
    _force_cpu_single_thread()

    seed = int(args["seed"])
    b = int(args["b"])
    n = int(args["n_time"])
    X = args["X"]              # np.ndarray (T,G)
    y_pred = args["y_pred"]    # np.ndarray (T,G)

    rng = np.random.default_rng(seed + 100000 * (b + 1))
    idx = rng.permutation(n)
    y_true_null = X[idx, :]

    r2 = r2_score(y_true_null, y_pred)
    e = rmse(y_true_null, y_pred)
    return float(r2), float(e)


def time_permutation_test_inter(
    model_predict_fn,
    model_obj,
    test_sample,
    n_perm: int = 200,
    seed: int = 0,
    out_prefix: str = "time_perm",
    plot: bool = True,
    perm_workers: int = 1,
    device_type: Optional[str] = None,   # "cpu"|"mps"|"cuda"|None
):
    """
    Model fixed. Permute only mapping between time index and expression rows.

    Safe policy:
      - if device_type in ("mps","cuda") => force sequential
      - else cpu: can use spawn pool with perm_workers
    """
    # ----- observed -----
    X_np = test_sample.X.detach().cpu().numpy()
    y_pred = model_predict_fn(model_obj, test_sample)
    y_pred_np = np.asarray(y_pred)

    r2_obs = r2_score(X_np, y_pred_np)
    rmse_obs = rmse(X_np, y_pred_np)

    # decide execution mode
    dev = device_type or getattr(test_sample.X.device, "type", "cpu")
    use_parallel = (dev == "cpu") and (perm_workers is not None) and (perm_workers > 1)

    r2_null = np.zeros(n_perm, dtype=float)
    rmse_null = np.zeros(n_perm, dtype=float)

    if use_parallel:
        ctx = mp.get_context("spawn")
        n_workers = min(int(perm_workers), n_perm)
        base = {
            "seed": int(seed),
            "n_time": int(X_np.shape[0]),
            "X": X_np.astype(np.float32, copy=False),
            "y_pred": y_pred_np.astype(np.float32, copy=False),
        }
        jobs = []
        for b in range(n_perm):
            a = dict(base)
            a["b"] = int(b)
            jobs.append(a)

        with ctx.Pool(processes=n_workers) as pool:
            out = pool.map(_time_perm_worker, jobs)

        for b, (r2b, eb) in enumerate(out):
            r2_null[b] = r2b
            rmse_null[b] = eb

    else:
        rng = np.random.default_rng(seed)
        n = X_np.shape[0]
        for b in range(n_perm):
            idx = rng.permutation(n)
            y_true_null = X_np[idx, :]
            r2_null[b] = r2_score(y_true_null, y_pred_np)
            rmse_null[b] = rmse(y_true_null, y_pred_np)

    # empirical p-values
    p_r2 = (1 + np.sum(r2_null >= r2_obs)) / (1 + n_perm)
    p_rmse = (1 + np.sum(rmse_null <= rmse_obs)) / (1 + n_perm)

    hist_paths = {}
    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(r2_null, bins=30)
        plt.axvline(r2_obs, linestyle="--")
        plt.title(f"Time Permutation (R2), p={p_r2:.4g}")
        plt.xlabel("R2")
        plt.ylabel("count")
        plt.tight_layout()
        path_r2 = f"{out_prefix}_r2_hist.png"
        plt.savefig(path_r2, dpi=200)
        plt.close()

        plt.figure()
        plt.hist(rmse_null, bins=30)
        plt.axvline(rmse_obs, linestyle="--")
        plt.title(f"Time Permutation (RMSE), p={p_rmse:.4g}")
        plt.xlabel("RMSE")
        plt.ylabel("count")
        plt.tight_layout()
        path_rmse = f"{out_prefix}_rmse_hist.png"
        plt.savefig(path_rmse, dpi=200)
        plt.close()

        hist_paths["r2_hist"] = path_r2
        hist_paths["rmse_hist"] = path_rmse

    return {
        "r2_obs": float(r2_obs),
        "rmse_obs": float(rmse_obs),
        "r2_null": r2_null.tolist(),
        "rmse_null": rmse_null.tolist(),
        "p_value_r2": float(p_r2),
        "p_value_rmse": float(p_rmse),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "hist_paths": hist_paths,
        "parallel_used": bool(use_parallel),
        "device_type": str(dev),
    }


# =========================================================
# 2) EDGE REWIRING + RETRAIN - SAFE CPU PARALLEL
# =========================================================
def _pack_sample_cpu(sample) -> Dict[str, Any]:
    return {
        "name": sample.name,
        "genes": list(getattr(sample, "genes", [])),
        "t": sample.t.detach().cpu().numpy().astype(np.float32),
        "X": sample.X.detach().cpu().numpy().astype(np.float32),
    }


def _rebuild_sample_cpu(pack: Dict[str, Any]):
    class _S:
        pass
    s = _S()
    s.name = pack["name"]
    s.genes = pack.get("genes", [])
    s.gene_to_idx = {g: i for i, g in enumerate(s.genes)}
    s.t = torch.tensor(pack["t"], dtype=torch.float32, device="cpu")
    s.X = torch.tensor(pack["X"], dtype=torch.float32, device="cpu")
    return s


def _pack_edges_cpu(edge_idx_by_lag) -> Dict[str, Any]:
    out = {}
    for lag, d in edge_idx_by_lag.items():
        out[str(int(lag))] = {
            "src": d["src"].detach().cpu().numpy().astype(np.int64),
            "dst": d["dst"].detach().cpu().numpy().astype(np.int64),
        }
    return out


def _sample_edges_no_self(rng: np.random.Generator, G: int, E: int) -> Tuple[np.ndarray, np.ndarray]:
    src_list = []
    dst_list = []
    while len(src_list) < E:
        need = E - len(src_list)
        src = rng.integers(0, G, size=need)
        dst = rng.integers(0, G, size=need)
        mask = src != dst
        src_list.extend(src[mask].tolist())
        dst_list.extend(dst[mask].tolist())
    return np.asarray(src_list[:E], dtype=np.int64), np.asarray(dst_list[:E], dtype=np.int64)


def _edge_rewire_worker(args: Dict[str, Any]) -> Tuple[float, float]:
    """
    One permutation: rewire edges lag-wise, retrain, evaluate.
    CPU only, single-thread torch.
    """
    _force_cpu_single_thread()

    b = int(args["b"])
    seed = int(args["seed"]) + b + 1

    rng = np.random.default_rng(seed)

    # rebuild samples
    train_a = _rebuild_sample_cpu(args["train_a"])
    train_b = _rebuild_sample_cpu(args["train_b"])
    test_s  = _rebuild_sample_cpu(args["test_sample"])

    tau_by_lag = {int(k): float(v) for k, v in args["tau_by_lag"].items()}
    tau_max = float(args["tau_max"])

    train_cfg = args["train_cfg"]
    test_cfg = args["test_cfg"]

    # rebuild edge counts from observed
    edge_pack = args["edge_idx_by_lag"]
    G = int(train_a.X.shape[1])

    edge_null = {}
    for lag_str, d in edge_pack.items():
        lag = int(lag_str)
        E = int(len(d["src"]))
        src_arr, dst_arr = _sample_edges_no_self(rng, G, E)
        edge_null[lag] = {
            "src": torch.tensor(src_arr, dtype=torch.long, device="cpu"),
            "dst": torch.tensor(dst_arr, dtype=torch.long, device="cpu"),
        }

    shared_null, _, _ = args["train_joint_fn"](
        train_a, train_b,
        edge_idx_by_lag=edge_null,
        tau_by_lag=tau_by_lag,
        tau_max=tau_max,
        device=torch.device("cpu"),
        cfg=train_cfg,
        seed=seed,
    )

    test_out_null = args["forecast_fn"](
        test_sample=test_s,
        shared=shared_null,
        tau_by_lag=tau_by_lag,
        tau_max=tau_max,
        device=torch.device("cpu"),
        cfg=test_cfg,
    )

    r2 = float(test_out_null["metrics_future"]["r2"])
    e = float(test_out_null["metrics_future"]["rmse"])
    return r2, e


def edge_rewiring_permutation_test(
    train_joint_fn,
    forecast_fn,
    train_a,
    train_b,
    test_sample,
    edge_idx_by_lag,
    tau_by_lag,
    tau_max,
    train_cfg,
    test_cfg,
    n_perm: int = 50,
    seed: int = 0,
    out_prefix: str = "edge_rewire_perm",
    plot: bool = True,
    perm_workers: int = 1,
    device_type: Optional[str] = None,
):
    """
    Safe policy:
      - if device_type in ("mps","cuda"): sequential (do not spawn)
      - if cpu and perm_workers>1: spawn parallel
    """
    dev = device_type or getattr(train_a.X.device, "type", "cpu")

    # ---- Observed model ----
    # (여긴 한 번만 학습)
    shared_obs, _, _ = train_joint_fn(
        train_a, train_b,
        edge_idx_by_lag=edge_idx_by_lag,
        tau_by_lag=tau_by_lag,
        tau_max=tau_max,
        device=train_a.X.device,
        cfg=train_cfg,
        seed=seed,
    )
    test_out_obs = forecast_fn(
        test_sample=test_sample,
        shared=shared_obs,
        tau_by_lag=tau_by_lag,
        tau_max=tau_max,
        device=test_sample.X.device,
        cfg=test_cfg,
    )
    r2_obs = float(test_out_obs["metrics_future"]["r2"])
    rmse_obs = float(test_out_obs["metrics_future"]["rmse"])

    r2_null = np.zeros(n_perm, dtype=float)
    rmse_null = np.zeros(n_perm, dtype=float)

    use_parallel = (dev == "cpu") and (perm_workers is not None) and (perm_workers > 1)

    if use_parallel:
        ctx = mp.get_context("spawn")
        n_workers = min(int(perm_workers), n_perm)

        base_args = {
            "seed": int(seed),
            "train_a": _pack_sample_cpu(train_a),
            "train_b": _pack_sample_cpu(train_b),
            "test_sample": _pack_sample_cpu(test_sample),
            "edge_idx_by_lag": _pack_edges_cpu(edge_idx_by_lag),
            "tau_by_lag": {int(k): float(v) for k, v in tau_by_lag.items()},
            "tau_max": float(tau_max),
            "train_cfg": train_cfg,
            "test_cfg": test_cfg,
            "train_joint_fn": train_joint_fn,
            "forecast_fn": forecast_fn,
        }

        jobs = []
        for b in range(n_perm):
            a = dict(base_args)
            a["b"] = int(b)
            jobs.append(a)

        with ctx.Pool(processes=n_workers) as pool:
            out = pool.map(_edge_rewire_worker, jobs)

        for b, (r2b, eb) in enumerate(out):
            r2_null[b] = r2b
            rmse_null[b] = eb

    else:
        # sequential (including mps/cuda safety)
        rng = np.random.default_rng(seed)
        device = train_a.X.device
        G = int(train_a.X.shape[1])

        # edge counts by lag
        lag_edge_counts = {int(lag): int(d["src"].numel()) for lag, d in edge_idx_by_lag.items()}

        def sample_edges(E):
            src_list = []
            dst_list = []
            while len(src_list) < E:
                need = E - len(src_list)
                src = rng.integers(0, G, size=need)
                dst = rng.integers(0, G, size=need)
                mask = src != dst
                src_list.extend(src[mask])
                dst_list.extend(dst[mask])
            return np.array(src_list[:E]), np.array(dst_list[:E])

        for b in range(n_perm):
            edge_null = {}
            for lag, E in lag_edge_counts.items():
                src_arr, dst_arr = sample_edges(E)
                edge_null[lag] = {
                    "src": torch.tensor(src_arr, device=device),
                    "dst": torch.tensor(dst_arr, device=device),
                }

            shared_null, _, _ = train_joint_fn(
                train_a, train_b,
                edge_idx_by_lag=edge_null,
                tau_by_lag=tau_by_lag,
                tau_max=tau_max,
                device=device,
                cfg=train_cfg,
                seed=seed + b + 1,
            )
            test_out_null = forecast_fn(
                test_sample=test_sample,
                shared=shared_null,
                tau_by_lag=tau_by_lag,
                tau_max=tau_max,
                device=device,
                cfg=test_cfg,
            )
            r2_null[b] = float(test_out_null["metrics_future"]["r2"])
            rmse_null[b] = float(test_out_null["metrics_future"]["rmse"])

    p_r2 = (1 + np.sum(r2_null >= r2_obs)) / (1 + n_perm)
    p_rmse = (1 + np.sum(rmse_null <= rmse_obs)) / (1 + n_perm)

    hist_paths = {}
    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(r2_null, bins=30)
        plt.axvline(r2_obs, linestyle="--")
        plt.title(f"Edge Rewiring (R2), p={p_r2:.4g}")
        plt.xlabel("R2")
        plt.ylabel("count")
        plt.tight_layout()
        path_r2 = f"{out_prefix}_r2_hist.png"
        plt.savefig(path_r2, dpi=200)
        plt.close()

        plt.figure()
        plt.hist(rmse_null, bins=30)
        plt.axvline(rmse_obs, linestyle="--")
        plt.title(f"Edge Rewiring (RMSE), p={p_rmse:.4g}")
        plt.xlabel("RMSE")
        plt.ylabel("count")
        plt.tight_layout()
        path_rmse = f"{out_prefix}_rmse_hist.png"
        plt.savefig(path_rmse, dpi=200)
        plt.close()

        hist_paths["r2_hist"] = path_r2
        hist_paths["rmse_hist"] = path_rmse

    return {
        "r2_obs": float(r2_obs),
        "rmse_obs": float(rmse_obs),
        "r2_null": r2_null.tolist(),
        "rmse_null": rmse_null.tolist(),
        "p_value_r2": float(p_r2),
        "p_value_rmse": float(p_rmse),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "hist_paths": hist_paths,
        "parallel_used": bool(use_parallel),
        "device_type": str(dev),
    }