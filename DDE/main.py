# main.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch

from module.data_io import (
    load_expression_csv, load_edge_csv,
    intersect_genes, subset_sample_to_genes,
    merge_edge_tables, filter_edges_to_genes,
    edges_to_index_by_lag
)
from module.time_delay import choose_delta_from_samples, DelayConfig
from module.train_joint import TrainConfig, train_joint_two_samples
from module.test_forecast import TestConfig, fit_history_and_forecast
from module.multistart import run_multistart, MultiStartConfig
from module.plot_gene_trajectories import plot_all_genes
from module.export_dde_latex import export_dde_equations_latex

from module.dde_rhs import SharedParams
from module.history import HistoryParam, HistoryGrid
from module.dde_solver import solve_dde_at_observation_times, SolverConfig, choose_internal_dt
from module.metrics import per_gene_metrics
from module.dde_validation import (
        time_permutation_test_inter,
        edge_rewiring_permutation_test
    )



# 설정값 고정
def parse_args():
    # 프로그램 실행 시 파라미터를 받아 처리를 간단히 할 수 있도록하는 라이브러리
    # ()안은 설명
    p = argparse.ArgumentParser("DBN→DDE pipeline (joint train + test history-fit forecast)")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
                help="device selection; on M1 recommend auto/mps")
    p.add_argument("--ms_workers", type=int, default=1,
                help="multistart workers (CPU only). if device is mps/cuda -> forced to 1 for safety")
    p.add_argument("--perm_workers", type=int, default=4,
                help="permutation workers (CPU spawn). recommended 2~6 on M1")

    # expression CSVs
    # --train_a_expr: 단기 발현양 파일 경로
    # --train_b_expr: 장기 발현양 파일 경로
    # --test_expr: 테스트 샘플 발현양 파일 경로
    p.add_argument("--train_a_expr", type=str, required=True, help="e.g., short_Bcell_proliferation.csv")
    p.add_argument("--train_b_expr", type=str, required=True, help="e.g., long_Bcell_proliferation.csv")
    p.add_argument("--test_expr", type=str, required=True, help="e.g., inter_max_log3.csv (or any test sample)")

    # edge CSVs (can be 1 or 2+)
    # --edge_csvs: DBN 결과로 얻은 엣지 파일 경로
    p.add_argument("--edge_csvs", type=str, nargs="+", required=True,
                   help="e.g., short_B_cell_proliferation_log3_edges.csv short_B_cell_proliferation_log3_edges.csv (merged then dedup)")

    # output
    # --out_dir: 결과 저장 경로
    p.add_argument("--out_dir", type=str, default="results_dde", help="output directory")

    # delta selection
    # --delta_mode: 델타값(불균등 시계열에서 대표 시간 간격) 결정
    # 기본은 median(중앙값), 평균(mean) 선택도 가능
    p.add_argument("--delta_mode", type=str, default="median", choices=["median", "mean"],
                   help="how to choose representative Δ from train observation gaps")

    # solver
    # DDE solver 내부 적분 step 제어
    # --dt_frac: 내부 dt = dt_frac * 중앙값 gap
    # --dt_max: 내부 dt 상한
    # --dt_min: 내부 dt 하한
    # --activation: DDE solver 활성화 함수
    p.add_argument("--dt_frac", type=float, default=0.10, help="internal dt = dt_frac * median_gap")
    p.add_argument("--dt_max", type=float, default=1.0, help="cap for internal dt")
    p.add_argument("--dt_min", type=float, default=1e-3, help="floor for internal dt")
    p.add_argument("--activation", type=str, default="tanh", choices=["tanh", "identity", "softsign"])

    # history
    p.add_argument("--M_hist", type=int, default=12, help="#history knots")

    # training
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr_shared", type=float, default=1e-2)
    p.add_argument("--lr_hist", type=float, default=1e-2)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=80)

    # testing (history fit)
    p.add_argument("--K_fit", type=int, default=2, help="use first K points to fit test history only")
    p.add_argument("--iters_hist", type=int, default=800)
    p.add_argument("--lr_hist_test", type=float, default=1e-2)

    # multistart
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="multi-start seeds")
    p.add_argument("--gpu_ids", type=int, nargs="*", default=None,
                   help="optional multi-GPU parallel: e.g., --gpu_ids 0 1 2 ; if omitted -> sequential")

    return p.parse_args()

def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# 결과 저장 경로 존재 확인
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# 예측 결과 저장
def save_prediction_csv(sample_name: str, t: torch.Tensor, Z_pred: torch.Tensor, genes: List[str], out_path: Path):
    """
    Save predictions as CSV with columns:
      Time, <gene1>, <gene2>, ...
    Time is in days (float).
    """
    t_cpu = t.detach().cpu().numpy()
    Z_cpu = Z_pred.detach().cpu().numpy()

    df = pd.DataFrame(Z_cpu, columns=genes)
    df.insert(0, "Time", t_cpu)
    df.to_csv(out_path, index=False)

# 예측 성능 평가
def save_metrics_csv(rmse_g: torch.Tensor, r2_g: torch.Tensor, genes: List[str], out_path: Path):
    df = pd.DataFrame({
        "GeneName": genes,
        "RMSE": rmse_g.detach().cpu().numpy(),
        "R2": r2_g.detach().cpu().numpy(),
    })
    df.to_csv(out_path, index=False)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = pick_device(args.device)
    print("[Info] device:", device)

    # 안전장치: MPS/CUDA에서는 multistart 병렬 금지 (subprocess에서 GPU 쓰면 터질 확률 큼)
    if device.type in ("mps", "cuda") and args.ms_workers > 1:
        print("[WARN] device is GPU(MPS/CUDA). For safety, forcing ms_workers=1.")
        args.ms_workers = 1

    # -------------------------
    # STEP A: Load expression
    # -------------------------
    train_a = load_expression_csv(args.train_a_expr, name=Path(args.train_a_expr).stem, device=device)
    train_b = load_expression_csv(args.train_b_expr, name=Path(args.train_b_expr).stem, device=device)
    test_s  = load_expression_csv(args.test_expr,   name=Path(args.test_expr).stem,   device=device)

    # -------------------------
    # STEP B: Gene intersection (all 3)
    # -------------------------
    common_genes = intersect_genes([train_a, train_b, test_s])
    print("[Info] common genes:", len(common_genes))

    train_a = subset_sample_to_genes(train_a, common_genes)
    train_b = subset_sample_to_genes(train_b, common_genes)
    test_s  = subset_sample_to_genes(test_s,  common_genes)

    # -------------------------
    # STEP C: Load + merge edges, then filter to gene intersection
    # -------------------------
    edge_tables = [load_edge_csv(p) for p in args.edge_csvs]
    edges_merged = merge_edge_tables(edge_tables)
    edges_filt = filter_edges_to_genes(edges_merged, set(common_genes))
    print("[Info] edges after merge+filter:", len(edges_filt.source))

    edge_idx_by_lag = edges_to_index_by_lag(edges_filt, train_a.gene_to_idx)
    lags = sorted(edge_idx_by_lag.keys())
    print("[Info] lags:", lags)
    for k in lags:
        print(f"  - lag {k}: E={edge_idx_by_lag[k]['src'].numel()}")

    # -------------------------
    # STEP D: Choose Δ and build delays
    # -------------------------
    # 델타값 결정: 실제 실험 시계열이 불균등 간격이나 DBN lag는 정수 step
    # 따라서 관측 간격의 대표값을 delta로 설정
    # tau_k = k* delta
    # k=0,1,2 몇 단계 전 관측값까지 현재에 영향을 끼칠 것인가
    # tau: 시간 지연
    delta = choose_delta_from_samples([train_a.t, train_b.t], mode=args.delta_mode)
    delay_cfg = DelayConfig(delta=delta, lags=lags)
    tau_by_lag = delay_cfg.tau_by_lag()
    tau_max = delay_cfg.tau_max()

    print(f"[Info] Δ({args.delta_mode}) = {delta:.6g} days")
    print(f"[Info] tau_max = {tau_max:.6g} days")
    print("[Info] tau_by_lag:", tau_by_lag)

    # -------------------------
    # STEP E: Build configs
    # -------------------------
    solver_cfg = SolverConfig(
        dt_frac_of_median_gap=args.dt_frac,
        dt_max=args.dt_max,
        dt_min=args.dt_min,
        activation=args.activation,
        store_every_step=True,
    )

    # -------------------------
    # FIXED INTERNAL DT (IMPORTANT)
    # -------------------------

    fixed_dt = choose_internal_dt(train_a.t, solver_cfg)
    solver_cfg.fixed_dt = fixed_dt

    print(f"[Info] fixed internal dt = {fixed_dt:.6g}")


    train_cfg = TrainConfig(
        lr_shared=args.lr_shared,
        lr_hist=args.lr_hist,
        epochs=args.epochs,
        sigma=args.sigma,
        M_hist=args.M_hist,
        solver_cfg=solver_cfg,  # avoid circular imports
        grad_clip=args.grad_clip,
        patience=args.patience,
    )

    test_cfg = TestConfig(
        K_fit=args.K_fit,
        iters_hist=args.iters_hist,
        lr_hist=args.lr_hist_test,
        sigma=args.sigma,
        M_hist=args.M_hist,
        solver_cfg=solver_cfg,
        grad_clip=args.grad_clip,
    )

    # Save run config
    run_config = {
        "train_a_expr": args.train_a_expr,
        "train_b_expr": args.train_b_expr,
        "test_expr": args.test_expr,
        "edge_csvs": args.edge_csvs,
        "common_genes_count": len(common_genes),
        "edges_count": len(edges_filt.source),
        "lags": lags,
        "delta_mode": args.delta_mode,
        "delta_days": delta,
        "tau_by_lag_days": tau_by_lag,
        "tau_max_days": tau_max,
        "solver_cfg": vars(solver_cfg),
        "train_cfg": {
            "lr_shared": args.lr_shared,
            "lr_hist": args.lr_hist,
            "epochs": args.epochs,
            "sigma": args.sigma,
            "M_hist": args.M_hist,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
        },
        "test_cfg": {
            "K_fit": args.K_fit,
            "iters_hist": args.iters_hist,
            "lr_hist_test": args.lr_hist_test,
            "sigma": args.sigma,
            "M_hist": args.M_hist,
        },
        "seeds": args.seeds,
        "gpu_ids": args.gpu_ids,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # -------------------------
    # STEP F: Run multistart (sequential by default; multi-GPU optional)
    # -------------------------
    ms_cfg = MultiStartConfig(seeds=args.seeds, gpu_ids=None)

    results = run_multistart(
        train_a=train_a, train_b=train_b, test_sample=test_s,
        edge_idx_by_lag=edge_idx_by_lag,
        tau_by_lag=tau_by_lag,
        tau_max=tau_max,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        ms_cfg=ms_cfg,
        device=device,
    )

    # -------------------------
    # STEP G: Select best seed (future-metrics priority, then all-metrics)
    # -------------------------
    def score(r: Dict[str, Any]) -> float:
        # lower is better: use future RMSE primarily (after K)
        m = r["test_metrics_future"]
        if m["rmse"] == m["rmse"]:  # not NaN
            return float(m["rmse"])
        return float(r["test_metrics_all"]["rmse"])

    results_sorted = sorted(results, key=score)
    best = results_sorted[0]

    # Save summary CSV for all starts
    rows = []
    for r in results_sorted:
        rows.append({
            "seed": r["seed"],
            "train_best_loss": r["train_info"]["best_loss"],
            "train_best_epoch": r["train_info"]["best_epoch"],
            "test_fit_loss_best": r["loss_fit_best"],
            "test_rmse_all": r["test_metrics_all"]["rmse"],
            "test_r2_all": r["test_metrics_all"]["r2"],
            "test_rmse_future": r["test_metrics_future"]["rmse"],
            "test_r2_future": r["test_metrics_future"]["r2"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "multistart_summary.csv", index=False)

    print("[Info] BEST seed:", best["seed"])
    print("[Info] BEST test future metrics:", best["test_metrics_future"])
    print("[Info] BEST test all metrics:", best["test_metrics_all"])

    # -------------------------
    # STEP H: Re-run best model on current device to export predictions/metrics cleanly
    #         (Because multi-GPU workers may have produced state_dicts on different devices.)
    # -------------------------
    # Rebuild shared + histories on this device and load state_dicts


    # ensure tensors are on device
    train_a.t, train_a.X = train_a.t.to(device), train_a.X.to(device)
    train_b.t, train_b.X = train_b.t.to(device), train_b.X.to(device)
    test_s.t,  test_s.X  = test_s.t.to(device),  test_s.X.to(device)
    edge_idx_local = {k: {"src": d["src"].to(device), "dst": d["dst"].to(device)} for k, d in edge_idx_by_lag.items()}

    shared = SharedParams(G=len(common_genes), edge_idx_by_lag=edge_idx_local).to(device)
    shared.load_state_dict(best["shared_state"])
    print("[DEBUG] shared.named_parameters():", [n for n,_ in shared.named_parameters()])
    print("[DEBUG] shared.state_dict keys:", list(shared.state_dict().keys())[:20])

    grid = HistoryGrid(M=args.M_hist)
    hist_a = HistoryParam(G=len(common_genes), grid=grid, init_from_first_obs=train_a.X[0]).to(device)
    hist_b = HistoryParam(G=len(common_genes), grid=grid, init_from_first_obs=train_b.X[0]).to(device)
    hist_a.load_state_dict(best["hist_train_state"][train_a.name])
    hist_b.load_state_dict(best["hist_train_state"][train_b.name])

    # For test history, load best-fitted
    hist_test = HistoryParam(G=len(common_genes), grid=grid, init_from_first_obs=test_s.X[0]).to(device)
    hist_test.load_state_dict(best["hist_test_state"])

    solver_cfg_obj = SolverConfig(
        dt_frac_of_median_gap=args.dt_frac,
        dt_max=args.dt_max,
        dt_min=args.dt_min,
        activation=args.activation,
        store_every_step=True,
    )
    # 반드시 동일 dt 사용
    solver_cfg_obj.fixed_dt = fixed_dt

    # predictions
    with torch.no_grad():
        Z_train_a = solve_dde_at_observation_times(train_a.t, shared, hist_a, tau_by_lag, tau_max, solver_cfg_obj)
        Z_train_b = solve_dde_at_observation_times(train_b.t, shared, hist_b, tau_by_lag, tau_max, solver_cfg_obj)
        Z_test    = solve_dde_at_observation_times(test_s.t,  shared, hist_test, tau_by_lag, tau_max, solver_cfg_obj)

    # Save predictions
    pred_dir = out_dir / "predictions"
    ensure_dir(pred_dir)
    save_prediction_csv(train_a.name, train_a.t, Z_train_a, common_genes, pred_dir / f"{train_a.name}_pred.csv")
    save_prediction_csv(train_b.name, train_b.t, Z_train_b, common_genes, pred_dir / f"{train_b.name}_pred.csv")
    save_prediction_csv(test_s.name,  test_s.t,  Z_test,    common_genes, pred_dir / f"{test_s.name}_pred.csv")

    # Save gene-level metrics on test
    rmse_g, r2_g = per_gene_metrics(test_s.X, Z_test)
    metrics_dir = out_dir / "metrics"
    ensure_dir(metrics_dir)
    save_metrics_csv(rmse_g, r2_g, common_genes, metrics_dir / "test_gene_level_metrics.csv")

    # Save checkpoints
    ckpt = {
        "best_seed": best["seed"],
        "shared_state": shared.state_dict(),
        "hist_train_state": {train_a.name: hist_a.state_dict(), train_b.name: hist_b.state_dict()},
        "hist_test_state": hist_test.state_dict(),
        "common_genes": common_genes,
        "tau_by_lag": tau_by_lag,
        "tau_max": tau_max,
        "delta": delta,
        "args": vars(args),
        "fixed_dt": float(fixed_dt),
    }
    torch.save(ckpt, out_dir / "best_checkpoint.pt")

    print("[Done] Saved:")
    print(" -", out_dir / "run_config.json")
    print(" -", out_dir / "multistart_summary.csv")
    print(" -", pred_dir)
    print(" -", metrics_dir / "test_gene_level_metrics.csv")
    print(" -", out_dir / "best_checkpoint.pt")

    samples = {
        train_a.name: train_a,
        train_b.name: train_b,
        test_s.name:  test_s,
    }

    preds = {
        train_a.name: Z_train_a,
        train_b.name: Z_train_b,
        test_s.name:  Z_test,
    }

    # ============================================================
    # VALIDATION: Time permutation + Edge rewiring (multi-seed)
    # ============================================================

    validation_dir = out_dir / "validation"
    ensure_dir(validation_dir)

    def predict_fn(model_obj, sample):
        dev = next(model_obj.parameters()).device

        class _Tmp:
            pass

        tmp = _Tmp()
        tmp.t = sample.t.to(dev)
        tmp.X = sample.X.to(dev)

        with torch.no_grad():
            Z = solve_dde_at_observation_times(
                tmp.t,
                model_obj,
                hist_test,
                tau_by_lag,
                tau_max,
                solver_cfg_obj,
            )
        return Z.detach().cpu().numpy()

    # ---------- 1) Time permutation ----------
    all_time = []
    for s in args.seeds:
        print(f"\n[Validation] Time permutation (seed={s})")

        res = time_permutation_test_inter(
            model_predict_fn=predict_fn,
            model_obj=shared,
            test_sample=test_s,
            n_perm=200,
            seed=s,
            out_prefix=str(validation_dir / f"time_perm_seed{s}"),
            plot=True,
            perm_workers=args.perm_workers,
            device_type=device.type,
        )

        (validation_dir / f"time_permutation_seed{s}.json").write_text(
            json.dumps(res, indent=2),
            encoding="utf-8"
        )

        print("[Validation] p(R2):", res["p_value_r2"],
              " p(RMSE):", res["p_value_rmse"])

        all_time.append(res)

    # ---------- 2) Edge rewiring ----------
    all_edge = []
    for s in args.seeds:
        print(f"\n[Validation] Edge rewiring (seed={s})")

        res = edge_rewiring_permutation_test(
            train_joint_fn=train_joint_two_samples,
            forecast_fn=fit_history_and_forecast,
            train_a=train_a,
            train_b=train_b,
            test_sample=test_s,
            edge_idx_by_lag=edge_idx_local,
            tau_by_lag=tau_by_lag,
            tau_max=tau_max,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            n_perm=100,
            seed=s,
            out_prefix=str(validation_dir / f"edge_rewire_seed{s}"),
            plot=True,
            perm_workers=args.perm_workers,
            device_type=device.type,
        )

        (validation_dir / f"edge_rewire_seed{s}.json").write_text(
            json.dumps(res, indent=2),
            encoding="utf-8"
        )

        print("[Validation] p(R2):", res["p_value_r2"],
              " p(RMSE):", res["p_value_rmse"])

        all_edge.append(res)

    # ---------- 3) aggregate save ----------
    (validation_dir / "time_permutation_all.json").write_text(
        json.dumps(all_time, indent=2),
        encoding="utf-8"
    )

    (validation_dir / "edge_rewire_permutation_all.json").write_text(
        json.dumps(all_edge, indent=2),
        encoding="utf-8"
    )

    print("\n[Validation completed]")
    print("Results saved in:", validation_dir)

    # -------------------------
    # STEP I: Forward simulation (extend time axis) for ALL samples
    # -------------------------

    FUTURE_DAYS = 7.0  # 얼마나 미래까지 볼지 (days)

    def make_future(sample, history):
        # 내부 dt는 solver에서 쓰는 값과 동일하게 맞춤
        dt_internal = solver_cfg_obj.fixed_dt

        # 미래 시간 벡터 생성 (균등 간격)
        t_last = sample.t[-1].item()
        t_future = torch.arange(
            t_last + dt_internal,
            t_last + FUTURE_DAYS + 1e-8,
            dt_internal,
            device=device,
        )

        # 관측 + 미래 시간 합치기
        t_extended = torch.cat([sample.t, t_future], dim=0)

        # extended timeline에서 DDE solve
        with torch.no_grad():
            Z_extended = solve_dde_at_observation_times(
                t_obs=t_extended,
                shared=shared,
                history=history,
                tau_by_lag=tau_by_lag,
                tau_max=tau_max,
                cfg=solver_cfg_obj,
            )

        # 미래 구간만 분리
        Z_future = Z_extended[sample.t.shape[0]:]  # (T_future, G)

        return (
            t_future.detach().cpu().numpy(),
            Z_future.detach().cpu().numpy(),
        )

    future_preds = {
        train_a.name: make_future(train_a, hist_a),
        train_b.name: make_future(train_b, hist_b),
        test_s.name:  make_future(test_s,  hist_test),
    }

    def solve_at_t_and_tplus(sample, history, horizon_days: float):
        """
        Returns:
        Z_at_t:     (T, G) prediction at observed times t
        Z_at_tplus: (T, G) prediction at times (t + horizon_days)
        """
        t_obs = sample.t
        t_plus = t_obs + horizon_days

        # solver는 정렬된 time vector를 기대하므로 합쳐서 unique+sort
        t_query = torch.cat([t_obs, t_plus], dim=0)
        t_query = torch.unique(t_query)  # unique (order may change)
        t_query, _ = torch.sort(t_query)

        with torch.no_grad():
            Z_query = solve_dde_at_observation_times(
                t_obs=t_query,
                shared=shared,
                history=history,
                tau_by_lag=tau_by_lag,
                tau_max=tau_max,
                cfg=solver_cfg_obj,
            )

        # t_query에서 t_obs, t_plus에 해당하는 인덱스를 찾아서 뽑기
        # (시간점이 정확히 일치하므로 equality 매칭 사용 가능)
        # 단, 부동소수 오차가 걱정되면 round/quantize 전략을 쓰면 됨.
        def index_map(t_from, t_to):
            # returns indices in t_from that match each value in t_to
            # assumes exact matches exist
            idx = torch.searchsorted(t_from, t_to)
            return idx

        idx_obs = index_map(t_query, t_obs)
        idx_plus = index_map(t_query, t_plus)

        Z_at_t = Z_query[idx_obs]
        Z_at_tplus = Z_query[idx_plus]
        return Z_at_t, Z_at_tplus


    def export_obs_pred_future7_csv(out_csv_path: Path):
        rows = []
        # sample별 history 파라미터 매핑
        sample_to_hist = {
            train_a.name: hist_a,
            train_b.name: hist_b,
            test_s.name:  hist_test,
        }

        for name, sample in samples.items():
            hist = sample_to_hist[name]

            # 이미 계산해둔 관측시점 예측(안전/일관성): preds[name]
            Z_pred_obs = preds[name]  # (T, G)

            # 7일 뒤 예측은 solver로 계산
            Z_obs_check, Z_pred_plus7 = solve_at_t_and_tplus(sample, hist, FUTURE_DAYS)

            # 혹시 preds[name]와 Z_obs_check가 다르면(수치적 이유) 무엇을 쓸지 선택 필요.
            # 여기서는 "기존 preds[name]"를 기준으로 저장(평가와 동일하게 유지)
            t_np = sample.t.detach().cpu().numpy()
            X_np = sample.X.detach().cpu().numpy()
            Z_np = Z_pred_obs.detach().cpu().numpy()
            Zp7_np = Z_pred_plus7.detach().cpu().numpy()

            for ti in range(len(t_np)):
                for gi, gname in enumerate(common_genes):
                    rows.append({
                        "Sample": name,
                        "Gene": gname,
                        "Time": float(t_np[ti]),
                        "Observed": float(X_np[ti, gi]),
                        "Predicted": float(Z_np[ti, gi]),
                        "Predicted_tplus7": float(Zp7_np[ti, gi]),
                        "Horizon_days": FUTURE_DAYS,
                    })

        df = pd.DataFrame(rows)
        df.to_csv(out_csv_path, index=False)


    # ---- 저장 실행 위치: plot_all_genes 호출 직전/직후 아무데나 가능 ----
    tables_dir = out_dir / "tables"
    ensure_dir(tables_dir)
    export_obs_pred_future7_csv(tables_dir / "obs_pred_predplus7_long.csv")
    print("[Done] Saved:", tables_dir / "obs_pred_predplus7_long.csv")


    plot_all_genes(
        samples=samples,
        preds=preds,
        gene_names=common_genes,
        out_dir=out_dir / "plots",
        max_genes=None,
        K_fit=args.K_fit,
        future=future_preds,
    )



    export_dde_equations_latex(
        shared=shared,
        gene_names=common_genes,
        edge_idx_by_lag=edge_idx_local,
        tau_by_lag=tau_by_lag,
        out_path=out_dir / "dde_equations.tex.txt",
        activation=args.activation,
        weight_threshold=0.0   # or e.g. 1e-3
    )


if __name__ == "__main__":
    main()
