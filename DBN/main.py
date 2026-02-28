# main_stability.py
from __future__ import annotations

import os
import csv
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import get_context
from typing import Dict, Tuple, List

from data_loader import DataLoader, Dataset
from lasso import LassoPreprocessor
from pearson import PearsonPreprocessor
from score import BICLPScorer
from anealing import SimulatedAnnealer
from graph import DBNGraph
from export import GraphExporter


EdgeKey = Tuple[str, str, int]  # (source_gene, target_gene, lag)

# candidates 타입(권장):
# - inter_candidates: Dict[int, Dict[int, List[int]]]
#   => inter_candidates[lag][target_idx] = [parent_idx, ...]   (lag는 1..order_l)
# - intra_candidates: Dict[int, List[int]]
#   => intra_candidates[target_idx] = [parent_idx, ...]

def build_inter_candidates_from_A(
    A: np.ndarray,
    top_k: int = 4,
    allow_self: bool = False,
) -> Dict[int, Dict[int, List[int]]]:
    """
    A: shape (order_l, n_genes, n_genes)
       A[lag-1, parent, target] in [0,1] (scaled abs lasso coef)
    return:
      inter_candidates[lag][target] = list(parent indices)  where lag in [1..order_l]
    """
    A = np.asarray(A)
    if A.ndim != 3:
        raise ValueError(f"A must be 3D (L,G,G). Got shape={A.shape}")

    L, G, G2 = A.shape
    if G != G2:
        raise ValueError(f"A must be (L,G,G). Got shape={A.shape}")

    inter_candidates: Dict[int, Dict[int, List[int]]] = {}

    for lag_idx in range(L):
        lag = lag_idx + 1
        inter_candidates[lag] = {}

        for tgt in range(G):
            scores = A[lag_idx, :, tgt].copy()

            # self-parent 제거(선택)
            if not allow_self:
                scores[tgt] = -np.inf

            # score>0인 후보만 우선
            pos_idx = np.where(scores > 0)[0]
            if pos_idx.size == 0:
                # 전부 0이면 fallback: top_k를 그냥 뽑아도 되지만,
                # 현재는 빈 리스트로 둬서 "가능한 inter parent 없음"으로 처리
                inter_candidates[lag][tgt] = []
                continue

            # 양수 후보 중 top_k
            pos_scores = scores[pos_idx]
            order = np.argsort(-pos_scores)  # desc
            chosen = pos_idx[order[: min(top_k, len(order))]]

            inter_candidates[lag][tgt] = chosen.tolist()

    return inter_candidates


def build_intra_candidates_from_C(
    C: np.ndarray,
    top_k: int = 6,
    allow_self: bool = False,
    use_abs: bool = True,
) -> Dict[int, List[int]]:
    """
    C: shape (n_genes, n_genes) Pearson correlation matrix (or similar)
       C[parent, target]
    return:
      intra_candidates[target] = list(parent indices)
    """
    C = np.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square 2D (G,G). Got shape={C.shape}")

    G = C.shape[0]
    intra_candidates: Dict[int, List[int]] = {}

    for tgt in range(G):
        scores = C[:, tgt].copy()
        if use_abs:
            scores = np.abs(scores)

        if not allow_self:
            scores[tgt] = -np.inf

        # 유효한 후보만 (nan/inf 제거)
        valid = np.isfinite(scores)
        if valid.sum() == 0:
            intra_candidates[tgt] = []
            continue

        idx = np.where(valid)[0]
        vals = scores[idx]
        order = np.argsort(-vals)  # desc
        chosen = idx[order[: min(top_k, len(order))]]

        # -inf 들어간 self는 자동 제외됨
        intra_candidates[tgt] = chosen.tolist()

    return intra_candidates

def extract_edges_named(graph: DBNGraph, genes: List[str]) -> List[EdgeKey]:
    n = len(genes)
    edges: List[EdgeKey] = []

    # intra
    for i in range(n):
        for j in range(n):
            if graph.G0[i, j]:
                edges.append((genes[i], genes[j], 0))

    # inter
    for lag in range(graph.order_l):
        for i in range(n):
            for j in range(n):
                if graph.Gt[lag, i, j]:
                    edges.append((genes[i], genes[j], lag + 1))

    return edges


def build_graph_from_edges(genes: List[str], order_l: int, edges: List[EdgeKey]) -> DBNGraph:
    idx = {g: i for i, g in enumerate(genes)}
    g = DBNGraph(n_genes=len(genes), order_l=order_l)
    for src, dst, lag in edges:
        g.add_edge(idx[src], idx[dst], lag)
    return g


def run_one_seed(args_tuple):
    seed, df, order_l, lasso_lam, sa_iter, max_parents = args_tuple

    dataset = Dataset(df)

    # 1️ Lasso
    lasso = LassoPreprocessor(
        order_l=order_l,
        lam=lasso_lam,
        normalize=True,
        top_k_parents=4,        # hard gating
        allow_self_parent=False
    )
    A = lasso.fit(dataset)

    # 2️ Pearson
    pear_gpu = PearsonPreprocessor()
    C = pear_gpu.fit(dataset)

    # 3️ 후보군 생성
    inter_candidates = build_inter_candidates_from_A(A)
    intra_candidates = build_intra_candidates_from_C(
        C,
        top_k=6,
        allow_self=False
    )

    # 4️ 점수 계산산
    scorer = BICLPScorer(dataset, A, C)

    # 5️ Annealer
    annealer = SimulatedAnnealer(
        scorer=scorer,
        n_genes=dataset.n_genes,
        order_l=order_l,
        T_init=10.0,
        T_min=1e-4,
        freezing_rate=0.995,
        max_iter=sa_iter,
        seed=seed,
        max_parents=max_parents,
        allow_self_loop=False,
        invalid_retry=30,
        inter_candidates=inter_candidates,
        intra_candidates=intra_candidates,
    )

    best_graph, best_score = annealer.run(verbose=False)

    edges = extract_edges_named(best_graph, dataset.genes)

    return {
        "seed": seed,
        "best_score": float(best_score),
        "n_edges": int(len(edges)),
        "edges": edges,
        "genes": dataset.genes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./dbn/results/k2/sa")
    parser.add_argument("--order_l", type=int, default=2)
    parser.add_argument("--lasso_lam", type=float, default=0.01)
    parser.add_argument("--sa_iter", type=int, default=1000)
    parser.add_argument("--max_parents", type=int, default=3)

    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--n_workers", type=int, default=6)

    # stability thresholds
    parser.add_argument("--thr_main", type=float, default=0.7)
    parser.add_argument("--thr_supp", type=float, nargs="*", default=[0.6, 0.8])

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== [1] Load CSV ===")
    loader = DataLoader(args.csv_path)
    df = loader.data
    if df is None:
        raise ValueError("CSV 파일을 불러오지 못했습니다.")
    if isinstance(df, dict):
        raise ValueError("csv_path는 단일 CSV여야 합니다. (현재는 디렉토리/다중 CSV로 인식됨)")

    dataset0 = Dataset(df)
    genes = dataset0.genes
    print(dataset0)

    print("\n=== [2] Multi-start SA (stability selection) ===")
    print(f"- output_dir: {args.output_dir}")
    print(f"- order_l: {args.order_l}")
    print(f"- lasso_lam: {args.lasso_lam}")
    print(f"- sa_iter: {args.sa_iter}")
    print(f"- max_parents: {args.max_parents}")
    print(f"- seeds: {args.seeds}")
    print(f"- n_workers: {args.n_workers}")

    # multiprocessing (macOS: spawn)
    ctx = get_context("spawn")
    pool = ctx.Pool(processes=args.n_workers)

    try:
        worker_args = [
            (s, df, args.order_l, args.lasso_lam, args.sa_iter, args.max_parents)
            for s in args.seeds
        ]
        results = pool.map(run_one_seed, worker_args)
    finally:
        pool.close()
        pool.join()

    # save per-seed edges & scores
    scores_rows = []
    all_edge_lists = []
    for r in results:
        seed = r["seed"]
        best_score = r["best_score"]
        n_edges = r["n_edges"]
        edges = r["edges"]
        all_edge_lists.append(edges)

        scores_rows.append({"seed": seed, "best_score": best_score, "n_edges": n_edges})

        # per-seed edge csv
        out_edge_path = os.path.join(args.output_dir, f"run_seed{seed}_edges.csv")
        with open(out_edge_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "target", "lag"])
            for (src, dst, lag) in edges:
                w.writerow([src, dst, lag])

    scores_df = pd.DataFrame(scores_rows).sort_values("best_score", ascending=False)
    scores_df.to_csv(os.path.join(args.output_dir, "multistart_scores.csv"), index=False)

    print("\nTop seeds by score:")
    print(scores_df.head(5))

    # edge frequency
    edge_counter = Counter()
    for edges in all_edge_lists:
        # IMPORTANT: count presence per-run (set) not multiplicity
        edge_counter.update(set(edges))

    S = len(all_edge_lists)
    freq_rows = []
    for (src, dst, lag), cnt in edge_counter.items():
        freq_rows.append({
            "source": src,
            "target": dst,
            "lag": lag,
            "count_runs": cnt,
            "freq": cnt / S
        })

    freq_df = pd.DataFrame(freq_rows).sort_values(["freq", "count_runs"], ascending=False)
    freq_df.to_csv(os.path.join(args.output_dir, "stability_edge_frequency.csv"), index=False)

    # consensus thresholds: main + supp
    thresholds = [args.thr_main] + [t for t in args.thr_supp if t != args.thr_main]
    thresholds = sorted(set(thresholds))

    exporter = GraphExporter(genes)

    for thr in thresholds:
        keep = freq_df[freq_df["freq"] >= thr]
        keep_edges = list(zip(keep["source"], keep["target"], keep["lag"]))

        consensus_graph = build_graph_from_edges(genes, args.order_l, keep_edges)

        # safety check
        if not consensus_graph.is_valid(max_parents=args.max_parents, allow_self_loop=False):
            print(f"[WARN] Consensus graph at thr={thr} violates constraints (unexpected). "
                  f"Edges kept={len(keep_edges)}. Consider raising thr or lowering density.")
        edge_path = os.path.join(args.output_dir, f"consensus_edges_thr{thr:.1f}.csv")
        graphml_path = os.path.join(args.output_dir, f"consensus_network_thr{thr:.1f}.graphml")

        exporter.save_edge_list(consensus_graph, edge_path)
        exporter.save_graphml(consensus_graph, graphml_path)

        print(f"[Saved] thr={thr:.1f} edges={len(keep_edges)}")
        print(" -", edge_path)
        print(" -", graphml_path)

    print("\nDone. Stability selection outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()