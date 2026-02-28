# data_io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import re
import pandas as pd
import torch

# PODX 같은 칼럼만 사용
POD_RE = re.compile(r"^POD(\d+)$", re.IGNORECASE)


@dataclass
class ExpressionSample:
    name: str                   # 샘플 이름 short, long 등등
    genes: List[str]            # length G, 유전자 순서 보존존
    t: torch.Tensor             # (N,) float32 시간축
    X: torch.Tensor             # (N, G) float32  (time-major) (time, gene) 구조조
    gene_to_idx: Dict[str, int] # gene name -> column index 매핑


@dataclass
class EdgeTable:
    # raw edge list
    source: List[str]
    target: List[str]
    lag: List[int]


def _parse_time_columns(df: pd.DataFrame) -> Tuple[List[str], torch.Tensor]:
    """
    Expect columns like: GeneName, POD1, POD3, ...
    Returns:
      - time_cols: ["POD1", "POD3", ...] in ascending POD order
      - t: tensor([1., 3., ...]) float32
    """
    time_cols = []
    pod_days = []
    for c in df.columns:
        if c == "GeneName":
            continue
        m = POD_RE.match(str(c).strip()) # 문자열 안전 처리
        if not m:
            raise ValueError(
                f"Unrecognized time column '{c}'. Expected POD<number> like POD1, POD3 ..."
            )
        d = int(m.group(1)) # POD 숫자 추출
        # 컬럼명과 실제 시간 분리 저장
        time_cols.append(c)
        pod_days.append(d)

    # sort by POD day
    order = sorted(range(len(pod_days)), key=lambda i: pod_days[i])
    time_cols_sorted = [time_cols[i] for i in order]
    t_sorted = torch.tensor([float(pod_days[i]) for i in order], dtype=torch.float32)
    return time_cols_sorted, t_sorted

# csv -> ExpressionSamople
def load_expression_csv(
    path: str,
    name: str,
    device: Optional[torch.device] = None,
) -> ExpressionSample:
    """
    Reads expression CSV with:
      GeneName,POD1,POD3,...
    Output X is (N,G) time-major float32 tensor.
    """
    df = pd.read_csv(path)
    if "GeneName" not in df.columns:
        raise ValueError(f"{path}: missing 'GeneName' column")

    # parse and sort time columns
    time_cols, t = _parse_time_columns(df)

    # genes 이름
    genes = df["GeneName"].astype(str).tolist()
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # matrix: rows=genes, cols=time -> transpose to (N,G)
    X_gene_time = df[time_cols].astype(float).values  # (G, N)
    X = torch.tensor(X_gene_time, dtype=torch.float32).T.contiguous()  # (N, G)

    if device is not None:
        t = t.to(device)
        X = X.to(device)

    return ExpressionSample(name=name, genes=genes, t=t, X=X, gene_to_idx=gene_to_idx)

# 엣지 csv 로드
def load_edge_csv(path: str) -> EdgeTable:
    """
    Reads edge CSV with:
      source,target,lag
    """
    df = pd.read_csv(path)
    # 컬럼 확인인
    for col in ["source", "target", "lag"]:
        if col not in df.columns:
            raise ValueError(f"{path}: missing '{col}' column")

    source = df["source"].astype(str).tolist()
    target = df["target"].astype(str).tolist()
    lag = df["lag"].astype(int).tolist()
    return EdgeTable(source=source, target=target, lag=lag)


def intersect_genes(samples: List[ExpressionSample]) -> List[str]:
    """
    Returns sorted list of genes common to all samples.
    Sorting is lexicographic for determinism.
    공통 유전자만 추출
    """
    if not samples:
        raise ValueError("samples is empty")
    # 유전자 순서 바뀜. 이후 재정렬
    common = set(samples[0].genes)
    for s in samples[1:]:
        common &= set(s.genes)
    # 유전자 1차 정렬
    genes = sorted(common)
    if len(genes) == 0:
        raise ValueError("No intersecting genes among samples")
    return genes

# 유전자 재정렬
def subset_sample_to_genes(sample: ExpressionSample, genes_keep: List[str]) -> ExpressionSample:
    """
    Reorders and subsets X to genes_keep (common gene list).
    각 샘플의 X를 "공통 유전자 순서"로 재정렬
    """
    idx = []
    for g in genes_keep:
        if g not in sample.gene_to_idx:
            raise ValueError(f"Sample {sample.name}: missing gene {g} during subset")
        # 인덱스 매핑
        idx.append(sample.gene_to_idx[g])
    # 불연속적 텐서 연속적으로 변환
    X_sub = sample.X[:, idx].contiguous()  # (N, G_common)
    gene_to_idx_new = {g: i for i, g in enumerate(genes_keep)}
    return ExpressionSample(
        name=sample.name,
        genes=genes_keep,
        t=sample.t,
        X=X_sub,
        gene_to_idx=gene_to_idx_new,
    )

# 공통 유전자에 속하지 않는 DBN 엣지 제거
def filter_edges_to_genes(edges: EdgeTable, genes_keep_set: set) -> EdgeTable:
    """
    Keeps edges whose source and target are both in genes_keep_set.
    """
    src2, tgt2, lag2 = [], [], []
    for s, t, k in zip(edges.source, edges.target, edges.lag):
        if (s in genes_keep_set) and (t in genes_keep_set):
            src2.append(s); tgt2.append(t); lag2.append(int(k))
    if len(src2) == 0:
        raise ValueError("After filtering, no edges remain (check gene intersection / edge genes).")
    return EdgeTable(source=src2, target=tgt2, lag=lag2)

# short + long edges -> 하나의 edge set
def merge_edge_tables(edge_tables: List[EdgeTable]) -> EdgeTable:
    """
    Simple concatenation, then de-duplicate identical (source,target,lag).
    """
    seen = set()
    src, tgt, lag = [], [], []
    for et in edge_tables:
        for s, t, k in zip(et.source, et.target, et.lag):
            key = (s, t, int(k))
            if key in seen:
                continue
            seen.add(key)
            src.append(s); tgt.append(t); lag.append(int(k))
    return EdgeTable(source=src, target=tgt, lag=lag)


def edges_to_index_by_lag(
    edges: EdgeTable,
    gene_to_idx: Dict[str, int],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Convert edges into index tensors grouped by lag:
      out[lag]["src"] -> (E_lag,) long
      out[lag]["dst"] -> (E_lag,) long
      병렬 연산을 위함
    """
    by_lag: Dict[int, List[Tuple[int,int]]] = {}
    for s, t, k in zip(edges.source, edges.target, edges.lag):
        k = int(k)
        si = gene_to_idx[s]
        ti = gene_to_idx[t]
        by_lag.setdefault(k, []).append((si, ti))

    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for k, pairs in by_lag.items():
        src_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        dst_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        out[k] = {"src": src_idx, "dst": dst_idx}
    return out
