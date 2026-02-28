# graph.py
from __future__ import annotations
import numpy as np
import random
from typing import List, Tuple, Optional


class DBNGraph:
    """
    Dynamic Bayesian Network (DBN) graph for higher-order models.
    - G0: intra-slice adjacency (n × n)
    - Gt: inter-slice adjacency (L × n × n), where lag=1..L maps to index 0..L-1
    """

    def __init__(self, n_genes: int, order_l: int = 1):
        self.n_genes = n_genes
        self.order_l = order_l
        self.G0 = np.zeros((n_genes, n_genes), dtype=bool)
        self.Gt = np.zeros((order_l, n_genes, n_genes), dtype=bool)

    def copy(self) -> "DBNGraph":
        new = DBNGraph(self.n_genes, self.order_l)
        new.G0 = self.G0.copy()
        new.Gt = self.Gt.copy()
        return new

    # -------------------
    # Edge operations
    # -------------------
    def has_edge(self, src: int, dst: int, lag: int = 0) -> bool:
        if lag == 0:
            return bool(self.G0[src, dst])
        return bool(self.Gt[lag - 1, src, dst])

    def add_edge(self, src: int, dst: int, lag: int = 0) -> None:
        if lag == 0:
            self.G0[src, dst] = True
        else:
            self.Gt[lag - 1, src, dst] = True

    def delete_edge(self, src: int, dst: int, lag: int = 0) -> None:
        if lag == 0:
            self.G0[src, dst] = False
        else:
            self.Gt[lag - 1, src, dst] = False

    def reverse_edge(self, src: int, dst: int, lag: int = 0) -> None:
        # reverse only if edge exists
        if not self.has_edge(src, dst, lag):
            return
        self.delete_edge(src, dst, lag)
        self.add_edge(dst, src, lag)

    def change_parent(self, old: int, new: int, child: int, lag: int = 0) -> None:
        # If old->child not present, it's a no-op to avoid accidental creation
        if self.has_edge(old, child, lag):
            self.delete_edge(old, child, lag)
            self.add_edge(new, child, lag)

    def change_child(self, parent: int, old: int, new: int, lag: int = 0) -> None:
        if self.has_edge(parent, old, lag):
            self.delete_edge(parent, old, lag)
            self.add_edge(parent, new, lag)

    # graph.py 안에 DBNGraph 클래스에 추가
    # top_k 위함
    def list_edges(self):
        edges = []
        # intra
        for src in range(self.n_genes):
            for dst in range(self.n_genes):
                if self.G0[src, dst]:
                    edges.append((src, dst, 0))
        # inter
        for lag in range(self.order_l):
            for src in range(self.n_genes):
                for dst in range(self.n_genes):
                    if self.Gt[lag, src, dst]:
                        edges.append((src, dst, lag + 1))
        return edges

    def num_parents(self, child: int) -> int:
        return len(self.parents_of(child))

    # -------------------
    # Queries / constraints
    # -------------------
    def parents_of(self, gene_idx: int) -> List[Tuple[int, int]]:
        parents = []
        # intra-slice parents
        for j in range(self.n_genes):
            if self.G0[j, gene_idx]:
                parents.append((j, 0))
        # inter-slice parents
        for lag in range(self.order_l):
            for j in range(self.n_genes):
                if self.Gt[lag, j, gene_idx]:
                    parents.append((j, lag + 1))
        return parents

    def parent_count(self, gene_idx: int) -> int:
        return len(self.parents_of(gene_idx))

    def is_valid(self, max_parents: Optional[int] = None, allow_self_loop: bool = False) -> bool:
        # self-loop check
        if not allow_self_loop:
            if np.any(np.diag(self.G0)):
                return False
            for lag in range(self.order_l):
                if np.any(np.diag(self.Gt[lag])):
                    return False

        # max parents check
        if max_parents is not None:
            for i in range(self.n_genes):
                if self.parent_count(i) > max_parents:
                    return False
        return True

    def roll_to_grn(self) -> np.ndarray:
        grn = self.G0.copy()
        for lag in range(self.order_l):
            grn |= self.Gt[lag]
        return grn

    def __repr__(self):
        return f"DBNGraph(n_genes={self.n_genes}, order_l={self.order_l})"


# -------------------
# Operations
# -------------------

class BaseOperation:
    def apply(self, graph: DBNGraph) -> DBNGraph:
        raise NotImplementedError


class AddEdgeOp(BaseOperation):
    def __init__(self, src: int, dst: int, lag: int):
        self.src = src
        self.dst = dst
        self.lag = lag

    def apply(self, graph: DBNGraph) -> DBNGraph:
        g = graph.copy()
        g.add_edge(self.src, self.dst, self.lag)
        return g


class DeleteEdgeOp(BaseOperation):
    def __init__(self, src: int, dst: int, lag: int):
        self.src = src
        self.dst = dst
        self.lag = lag

    def apply(self, graph: DBNGraph) -> DBNGraph:
        g = graph.copy()
        g.delete_edge(self.src, self.dst, self.lag)
        return g


class ReverseEdgeOp(BaseOperation):
    def __init__(self, src: int, dst: int, lag: int):
        self.src = src
        self.dst = dst
        self.lag = lag

    def apply(self, graph: DBNGraph) -> DBNGraph:
        g = graph.copy()
        g.reverse_edge(self.src, self.dst, self.lag)
        return g


class ChangeParentOp(BaseOperation):
    def __init__(self, old: int, new: int, child: int, lag: int):
        self.old = old
        self.new = new
        self.child = child
        self.lag = lag

    def apply(self, graph: DBNGraph) -> DBNGraph:
        g = graph.copy()
        g.change_parent(self.old, self.new, self.child, self.lag)
        return g


class ChangeChildOp(BaseOperation):
    def __init__(self, parent: int, old: int, new: int, lag: int):
        self.parent = parent
        self.old = old
        self.new = new
        self.lag = lag

    def apply(self, graph: DBNGraph) -> DBNGraph:
        g = graph.copy()
        g.change_child(self.parent, self.old, self.new, self.lag)
        return g

# graph.py (OperationFactory만 교체)

import random
from typing import Dict, List, Optional

class OperationFactory:
    def __init__(
        self,
        n_genes: int,
        order_l: int,
        seed: int,
        inter_candidates: Optional[Dict[int, List[List[int]]]] = None,  # lag -> child -> parents
        intra_candidates: Optional[List[List[int]]] = None,             # child -> parents
        p_add: float = 0.45,
        p_del: float = 0.35,
        p_chg_parent: float = 0.20,
    ):
        self.n_genes = n_genes
        self.order_l = order_l
        self.rng = random.Random(seed)

        self.inter_candidates = inter_candidates
        self.intra_candidates = intra_candidates

        self.p_add = p_add
        self.p_del = p_del
        self.p_chg_parent = p_chg_parent

    def _pick_lag(self):
        return self.rng.randrange(0, self.order_l + 1)

    def _candidate_parents(self, child: int, lag: int):
        # 후보군이 없으면 fallback: 전체 허용
        if lag == 0:
            if self.intra_candidates is None:
                return list(range(self.n_genes))
            return self.intra_candidates[child]
        else:
            if self.inter_candidates is None:
                return list(range(self.n_genes))
            return self.inter_candidates[lag][child]

    def random_operation(self, graph, max_parents: int, allow_self_loop: bool):
        u = self.rng.random()

        if u < self.p_add:
            lag = self._pick_lag()
            child = self.rng.randrange(self.n_genes)
            cand = self._candidate_parents(child, lag)

            if not allow_self_loop:
                cand = [p for p in cand if p != child]

            # max_parents constraint
            if len(graph.parents_of(child)) >= max_parents:
                return self._delete_op(graph)

            self.rng.shuffle(cand)
            for parent in cand:
                if lag == 0:
                    if not graph.G0[parent, child]:
                        return AddEdgeOp(parent, child, 0)
                else:
                    if not graph.Gt[lag - 1, parent, child]:
                        return AddEdgeOp(parent, child, lag)

            return self._delete_op(graph)

        elif u < self.p_add + self.p_del:
            return self._delete_op(graph)

        else:
            edges = graph.list_edges()
            if not edges:
                return self._delete_op(graph)

            src, child, lag = self.rng.choice(edges)

            cand = self._candidate_parents(child, lag)
            if not allow_self_loop:
                cand = [p for p in cand if p != child]

            self.rng.shuffle(cand)
            for new_parent in cand:
                if new_parent == src:
                    continue
                if lag == 0:
                    if not graph.G0[new_parent, child]:
                        return ChangeParentOp(src, new_parent, child, 0)
                else:
                    if not graph.Gt[lag - 1, new_parent, child]:
                        return ChangeParentOp(src, new_parent, child, lag)

            return self._delete_op(graph)

    def _delete_op(self, graph):
        edges = graph.list_edges()
        if not edges:
            src = self.rng.randrange(self.n_genes)
            dst = self.rng.randrange(self.n_genes)
            lag = self._pick_lag()
            return AddEdgeOp(src, dst, lag)
        src, dst, lag = self.rng.choice(edges)
        return DeleteEdgeOp(src, dst, lag)