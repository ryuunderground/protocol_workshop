# anealing.py

from graph import DBNGraph, OperationFactory
from score import BICLPScorer
import numpy as np
import random

class SimulatedAnnealer:
    """
    SA-based DBN structure learner.
    """

    def __init__(
        self,
        scorer: BICLPScorer,
        n_genes: int,
        order_l: int,
        initial_graph: DBNGraph = None,
        T_init: float = 5.0,
        T_min: float = 1e-4,
        freezing_rate: float = 0.98,
        max_iter: int = 2000,
        seed: int = 42,

        # main.py에서 넘기고 있는 옵션들
        max_parents: int = 3,
        allow_self_loop: bool = False,
        invalid_retry: int = 30,

        # 정의되지 않은 후보군 우회
        inter_candidates=None,
        intra_candidates=None,
    ):
        self.scorer = scorer
        self.n_genes = n_genes
        self.order_l = order_l

        self.T = T_init
        self.T_min = T_min
        self.freezing_rate = freezing_rate
        self.max_iter = max_iter

        self.seed = seed
        self.rng = random.Random(seed)

        self.max_parents = max_parents
        self.allow_self_loop = allow_self_loop
        self.invalid_retry = invalid_retry

        # 저장
        self.inter_candidates = inter_candidates
        self.intra_candidates = intra_candidates

        self.graph = initial_graph if initial_graph else DBNGraph(n_genes, order_l)

        self.op_factory = OperationFactory(
            n_genes=n_genes,
            order_l=order_l,
            seed=seed,
            inter_candidates=self.inter_candidates,
            intra_candidates=self.intra_candidates,
        )

        self.best_graph = self.graph.copy()
        self.best_score = -np.inf

    def acceptance_probability(self, old_score, new_score):
        if new_score > old_score:
            return 1.0
        return np.exp((new_score - old_score) / max(self.T, 1e-8))

    def step(self):
        old_score = self.scorer.score(self.graph)

        # 새 OperationFactory 시그니처에 맞춰 호출
        op = self.op_factory.random_operation(
            graph=self.graph,
            max_parents=self.max_parents,
            allow_self_loop=self.allow_self_loop,
        )

        new_graph = op.apply(self.graph)
        new_score = self.scorer.score(new_graph)

        p = self.acceptance_probability(old_score, new_score)
        if self.rng.random() < p:
            self.graph = new_graph

        if new_score > self.best_score:
            self.best_score = new_score
            self.best_graph = new_graph.copy()

    def run(self, verbose=True):
        iter_count = 0
        if verbose:
            print("Starting Simulated Annealing...")
            print(f"Initial T = {self.T}, stopping at T <= {self.T_min}")

        while self.T > self.T_min and iter_count < self.max_iter:
            self.step()
            self.T *= self.freezing_rate
            iter_count += 1

            if verbose and iter_count % 200 == 0:
                print(f"[Iter {iter_count}] T={self.T:.4f}  Best score={self.best_score:.4f}")

        if verbose:
            print("SA Finished.")
            print("Best score:", self.best_score)

        return self.best_graph, self.best_score