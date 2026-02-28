# score.py

from sklearn.linear_model import LinearRegression
from data_loader import Dataset
from graph import DBNGraph
import numpy as np

class BICLPScorer:
    """
    BIC-LP 점수 계산기
    - Gaussian likelihood
    - BIC penalty
    - A(라쏘) / C(피어슨) 기반 prior 점수
    """

    def __init__(self, dataset: Dataset, A_matrix, C_matrix, m=0.8, gamma=0.05):
        self.df = dataset.df
        self.genes = dataset.genes
        self.n_genes = dataset.n_genes
        self.time_values = dataset.time_values
        self.T = self.df.shape[1]

        self.A = A_matrix
        self.C = C_matrix
        self.m = m
        self.gamma = gamma

    def _log_likelihood_i(self, graph: DBNGraph, gene_i: int):
        parents = graph.parents_of(gene_i)
        Xi = self.df.values[gene_i]
        # 부모 없으면: 단순 Gaussian 모델
        if len(parents) == 0:
            # 분산 0 방지로 매우 작은 값 추가
             
            var = np.var(Xi) + 1e-8
            ll = -0.5 * self.T * np.log(2 * np.pi * var) \
                - 0.5 * np.sum((Xi - Xi.mean())**2) / var
            return ll
            
        # 부모 있는 경우: 회귀 설계
        X_reg = []
        y_reg = []
        for t in range(self.T):
            row = []
            lag_has_no_problem = True
            for (parent, lag) in parents:
                if t - lag < 0:
                    lag_has_no_problem = False
                    break
                row.append(self.df.values[parent, t - lag])
            if lag_has_no_problem:
                X_reg.append(row)
                y_reg.append(Xi[t])

        # 데이터가 너무 적으면, 이 그래프에서는 유효한 회귀 불가
        #    → 부모 없는 모델처럼 취급 (fallback)
        if len(y_reg) < 2:
            var = np.var(Xi) + 1e-8
            ll = -0.5 * self.T * np.log(2 * np.pi * var) \
                - 0.5 * np.sum((Xi - Xi.mean())**2) / var
            return ll

        X_reg = np.array(X_reg)
        y_reg = np.array(y_reg)
        # 가우시안 우도 회귀
        model = LinearRegression().fit(X_reg, y_reg)
        pred = model.predict(X_reg)
        residual = y_reg - pred
        var = np.var(residual) + 1e-8
        ll = -0.5 * len(y_reg) * np.log(2 * np.pi * var) \
            - 0.5 * np.sum(residual**2) / var
        return ll

    # 구조 복잡도 억제
    # 논문의 Score_BIC-LP Sigma 속 2번 째 항
    def _bic_penalty_i(self, graph: DBNGraph, gene_i: int):
        k = len(graph.parents_of(gene_i))
        return (k + 2) / 2 * np.log(self.T)

    # 논문의 Scroe_exist
    # lasso.py 에서 넘겨준 beta= A[lag, j,i]가 클 수록, 즉 라쏘 회귀 결과과 영향력이 클 수록 점수 증가
    def _score_exist(self, graph: DBNGraph):
        total = 0.0
        # inter-slice
        L = self.A.shape[0]
        for lag in range(L):
            for i in range(self.n_genes):
                parents = graph.parents_of(i)
                parents_lag = [p for p in parents if p[1] == lag + 1]
                for (j, _) in parents_lag:
                    beta = self.A[lag, j, i]
                    total += np.log(self.m * beta + self.gamma)

        # intra-slice
        for i in range(self.n_genes):
            parents0 = [p for p in graph.parents_of(i) if p[1] == 0]
            for (j, _) in parents0:
                cij = self.C[j, i]
                total += np.log(self.m * cij + self.gamma)
        return total
    
    # 논문의 score_nonexist
    # beta가 크면 있어야 할 것을 없앴다.
    # beta가 작으면 없어도 되는 걸 없앴다.
    def _score_nonexist(self, graph: DBNGraph):
        total = 0.0
        L = self.A.shape[0]

        # inter-slice
        for lag in range(L):
            for i in range(self.n_genes):
                parents = graph.parents_of(i)
                parents_lag = {p[0] for p in parents if p[1] == lag + 1}
                for j in range(self.n_genes):
                    if j not in parents_lag:
                        beta = self.A[lag, j, i]
                        total += np.log(self.m * (1 - beta) + self.gamma)

        # intra-slice
        for i in range(self.n_genes):
            parents0 = {p[0] for p in graph.parents_of(i) if p[1] == 0}
            for j in range(self.n_genes):
                if j not in parents0:
                    cij = self.C[j, i]
                    total += np.log(self.m * (1 - cij) + self.gamma)

        return total
    # 모든 항 결합
    def score(self, graph: DBNGraph):
        score = 0.0
        for i in range(self.n_genes):
            ll = self._log_likelihood_i(graph, i)
            bic_penalty = self._bic_penalty_i(graph, i)
            score += ll - bic_penalty

        scorebp = self._score_exist(graph) + self._score_nonexist(graph)
        return score + scorebp
