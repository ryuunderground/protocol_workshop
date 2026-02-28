# lasso.py

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from joblib import Parallel, delayed

class LassoPreprocessor:
    def __init__(
        self,
        order_l=2,
        lam=None,
        scale=True,
        normalize=False,
        n_jobs=-1,
        top_k_parents: int | None = None,   # ✅ 추가
        allow_self_parent: bool = False,    # ✅ 추가
    ):
        self.order_l = order_l
        self.scale = scale
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.top_k_parents = top_k_parents
        self.allow_self_parent = allow_self_parent

        if lam is not None:
            self.alphas = [lam]
        else:
            self.alphas = np.logspace(-6, -1, 10)

        self.A = None
        self.genes = None
        self.n_genes = None

    def _scale_matrix(self, A):
        L, G, _ = A.shape
        A_scaled = np.zeros_like(A)
        for lag in range(L):
            for i in range(G):
                col = A[lag, :, i]
                maxv = col.max()
                if maxv > 0:
                    A_scaled[lag, :, i] = col / maxv
                else:
                    A_scaled[lag, :, i] = 0.0
        return A_scaled

    def _apply_topk_gating(self, A: np.ndarray) -> np.ndarray:
        """
        A: (L, G, G) where A[lag, parent, child]
        Keep only top_k_parents per (lag, child) among A>0.
        """
        if self.top_k_parents is None:
            return A

        L, G, _ = A.shape
        A2 = np.zeros_like(A)

        for lag in range(L):
            for child in range(G):
                scores = A[lag, :, child].copy()

                if not self.allow_self_parent:
                    scores[child] = 0.0

                # keep only positive candidates
                pos_idx = np.where(scores > 0)[0]
                if len(pos_idx) == 0:
                    continue

                # pick topK by score
                k = min(self.top_k_parents, len(pos_idx))
                top_idx = pos_idx[np.argsort(scores[pos_idx])[::-1][:k]]
                A2[lag, top_idx, child] = A[lag, top_idx, child]

        return A2

    def _create_lagged_matrix(self, values: np.ndarray, lag: int):
        if lag >= values.shape[1]:
            raise ValueError(f"Lag {lag} >= number of timepoints")
        return values[:, :-lag].T

    def _fit_single_target(self, i, df_proc_values, lag, alphas):
        X_lag = self._create_lagged_matrix(df_proc_values, lag)
        y = df_proc_values[i, lag:]

        best_coef = None
        best_loss = np.inf

        for lam in alphas:
            model = Lasso(alpha=lam, fit_intercept=True, max_iter=80000, tol=1e-3)
            model.fit(X_lag, y)
            y_pred = model.predict(X_lag)
            loss = np.mean((y - y_pred) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_coef = np.abs(model.coef_)

        return best_coef

    def fit(self, dataset):
        df = dataset.df
        self.genes = dataset.genes
        self.n_genes = dataset.n_genes

        vals = df.values
        if self.normalize:
            mean = vals.mean(axis=1, keepdims=True)
            std = vals.std(axis=1, keepdims=True) + 1e-8
            vals = (vals - mean) / std

        A = np.zeros((self.order_l, self.n_genes, self.n_genes))

        for lag in range(1, self.order_l + 1):
            print(f"[Lasso] lag = {lag}")
            coefs_list = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_single_target)(i, vals, lag, self.alphas)
                for i in range(self.n_genes)
            )
            for i, coef in enumerate(coefs_list):
                A[lag - 1, :, i] = coef

        if self.scale:
            A = self._scale_matrix(A)

        # ✅ 핵심: topK gating 적용
        A = self._apply_topk_gating(A)

        self.A = A
        return A

    def get_matrix(self):
        return self.A