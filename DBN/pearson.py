import numpy as np
from data_loader import Dataset

class PearsonPreprocessor:
    """
    GPU 없이 NumPy 기반 Pearson 상관계수 계산
    - 유전자(row) 간 상관계수 → C matrix
    - [-1,1] Pearson r을 (r+1)/2로 [0,1] 선형 변환
    """

    def __init__(self, scale: bool = True):
        self.scale = scale
        self.C: np.ndarray = None
        self.genes = None
        self.n_genes = None

    def fit(self, dataset: Dataset) -> np.ndarray:
        self.genes = dataset.genes
        self.n_genes = dataset.n_genes

        arr = dataset.df.values  # (n_genes, T)
        
        # NumPy 기반 Pearson r 계산
        corr = np.corrcoef(arr)

        # r in [-1,1] → [0,1]
        corr = (corr + 1.0) / 2.0
        
        # diagonal 제거
        # self-loop 억제를 위해 0
        np.fill_diagonal(corr, 0.0)

        self.C = corr
        return self.C

    def get_matrix(self):
        return self.C

