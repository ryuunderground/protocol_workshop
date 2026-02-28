import os
import numpy as np
import pandas as pd
from typing import Dict, Union, List
"""
Union[...]은 **“이 변수(또는 반환값)가 여러 타입 중 하나일 수 있다”**는 것을 명시적으로 선언하기 위한 타입 힌트(type hint) 입니다.
실행 결과를 바꾸는 기능은 없고, 의미·의도·사용 규칙을 사람과 도구에게 알려주는 역할을 합니다
"""

# ---------------- DataLoader ----------------

class DataLoader:
    """
    csv 파일을 읽고 df 또는 dict 형태로 반환하는 클래스
    """

    # path: csv 파일 혹은 csv 파일이 들어있는 디렉토리
    # self.data: 3 중 하나
    # 1. pd.DataFrame : 단일 csv를 읽었을 경우
    # 2. Dict[str, pd.DataFrame]: 여러개의 csv를 읽었을 경우
    # 3. None: 에러 났을 경우
    def __init__(self, path: str):
        self.path = path
        self.data: Union[pd.DataFrame, Dict[str, pd.DataFrame], None] = None
        self._load()

    # csv 파일 하나 읽어오는 함수
    def _load_csv_file(self, file_path: str) -> Union[pd.DataFrame, None]:
        # 입력된 경로가 .csv 확장자가 아닐 경우
        if not file_path.lower().endswith('.csv'):
            print(f"오류: '{file_path}'는 CSV 파일이 아닙니다.")
            return None
        # 입력된 경로의  csv 읽고 df 형태로 반환
        print(f"파일을 읽는 중...: '{file_path}'")
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"오류: '{file_path}' 파일을 읽는 중 에러 발생: {e}")
            return None

    # csv 파일들이 들어있는 디렉토리에서 모든 csv를 읽어오는 함수
    def _load_directory(self, dir_path: str) -> Union[Dict[str, pd.DataFrame], None]:
        print(f"디렉토리에서 csv 파일을 찾는 중... '{dir_path}'")
        try:
            # os.listdir(dir_path): dir_path 내부 모든 파일 목록을 가져옴
            # .csv로 끝나는 파일들을 sorted()로 정렬
            csv_files = sorted(
                [f for f in os.listdir(dir_path) if f.lower().endswith('.csv')]
            )
            # 디렉토리가 없을 시 에러
        except FileNotFoundError:
            print(f"오류: 디렉토리를 찾을 수 없습니다. -> '{dir_path}'")
            return None
        # csv 파일이 없을 시 에러
        if not csv_files:
            print(f"오류: '{dir_path}' 안에서 CSV 파일을 찾지 못했습니다.")
            return None
        # csv 파일 존재 시 data_frame 형태로 저장
        data_frames = {}
        for csv_file in csv_files:
            file_path = os.path.join(dir_path, csv_file)
            df = self._load_csv_file(file_path)
            if df is not None:
                # key_name: 확장자 제거 ex) abc.csv -> abc
                key_name = os.path.splitext(csv_file)[0]
                # 반환 형태:
                # {
                #     "inter": pd.DataFrame(...),
                #     "long":  pd.DataFrame(...),
                #     "short": pd.DataFrame(...)
                # }
                data_frames[key_name] = df
        return data_frames if data_frames else None

    def _load(self):
        # 경로 부재 시 에러
        if not os.path.exists(self.path):
            print(f"오류: 경로를 찾을 수 없습니다. -> '{self.path}'")
            return
        # 입력된 경로가 파일일 경우
        if os.path.isfile(self.path):
            self.data = self._load_csv_file(self.path)
        # 입력된 경로가 디렉토리일 경우
        elif os.path.isdir(self.path):
            self.data = self._load_directory(self.path)
        # 에러
        else:
            print(f"오류: '{self.path}'는 유효한 파일이나 폴더가 아닙니다.")


# ---------------- Dataset ----------------

class Dataset:
    """
    유전자 × 시간(Time-series) 데이터 구조 클래스
    - df: DataFrame (index = GeneName, columns = timepoints)
    - genes: 유전자 이름 리스트
    - timepoints: POD 시간대 리스트
    - time_values: POD1=1, POD3=3 같은 숫자 배열
    """
    # 입력값: DataFrame 하나
    def __init__(self, df: pd.DataFrame):
        # 열 이름으로 "GeneName"이 있을 경우 index로 지정
        if "GeneName" in df.columns:
            df = df.set_index("GeneName")

        self.df: pd.DataFrame = df.astype(float)
        self.genes: List[str] = list(self.df.index)
        self.n_genes: int = len(self.genes)
        self.timepoints: List[str] = list(self.df.columns)
        self.n_timepoints: int = len(self.timepoints)
        self.time_values: np.ndarray = self._parse_timepoints(self.timepoints)
        self.expression_matrix: np.ndarray = self.df.values
    # 시간 포인트 변환 ex POD1 -> 1
    def _parse_timepoints(self, cols: List[str]) -> np.ndarray:
        times = []
        for c in cols:
            if c.lower() == "control":
                times.append(0)
            elif c.upper().startswith("POD"):
                t = int(c.upper().replace("POD", ""))
                times.append(t)
            else:
                raise ValueError(f"시간 컬럼 이름을 해석할 수 없습니다: {c}")
        # 정수로 된 숫자들을 어레이 형태로 반환
        return np.array(times, dtype=int)
    """ 유전자 이름에 해당하는 발현량 추출
    ex)
    GeneName,Control,POD3,POD7
    TSPAN8,1.2,2.1,2.8
    ITGA8,0.5,1.0,1.4
    IFI44L,3.1,3.8,4.2

    expr = dataset.get_gene_expression("TSPAN8")
    print(expr)
    결과: [1.2 2.1 2.8]
    """

    # def get_gene_expression(self, gene: str) -> np.ndarray:
    #     return self.df.loc[gene].values

    # 주어진 유전자만 골라서 새로운 Dataset 생성
    """
    sub_ds = dataset.subset_genes(["TSPAN8", "IFI44L"])
    print(sub_ds)
    결과: Dataset(n_genes=2, n_timepoints=3)
    """
    # def subset_genes(self, gene_list: List[str]) -> "Dataset":
    #     new_df = self.df.loc[gene_list]
    #     return Dataset(new_df)
    # 새로운 Dataset 생성, 원본과 독립적으로 사용
    def copy(self) -> "Dataset":
        return Dataset(self.df.copy())
    # 요약 정보 출력
    def __repr__(self):
        return f"Dataset(n_genes={self.n_genes}, n_timepoints={self.n_timepoints})"