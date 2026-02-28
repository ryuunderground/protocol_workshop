## protocol_workshop

DBN(Dynamic Bayesian Network)으로 **시계열 발현 데이터에서 후보 네트워크를 추정**하고, 그 네트워크를 고정한 채 DDE(Delay Differential Equation) 기반 연속시간 모델로 **학습/예측/검증(permutation test)** 하는 파이프라인입니다.

### 폴더 구조
- **`DBN/`**: 시계열 발현량 CSV → 네트워크(엣지 리스트/GraphML) 추정 + stability selection
- **`DDE/`**: (DBN에서 얻은) 엣지 CSV + 발현량 CSV → DDE 파라미터 학습/예측 + time/edge permutation validation

### 실행 환경
- **Python 3.10+** (타입힌트 `int | None` 사용)
- **주요 의존 패키지(예시)**: `numpy`, `pandas`, `scikit-learn`, `joblib`, `networkx`, `torch`, `matplotlib`
  - Torch 설치는 환경(CPU/CUDA)에 따라 달라서, 필요하면 PyTorch 공식 설치 가이드를 참고하세요.

---

## 데이터 형식

### DBN 입력 CSV (`DBN/main.py --csv_path`)
- **행**: 유전자 (CSV에 `GeneName` 컬럼이 있으면 자동으로 index로 사용)
- **열**: 시간 포인트
  - `Control`(0으로 처리) 또는 `POD1`, `POD3` 같은 형태를 지원합니다.

### DDE 입력 CSV (`DDE/main.py --train_a_expr/--train_b_expr/--test_expr`)
- **컬럼**: `GeneName, POD1, POD3, ...` (반드시 `POD<number>` 형식)
- **주의**: DDE 쪽 로더는 `Control` 같은 컬럼을 허용하지 않습니다. (POD 컬럼만 인식)

### 엣지 CSV 형식 (DBN 출력 = DDE 입력)
- **헤더**: `source,target,lag`
- **lag**:
  - `0` = intra-slice(동일 시점)
  - `1..order_l` = inter-slice(지연 효과)

---

## 1) DBN 실행 (네트워크 추정 + stability selection)

### 엔트리포인트
- **`DBN/main.py`**

### 주요 인자
- **`--csv_path`**: 입력 발현량 CSV
- **`--output_dir`**: 결과 저장 폴더
- **`--order_l`**: 최대 lag(지연 단계) 수
- **`--lasso_lam`**: Lasso 규제 계수(라쏘 기반 prior 생성에 사용)
- **`--sa_iter`**: simulated annealing 반복 횟수
- **`--max_parents`**: 노드당 최대 부모 수 제약
- **`--seeds` / `--n_workers`**: multi-start seed 목록 및 병렬 워커 수
- **`--thr_main` / `--thr_supp`**: stability selection 임계값(예: 0.6/0.7/0.8)

### 출력물(예: `--output_dir` 아래)
- **seed별 결과**
  - `run_seed{seed}_edges.csv`: 해당 seed에서 찾은 best 그래프의 엣지 리스트
  - `multistart_scores.csv`: seed별 best_score 등 요약
- **stability selection 결과**
  - `stability_edge_frequency.csv`: 엣지별 출현 횟수/비율(`freq = count_runs / #seeds`)
  - `consensus_edges_thr0.6.csv`, `consensus_edges_thr0.7.csv`, `consensus_edges_thr0.8.csv` ...
  - `consensus_network_thr0.6.graphml` 등(GraphML)

### 실행 예시(단일 라인)

```bash
python DBN/main.py --csv_path .\data\renew_inputs\inter_DBN_input.csv --output_dir .\dbn_results --order_l 2 --lasso_lam 0.01 --sa_iter 5000 --max_parents 3 --seeds 0 1 2 3 4 --n_workers 5 --thr_main 0.7 --thr_supp 0.6 0.8
```

이후 DDE에는 보통 `consensus_edges_thr0.7.csv` 같은 **consensus 엣지 파일**을 넣어 사용합니다.

---

## 2) DDE 실행 (학습/예측 + permutation validation)

### 엔트리포인트
- **`DDE/main.py`**

### 처리 흐름(코드 기준)
- **발현 데이터 로드**: `module/data_io.py: load_expression_csv`
- **엣지 로드/병합/필터링**:
  - `load_edge_csv` → `merge_edge_tables`(중복 제거) → `filter_edges_to_genes`(공통 유전자만 유지)
- **지연 시간 설정**:
  - `choose_delta_from_samples`로 대표 Δ 선택 → `tau_by_lag[k] = k * Δ`
- **multistart 학습/평가**:
  - `module/multistart.py: run_multistart`로 seed별 학습 후 성능 비교
  - 기본 선택 기준은 `test_metrics_future.rmse`(가능하면) 우선입니다.
- **best seed 재구성 후 내보내기**:
  - `predictions/`, `metrics/`, `best_checkpoint.pt`, `dde_equations.tex.txt` 등 저장
- **Validation(permutation test)**:
  - `module/dde_validation.py`의 `time_permutation_test_inter`, `edge_rewiring_permutation_test`

### 주요 인자
- **데이터/엣지**
  - `--train_a_expr`, `--train_b_expr`, `--test_expr`
  - `--edge_csvs`(여러 개 가능): DBN에서 나온 엣지 CSV들을 넣으면 병합+중복제거 후 사용
- **출력**
  - `--out_dir`
- **학습/평가**
  - `--seeds`(multistart)
  - `--K_fit`(test history fit에 사용할 관측점 개수)
- **병렬/디바이스**
  - `--device` (`auto|cpu|mps|cuda`)
  - `--ms_workers`(multistart CPU 병렬, GPU에서는 안전상 1로 강제)
  - `--perm_workers`(permutation test CPU 병렬)

### 출력물(예: `--out_dir` 아래)
- `multistart_summary.csv`
- `predictions/*.csv`
- `metrics/test_gene_level_metrics.csv`
- `validation/`
  - `time_permutation_seed*.json`, `edge_rewire_seed*.json`
  - `time_permutation_all.json`, `edge_rewire_permutation_all.json`
  - `*_r2_hist.png`, `*_rmse_hist.png`
- `best_checkpoint.pt`
- `dde_equations.tex.txt`

### 실행 예시(단일 라인)

```bash
python DDE/main.py --device cpu --ms_workers 1 --perm_workers 4 --train_a_expr ..\data\renew_inputs\short_input.csv --train_b_expr ..\data\renew_inputs\long_input.csv --test_expr ..\data\renew_inputs\inter_input.csv --edge_csvs .\dbn_results\consensus_edges_thr0.7.csv --out_dir .\results_dde\run1 --seeds 0 1 2 3 4 --K_fit 2
```

---

## permutation test의 p-value 계산(요약)

두 테스트 모두 **empirical p-value**를 다음처럼 계산합니다.

- **R² (클수록 좋음)**:
  - \(p = (1 + \#\{R2_{null} \ge R2_{obs}\}) / (1 + n_{perm})\)
- **RMSE (작을수록 좋음)**:
  - \(p = (1 + \#\{RMSE_{null} \le RMSE_{obs}\}) / (1 + n_{perm})\)

---

## 파일별 설명

### DBN 폴더

- **`DBN/main.py`**  
  - 시계열 발현 CSV를 입력 받아, Lasso + Pearson + BIC-LP 점수를 조합해 simulated annealing으로 **DBN 구조를 탐색**하고, multi-start + stability selection으로 **seed별 그래프/consensus 그래프를 저장**하는 엔트리포인트입니다.
- **`DBN/lasso.py`**  
  - `LassoPreprocessor` 클래스: 각 유전자에 대해 Lasso 회귀를 돌려 **시간지연 간선 후보의 강도 행렬 A(lag, parent, child)** 를 추정하고, 정규화 + top‑k gating으로 sparse한 prior를 만듭니다.
- **`DBN/pearson.py`**  
  - Pearson 상관계수를 사용해 **intra-slice(동일 시점) 후보 간선 점수 행렬 C(parent, target)** 를 계산하는 전처리 모듈입니다.
- **`DBN/score.py`**  
  - `BICLPScorer` 클래스: 주어진 DBNGraph에 대해 **Gaussian likelihood + BIC penalty + Lasso/ Pearson 기반 prior 점수**를 합쳐 하나의 스칼라 점수를 반환합니다.
- **`DBN/anealing.py`**  
  - `SimulatedAnnealer` 클래스: add/remove/reverse 등 그래프 연산을 적용하면서 `BICLPScorer` 점수를 최대화하는 **simulated annealing 최적화 루프**를 구현합니다.
- **`DBN/graph.py`**  
  - `DBNGraph` 및 관련 연산(flip/add/remove edge, 유효성 검사, parent 목록 계산 등)을 정의하는 **그래프 자료구조/조작 유틸 모듈**입니다.
- **`DBN/export.py`**  
  - `GraphExporter` 클래스: `DBNGraph`를 **엣지 리스트 CSV** 및 **GraphML** 파일로 저장하는 기능을 제공합니다.
- **`DBN/data_loader.py`**  
  - `DataLoader`/`Dataset` 클래스: CSV 또는 디렉터리에서 **발현 데이터프레임을 읽고, 유전자 × 시간 구조로 정리**하여 DBN 쪽 코드가 사용하기 쉬운 형태로 변환합니다.

### DDE 폴더

- **`DDE/main.py`**  
  - 전체 DDE 파이프라인의 엔트리포인트로,  
    - 발현 CSV/엣지 CSV 로드 → 공통 유전자/엣지 필터링  
    - delay 구성(Δ, τ_k) 및 solver/config 세팅  
    - `run_multistart` 로 **joint train + test history fit** 수행 후 best seed 선택  
    - 예측/메트릭/체크포인트/LaTeX 방정식/forward simulation/표/플롯/validation(time & edge permutation)까지 **모든 산출물**을 생성합니다.
- **`DDE/module/data_io.py`**  
  - `ExpressionSample`, `EdgeTable` 및 `load_expression_csv`, `load_edge_csv`, `intersect_genes`, `filter_edges_to_genes`, `merge_edge_tables`, `edges_to_index_by_lag` 등을 정의하는 **입력 데이터/엣지 로딩 & 전처리 모듈**입니다.
- **`DDE/module/time_delay.py`**  
  - `DelayConfig`, `choose_delta_from_samples`로 **불균등 시간 간격에서 대표 Δ를 선택**하고, lag별 지연시간 `tau_by_lag` 및 `tau_max`를 계산합니다.
- **`DDE/module/dde_rhs.py`**  
  - `SharedParams` 등 DDE의 우변(연속시간 동역학)을 파라미터화한 **공유 파라미터 네트워크 모델**을 정의합니다. DBN에서 온 엣지 인덱스를 사용해 어떤 유전자가 어떤 지연으로 어떤 유전자에 영향을 주는지 표현합니다.
- **`DDE/module/dde_solver.py`**  
  - `SolverConfig` 및 `solve_dde_at_observation_times`를 정의하는 **DDE 수치해석 모듈**로, 주어진 시간축에서 shared+history를 이용해 예측 trajectory를 적분합니다.
- **`DDE/module/history.py`**  
  - `HistoryGrid`, `HistoryParam` 클래스로, 과거 구간 \([t_0-\tau_{max}, t_0]\)의 히스토리를 **M개 knot로 파라미터화하여 학습 가능한 텐서로 표현**합니다.
- **`DDE/module/loss.py`**  
  - `gaussian_nll_from_predictions` 등을 제공하는 **손실 함수 모듈**로, 관측값과 예측값 사이의 Gaussian negative log-likelihood를 계산합니다.
- **`DDE/module/metrics.py`**  
  - `rmse`, `r2_score`, `per_gene_metrics` 등 **전체/유전자별 RMSE, R²** 계산 유틸입니다.
- **`DDE/module/train_joint.py`**  
  - `TrainConfig`, `train_joint_two_samples`를 정의하며, 두 학습 샘플(train_a, train_b)에 대해 **shared parameters + history_a + history_b를 공동으로 최적화하는 joint MLE 학습 루프**를 구현합니다.
- **`DDE/module/test_forecast.py`**  
  - `TestConfig`, `fit_history_and_forecast`를 정의하며, **shared는 고정**한 채 test 샘플의 history만 최적화하고, 전체/미래 구간에 대한 **RMSE/R² 및 per‑gene metrics**를 계산합니다.
- **`DDE/module/multistart.py`**  
  - `MultiStartConfig`, `run_multistart`를 통해 여러 seed에 대해 **joint train + test forecast를 반복 실행**하고, CPU/GPU 환경에 따라 안전하게 병렬화를 제어하는 모듈입니다.
- **`DDE/module/dde_validation.py`**  
  - `time_permutation_test_inter`(시간 인덱스 섞기)와 `edge_rewiring_permutation_test`(엣지 재배선+재학습)를 포함해, **RMSE/R² 기반 permutation test 및 p-value 계산**을 담당합니다.
- **`DDE/module/analyze_perm.py`**  
  - 이미 저장된 permutation 결과(JSON)를 로드해 **p-value, 신뢰구간, effect size(Δ, z-score) 등을 집계하여 CSV로 요약**하는 분석 스크립트입니다.
- **`DDE/module/plot_gene_trajectories.py`**  
  - `gene_metrics`(NRMSE, R²)와 `plot_all_genes`/`plot_single_gene`를 통해 관측/예측/forward trajectory를 **유전자별로 플롯하여 저장**합니다.
- **`DDE/module/export_dde_latex.py`**  
  - 학습된 shared 파라미터와 엣지/지연 정보를 이용해 **DDE 방정식을 LaTeX 형태로 문자열로 내보내는 유틸**입니다.
- **`DDE/module/interp.py`**  
  - `linear_interp_1d` 등 히스토리 보간에 사용되는 **1D 선형 보간 함수**를 구현합니다.
- **`DDE/module/ode_rk.py`**  
  - ODE/DDE 수치해석에 사용하는 **Runge–Kutta 계열 적분기 구현**(저수준 solver 루틴)입니다.
