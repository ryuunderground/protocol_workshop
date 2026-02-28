# summarize_edge_perm.py
import json
import glob
import math
import numpy as np
import pandas as pd
from pathlib import Path

def wilson_ci(b, n, alpha=0.05):
    # Wilson score interval for proportion b/n
    # Here we assume p_hat = b/n (if you use +1 correction, apply to b,n before calling)
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054  # approx for 95%
    phat = b / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat)/n) + (z**2/(4*n**2)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def load_one(path):
    d = json.loads(Path(path).read_text())
    r2_obs = float(d["r2_obs"])
    rmse_obs = float(d["rmse_obs"])
    r2_null = np.array(d["r2_null"], dtype=float)
    rmse_null = np.array(d["rmse_null"], dtype=float)
    n_perm = int(d["n_perm"])
    seed = int(d.get("seed", -1))

    # counts for p-value definition
    b_r2 = int(np.sum(r2_null >= r2_obs))
    b_rmse = int(np.sum(rmse_null <= rmse_obs))

    # if your code uses (b+1)/(n+1), reflect that for CI too
    p_r2 = (b_r2 + 1) / (n_perm + 1)
    p_rmse = (b_rmse + 1) / (n_perm + 1)

    # Wilson CI on the corrected counts
    ci_r2 = wilson_ci(b_r2 + 1, n_perm + 1)
    ci_rmse = wilson_ci(b_rmse + 1, n_perm + 1)

    # effect sizes
    delta_r2 = r2_obs - np.median(r2_null)
    delta_rmse = np.median(rmse_null) - rmse_obs

    z_r2 = (r2_obs - np.mean(r2_null)) / (np.std(r2_null, ddof=1) + 1e-12)
    z_rmse = (np.mean(rmse_null) - rmse_obs) / (np.std(rmse_null, ddof=1) + 1e-12)

    return {
        "seed": seed,
        "n_perm": n_perm,
        "r2_obs": r2_obs,
        "rmse_obs": rmse_obs,
        "p_value_r2": p_r2,
        "p_value_r2_ci_lo": ci_r2[0],
        "p_value_r2_ci_hi": ci_r2[1],
        "p_value_rmse": p_rmse,
        "p_value_rmse_ci_lo": ci_rmse[0],
        "p_value_rmse_ci_hi": ci_rmse[1],
        "delta_r2_vs_median_null": delta_r2,
        "delta_rmse_vs_median_null": delta_rmse,
        "z_r2": z_r2,
        "z_rmse": z_rmse,
    }

def main():
    paths = sorted(glob.glob("results_dde/results/validation/edge_rewire_permutation_seed*.json"))
    if not paths:
        # fallback: maybe single file name
        paths = sorted(glob.glob("results_dde/results/validation/edge_rewire_permutation*.json"))
    rows = [load_one(p) for p in paths]
    df = pd.DataFrame(rows).sort_values("seed")
    out_dir = Path("results_dde/results/validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "edge_perm_seed_summary.csv", index=False)

    # overall summary
    num_cols = [c for c in df.columns if c not in ("seed",)]
    summary = []
    for c in num_cols:
        v = df[c].astype(float).values
        summary.append({
            "metric": c,
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        })
    pd.DataFrame(summary).to_csv(out_dir / "edge_perm_overall_summary.csv", index=False)

    print("[OK] Saved:",
          out_dir / "edge_perm_seed_summary.csv",
          out_dir / "edge_perm_overall_summary.csv")

if __name__ == "__main__":
    main()