# plot_gene_trajectories.py

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def gene_metrics(y_true, y_pred, eps: float = 1e-12):
    """
    Returns:
      nrmse: RMSE / std(y_true)
      r2:    coefficient of determination
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.std(y_true)

    nrmse = rmse / (denom + eps)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)

    return nrmse, r2


def plot_single_gene(
    gene_name: str,
    samples: dict,
    preds: dict,
    gene_idx: int,
    out_path: Path,
    K_fit: int | None = None,
    future: dict | None = None,   # {sample_name: (t_future, z_future)}
):
    """
    future:
      dict[sample_name] = (t_future, z_future[:, gene_idx])
    """
    

    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (name, sample) in zip(axes, samples.items()):
        t = sample.t.detach().cpu().numpy()
        y_true = sample.X[:, gene_idx].detach().cpu().numpy()
        y_pred = preds[name][:, gene_idx].detach().cpu().numpy()

        # observed
        ax.scatter(t, y_true, color="black", label="Observed", zorder=3)

        # ===== Always draw one continuous prediction line =====
        ax.plot(
            t,
            y_pred,
            color="red",
            linewidth=2,
            label="Predicted"
        )

        # Metrics
        if K_fit is not None and "test" in name.lower() and K_fit < len(t):
            nrmse_f, r2_f = gene_metrics(y_true[K_fit:], y_pred[K_fit:])
            title_metric = f"Forecast R²={r2_f:.2f}, NRMSE={nrmse_f:.2f}"
        else:
            nrmse, r2 = gene_metrics(y_true, y_pred)
            title_metric = f"R²={r2:.2f}, NRMSE={nrmse:.2f}"

        # forward simulation (future)
        if future is not None and name in future:
            t_fut, z_fut = future[name]

            # 안전하게 numpy 변환
            t_fut = np.asarray(t_fut)
            z_fut = np.asarray(z_fut)

            # gene_idx 슬라이싱 안정화
            if z_fut.ndim == 2:
                y_fut = z_fut[:, gene_idx]
            else:
                y_fut = z_fut

            ax.plot(
                np.concatenate([t[-1:], t_fut]),
                np.concatenate([[y_pred[-1]], y_fut]),
                color="blue",
                linewidth=2,
                label="Forward (7d)"
            )

        ax.set_title(f"{name} | {title_metric}", fontsize=11)
        ax.set_ylabel("log2(TMM)")
        ax.grid(alpha=0.3)

        

    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"Gene: {gene_name}", fontsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_all_genes(
    samples: dict,
    preds: dict,
    gene_names: list,
    out_dir: str,
    max_genes: int | None = None,
    K_fit: int | None = None,
    future: dict | None = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = len(gene_names)
    genes_to_plot = range(G) if max_genes is None else range(min(G, max_genes))

    for i in genes_to_plot:
        gene = gene_names[i]
        fname = f"gene_{i:04d}_{gene}.png"
        plot_single_gene(
            gene_name=gene,
            samples=samples,
            preds=preds,
            gene_idx=i,
            out_path=out_dir / fname,
            K_fit=K_fit,
            future=future,
        )
