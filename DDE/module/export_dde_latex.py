# export_dde_latex.py
from pathlib import Path
import torch


def export_dde_equations_latex(
    shared,
    gene_names: list[str],
    edge_idx_by_lag: dict,
    tau_by_lag: dict,
    out_path: str | Path,
    activation: str = "tanh",
    weight_threshold: float = 0.0,
):
    """
    shared: trained SharedParams
    gene_names: list of gene names (length G)
    edge_idx_by_lag: {lag: {"src": tensor, "dst": tensor}}
    tau_by_lag: {lag: tau_value}
    out_path: .txt file path
    """

    out_path = Path(out_path)

    # get actual parameter values (detach from graph)
    lam = shared.lam().detach().cpu().numpy()    # (G,)
    b   = shared.b.detach().cpu().numpy()        # (G,)

    # weights per lag
    w_by_lag = {}
    for k in shared.lags:
        w_by_lag[k] = shared.w(k).detach().cpu().numpy()

    if activation == "tanh":
        act_tex = r"\tanh"
    elif activation == "identity":
        act_tex = ""
    else:
        act_tex = activation

    lines = []
    lines.append(r"% Automatically generated DDE system")
    lines.append(r"% dz_i(t)/dt equations")
    lines.append("")

    G = len(gene_names)

    for i in range(G):
        gene_i = gene_names[i]
        terms = []

        # decay
        terms.append(f"- {lam[i]:.4f} \\, z_{{{gene_i}}}(t)")

        # regulatory terms
        for k in shared.lags:
            tau = tau_by_lag[k]
            src = edge_idx_by_lag[k]["src"].tolist()
            dst = edge_idx_by_lag[k]["dst"].tolist()
            w   = w_by_lag[k]

            for e, (j, ii) in enumerate(zip(src, dst)):
                if ii != i:
                    continue
                if abs(w[e]) <= weight_threshold:
                    continue

                gene_j = gene_names[j]

                if tau == 0:
                    z_term = f"z_{{{gene_j}}}(t)"
                else:
                    z_term = f"z_{{{gene_j}}}(t - {tau:.2f})"

                if act_tex:
                    z_term = f"{act_tex}\\!\\left({z_term}\\right)"

                terms.append(f"+ {w[e]:.4f} \\, {z_term}")

        # bias
        terms.append(f"+ {b[i]:.4f}")

        rhs = " ".join(terms)
        eq = rf"\frac{{d z_{{{gene_i}}}(t)}}{{dt}} = {rhs}"
        lines.append(eq)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
