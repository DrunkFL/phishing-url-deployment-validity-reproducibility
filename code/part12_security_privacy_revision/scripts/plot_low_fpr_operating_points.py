from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "low_fpr_operating_point_summary.csv"
OUTPUT = ROOT / "figures" / "fig5_low_fpr_threshold_transfer"


def main() -> None:
    data = pd.read_csv(INPUT)
    data = data[data["evaluation_scope"].isin(["source_s3_test", "external_primary"])].copy()
    data["direction"] = data["source_dataset"].map(
        {
            "iscx_url2016_binary": "ISCX to PhiUSIIL",
            "phiusiil": "PhiUSIIL to ISCX",
        }
    )

    models = ["lr", "rf", "xgb"]
    model_labels = {"lr": "LR", "rf": "RF", "xgb": "XGBoost"}
    directions = ["ISCX to PhiUSIIL", "PhiUSIIL to ISCX"]
    colors = {"ISCX to PhiUSIIL": "#1f77b4", "PhiUSIIL to ISCX": "#c43c39"}
    markers = {0.001: "o", 0.01: "s"}
    budget_labels = {0.001: "0.1% source budget", 0.01: "1% source budget"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    scopes = [
        ("source_s3_test", "Frozen threshold on source S3 test"),
        ("external_primary", "Frozen threshold on external source"),
    ]
    x = np.arange(len(models), dtype=float)

    for axis, (scope, title) in zip(axes, scopes):
        subset = data[data["evaluation_scope"] == scope]
        for direction_index, direction in enumerate(directions):
            for budget_index, budget in enumerate(markers):
                rows = (
                    subset[
                        (subset["direction"] == direction)
                        & np.isclose(subset["fpr_budget"], budget)
                    ]
                    .set_index("model")
                    .loc[models]
                )
                offset = (-0.15 if direction_index == 0 else 0.15) + (
                    -0.035 if budget_index == 0 else 0.035
                )
                axis.plot(
                    x + offset,
                    rows["fpr_mean"],
                    linestyle="none",
                    marker=markers[budget],
                    markersize=7,
                    markerfacecolor=colors[direction],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    color=colors[direction],
                    label=f"{direction}, {budget_labels[budget]}",
                )

        axis.axhline(0.001, color="#555555", linewidth=1, linestyle=":")
        axis.axhline(0.01, color="#555555", linewidth=1, linestyle="--")
        axis.set_yscale("log")
        axis.set_ylim(5e-4, 1.25)
        axis.set_xticks(x, [model_labels[name] for name in models])
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", which="both", linewidth=0.5, alpha=0.25)
        axis.set_xlabel("Model")

    axes[0].set_ylabel("Realized false-positive rate (log scale)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Source-selected false-positive budgets do not transfer", fontsize=12)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
