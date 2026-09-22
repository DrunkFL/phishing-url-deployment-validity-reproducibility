from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    metrics = pd.read_csv(RESULTS / "stage11f_external_metrics.csv")
    primary = metrics.loc[metrics["cohort"] == "source_domain_filtered"].copy()
    index = ["source_dataset", "repetition", "model"]
    paired = primary.pivot(index=index, columns="feature_set", values="roc_auc").reset_index()
    paired["delta_dcss15_minus_stable"] = paired["f_dcss_15"] - paired["f_stable"]
    paired["delta_dcss20_minus_stable"] = paired["f_dcss_20"] - paired["f_stable"]
    paired.to_csv(RESULTS / "stage11f_primary_auc_pairs.csv", index=False)

    by_model = (
        paired.groupby(["source_dataset", "model"])
        .agg(
            n_repetitions=("repetition", "size"),
            dcss15_minus_stable_mean=("delta_dcss15_minus_stable", "mean"),
            dcss15_minus_stable_std=("delta_dcss15_minus_stable", "std"),
            dcss15_wins=("delta_dcss15_minus_stable", lambda x: int((x > 0).sum())),
            dcss20_minus_stable_mean=("delta_dcss20_minus_stable", "mean"),
        )
        .reset_index()
    )
    by_model.to_csv(RESULTS / "stage11f_primary_auc_by_model.csv", index=False)

    by_rep = (
        paired.groupby(["source_dataset", "repetition"])[
            ["delta_dcss15_minus_stable", "delta_dcss20_minus_stable"]
        ]
        .mean()
        .reset_index()
    )
    by_direction = (
        by_rep.groupby("source_dataset")
        .agg(
            n_repetitions=("repetition", "size"),
            dcss15_minus_stable_mean=("delta_dcss15_minus_stable", "mean"),
            dcss15_minus_stable_std=("delta_dcss15_minus_stable", "std"),
            dcss15_positive_repetitions=("delta_dcss15_minus_stable", lambda x: int((x > 0).sum())),
            dcss20_minus_stable_mean=("delta_dcss20_minus_stable", "mean"),
        )
        .reset_index()
    )
    by_direction.to_csv(RESULTS / "stage11f_primary_auc_by_direction.csv", index=False)

    summary = (
        metrics.groupby(["source_dataset", "model", "feature_set", "cohort"])
        .agg(
            n_repetitions=("repetition", "size"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            recall_mean=("recall", "mean"),
            fpr_mean=("fpr", "mean"),
            ece_15_mean=("ece_15", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "stage11f_performance_summary.csv", index=False)

    audit = json.loads((RESULTS / "stage11f_source_audit.json").read_text(encoding="utf-8"))
    interpretation = {
        "evidential_status": "exploratory_sensitivity_only",
        "metadata_count_mismatch": audit["raw_rows"] != audit["advertised_rows"],
        "domain_filtered_rows": audit["source_domain_filtered_rows"],
        "direction_results": by_direction.to_dict(orient="records"),
        "bidirectional_positive_dcss15": bool(
            len(by_direction) == 2 and (by_direction["dcss15_minus_stable_mean"] > 0).all()
        ),
    }
    (RESULTS / "stage11f_analysis_summary.json").write_text(
        json.dumps(interpretation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(interpretation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
