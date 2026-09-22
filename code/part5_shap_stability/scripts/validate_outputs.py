from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    complete = list(ROOT.glob("runs/*/*/*/*/*/complete.json"))
    internal_values = list(ROOT.glob("runs/*/*/*/*/*/internal/shap_values.parquet"))
    external_values = list(ROOT.glob("runs/partition/*/s3/*/*/external_common/shap_values.parquet"))
    assert len(complete) == 360, len(complete)
    assert len(internal_values) == 360, len(internal_values)
    assert len(external_values) == 60, len(external_values)

    for path in [*internal_values, *external_values]:
        frame = pd.read_parquet(path)
        assert len(frame) == 200, path
        assert frame["label"].value_counts().to_dict() == {0: 100, 1: 100}, path
        shap_columns = [column for column in frame if column.startswith("shap__")]
        value_columns = [column for column in frame if column.startswith("value__")]
        assert len(shap_columns) == 35 and len(value_columns) == 35
        assert np.isfinite(frame[shap_columns + value_columns].to_numpy(dtype=float)).all()

    pairwise = pd.read_csv(ROOT / "results" / "pairwise_stability.csv")
    expected = {"seed": 540, "partition": 1080, "regime": 360, "source": 60}
    assert pairwise.groupby("estimand").size().to_dict() == expected
    bounded = ["top10_jaccard", "top15_jaccard", "top20_jaccard", "direction_agreement_all", "direction_agreement_top20_union"]
    assert pairwise[bounded].apply(lambda column: column.between(0, 1).all()).all()
    assert pairwise["rank_spearman"].between(-1, 1).all()

    global_rows = pd.read_csv(ROOT / "results" / "all_global_feature_summaries.csv")
    assert len(global_rows) == 420 * 35
    assert global_rows.groupby(
        ["estimand", "dataset", "scenario", "model", "run_id", "cohort"], observed=True
    ).size().eq(35).all()

    benchmark = pd.read_csv(ROOT / "results" / "shap_scaling_benchmark.csv")
    convergence = pd.read_csv(ROOT / "results" / "cohort_size_rank_convergence.csv")
    assert len(benchmark) == 36
    assert len(convergence) == 36
    assert set(benchmark["n_samples"]) == {100, 500, 1000}

    # Seed cohorts are fixed; partition cohorts are shared across model families.
    for dataset in ("phiusiil", "iscx_url2016_binary"):
        for scenario in ("s0", "s3"):
            reference = None
            for model in ("lr", "rf", "xgb"):
                for run in (f"r{index:02d}" for index in range(10)):
                    path = ROOT / "runs" / "seed" / dataset / scenario / model / run / "internal" / "shap_values.parquet"
                    ids = tuple(pd.read_parquet(path, columns=["sample_id"])["sample_id"])
                    reference = ids if reference is None else reference
                    assert ids == reference
        external_reference = None
        for model in ("lr", "rf", "xgb"):
            for run in (f"r{index:02d}" for index in range(10)):
                path = ROOT / "runs" / "partition" / dataset / "s3" / model / run / "external_common" / "shap_values.parquet"
                ids = tuple(pd.read_parquet(path, columns=["sample_id"])["sample_id"])
                external_reference = ids if external_reference is None else external_reference
                assert ids == external_reference

    manifest = pd.read_csv(ROOT / "results" / "result_manifest.csv")
    for row in manifest.itertuples():
        path = ROOT / row.path
        assert path.stat().st_size == row.bytes
        assert sha256_file(path) == row.sha256

    result = {
        "complete_runs": len(complete),
        "internal_shap_files": len(internal_values),
        "external_shap_files": len(external_values),
        "pairwise_rows": len(pairwise),
        "benchmark_rows": len(benchmark),
        "status": "PASS",
    }
    (ROOT / "results" / "validation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
