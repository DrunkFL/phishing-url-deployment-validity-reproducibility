from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_info


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent
PART2 = ARCHIVE / "part2_url_normalization_leakage_audit"
PART3 = ARCHIVE / "part3_s0_s4_data_splits"
PART4 = ARCHIVE / "part4_baseline_models"
sys.path.insert(0, str(PART4 / "scripts"))
from model_utils import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a frozen source-model replay")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def main() -> None:
    args = parse_args()
    source = "phiusiil"
    repetition = "r00"
    model_name = "lr"
    names = (
        pd.read_csv(PART2 / "results" / "feature_dictionary.csv")
        .sort_values("feature_order")["feature"]
        .tolist()
    )
    features = pd.read_parquet(
        PART2 / "data" / "phiusiil_master_features.parquet",
        columns=["source_row", "raw_url_sha256", "label", *names],
    )
    assignments = pd.read_parquet(
        PART3 / "data" / "assignments" / "phiusiil_master_s3_domain_assignments.parquet",
        columns=[
            "sample_id",
            "source_row",
            "raw_url_sha256",
            "label",
            "split_r00",
        ],
    )
    merged = assignments.merge(
        features.drop(columns="label"),
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    train = merged.loc[merged["split_r00"] == "train"]
    test = merged.loc[merged["split_r00"] == "test"]
    X_train = train[names].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int8)
    X_test = test[names].to_numpy(dtype=np.float32)
    y_test = test["label"].to_numpy(dtype=np.int8)

    saved_path = (
        PART4
        / "runs"
        / "internal"
        / source
        / "s3"
        / repetition
        / model_name
        / "internal_metrics.json"
    )
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    archived = pd.read_parquet(
        PART4
        / "predictions"
        / "internal"
        / source
        / "s3"
        / repetition
        / f"{model_name}.parquet"
    )
    if not archived["sample_id"].equals(test["sample_id"].reset_index(drop=True)):
        raise RuntimeError("Archived prediction order differs from reconstructed test order")
    archived_probability = archived["probability_phishing"].to_numpy(dtype=np.float64)
    threshold = float(saved["threshold"])

    rows = []
    predictions = []
    for repeat in range(args.repeats):
        model = build_model(
            model_name,
            json.loads(saved["selected_params_json"]),
            int(saved["model_seed"]),
            args.n_jobs,
        )
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        predictions.append(probability)
        classifier = model.named_steps["classifier"]
        rows.append(
            {
                "repeat": repeat,
                "roc_auc": float(roc_auc_score(y_test, probability)),
                "roc_auc_delta_saved": float(roc_auc_score(y_test, probability) - saved["roc_auc"]),
                "macro_f1": float(
                    f1_score(y_test, probability >= threshold, average="macro")
                ),
                "macro_f1_delta_saved": float(
                    f1_score(y_test, probability >= threshold, average="macro")
                    - saved["macro_f1"]
                ),
                "prediction_sha256_float64": array_sha256(probability),
                "coefficient_sha256_float64": array_sha256(classifier.coef_),
                "intercept_sha256_float64": array_sha256(classifier.intercept_),
                "n_iter": classifier.n_iter_.tolist(),
                "max_abs_probability_delta_archived_float32": float(
                    np.max(np.abs(probability - archived_probability))
                ),
                "threshold_class_changes_vs_archived": int(
                    np.count_nonzero(
                        (probability >= threshold)
                        != (archived_probability >= threshold)
                    )
                ),
            }
        )

    repeat_differences = []
    for index in range(1, len(predictions)):
        repeat_differences.append(
            {
                "left_repeat": 0,
                "right_repeat": index,
                "max_abs_probability_delta": float(
                    np.max(np.abs(predictions[0] - predictions[index]))
                ),
                "different_float64_probabilities": int(
                    np.count_nonzero(predictions[0] != predictions[index])
                ),
            }
        )

    payload = {
        "source": source,
        "repetition": repetition,
        "model": model_name,
        "n_jobs": args.n_jobs,
        "train_rows": len(train),
        "test_rows": len(test),
        "X_train_sha256_float32": array_sha256(X_train),
        "X_test_sha256_float32": array_sha256(X_test),
        "saved_metrics": {
            "roc_auc": saved["roc_auc"],
            "macro_f1": saved["macro_f1"],
            "threshold": threshold,
        },
        "archived_prediction_sha256_float32": array_sha256(
            archived["probability_phishing"].to_numpy(dtype=np.float32)
        ),
        "threadpools": threadpool_info(),
        "fits": rows,
        "repeat_differences": repeat_differences,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
