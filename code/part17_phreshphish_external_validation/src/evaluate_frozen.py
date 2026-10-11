from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT.parent
CONFIG_PATH = ROOT / "config" / "experiment.json"
SOURCE_LAYOUT = {
    "phiusiil": {
        "features": "phiusiil_master_features.parquet",
        "assignment": "phiusiil_master_s3_domain_assignments.parquet",
    },
    "iscx_url2016_binary": {
        "features": "iscx_url2016_binary_master_features.parquet",
        "assignment": "iscx_url2016_binary_master_s3_domain_assignments.parquet",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen source selections on PhreshPhish")
    parser.add_argument("target_features", type=Path)
    parser.add_argument("--source-experiment-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--sources", nargs="+", choices=SOURCE_LAYOUT, default=list(SOURCE_LAYOUT))
    parser.add_argument("--repetitions", nargs="+", default=[f"r{i:02d}" for i in range(10)])
    parser.add_argument("--models", nargs="+", choices=("lr", "rf", "xgb"), default=["lr", "rf", "xgb"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--replay-tolerance", type=float, default=2e-5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_fpr_threshold(labels: np.ndarray, probabilities: np.ndarray, budget: float) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    valid = np.flatnonzero(fpr <= budget + 1e-15)
    if len(valid) == 0:
        return float(np.nextafter(np.max(probabilities), np.inf))
    best_tpr = np.max(tpr[valid])
    tied = valid[np.isclose(tpr[valid], best_tpr, rtol=0.0, atol=1e-15)]
    return float(np.min(thresholds[tied]))


def domain_equal_weights(groups: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(groups.astype(str), return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
    sample_weight: np.ndarray | None,
) -> float:
    weights = np.ones(len(labels), dtype=float) if sample_weight is None else sample_weight.astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)
    total = weights.sum()
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if not mask.any():
            continue
        bin_weight = weights[mask].sum()
        confidence = np.average(probabilities[mask], weights=weights[mask])
        observed = np.average(labels[mask], weights=weights[mask])
        value += (bin_weight / total) * abs(confidence - observed)
    return float(value)


def metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    bins: int,
    weighting: str,
    groups: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation cohort must contain both labels")
    weights = None if weighting == "row" else domain_equal_weights(np.asarray(groups))
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1], sample_weight=weights).ravel()
    fpr = fp / (fp + tn) if fp + tn else math.nan
    return {
        "weighting": weighting,
        "n_samples": int(len(labels)),
        "n_benign": int((labels == 0).sum()),
        "n_phishing": int((labels == 1).sum()),
        "effective_weight_sum": float(len(labels) if weights is None else weights.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions, sample_weight=weights)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions, sample_weight=weights)),
        "precision": float(precision_score(labels, predictions, zero_division=0, sample_weight=weights)),
        "recall": float(recall_score(labels, predictions, zero_division=0, sample_weight=weights)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0, sample_weight=weights)),
        "roc_auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        "pr_auc": float(average_precision_score(labels, probabilities, sample_weight=weights)),
        "brier": float(brier_score_loss(labels, probabilities, sample_weight=weights)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1], sample_weight=weights)),
        "ece": expected_calibration_error(labels, probabilities, bins, weights),
        "fpr": float(fpr),
        "alerts_per_10000_benign": float(fpr * 10000.0),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    target_path = args.target_features.resolve()
    target_metadata_path = target_path.with_suffix(".metadata.json")
    target_metadata = json.loads(target_metadata_path.read_text(encoding="utf-8"))
    if target_metadata["dataset_revision"] != config["dataset"]["revision"]:
        raise RuntimeError("Target feature revision does not match the frozen experiment")
    if target_metadata["scope"] != "full_split" and not args.allow_smoke:
        raise RuntimeError("Refusing to report a smoke subset as an experiment; pass --allow-smoke for pipeline testing")

    source_root = args.source_experiment_root.resolve()
    part2 = source_root / "part2_url_normalization_leakage_audit"
    part3 = source_root / "part3_s0_s4_data_splits"
    part4 = source_root / "part4_baseline_models"
    sys.path.insert(0, str(part4 / "scripts"))
    from model_utils import build_model

    dictionary = pd.read_csv(part2 / "results" / "feature_dictionary.csv").sort_values("feature_order")
    feature_names = dictionary["feature"].tolist()
    if feature_names != target_metadata["feature_names"]:
        raise RuntimeError("Target feature order differs from the frozen source feature order")
    target = pd.read_parquet(target_path)
    if len(target) != target_metadata["rows_written"]:
        raise RuntimeError("Target feature table row count differs from its metadata")
    target_values = target[feature_names].to_numpy(dtype=np.float32)
    target_labels = target["label"].to_numpy(dtype=np.int8)
    target_groups = target["registrable_domain_sha256"].astype(str).to_numpy()

    run_root = ROOT / "results" / target_metadata["scope"]
    result_path = run_root / "external_metrics.csv"
    metadata_path = run_root / "run_metadata.json"
    if (result_path.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError(f"Results already exist; use --force to replace: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for source in args.sources:
        source_features_path = part2 / "data" / SOURCE_LAYOUT[source]["features"]
        source_features = pd.read_parquet(
            source_features_path,
            columns=[
                "source_row",
                "raw_url_sha256",
                "label",
                "normalized_url_sha256",
                "host",
                "registrable_domain",
                *feature_names,
            ],
        )
        assignment_path = part3 / "data" / "assignments" / SOURCE_LAYOUT[source]["assignment"]
        assignments = pd.read_parquet(
            assignment_path,
            columns=["source_row", "raw_url_sha256", *[f"split_{rep}" for rep in args.repetitions]],
        )
        merged = assignments.merge(
            source_features,
            on=["source_row", "raw_url_sha256"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(source_features):
            raise RuntimeError(f"Source assignment mismatch for {source}")
        source_X = merged[feature_names].to_numpy(dtype=np.float32)
        source_y = merged["label"].to_numpy(dtype=np.int8)

        for repetition in args.repetitions:
            split = merged[f"split_{repetition}"].astype(str).to_numpy()
            train = split == "train"
            validation = split == "validation"
            test = split == "test"
            source_scope = train | validation
            source_url_hashes = set(merged.loc[source_scope, "normalized_url_sha256"].astype(str))
            source_host_hashes = {text_sha256(value) for value in merged.loc[source_scope, "host"].fillna("").astype(str)}
            source_domain_hashes = {
                text_sha256(value) for value in merged.loc[source_scope, "registrable_domain"].fillna("").astype(str)
            }
            url_overlap = target["normalized_url_sha256"].isin(source_url_hashes).to_numpy()
            host_overlap = target["host_sha256"].isin(source_host_hashes).to_numpy()
            domain_overlap = target["registrable_domain_sha256"].isin(source_domain_hashes).to_numpy()
            primary = ~domain_overlap

            for model_name in args.models:
                saved_path = part4 / "runs" / "internal" / source / "s3" / repetition / model_name / "internal_metrics.json"
                saved = json.loads(saved_path.read_text(encoding="utf-8"))
                params = json.loads(saved["selected_params_json"])
                model = build_model(model_name, params, int(saved["model_seed"]), args.n_jobs)
                model.fit(source_X[train], source_y[train])
                if list(model.classes_) != [0, 1]:
                    raise RuntimeError(f"Unexpected model class order for {source}/{repetition}/{model_name}")
                source_test_probabilities = model.predict_proba(source_X[test])[:, 1]
                replay_auc = float(roc_auc_score(source_y[test], source_test_probabilities))
                replay_macro_f1 = float(
                    f1_score(
                        source_y[test],
                        source_test_probabilities >= float(saved["threshold"]),
                        average="macro",
                    )
                )
                auc_delta = replay_auc - float(saved["roc_auc"])
                f1_delta = replay_macro_f1 - float(saved["macro_f1"])
                replay_rows.append(
                    {
                        "source_dataset": source,
                        "repetition": repetition,
                        "model": model_name,
                        "saved_roc_auc": saved["roc_auc"],
                        "replayed_roc_auc": replay_auc,
                        "roc_auc_delta": auc_delta,
                        "saved_macro_f1": saved["macro_f1"],
                        "replayed_macro_f1": replay_macro_f1,
                        "macro_f1_delta": f1_delta,
                    }
                )
                if max(abs(auc_delta), abs(f1_delta)) > args.replay_tolerance:
                    raise RuntimeError(
                        f"Source replay drift exceeds tolerance for {source}/{repetition}/{model_name}: "
                        f"AUC delta={auc_delta:.3g}, Macro-F1 delta={f1_delta:.3g}"
                    )

                validation_probabilities = model.predict_proba(source_X[validation])[:, 1]
                target_probabilities = model.predict_proba(target_values)[:, 1]
                target_auc = roc_auc_score(target_labels, target_probabilities)
                auc_complement_identity_error = abs(
                    target_auc
                    + roc_auc_score(target_labels, 1.0 - target_probabilities)
                    - 1.0
                )
                auc_negation_identity_error = abs(
                    target_auc
                    + roc_auc_score(target_labels, -target_probabilities)
                    - 1.0
                )
                thresholds = [("macro_f1", float(saved["threshold"]), None)]
                for budget in config["fpr_budgets"]:
                    thresholds.append(
                        (
                            f"fpr_{budget:g}",
                            select_fpr_threshold(source_y[validation], validation_probabilities, float(budget)),
                            float(budget),
                        )
                    )

                prediction_path = run_root / "predictions" / source / repetition / f"{model_name}.parquet"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "sample_id": target["sample_id"],
                        "label": target_labels,
                        "probability_phishing": target_probabilities.astype(np.float64),
                        "included_in_primary": primary,
                    }
                ).to_parquet(prediction_path, index=False, compression="zstd")

                for cohort, mask in (("unfiltered", np.ones(len(target), dtype=bool)), ("primary_domain_filtered", primary)):
                    if not mask.any():
                        raise RuntimeError(f"Empty target cohort for {source}/{repetition}/{model_name}/{cohort}")
                    for threshold_kind, threshold, budget in thresholds:
                        for weighting in ("row", "registrable_domain_equal"):
                            row = metrics(
                                target_labels[mask],
                                target_probabilities[mask],
                                threshold,
                                int(config["ece_bins"]),
                                weighting,
                                target_groups[mask],
                            )
                            rows.append(
                                {
                                    "result_scope": target_metadata["scope"],
                                    "source_dataset": source,
                                    "target_dataset": "phreshphish",
                                    "target_revision": target_metadata["dataset_revision"],
                                    "repetition": repetition,
                                    "model": model_name,
                                    "selected_candidate": saved["selected_candidate"],
                                    "cohort": cohort,
                                    "threshold_kind": threshold_kind,
                                    "source_validation_fpr_budget": budget,
                                    "target_normalized_url_overlap_n": int(url_overlap.sum()),
                                    "target_host_overlap_n": int(host_overlap.sum()),
                                    "target_registrable_domain_overlap_n": int(domain_overlap.sum()),
                                    "auc_complement_identity_error": float(auc_complement_identity_error),
                                    "auc_negation_identity_error": float(auc_negation_identity_error),
                                    "probability_zero_n": int((target_probabilities == 0.0).sum()),
                                    "probability_one_n": int((target_probabilities == 1.0).sum()),
                                    **row,
                                }
                            )

    result = pd.DataFrame(rows)
    replay = pd.DataFrame(replay_rows)
    result_tmp = result_path.with_suffix(".csv.tmp")
    replay_path = run_root / "source_replay_check.csv"
    replay_tmp = replay_path.with_suffix(".csv.tmp")
    result.to_csv(result_tmp, index=False)
    replay.to_csv(replay_tmp, index=False)
    result_tmp.replace(result_path)
    replay_tmp.replace(replay_path)
    metadata = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SMOKE_PASS" if target_metadata["scope"] == "smoke_only" else "FULL_RUN_PASS",
        "interpretation_boundary": (
            "Pipeline-only smoke test; metrics are not manuscript evidence."
            if target_metadata["scope"] == "smoke_only"
            else "Independent fixed-test evaluation; manuscript claims still require aggregate review and SHAP validation."
        ),
        "source_experiment_root": str(source_root),
        "target_features": str(target_path),
        "target_features_sha256": sha256_file(target_path),
        "rows": len(result),
        "source_replay_rows": len(replay),
        "max_abs_source_replay_auc_delta": float(replay["roc_auc_delta"].abs().max()),
        "max_abs_source_replay_macro_f1_delta": float(replay["macro_f1_delta"].abs().max()),
        "elapsed_seconds": time.perf_counter() - started,
        "n_jobs": args.n_jobs,
        "results": str(result_path),
        "results_sha256": sha256_file(result_path),
        "source_replay_check": str(replay_path),
        "source_replay_check_sha256": sha256_file(replay_path),
    }
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_tmp.replace(metadata_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
