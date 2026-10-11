from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_NUMERICAL_THREADS = 12
THREAD_ENV_BEFORE_OVERRIDE = {
    name: os.environ.get(name)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
}
for _thread_variable in THREAD_ENV_BEFORE_OVERRIDE:
    os.environ[_thread_variable] = str(EXPECTED_NUMERICAL_THREADS)

import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_info


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT.parent
PART2 = ARCHIVE_ROOT / "part2_url_normalization_leakage_audit"
PART3 = ARCHIVE_ROOT / "part3_s0_s4_data_splits"
PART4 = ARCHIVE_ROOT / "part4_baseline_models"
PART5 = ARCHIVE_ROOT / "part5_shap_stability"
CONFIG_PATH = ROOT / "config" / "experiment.json"
DEFAULT_TARGET = ROOT / "data" / "features" / "server_full" / "phreshphish_test.parquet"
DEFAULT_OUTPUT = ROOT / "results" / "shap_external"

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
MODELS = ("lr", "rf", "xgb")
REPETITIONS = tuple(f"r{index:02d}" for index in range(10))

sys.path.insert(0, str(PART4 / "scripts"))
sys.path.insert(0, str(PART5 / "scripts"))

from model_utils import build_model  # noqa: E402
from shap_utils import (  # noqa: E402
    build_explainer,
    deterministic_stratified_sample,
    explain,
    global_feature_summary,
    ranking_comparison,
    shap_value_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare source-test and PhreshPhish SHAP explanations for frozen S3 models"
    )
    parser.add_argument("--target-features", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources", nargs="+", choices=SOURCE_LAYOUT, default=list(SOURCE_LAYOUT))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=list(REPETITIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--cohort-size", type=int, default=200)
    parser.add_argument("--background-size", type=int, default=200)
    parser.add_argument("--replay-tolerance", type=float, default=2e-5)
    parser.add_argument("--prediction-replay-tolerance", type=float, default=1e-7)
    parser.add_argument("--scope", choices=("full", "smoke"), default="full")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def validated_threadpools() -> list[dict[str, Any]]:
    pools = threadpool_info()
    relevant = [pool for pool in pools if pool.get("user_api") in {"blas", "openmp"}]
    if not relevant:
        raise RuntimeError("No BLAS/OpenMP thread pool was detected")
    mismatches = [
        pool
        for pool in relevant
        if int(pool.get("num_threads", -1)) != EXPECTED_NUMERICAL_THREADS
    ]
    if mismatches:
        details = [
            {
                "user_api": pool.get("user_api"),
                "internal_api": pool.get("internal_api"),
                "num_threads": pool.get("num_threads"),
                "filepath": pool.get("filepath"),
            }
            for pool in mismatches
        ]
        raise RuntimeError(
            f"Numerical thread topology differs from the archived model path: {details}"
        )
    return relevant


def sample_id_sha256(frame: pd.DataFrame) -> str:
    joined = "\n".join(frame["sample_id"].astype(str)) + "\n"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def feature_names() -> list[str]:
    dictionary = pd.read_csv(PART2 / "results" / "feature_dictionary.csv")
    names = dictionary.sort_values("feature_order")["feature"].tolist()
    if len(names) != 35 or len(names) != len(set(names)):
        raise RuntimeError("Expected the frozen set of 35 unique features")
    return names


def load_source(source: str, names: list[str]) -> pd.DataFrame:
    features = pd.read_parquet(
        PART2 / "data" / SOURCE_LAYOUT[source]["features"],
        columns=["source_row", "raw_url_sha256", "label", *names],
    )
    assignments = pd.read_parquet(
        PART3 / "data" / "assignments" / SOURCE_LAYOUT[source]["assignment"],
        columns=[
            "sample_id",
            "source_row",
            "raw_url_sha256",
            "label",
            *[f"split_{repetition}" for repetition in REPETITIONS],
        ],
    )
    merged = assignments.merge(
        features.drop(columns="label"),
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(assignments) or set(merged["label"].unique()) != {0, 1}:
        raise RuntimeError(f"Source merge failed for {source}")
    if not np.isfinite(merged[names].to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite source features for {source}")
    return merged


def load_target(path: Path, names: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = path.resolve()
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if metadata["scope"] != "full_split":
        raise RuntimeError("SHAP analysis requires the full target split")
    if metadata["dataset_revision"] != config["dataset"]["revision"]:
        raise RuntimeError("Target revision differs from the frozen experiment")
    if sha256_file(path) != metadata["output_sha256"]:
        raise RuntimeError("Target feature SHA-256 differs from metadata")
    target = pd.read_parquet(path)
    if len(target) != metadata["rows_written"] or metadata["invalid_url_rows"] != 0:
        raise RuntimeError("Target row-count or invalid-row contract failed")
    if names != metadata["feature_names"]:
        raise RuntimeError("Target feature order differs from the frozen source order")
    if not np.isfinite(target[names].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite target features")
    return target, metadata


def build_target_common_cohort(
    source: str,
    target: pd.DataFrame,
    repetitions: list[str],
    models: list[str],
    cohort_size: int,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_root = ROOT / "results" / "full_split" / "predictions" / source
    common_mask = np.ones(len(target), dtype=bool)
    per_repetition_counts: dict[str, int] = {}
    for repetition in repetitions:
        reference_mask = None
        for model in models:
            prediction_path = prediction_root / repetition / f"{model}.parquet"
            prediction = pd.read_parquet(
                prediction_path,
                columns=["sample_id", "label", "included_in_primary"],
            )
            if not prediction["sample_id"].equals(target["sample_id"]):
                raise RuntimeError(f"Prediction sample order differs for {source}/{repetition}/{model}")
            if not prediction["label"].equals(target["label"]):
                raise RuntimeError(f"Prediction labels differ for {source}/{repetition}/{model}")
            mask = prediction["included_in_primary"].to_numpy(dtype=bool)
            if reference_mask is None:
                reference_mask = mask
            elif not np.array_equal(reference_mask, mask):
                raise RuntimeError(f"Primary mask differs across models for {source}/{repetition}")
        if reference_mask is None:
            raise RuntimeError(f"No primary mask for {source}/{repetition}")
        per_repetition_counts[repetition] = int(reference_mask.sum())
        common_mask &= reference_mask

    common_pool = target.loc[common_mask].copy()
    cohort = deterministic_stratified_sample(
        common_pool,
        cohort_size,
        f"phreshphish:{source}:common-primary",
    )
    if cohort["label"].value_counts().sort_index().to_dict() != {0: cohort_size // 2, 1: cohort_size // 2}:
        raise RuntimeError(f"Target cohort is not class-balanced for {source}")
    cohort_path = output_root / "cohorts" / f"{source}_target_common.parquet"
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    cohort[[
        "sample_id",
        "provider_sha256",
        "label",
        "split_row",
        "raw_url_sha256",
        "registrable_domain_sha256",
    ]].to_parquet(cohort_path, index=False, compression="zstd")
    metadata = {
        "source_dataset": source,
        "purpose": "post_hoc_explanation_only",
        "selection": "intersection of all requested source-specific primary domain-filtered masks",
        "common_pool_rows": len(common_pool),
        "common_pool_label_counts": common_pool["label"].value_counts().sort_index().to_dict(),
        "cohort_rows": len(cohort),
        "cohort_label_counts": cohort["label"].value_counts().sort_index().to_dict(),
        "cohort_sample_id_sha256": sample_id_sha256(cohort),
        "per_repetition_primary_rows": per_repetition_counts,
        "cohort_file": str(cohort_path.resolve()),
        "cohort_file_sha256": sha256_file(cohort_path),
    }
    write_json(cohort_path.with_suffix(".metadata.json"), metadata)
    return cohort, metadata


def save_explanation(
    output_dir: Path,
    cohort: pd.DataFrame,
    names: list[str],
    X: np.ndarray,
    shap_values: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "global_summary.csv"
    values_path = output_dir / "shap_values.parquet"
    global_feature_summary(X, shap_values, names).to_csv(summary_path, index=False)
    serialized_cohort = cohort.copy()
    if "source_row" not in serialized_cohort and "split_row" in serialized_cohort:
        serialized_cohort["source_row"] = serialized_cohort["split_row"]
    shap_value_frame(serialized_cohort, X, shap_values, names).to_parquet(
        values_path,
        index=False,
        compression="zstd",
    )
    write_json(output_dir / "metadata.json", metadata)
    return summary_path, values_path


def direction_conflicts(
    source_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "source_dataset",
        "repetition",
        "model",
        "feature",
        "source_rank",
        "target_rank",
        "source_direction_correlation",
        "target_direction_correlation",
        "source_direction_sign",
        "target_direction_sign",
        "source_mean_abs_shap",
        "target_mean_abs_shap",
    ]
    left = source_summary.set_index("feature")
    right = target_summary.set_index("feature")
    top_union = set(left.nsmallest(15, "rank").index) | set(right.nsmallest(15, "rank").index)
    rows = []
    for feature in sorted(top_union):
        source_sign = int(left.loc[feature, "direction_sign"])
        target_sign = int(right.loc[feature, "direction_sign"])
        if source_sign == 0 or target_sign == 0 or source_sign == target_sign:
            continue
        rows.append({
            **context,
            "feature": feature,
            "source_rank": int(left.loc[feature, "rank"]),
            "target_rank": int(right.loc[feature, "rank"]),
            "source_direction_correlation": float(left.loc[feature, "direction_correlation"]),
            "target_direction_correlation": float(right.loc[feature, "direction_correlation"]),
            "source_direction_sign": source_sign,
            "target_direction_sign": target_sign,
            "source_mean_abs_shap": float(left.loc[feature, "mean_abs_shap"]),
            "target_mean_abs_shap": float(right.loc[feature, "mean_abs_shap"]),
        })
    return pd.DataFrame(rows, columns=columns)


def external_metric_row(source: str, repetition: str, model: str) -> pd.Series:
    metrics = pd.read_csv(ROOT / "results" / "full_split" / "external_metrics.csv")
    selected = metrics.loc[
        (metrics["source_dataset"] == source)
        & (metrics["repetition"] == repetition)
        & (metrics["model"] == model)
        & (metrics["cohort"] == "primary_domain_filtered")
        & (metrics["threshold_kind"] == "macro_f1")
        & (metrics["weighting"] == "row")
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one external metric row for {source}/{repetition}/{model}")
    return selected.iloc[0]


def verify_archived_source_predictions(
    source: str,
    repetition: str,
    model_name: str,
    test: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    tolerance: float,
) -> dict[str, Any]:
    prediction_path = (
        PART4
        / "predictions"
        / "internal"
        / source
        / "s3"
        / repetition
        / f"{model_name}.parquet"
    )
    archived = pd.read_parquet(
        prediction_path,
        columns=["sample_id", "source_row", "label", "probability_phishing", "prediction"],
    )
    current = test[["sample_id", "source_row", "label"]].reset_index(drop=True)
    normalized_values = {
        "sample_id": (
            current["sample_id"].astype(str).to_numpy(),
            archived["sample_id"].astype(str).to_numpy(),
        ),
        "source_row": (
            current["source_row"].to_numpy(dtype=np.int64),
            archived["source_row"].to_numpy(dtype=np.int64),
        ),
        "label": (
            current["label"].to_numpy(dtype=np.int8),
            archived["label"].to_numpy(dtype=np.int8),
        ),
    }
    for column, (current_values, archived_values) in normalized_values.items():
        if not np.array_equal(current_values, archived_values):
            raise RuntimeError(
                f"Archived source prediction {column} differs for "
                f"{source}/{repetition}/{model_name}"
            )
    archived_probabilities = archived["probability_phishing"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or not np.isfinite(archived_probabilities).all():
        raise RuntimeError(f"Non-finite source probability for {source}/{repetition}/{model_name}")
    max_abs_delta = float(np.max(np.abs(probabilities - archived_probabilities)))
    current_predictions = (probabilities >= threshold).astype(np.int8)
    archived_predictions = archived["prediction"].to_numpy(dtype=np.int8)
    threshold_class_changes = int(np.count_nonzero(current_predictions != archived_predictions))
    archived_threshold_inconsistencies = int(
        np.count_nonzero(
            (archived_probabilities >= threshold).astype(np.int8) != archived_predictions
        )
    )
    float32_exact = bool(
        np.array_equal(
            probabilities.astype(np.float32),
            archived["probability_phishing"].to_numpy(dtype=np.float32),
        )
    )
    if max_abs_delta > tolerance:
        raise RuntimeError(
            f"Archived source probability replay drift for {source}/{repetition}/{model_name}: "
            f"max_abs_delta={max_abs_delta:.3g}"
        )
    if threshold_class_changes or archived_threshold_inconsistencies:
        raise RuntimeError(
            f"Archived source threshold replay differs for {source}/{repetition}/{model_name}: "
            f"class_changes={threshold_class_changes}, "
            f"archived_inconsistencies={archived_threshold_inconsistencies}"
        )
    return {
        "archived_prediction_file": str(prediction_path.resolve()),
        "archived_prediction_file_sha256": sha256_file(prediction_path),
        "archived_probability_sha256_float32": array_sha256(
            archived["probability_phishing"].to_numpy(dtype=np.float32)
        ),
        "replayed_probability_sha256_float64": array_sha256(probabilities.astype(np.float64)),
        "max_abs_probability_delta_archived_float32": max_abs_delta,
        "threshold_class_changes_vs_archived": threshold_class_changes,
        "archived_threshold_inconsistencies": archived_threshold_inconsistencies,
        "float32_probability_vector_exact": float32_exact,
    }


def run_one(
    source: str,
    repetition: str,
    model_name: str,
    source_frame: pd.DataFrame,
    target_cohort: pd.DataFrame,
    target_cohort_metadata: dict[str, Any],
    names: list[str],
    args: argparse.Namespace,
) -> None:
    run_dir = args.output_root / "runs" / source / repetition / model_name
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not args.force:
        print(f"SKIP {source} {repetition} {model_name}", flush=True)
        return

    split = source_frame[f"split_{repetition}"].astype(str)
    train = source_frame.loc[split == "train"]
    test = source_frame.loc[split == "test"]
    background = deterministic_stratified_sample(
        train,
        args.background_size,
        f"phreshphish-external:{source}:{repetition}:background",
    )
    source_cohort = deterministic_stratified_sample(
        test,
        args.cohort_size,
        f"phreshphish-external:{source}:{repetition}:source-test",
    )

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
    params = json.loads(saved["selected_params_json"])
    X_train = train[names].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int8)
    model = build_model(model_name, params, int(saved["model_seed"]), args.n_jobs)
    fit_started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    test_probabilities = model.predict_proba(test[names].to_numpy(dtype=np.float32))[:, 1]
    replay_auc = float(roc_auc_score(test["label"], test_probabilities))
    replay_f1 = float(
        f1_score(
            test["label"],
            test_probabilities >= float(saved["threshold"]),
            average="macro",
        )
    )
    auc_delta = replay_auc - float(saved["roc_auc"])
    f1_delta = replay_f1 - float(saved["macro_f1"])
    if max(abs(auc_delta), abs(f1_delta)) > args.replay_tolerance:
        raise RuntimeError(
            f"Source replay drift for {source}/{repetition}/{model_name}: "
            f"AUC={auc_delta:.3g}, Macro-F1={f1_delta:.3g}"
        )
    archived_replay = verify_archived_source_predictions(
        source,
        repetition,
        model_name,
        test,
        test_probabilities,
        float(saved["threshold"]),
        args.prediction_replay_tolerance,
    )

    X_background = background[names].to_numpy(dtype=np.float32)
    explainer = build_explainer(model_name, model, X_background)
    X_source = source_cohort[names].to_numpy(dtype=np.float32)
    source_values, source_seconds, source_warnings = explain(explainer, X_source)
    X_target = target_cohort[names].to_numpy(dtype=np.float32)
    target_values, target_seconds, target_warnings = explain(explainer, X_target)
    common_metadata = {
        "purpose": "post_hoc_explanation_only",
        "source_dataset": source,
        "target_dataset": "phreshphish",
        "repetition": repetition,
        "model": model_name,
        "model_seed": int(saved["model_seed"]),
        "selected_candidate": saved["selected_candidate"],
        "selected_params": params,
        "feature_names": names,
        "fit_seconds": fit_seconds,
        "background_size": len(background),
        "background_sample_id_sha256": sample_id_sha256(background),
        "explainer_setup_seconds": explainer.setup_seconds,
        "model_output_scale": explainer.model_output_scale,
        "perturbation": explainer.perturbation,
        "expected_value": explainer.expected_value,
        "shap_version": shap.__version__,
        "completed_utc": utc_now(),
        "archived_source_prediction_replay": archived_replay,
    }
    source_summary_path, source_values_path = save_explanation(
        run_dir / "source_internal",
        source_cohort,
        names,
        X_source,
        source_values,
        {
            **common_metadata,
            "cohort": "source_internal_test",
            "cohort_size": len(source_cohort),
            "cohort_sample_id_sha256": sample_id_sha256(source_cohort),
            "explain_seconds": source_seconds,
            "warnings": source_warnings,
        },
    )
    target_summary_path, target_values_path = save_explanation(
        run_dir / "target_common",
        target_cohort,
        names,
        X_target,
        target_values,
        {
            **common_metadata,
            "cohort": "target_common_primary",
            "cohort_size": len(target_cohort),
            "cohort_sample_id_sha256": sample_id_sha256(target_cohort),
            "common_pool_rows": target_cohort_metadata["common_pool_rows"],
            "explain_seconds": target_seconds,
            "warnings": target_warnings,
        },
    )

    source_summary = pd.read_csv(source_summary_path)
    target_summary = pd.read_csv(target_summary_path)
    comparison = ranking_comparison(source_summary, target_summary)
    external = external_metric_row(source, repetition, model_name)
    context = {"source_dataset": source, "repetition": repetition, "model": model_name}
    conflicts = direction_conflicts(source_summary, target_summary, context)
    conflict_path = run_dir / "direction_conflicts.csv"
    conflicts.to_csv(conflict_path, index=False)
    comparison_payload = {
        **context,
        **comparison,
        "top15_direction_conflict_n": len(conflicts),
        "source_test_roc_auc": float(saved["roc_auc"]),
        "target_roc_auc": float(external["roc_auc"]),
        "roc_auc_loss": float(saved["roc_auc"] - external["roc_auc"]),
        "source_test_macro_f1": float(saved["macro_f1"]),
        "target_macro_f1": float(external["macro_f1"]),
        "macro_f1_loss": float(saved["macro_f1"] - external["macro_f1"]),
        "target_fpr": float(external["fpr"]),
        "target_recall": float(external["recall"]),
        "source_replay_roc_auc_delta": auc_delta,
        "source_replay_macro_f1_delta": f1_delta,
        "source_prediction_max_abs_delta_archived_float32": archived_replay[
            "max_abs_probability_delta_archived_float32"
        ],
        "source_threshold_class_changes_vs_archived": archived_replay[
            "threshold_class_changes_vs_archived"
        ],
        "source_archived_threshold_inconsistencies": archived_replay[
            "archived_threshold_inconsistencies"
        ],
        "source_float32_probability_vector_exact": archived_replay[
            "float32_probability_vector_exact"
        ],
        "source_explain_seconds": source_seconds,
        "target_explain_seconds": target_seconds,
        "warning_count": len(source_warnings) + len(target_warnings),
        "source_cohort_sha256": sample_id_sha256(source_cohort),
        "target_cohort_sha256": sample_id_sha256(target_cohort),
    }
    write_json(run_dir / "comparison.json", comparison_payload)
    write_json(
        complete_path,
        {
            "completed_utc": utc_now(),
            "comparison": comparison_payload,
            "artifacts": {
                "source_summary_sha256": sha256_file(source_summary_path),
                "source_values_sha256": sha256_file(source_values_path),
                "target_summary_sha256": sha256_file(target_summary_path),
                "target_values_sha256": sha256_file(target_values_path),
                "direction_conflicts_sha256": sha256_file(conflict_path),
            },
        },
    )
    print(
        f"DONE {source} {repetition} {model_name} "
        f"fit={fit_seconds:.1f}s source_shap={source_seconds:.1f}s "
        f"target_shap={target_seconds:.1f}s top15={comparison['top15_jaccard']:.3f}",
        flush=True,
    )
    del model, explainer, source_values, target_values
    gc.collect()


def aggregate_results(args: argparse.Namespace) -> dict[str, Any]:
    comparison_files = sorted((args.output_root / "runs").rglob("comparison.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in comparison_files]
    comparisons = pd.DataFrame(rows)
    expected = len(args.sources) * len(args.repetitions) * len(args.models)
    if len(comparisons) != expected:
        raise RuntimeError(f"Expected {expected} comparison rows, found {len(comparisons)}")
    key = ["source_dataset", "repetition", "model"]
    if comparisons[key].duplicated().any():
        raise RuntimeError("Duplicate SHAP comparison keys")
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_root / "comparison_metrics.csv"
    comparisons.sort_values(key).to_csv(comparison_path, index=False)

    conflict_files = sorted((args.output_root / "runs").rglob("direction_conflicts.csv"))
    conflict_frames = []
    for path in conflict_files:
        frame = pd.read_csv(path)
        if len(frame):
            conflict_frames.append(frame)
    conflicts = pd.concat(conflict_frames, ignore_index=True) if conflict_frames else pd.DataFrame()
    conflict_path = args.output_root / "direction_conflicts.csv"
    conflicts.to_csv(conflict_path, index=False)
    if len(conflicts):
        conflict_summary = (
            conflicts.groupby(["source_dataset", "model", "feature"], as_index=False)
            .agg(
                conflict_repetitions=("repetition", "nunique"),
                source_rank_mean=("source_rank", "mean"),
                target_rank_mean=("target_rank", "mean"),
            )
            .sort_values(["source_dataset", "model", "conflict_repetitions", "feature"], ascending=[True, True, False, True])
        )
    else:
        conflict_summary = pd.DataFrame(
            columns=["source_dataset", "model", "feature", "conflict_repetitions"]
        )
    conflict_summary_path = args.output_root / "feature_conflict_summary.csv"
    conflict_summary.to_csv(conflict_summary_path, index=False)

    summary_columns = [
        "top10_jaccard",
        "top15_jaccard",
        "top20_jaccard",
        "rank_spearman",
        "direction_agreement_all",
        "direction_agreement_top20_union",
        "top15_direction_conflict_n",
        "source_test_roc_auc",
        "target_roc_auc",
        "roc_auc_loss",
        "source_test_macro_f1",
        "target_macro_f1",
        "macro_f1_loss",
        "target_fpr",
        "target_recall",
    ]
    summary = comparisons.groupby(["source_dataset", "model"])[summary_columns].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = args.output_root / "aggregate_summary.csv"
    summary.to_csv(summary_path, index=False)

    association_rows = []
    for scope_name, frame in [("all", comparisons), *comparisons.groupby("source_dataset")]:
        for stability_metric in ("top15_jaccard", "rank_spearman", "direction_agreement_top20_union"):
            for loss_metric in ("roc_auc_loss", "macro_f1_loss"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho = spearmanr(frame[stability_metric], frame[loss_metric]).statistic
                association_rows.append({
                    "scope": scope_name,
                    "stability_metric": stability_metric,
                    "loss_metric": loss_metric,
                    "n": len(frame),
                    "spearman_rho_descriptive": float(rho) if np.isfinite(rho) else np.nan,
                })
    association = pd.DataFrame(association_rows)
    association_path = args.output_root / "stability_performance_association.csv"
    association.to_csv(association_path, index=False)

    expected_keys = {
        (source, repetition, model)
        for source in args.sources
        for repetition in args.repetitions
        for model in args.models
    }
    actual_keys = set(map(tuple, comparisons[key].itertuples(index=False, name=None)))
    shap_value_files = sorted((args.output_root / "runs").rglob("shap_values.parquet"))
    summary_files = sorted((args.output_root / "runs").rglob("global_summary.csv"))
    value_rows = [len(pd.read_parquet(path, columns=["sample_id"])) for path in shap_value_files]
    validation = {
        "status": "PASS",
        "scope": args.scope,
        "expected_runs": expected,
        "comparison_rows": len(comparisons),
        "run_keys_exact": actual_keys == expected_keys,
        "shap_value_files": len(shap_value_files),
        "global_summary_files": len(summary_files),
        "shap_value_row_counts": sorted(set(value_rows)),
        "comparison_missing_values": int(comparisons.isna().sum().sum()),
        "max_abs_source_replay_auc_delta": float(comparisons["source_replay_roc_auc_delta"].abs().max()),
        "max_abs_source_replay_macro_f1_delta": float(comparisons["source_replay_macro_f1_delta"].abs().max()),
        "max_source_probability_delta_archived_float32": float(
            comparisons["source_prediction_max_abs_delta_archived_float32"].max()
        ),
        "source_threshold_class_changes_total": int(
            comparisons["source_threshold_class_changes_vs_archived"].sum()
        ),
        "source_archived_threshold_inconsistencies_total": int(
            comparisons["source_archived_threshold_inconsistencies"].sum()
        ),
        "source_float32_probability_vector_exact_all": bool(
            comparisons["source_float32_probability_vector_exact"].all()
        ),
        "warning_count": int(comparisons["warning_count"].sum()),
        "target_cohort_hashes_per_source": comparisons.groupby("source_dataset")["target_cohort_sha256"].nunique().to_dict(),
        "comparison_metrics_sha256": sha256_file(comparison_path),
        "aggregate_summary_sha256": sha256_file(summary_path),
        "direction_conflicts_sha256": sha256_file(conflict_path),
        "feature_conflict_summary_sha256": sha256_file(conflict_summary_path),
        "association_sha256": sha256_file(association_path),
    }
    if not validation["run_keys_exact"]:
        raise RuntimeError("SHAP run keys are incomplete")
    if validation["shap_value_files"] != 2 * expected or validation["global_summary_files"] != 2 * expected:
        raise RuntimeError("SHAP artifact count differs from the protocol")
    if validation["shap_value_row_counts"] != [args.cohort_size]:
        raise RuntimeError("SHAP cohort row counts differ from the protocol")
    if validation["comparison_missing_values"] != 0:
        raise RuntimeError("Missing comparison values")
    if validation["max_abs_source_replay_auc_delta"] > args.replay_tolerance:
        raise RuntimeError("Source AUC replay drift exceeds tolerance")
    if validation["max_abs_source_replay_macro_f1_delta"] > args.replay_tolerance:
        raise RuntimeError("Source Macro-F1 replay drift exceeds tolerance")
    if validation["max_source_probability_delta_archived_float32"] > args.prediction_replay_tolerance:
        raise RuntimeError("Source probability replay drift exceeds tolerance")
    if validation["source_threshold_class_changes_total"] != 0:
        raise RuntimeError("Source threshold predictions differ from the archive")
    if validation["source_archived_threshold_inconsistencies_total"] != 0:
        raise RuntimeError("Archived source predictions disagree with their frozen thresholds")
    if any(value != 1 for value in validation["target_cohort_hashes_per_source"].values()):
        raise RuntimeError("Target cohort changed within a source direction")
    write_json(args.output_root / "validation_summary.json", validation)
    return validation


def main() -> None:
    args = parse_args()
    args.target_features = args.target_features.resolve()
    args.output_root = args.output_root.resolve()
    if args.cohort_size % 2 or args.background_size % 2:
        raise ValueError("Cohort and background sizes must be even")
    if args.scope == "full":
        if set(args.sources) != set(SOURCE_LAYOUT) or set(args.repetitions) != set(REPETITIONS) or set(args.models) != set(MODELS):
            raise RuntimeError("A full run requires both sources, ten repetitions, and all three models")

    names = feature_names()
    target, target_metadata = load_target(args.target_features, names)
    numerical_threadpools = validated_threadpools()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "run_metadata.json",
        {
            "started_utc": utc_now(),
            "scope": args.scope,
            "arguments": vars(args),
            "target_features": str(args.target_features),
            "target_features_sha256": target_metadata["output_sha256"],
            "target_revision": target_metadata["dataset_revision"],
            "feature_names": names,
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "shap": shap.__version__,
                "xgboost": xgboost.__version__,
            },
            "numerical_replay": {
                "expected_threads": EXPECTED_NUMERICAL_THREADS,
                "environment_before_override": THREAD_ENV_BEFORE_OVERRIDE,
                "environment_effective": {
                    name: os.environ.get(name) for name in THREAD_ENV_BEFORE_OVERRIDE
                },
                "threadpools": numerical_threadpools,
                "prediction_replay_tolerance": args.prediction_replay_tolerance,
            },
        },
    )

    started = time.perf_counter()
    sources = {source: load_source(source, names) for source in args.sources}
    target_cohorts = {}
    target_cohort_metadata = {}
    for source in args.sources:
        cohort, metadata = build_target_common_cohort(
            source,
            target,
            args.repetitions,
            args.models,
            args.cohort_size,
            args.output_root,
        )
        target_cohorts[source] = cohort
        target_cohort_metadata[source] = metadata

    for source in args.sources:
        for repetition in args.repetitions:
            for model_name in args.models:
                run_one(
                    source,
                    repetition,
                    model_name,
                    sources[source],
                    target_cohorts[source],
                    target_cohort_metadata[source],
                    names,
                    args,
                )

    validation = aggregate_results(args)
    final_metadata_path = args.output_root / "run_metadata.json"
    final_metadata = json.loads(final_metadata_path.read_text(encoding="utf-8"))
    final_metadata.update({
        "completed_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "FULL_RUN_PASS" if args.scope == "full" else "SMOKE_PASS",
        "validation": validation,
    })
    write_json(final_metadata_path, final_metadata)
    print(json.dumps({
        "status": final_metadata["status"],
        "elapsed_seconds": final_metadata["elapsed_seconds"],
        "validation": validation,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
