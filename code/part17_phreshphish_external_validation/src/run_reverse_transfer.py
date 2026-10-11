from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT.parent
CONFIG_PATH = ROOT / "config" / "experiment.json"
TARGETS = {
    "iscx_master": "iscx_url2016_binary_master_features.parquet",
    "iscx_deduplicated": "iscx_url2016_binary_deduplicated_features.parquet",
    "phiusiil_master": "phiusiil_master_features.parquet",
    "phiusiil_deduplicated": "phiusiil_deduplicated_features.parquet",
}
REPETITIONS = tuple(f"r{i:02d}" for i in range(10))
MODELS = ("lr", "rf", "xgb")
PARTITION_SEEDS = {repetition: 20260902 + index for index, repetition in enumerate(REPETITIONS)}
MODEL_SEEDS = {repetition: 20261001 + index for index, repetition in enumerate(REPETITIONS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on PhreshPhish and evaluate reciprocal transfer")
    parser.add_argument(
        "source_features",
        type=Path,
        nargs="?",
        default=ROOT / "data" / "features" / "server_train" / "phreshphish_train.parquet",
    )
    parser.add_argument("--source-experiment-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=list(REPETITIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--output-root", type=Path)
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def balanced_limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.reset_index(drop=True)
    per_class = limit // 2
    selected = (
        frame.sort_values(["label", "source_row"], kind="stable")
        .groupby("label", sort=True, group_keys=False)
        .head(per_class)
    )
    if len(selected) != 2 * per_class:
        raise ValueError("Target smoke limit could not retain both labels equally")
    return selected.reset_index(drop=True)


def load_targets(
    part2: Path,
    target_names: list[str],
    feature_names: list[str],
    target_limit: int | None,
) -> dict[str, pd.DataFrame]:
    columns = ["source_row", "label", "normalized_url_sha256", "host", "registrable_domain", *feature_names]
    loaded: dict[str, pd.DataFrame] = {}
    for target_name in target_names:
        target_columns = list(columns)
        if target_name.startswith("iscx_"):
            target_columns.insert(1, "source_file")
        frame = pd.read_parquet(part2 / "data" / TARGETS[target_name], columns=target_columns)
        frame = balanced_limit(frame, target_limit)
        if set(frame["label"].unique()) != {0, 1}:
            raise ValueError(f"Target {target_name} does not contain both labels")
        if "source_file" in frame.columns:
            frame["sample_id"] = (
                target_name
                + ":"
                + frame["source_file"].astype(str)
                + ":"
                + frame["source_row"].astype(str)
            )
        else:
            frame["sample_id"] = target_name + ":" + frame["source_row"].astype(str)
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Target {target_name} sample_id is not unique")
        frame["host_sha256"] = frame["host"].fillna("").astype(str).map(text_sha256)
        frame["registrable_domain_sha256"] = (
            frame["registrable_domain"].fillna("").astype(str).map(text_sha256)
        )
        loaded[target_name] = frame
    return loaded


def build_assignments(
    source: pd.DataFrame,
    repetitions: list[str],
    grouped_stratified_assignment: Any,
    assert_group_disjoint: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = source[
        ["sample_id", "provider_sha256", "label", "registrable_domain_sha256"]
    ].copy()
    audits: list[dict[str, object]] = []
    labels = source["label"].to_numpy(dtype=np.int8)
    groups = source["registrable_domain_sha256"].astype(str).to_numpy()
    for repetition in repetitions:
        split = grouped_stratified_assignment(labels, groups, PARTITION_SEEDS[repetition])
        assert_group_disjoint(split, groups)
        assignments[f"split_{repetition}"] = split
        for split_name in ("train", "validation", "test"):
            mask = split == split_name
            split_labels = labels[mask]
            if set(np.unique(split_labels)) != {0, 1}:
                raise ValueError(f"{repetition}/{split_name} does not contain both labels")
            audits.append(
                {
                    "repetition": repetition,
                    "partition_seed": PARTITION_SEEDS[repetition],
                    "split": split_name,
                    "n_samples": int(mask.sum()),
                    "n_benign": int((split_labels == 0).sum()),
                    "n_phishing": int((split_labels == 1).sum()),
                    "n_domains": int(np.unique(groups[mask]).size),
                    "proportion": float(mask.mean()),
                }
            )
    return assignments, pd.DataFrame(audits)


def run_one(
    source: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    feature_names: list[str],
    repetition: str,
    model_name: str,
    config: dict[str, Any],
    output_root: Path,
    select_model: Any,
    select_fpr_threshold: Any,
    metrics: Any,
    n_jobs: int,
    force: bool,
) -> None:
    run_dir = output_root / "runs" / repetition / model_name
    complete_path = run_dir / "complete.json"
    expected_files = [
        run_dir / "internal_metrics.csv",
        run_dir / "external_metrics.csv",
        run_dir / "tuning_results.csv",
        run_dir / "source_test_predictions.parquet",
    ]
    if complete_path.exists() and not force:
        if not all(path.exists() for path in expected_files):
            raise RuntimeError(f"Complete marker exists but run files are missing: {run_dir}")
        return
    if any(path.exists() for path in expected_files) and not force:
        raise FileExistsError(f"Partial run exists; inspect it or pass --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    split = source[f"split_{repetition}"].astype(str).to_numpy()
    masks = {name: split == name for name in ("train", "validation", "test")}
    source_X = source[feature_names].to_numpy(dtype=np.float32)
    source_y = source["label"].to_numpy(dtype=np.int8)
    source_groups = source["registrable_domain_sha256"].astype(str).to_numpy()
    started = time.perf_counter()
    selection = select_model(
        model_name,
        source_X[masks["train"]],
        source_y[masks["train"]],
        source_X[masks["validation"]],
        source_y[masks["validation"]],
        MODEL_SEEDS[repetition],
        n_jobs,
    )
    if list(selection.model.classes_) != [0, 1]:
        raise RuntimeError(f"Unexpected model classes for {repetition}/{model_name}")

    validation_probabilities = selection.model.predict_proba(source_X[masks["validation"]])[:, 1]
    thresholds: list[tuple[str, float, float | None]] = [
        ("macro_f1", float(selection.threshold), None)
    ]
    for budget in config["fpr_budgets"]:
        thresholds.append(
            (
                f"fpr_{float(budget):g}",
                select_fpr_threshold(source_y[masks["validation"]], validation_probabilities, float(budget)),
                float(budget),
            )
        )

    test_probabilities = selection.model.predict_proba(source_X[masks["test"]])[:, 1]
    source_prediction = pd.DataFrame(
        {
            "sample_id": source.loc[masks["test"], "sample_id"].to_numpy(),
            "label": source_y[masks["test"]],
            "registrable_domain_sha256": source_groups[masks["test"]],
            "probability_phishing": test_probabilities,
        }
    )
    source_prediction.to_parquet(
        run_dir / "source_test_predictions.parquet", index=False, compression="zstd"
    )

    common = {
        "source_dataset": "phreshphish_train",
        "repetition": repetition,
        "partition_seed": PARTITION_SEEDS[repetition],
        "model": model_name,
        "model_seed": MODEL_SEEDS[repetition],
        "selected_candidate": selection.selected_candidate,
        "selected_params_json": json.dumps(selection.selected_params, sort_keys=True),
        "source_train_n": int(masks["train"].sum()),
        "source_validation_n": int(masks["validation"].sum()),
        "source_test_n": int(masks["test"].sum()),
    }
    internal_rows: list[dict[str, object]] = []
    for threshold_kind, threshold, budget in thresholds:
        for weighting in ("row", "registrable_domain_equal"):
            internal_rows.append(
                {
                    **common,
                    "evaluation_dataset": "phreshphish_internal_domain_test",
                    "threshold_kind": threshold_kind,
                    "source_validation_fpr_budget": budget,
                    **metrics(
                        source_y[masks["test"]],
                        test_probabilities,
                        threshold,
                        int(config["ece_bins"]),
                        weighting,
                        source_groups[masks["test"]],
                    ),
                }
            )

    fit_scope = masks["train"] | masks["validation"]
    fit_url_hashes = set(source.loc[fit_scope, "normalized_url_sha256"].astype(str))
    fit_host_hashes = set(source.loc[fit_scope, "host_sha256"].astype(str))
    fit_domain_hashes = set(source.loc[fit_scope, "registrable_domain_sha256"].astype(str))
    external_rows: list[dict[str, object]] = []
    prediction_hashes: dict[str, str] = {}
    for target_name, target in targets.items():
        target_X = target[feature_names].to_numpy(dtype=np.float32)
        target_y = target["label"].to_numpy(dtype=np.int8)
        target_groups = target["registrable_domain_sha256"].astype(str).to_numpy()
        probabilities = selection.model.predict_proba(target_X)[:, 1]
        url_overlap = target["normalized_url_sha256"].isin(fit_url_hashes).to_numpy()
        host_overlap = target["host_sha256"].isin(fit_host_hashes).to_numpy()
        domain_overlap = target["registrable_domain_sha256"].isin(fit_domain_hashes).to_numpy()
        primary = ~domain_overlap
        if not primary.any() or set(np.unique(target_y[primary])) != {0, 1}:
            raise RuntimeError(f"Invalid primary cohort for {target_name}/{repetition}/{model_name}")
        target_auc = float(roc_auc_score(target_y, probabilities))
        complement_error = abs(target_auc + roc_auc_score(target_y, 1.0 - probabilities) - 1.0)
        negation_error = abs(target_auc + roc_auc_score(target_y, -probabilities) - 1.0)

        prediction_path = run_dir / "target_predictions" / f"{target_name}.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "sample_id": target["sample_id"],
                "label": target_y,
                "registrable_domain_sha256": target_groups,
                "probability_phishing": probabilities,
                "included_in_primary": primary,
            }
        ).to_parquet(prediction_path, index=False, compression="zstd")
        prediction_hashes[target_name] = sha256_file(prediction_path)

        for cohort, mask in (
            ("unfiltered", np.ones(len(target), dtype=bool)),
            ("primary_domain_filtered", primary),
        ):
            for threshold_kind, threshold, budget in thresholds:
                for weighting in ("row", "registrable_domain_equal"):
                    external_rows.append(
                        {
                            **common,
                            "target_dataset": target_name,
                            "cohort": cohort,
                            "threshold_kind": threshold_kind,
                            "source_validation_fpr_budget": budget,
                            "target_normalized_url_overlap_n": int(url_overlap.sum()),
                            "target_host_overlap_n": int(host_overlap.sum()),
                            "target_registrable_domain_overlap_n": int(domain_overlap.sum()),
                            "auc_complement_identity_error": float(complement_error),
                            "auc_negation_identity_error": float(negation_error),
                            "probability_zero_n": int((probabilities == 0.0).sum()),
                            "probability_one_n": int((probabilities == 1.0).sum()),
                            **metrics(
                                target_y[mask],
                                probabilities[mask],
                                threshold,
                                int(config["ece_bins"]),
                                weighting,
                                target_groups[mask],
                            ),
                        }
                    )

    internal_frame = pd.DataFrame(internal_rows)
    external_frame = pd.DataFrame(external_rows)
    tuning_frame = pd.DataFrame(selection.tuning_rows)
    write_csv(run_dir / "internal_metrics.csv", internal_frame)
    write_csv(run_dir / "external_metrics.csv", external_frame)
    write_csv(run_dir / "tuning_results.csv", tuning_frame)
    write_json(run_dir / "warnings.json", selection.warnings)
    write_json(
        complete_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "repetition": repetition,
            "model": model_name,
            "elapsed_seconds": time.perf_counter() - started,
            "internal_metrics_sha256": sha256_file(run_dir / "internal_metrics.csv"),
            "external_metrics_sha256": sha256_file(run_dir / "external_metrics.csv"),
            "tuning_results_sha256": sha256_file(run_dir / "tuning_results.csv"),
            "source_predictions_sha256": sha256_file(run_dir / "source_test_predictions.parquet"),
            "target_prediction_sha256": prediction_hashes,
            "warning_count": len(selection.warnings),
        },
    )


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_path = args.source_features.resolve()
    metadata_path = source_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["dataset_revision"] != config["dataset"]["revision"]:
        raise RuntimeError("Source revision does not match the locked experiment")
    is_full = metadata["scope"] == "full_split" and metadata["split"] == "train"
    if not is_full and not args.allow_smoke:
        raise RuntimeError("Only the full PhreshPhish train split is evidence; use --allow-smoke for testing")
    if args.target_limit is not None and not args.allow_smoke:
        raise RuntimeError("--target-limit is permitted only for smoke testing")

    source_root = args.source_experiment_root.resolve()
    part2 = source_root / "part2_url_normalization_leakage_audit"
    part3 = source_root / "part3_s0_s4_data_splits"
    part4 = source_root / "part4_baseline_models"
    sys.path.insert(0, str(part3 / "scripts"))
    sys.path.insert(0, str(part4 / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    from split_utils import assert_group_disjoint, grouped_stratified_assignment
    from model_utils import select_model
    from evaluate_frozen import metrics, select_fpr_threshold

    dictionary = pd.read_csv(part2 / "results" / "feature_dictionary.csv").sort_values("feature_order")
    feature_names = dictionary["feature"].tolist()
    if feature_names != metadata["feature_names"] or len(feature_names) != 35:
        raise RuntimeError("Source feature contract differs from the frozen 35-feature contract")
    required = [
        "sample_id",
        "provider_sha256",
        "label",
        "normalized_url_sha256",
        "host_sha256",
        "registrable_domain_sha256",
        *feature_names,
    ]
    source = pd.read_parquet(source_path, columns=required)
    if len(source) != int(metadata["rows_written"]):
        raise RuntimeError("Source row count differs from metadata")
    if source["sample_id"].duplicated().any():
        raise RuntimeError("Source sample_id is not unique")
    if set(source["label"].unique()) != {0, 1}:
        raise RuntimeError("Source does not contain both labels")

    scope = "full_split" if is_full else "smoke_only"
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else ROOT / "results" / "reverse_transfer" / scope
    )
    output_root.mkdir(parents=True, exist_ok=True)
    assignments, split_audit = build_assignments(
        source,
        args.repetitions,
        grouped_stratified_assignment,
        assert_group_disjoint,
    )
    for column in [f"split_{repetition}" for repetition in args.repetitions]:
        source[column] = assignments[column]
    assignment_path = output_root / "phreshphish_train_s3_domain_assignments.parquet"
    audit_path = output_root / "split_audit.csv"
    if (assignment_path.exists() or audit_path.exists()) and not args.force:
        existing = pd.read_parquet(assignment_path)
        if not existing.equals(assignments):
            raise RuntimeError("Existing source assignments differ from deterministic regeneration")
    else:
        assignments.to_parquet(assignment_path, index=False, compression="zstd")
        write_csv(audit_path, split_audit)

    targets = load_targets(part2, args.targets, feature_names, args.target_limit)
    started = time.perf_counter()
    expected_runs: list[Path] = []
    for repetition in args.repetitions:
        for model_name in args.models:
            run_one(
                source,
                targets,
                feature_names,
                repetition,
                model_name,
                config,
                output_root,
                select_model,
                select_fpr_threshold,
                metrics,
                args.n_jobs,
                args.force,
            )
            expected_runs.append(output_root / "runs" / repetition / model_name)

    internal = pd.concat([pd.read_csv(path / "internal_metrics.csv") for path in expected_runs], ignore_index=True)
    external = pd.concat([pd.read_csv(path / "external_metrics.csv") for path in expected_runs], ignore_index=True)
    tuning = pd.concat(
        [
            pd.read_csv(path / "tuning_results.csv").assign(
                repetition=path.parent.name, model=path.name
            )
            for path in expected_runs
        ],
        ignore_index=True,
    )
    internal_path = output_root / "internal_metrics.csv"
    external_path = output_root / "external_metrics.csv"
    tuning_path = output_root / "tuning_results.csv"
    write_csv(internal_path, internal)
    write_csv(external_path, external)
    write_csv(tuning_path, tuning)
    complete_files = [path / "complete.json" for path in expected_runs]
    warning_count = sum(json.loads(path.read_text(encoding="utf-8"))["warning_count"] for path in complete_files)
    run_metadata = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FULL_RUN_PASS" if is_full else "SMOKE_PASS",
        "interpretation_boundary": (
            "Reciprocal-transfer diagnostic; ISCX and PhiUSIIL are prior-study corpora, not new independent validation."
            if is_full
            else "Pipeline-only smoke test; outputs are not manuscript evidence."
        ),
        "source_features": str(source_path),
        "source_features_sha256": sha256_file(source_path),
        "source_metadata_sha256": sha256_file(metadata_path),
        "source_rows": len(source),
        "feature_count": len(feature_names),
        "repetitions": args.repetitions,
        "models": args.models,
        "targets": args.targets,
        "target_limit": args.target_limit,
        "completed_runs": len(expected_runs),
        "internal_metric_rows": len(internal),
        "external_metric_rows": len(external),
        "tuning_rows": len(tuning),
        "prediction_files": len(expected_runs) * (1 + len(args.targets)),
        "warning_count": warning_count,
        "max_auc_complement_identity_error": float(external["auc_complement_identity_error"].max()),
        "max_auc_negation_identity_error": float(external["auc_negation_identity_error"].max()),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "n_jobs": args.n_jobs,
        "assignment_sha256": sha256_file(assignment_path),
        "split_audit_sha256": sha256_file(audit_path),
        "internal_metrics_sha256": sha256_file(internal_path),
        "external_metrics_sha256": sha256_file(external_path),
        "tuning_results_sha256": sha256_file(tuning_path),
    }
    write_json(output_root / "run_metadata.json", run_metadata)
    print(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()
