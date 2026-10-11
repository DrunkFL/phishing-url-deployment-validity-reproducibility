from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from http_parquet import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_READ_TIMEOUT,
    HTTPRangeReader,
    RETRYABLE_STATUS_CODES,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT.parent
CONFIG_PATH = ROOT / "config" / "experiment.json"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "phreshphish_files.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the URL-only PhreshPhish feature table")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test row limit; omit for the full split")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-experiment-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_frozen_feature_code(source_root: Path):
    scripts = source_root / "part2_url_normalization_leakage_audit" / "scripts"
    if not scripts.is_dir():
        raise FileNotFoundError(f"Frozen Part 2 scripts not found: {scripts}")
    sys.path.insert(0, str(scripts))
    from feature_extraction import FEATURE_NAMES, FEATURE_SPECS, extract_url_features, feature_vector_sha256
    from url_normalization import PSL_SNAPSHOT, normalize_url

    return FEATURE_NAMES, FEATURE_SPECS, extract_url_features, feature_vector_sha256, PSL_SNAPSHOT, normalize_url


def output_schema(feature_specs) -> pa.Schema:
    fields = [
        pa.field("sample_id", pa.string()),
        pa.field("provider_sha256", pa.string()),
        pa.field("split", pa.string()),
        pa.field("split_row", pa.int64()),
        pa.field("label", pa.int8()),
        pa.field("date", pa.date32()),
        pa.field("raw_url_sha256", pa.string()),
        pa.field("normalized_url_sha256", pa.string()),
        pa.field("host_sha256", pa.string()),
        pa.field("registrable_domain_sha256", pa.string()),
        pa.field("parse_status", pa.string()),
        pa.field("feature_vector_sha256", pa.string()),
    ]
    fields.extend(
        pa.field(spec.name, pa.int64() if spec.dtype == "int" else pa.float64())
        for spec in feature_specs
    )
    return pa.schema(fields)


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["revision"] != config["dataset"]["revision"]:
        raise RuntimeError("Dataset manifest revision does not match the frozen configuration")
    selected_columns = config["dataset"]["selected_columns"]
    forbidden_columns = set(config["dataset"]["forbidden_columns"])
    if forbidden_columns.intersection(selected_columns):
        raise RuntimeError("A forbidden column was selected")

    (
        feature_names,
        feature_specs,
        extract_url_features,
        feature_vector_sha256,
        psl_snapshot,
        normalize_url,
    ) = import_frozen_feature_code(args.source_experiment_root.resolve())
    if len(feature_names) != config["feature_count"]:
        raise RuntimeError(f"Expected {config['feature_count']} features, found {len(feature_names)}")

    suffix = f"_{args.limit}_row_smoke" if args.limit is not None else ""
    output = args.output or ROOT / "data" / "features" / f"phreshphish_{args.split}{suffix}.parquet"
    metadata_output = output.with_suffix(".metadata.json")
    if (output.exists() or metadata_output.exists()) and not args.force:
        raise FileExistsError(f"Output already exists; use --force to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    shards = [item for item in manifest["files"] if item["split"] == args.split]
    expected_rows = int(config["dataset"]["expected_rows"][args.split])
    label_mapping = config["dataset"]["label_mapping"]
    schema = output_schema(feature_specs)
    writer = pq.ParquetWriter(temporary, schema=schema, compression="zstd")
    invalid_rows = []
    label_counts: Counter[int] = Counter()
    rows_scanned = 0
    rows_written = 0
    shard_offset = 0
    opened_shards = []
    started = datetime.now(timezone.utc)
    try:
        for shard in shards:
            if args.limit is not None and rows_scanned >= args.limit:
                break
            reader = HTTPRangeReader(shard["url"])
            python_file = pa.PythonFile(reader, mode="r")
            parquet = pq.ParquetFile(python_file)
            shard_rows = parquet.metadata.num_rows
            opened_shards.append(
                {
                    "path": shard["path"],
                    "bytes": shard["bytes"],
                    "sha256_from_hub_lfs": shard["sha256"],
                    "rows": shard_rows,
                }
            )
            local_row = 0
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=selected_columns):
                values = batch.to_pydict()
                output_rows = []
                for index in range(batch.num_rows):
                    if args.limit is not None and rows_scanned >= args.limit:
                        break
                    split_row = shard_offset + local_row
                    local_row += 1
                    rows_scanned += 1
                    url = "" if values["url"][index] is None else str(values["url"][index])
                    label_text = str(values["label"][index])
                    if label_text not in label_mapping:
                        raise ValueError(f"Unexpected label {label_text!r} at {args.split}:{split_row}")
                    label = int(label_mapping[label_text])
                    normalized = normalize_url(url, keep_fragment=False)
                    if normalized.parse_status != "ok":
                        invalid_rows.append(
                            {
                                "split_row": split_row,
                                "provider_sha256": values["sha256"][index],
                                "label": label,
                                "parse_status": normalized.parse_status,
                            }
                        )
                        continue
                    features = extract_url_features(url)
                    output_rows.append(
                        {
                            "sample_id": f"phreshphish:{args.split}:{split_row}",
                            "provider_sha256": values["sha256"][index],
                            "split": args.split,
                            "split_row": split_row,
                            "label": label,
                            "date": values["date"][index],
                            "raw_url_sha256": text_sha256(url),
                            "normalized_url_sha256": text_sha256(normalized.value),
                            "host_sha256": text_sha256(normalized.host),
                            "registrable_domain_sha256": text_sha256(normalized.registrable_domain),
                            "parse_status": normalized.parse_status,
                            "feature_vector_sha256": feature_vector_sha256(features),
                            **features,
                        }
                    )
                    label_counts[label] += 1
                if output_rows:
                    writer.write_table(pa.Table.from_pylist(output_rows, schema=schema))
                    rows_written += len(output_rows)
                if args.limit is not None and rows_scanned >= args.limit:
                    break
            python_file.close()
            shard_offset += shard_rows
    finally:
        writer.close()

    if args.limit is None and rows_scanned != expected_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {expected_rows} rows, scanned {rows_scanned}")
    temporary.replace(output)
    scope = "full_split" if args.limit is None else "smoke_only"
    metadata = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "dataset_repository": manifest["repository"],
        "dataset_release": manifest["release"],
        "dataset_revision": manifest["revision"],
        "split": args.split,
        "expected_full_split_rows": expected_rows,
        "rows_scanned": rows_scanned,
        "rows_written": rows_written,
        "invalid_url_rows": len(invalid_rows),
        "invalid_rows": invalid_rows,
        "label_counts_written": dict(sorted(label_counts.items())),
        "selected_columns": selected_columns,
        "forbidden_columns_not_read": sorted(forbidden_columns),
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "feature_code": str(args.source_experiment_root.resolve() / "part2_url_normalization_leakage_audit" / "scripts"),
        "psl_snapshot": str(psl_snapshot),
        "psl_sha256": sha256_file(psl_snapshot),
        "remote_shards": opened_shards,
        "remote_reader": {
            "block_size_bytes": DEFAULT_BLOCK_SIZE,
            "connect_timeout_seconds": DEFAULT_CONNECT_TIMEOUT,
            "read_timeout_seconds": DEFAULT_READ_TIMEOUT,
            "max_retries_after_initial_attempt": DEFAULT_MAX_RETRIES,
            "backoff_factor_seconds": DEFAULT_BACKOFF_FACTOR,
            "retryable_status_codes": list(RETRYABLE_STATUS_CODES),
        },
        "remote_integrity_boundary": "Shard SHA-256 values come from the pinned Hugging Face LFS manifest; full remote shard bytes were not downloaded locally.",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    write_json(metadata_output, metadata)
    print(json.dumps({key: metadata[key] for key in ("scope", "rows_scanned", "rows_written", "invalid_url_rows", "label_counts_written", "output", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
