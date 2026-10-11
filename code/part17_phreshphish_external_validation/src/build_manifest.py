from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment.json"
DEFAULT_OUTPUT = ROOT / "data" / "manifests" / "phreshphish_files.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pin the PhreshPhish repository and Parquet files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    repository = config["dataset"]["repository"]
    revision = config["dataset"]["revision"]
    encoded_repository = quote(repository, safe="/")

    metadata_url = f"https://huggingface.co/api/datasets/{encoded_repository}/revision/{revision}"
    tree_url = (
        f"https://huggingface.co/api/datasets/{encoded_repository}/tree/"
        f"{revision}/data?recursive=true&expand=true&limit=100"
    )
    session = requests.Session()
    metadata_response = session.get(metadata_url, timeout=60)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    if metadata.get("sha") != revision:
        raise RuntimeError(f"Resolved revision {metadata.get('sha')} does not match {revision}")

    tree_response = session.get(tree_url, timeout=60)
    tree_response.raise_for_status()
    entries = tree_response.json()
    files = []
    for entry in entries:
        path = entry.get("path", "")
        if not path.endswith(".parquet") or not path.startswith("data/"):
            continue
        filename = Path(path).name
        split = filename.split("-", 1)[0]
        lfs = entry.get("lfs") or {}
        files.append(
            {
                "path": path,
                "split": split,
                "bytes": int(entry["size"]),
                "git_oid": entry.get("oid"),
                "sha256": lfs.get("oid"),
                "url": (
                    f"https://huggingface.co/datasets/{repository}/resolve/"
                    f"{revision}/{path}"
                ),
            }
        )

    files.sort(key=lambda item: item["path"])
    counts = {split: sum(item["split"] == split for item in files) for split in ("train", "test")}
    if counts != {"train": 56, "test": 21}:
        raise RuntimeError(f"Unexpected pinned shard counts: {counts}")
    if any(not item["sha256"] for item in files):
        raise RuntimeError("At least one Parquet shard lacks an LFS SHA-256 identifier")

    payload = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "revision": revision,
        "release": config["dataset"]["release"],
        "license": config["dataset"]["license"],
        "repository_last_modified": metadata.get("lastModified"),
        "private": metadata.get("private"),
        "gated": metadata.get("gated"),
        "shard_counts": counts,
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    write_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("repository", "revision", "shard_counts", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
