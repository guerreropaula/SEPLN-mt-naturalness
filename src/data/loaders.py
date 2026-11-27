"""Loaders for the training and evaluation corpora."""

from __future__ import annotations

import logging
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset as hf_load_dataset

from src.utils.config import get_dataset_entry
from src.utils.errors import PipelineError
from src.utils.hf_auth import get_hf_token
from src.utils.io import save_dataframe_jsonl, save_json

logger = logging.getLogger(__name__)


def _with_metadata(df: pd.DataFrame, dataset_config: dict[str, Any]) -> pd.DataFrame:
    df.attrs["dataset_metadata"] = {
        "dataset_name": dataset_config["name"],
        "corpus_id": dataset_config["corpus_id"],
        "source_lang": dataset_config["source_lang"],
        "target_lang": dataset_config["target_lang"],
        "input_rows": len(df),
    }
    return df


def load_ehu_hac(dataset_config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Load aligned EHU-HAC files in their original order."""
    paths = dataset_config["paths"]
    root = Path(paths["root_dir"])
    source_path = root / paths["source_file"]
    target_path = root / paths["target_file"]

    with source_path.open(encoding="utf-8") as source_file:
        source_lines = [line.rstrip("\n") for line in source_file]
    with target_path.open(encoding="utf-8") as target_file:
        target_lines = [line.rstrip("\n") for line in target_file]
    if len(source_lines) != len(target_lines):
        raise PipelineError(
            f"EHU-HAC source/target line-count mismatch: {len(source_lines)} vs {len(target_lines)}"
        )
    if limit is not None:
        source_lines = source_lines[:limit]
        target_lines = target_lines[:limit]

    return _with_metadata(
        pd.DataFrame(
            {
                "original_index": range(len(source_lines)),
                "source": source_lines,
                "target": target_lines,
                "corpus": dataset_config["corpus_id"],
            }
        ),
        dataset_config,
    )


def load_local_jsonl(dataset_config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Load a normalized local JSONL parallel corpus without reordering rows."""
    path = Path(dataset_config["paths"]["file"])
    frame = pd.read_json(path, lines=True)
    if limit is not None:
        frame = frame.head(int(limit)).copy()
    frame = frame.copy()
    frame.insert(0, "original_index", range(len(frame)))
    frame["corpus"] = dataset_config["corpus_id"]
    return _with_metadata(frame, dataset_config)


def load_hf_dataset(dataset_config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Load configured bilingual and metadata columns in source order."""
    mapping = dataset_config["column_mapping"]
    source_column = mapping["source"]
    target_column = mapping["target"]
    metadata_mapping = dict(dataset_config.get("metadata_mapping", {}))
    dataset = hf_load_dataset(
        dataset_config["hf_repo_id"],
        name=dataset_config.get("subset"),
        split=dataset_config.get("split", "train"),
        token=get_hf_token(),
    )

    selected_columns = [source_column, target_column, *metadata_mapping.values()]
    dataset = dataset.select_columns(selected_columns)
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    records = dataset.to_pandas()
    records = records.rename(
        columns={
            source_column: "source",
            target_column: "target",
            **{source_name: output_name for output_name, source_name in metadata_mapping.items()},
        }
    )
    records.insert(0, "original_index", range(len(records)))
    records["corpus"] = dataset_config["corpus_id"]
    return _with_metadata(records, dataset_config)


def load_flores_plus(dataset_config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Join language-specific FLORES+ devtest configurations by sentence ID."""
    split = str(dataset_config.get("split", "devtest"))
    frames = []
    for role in ("source", "target"):
        dataset = hf_load_dataset(
            dataset_config["hf_repo_id"],
            name=dataset_config[f"{role}_subset"],
            split=split,
            token=get_hf_token(),
            revision=dataset_config.get("revision"),
        )
        frame = dataset.to_pandas()
        if "split" in frame and not frame["split"].eq(split).all():
            raise PipelineError(f"FLORES+ returned rows outside the requested {split} split.")
        frame = frame[["id", "text"]].rename(columns={"text": role})
        frame["id"] = frame["id"].astype(str)
        frames.append(frame)
    if set(frames[0]["id"]) != set(frames[1]["id"]):
        raise PipelineError("FLORES+ source/target sentence IDs do not match.")
    frame = frames[0].merge(frames[1], on="id", how="left", sort=False, validate="one_to_one")
    expected = dataset_config.get("expected_rows")
    if expected is not None and len(frame) != int(expected):
        raise PipelineError(f"FLORES+ {split} has {len(frame)} rows; expected {expected}.")
    frame["sentence_id"] = str(dataset_config["corpus_id"]) + "_" + frame["id"]
    frame["id"] = frame["sentence_id"]
    if limit is not None:
        frame = frame.head(int(limit)).copy()
    frame.insert(0, "original_index", range(len(frame)))
    frame["corpus"] = dataset_config["corpus_id"]
    return _with_metadata(frame, dataset_config)


_LOADERS = {
    "ehu_hac": load_ehu_hac,
    "hf_dataset": load_hf_dataset,
    "local_jsonl": load_local_jsonl,
    "flores_plus": load_flores_plus,
}


def load_dataset(
    dataset_key: str,
    datasets_config_path: str | Path = "configs/datasets.yaml",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load one configured corpus without filtering or reordering rows."""
    config = get_dataset_entry(dataset_key, datasets_config_path)
    loader = _LOADERS[config["loader"]]
    logger.info("Loading %s in original corpus order", dataset_key)
    return loader(config, limit=limit)


def load_processed_split(
    dataset_key: str,
    split: str,
    processed_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Load a processed split saved as JSONL or Parquet."""
    directory = Path(processed_dir) / dataset_key
    jsonl_path = directory / f"{split}.jsonl"
    parquet_path = directory / f"{split}.parquet"
    if jsonl_path.exists():
        frame = pd.read_json(jsonl_path, lines=True)
    elif parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    else:
        raise PipelineError(f"No processed {split!r} split found for {dataset_key!r}.")
    if "sentence_id" not in frame and "id" in frame:
        frame["sentence_id"] = frame["id"].astype(str)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a configured parallel dataset to JSONL.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default="configs/datasets.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"Dataset output exists: {output}")
    frame = load_dataset(args.dataset, args.dataset_config, args.limit)
    save_dataframe_jsonl(frame, output)
    save_json(frame.attrs.get("dataset_metadata", {}), output.with_suffix(".metadata.json"))


if __name__ == "__main__":
    main()
