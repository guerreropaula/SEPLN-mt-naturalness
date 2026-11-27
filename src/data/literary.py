"""Prepare the paper's Stanza/Vecalign literary test set from supplied book texts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.utils.errors import PipelineError


def segment_text(text: str, language: str, pipeline: Any = None) -> list[str]:
    """Produce one non-empty Stanza sentence per line for Vecalign input."""
    if pipeline is None:
        import stanza

        pipeline = stanza.Pipeline(
            lang=language, processors="tokenize", use_gpu=False, verbose=False
        )
    return [
        " ".join(sentence.text.split())
        for sentence in pipeline(text).sentences
        if sentence.text.strip()
    ]


def build_literary_pairs(
    sources: list[str],
    targets: list[str],
    alignment_lines: Iterable[str],
    target_lang: str,
    limit: int = 2500,
    require_full: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the first N non-empty 1:1 Vecalign pairs with character ratio 0.25--4."""
    if target_lang not in {"eu", "ca"} or limit <= 0:
        raise PipelineError("Literary preparation requires target eu/ca and a positive limit.")
    records = []
    rejected = Counter()
    seen_sources, seen_targets = set(), set()
    for line_number, line in enumerate(alignment_lines, 1):
        if not line.strip():
            continue
        try:
            left, right, _cost = line.strip().split(":", 2)
            source_indexes, target_indexes = ast.literal_eval(left), ast.literal_eval(right)
            if not isinstance(source_indexes, list) or not isinstance(target_indexes, list):
                raise ValueError("Expected lists of sentence indexes")
        except (ValueError, SyntaxError) as exc:
            raise PipelineError(f"Invalid Vecalign record on line {line_number}: {line!r}") from exc
        if len(source_indexes) != 1 or len(target_indexes) != 1:
            rejected["not_one_to_one"] += 1
            continue
        i, j = source_indexes[0], target_indexes[0]
        if (
            type(i) is not int
            or type(j) is not int
            or not 0 <= i < len(sources)
            or not 0 <= j < len(targets)
        ):
            raise PipelineError(f"Vecalign indexes out of range on line {line_number}.")
        if i in seen_sources or j in seen_targets:
            raise PipelineError("Vecalign reuses a sentence in multiple 1:1 alignments.")
        seen_sources.add(i)
        seen_targets.add(j)
        source, target = sources[i].strip(), targets[j].strip()
        if not source or not target:
            rejected["empty"] += 1
            continue
        if not 0.25 <= len(target) / len(source) <= 4.0:
            rejected["character_ratio"] += 1
            continue
        identifier = f"woolf_full_en_{target_lang}_{len(records) + 1:06d}"
        records.append(
            {
                "id": identifier,
                "sentence_id": identifier,
                "source": source,
                "target": target,
                "source_sentence_index": i,
                "target_sentence_index": j,
            }
        )
        if len(records) == limit:
            break
    if require_full and len(records) < limit:
        raise PipelineError(
            f"Only {len(records)} eligible literary pairs; the paper requires {limit}."
        )
    return records, {
        "retained_pairs": len(records),
        "requested_pairs": limit,
        "rejections": dict(rejected),
        "selection": "first valid pairs in alignment order",
        "character_ratio": [0.25, 4.0],
        "alignment": "Vecalign 1:1 only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    segment = commands.add_parser("segment")
    segment.add_argument("--input", required=True)
    segment.add_argument("--lang", required=True, choices=("en", "eu", "ca"))
    build = commands.add_parser("build")
    build.add_argument("--source-sentences", required=True)
    build.add_argument("--target-sentences", required=True)
    build.add_argument("--alignments", required=True)
    build.add_argument("--target-lang", required=True, choices=("eu", "ca"))
    build.add_argument("--limit", type=int, default=2500)
    for command in (segment, build):
        command.add_argument("--output", required=True)
        command.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"Output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "segment":
        sentences = segment_text(Path(args.input).read_text(encoding="utf-8"), args.lang)
        output.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    else:
        sources = Path(args.source_sentences).read_text(encoding="utf-8").splitlines()
        targets = Path(args.target_sentences).read_text(encoding="utf-8").splitlines()
        records, report = build_literary_pairs(
            sources,
            targets,
            Path(args.alignments).read_text(encoding="utf-8").splitlines(),
            args.target_lang,
            args.limit,
        )
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )
        report["input_sha256"] = {
            key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
            for key, value in vars(args).items()
            if key in {"source_sentences", "target_sentences", "alignments"}
        }
        output.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
