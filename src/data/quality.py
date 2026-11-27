"""Ordered, batched reference-free alignment filtering for the clean corpus pool."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from src.utils.errors import PipelineError


def filter_alignment_rows(
    rows: Iterable[dict[str, Any]],
    settings: Mapping[str, Any],
    scorer: Callable[[list[str], list[str]], list[float]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Filter otherwise-valid pairs, preserving order and all rejection records."""
    batch_size = int(settings.get("batch_size", 16))
    if batch_size < 1:
        raise PipelineError("alignment_quality.batch_size must be positive.")
    threshold = float(settings["threshold"])
    if not math.isfinite(threshold):
        raise PipelineError("alignment_quality.threshold must be finite.")
    model = None

    def score(sources: list[str], targets: list[str]) -> list[float]:
        nonlocal model
        if scorer is not None:
            return scorer(sources, targets)
        if model is None:
            from comet import download_model, load_from_checkpoint
            model = load_from_checkpoint(download_model(str(settings["model"])))
        outputs = model.predict(
            [
                {"src": source, "mt": target}
                for source, target in zip(sources, targets, strict=True)
            ],
            batch_size=batch_size,
            gpus=int(settings.get("gpus", 0)),
            progress_bar=False,
        )
        return list(outputs.scores if hasattr(outputs, "scores") else outputs["scores"])

    def flush(pending: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        candidates = [row for row in pending if row["status"] == "valid"]
        if candidates:
            scores = score(
                [row["source"] for row in candidates], [row["target"] for row in candidates]
            )
            for row, value in zip(candidates, scores, strict=True):
                value = float(value)
                if not math.isfinite(value):
                    raise PipelineError("Alignment scorer returned a non-finite COMETKiwi score.")
                row["alignment_cometkiwi"] = value
                if value < threshold:
                    row.update(status="invalid", reason="alignment_cometkiwi_below_threshold")
        yield from pending

    pending = []
    valid_candidates = 0
    for row in rows:
        pending.append(row)
        valid_candidates += row["status"] == "valid"
        if valid_candidates >= batch_size or len(pending) >= batch_size * 8:
            yield from flush(pending)
            pending = []
            valid_candidates = 0
    if pending:
        yield from flush(pending)
