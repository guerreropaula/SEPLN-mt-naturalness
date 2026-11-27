"""GRPO checkpoints"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.grpo.checkpoint_selection import compute_selection_score
from src.utils.errors import PipelineError


def score_analysis_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "mtld": ("lexical_diversity", "mtld"),
        "inverse_simpson": ("morphological_diversity", "morph_inverse_simpson_optional"),
        "ptf": ("synonym_frequency_analysis", "sfa_ptf"),
    }
    predictions, references = {}, {}
    for name, (section, leaf) in fields.items():
        predictions[name] = summary.get(f"prediction_{section}", {}).get(leaf)
        references[name] = summary.get(f"reference_{section}", {}).get(leaf)
    predictions["sacr"] = summary.get("astred_source_prediction", {}).get("sacr_cross")
    references["sacr"] = summary.get("astred_source_reference", {}).get("sacr_cross")
    for name in ("cometkiwi", "metricx"):
        predictions[name] = summary.get("adequacy", {}).get(name)
    return compute_selection_score(predictions, references)


def run_request(request_path: str | Path) -> dict[str, Any]:
    import pandas as pd

    from src.evaluation.run import analyze_predictions

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    settings = request["settings"]
    predictions = pd.read_json(request["predictions_path"], lines=True)
    if len(predictions) != request["development_examples"]:
        raise PipelineError("Checkpoint predictions do not cover the full development set.")
    result = analyze_predictions(
        predictions,
        cometkiwi_model=settings["cometkiwi_model"],
        metricx_model=settings["metricx_model"],
        comet_batch_size=int(settings.get("metric_batch_size", 4)),
        comet_gpus=int(settings.get("comet_gpus", 0)),
        stanza_lang=request["target_lang"],
        sfa_target_lang=request["target_lang"],
        sfa_source_lang="en",
        sfa_dictionary_path=settings.get("sfa_dictionary_path"),
        sfa_dictionary_invert=settings.get("sfa_dictionary_invert"),
        astred_source_prediction=True,
        astred_source_prediction_source_model_or_lang="en",
        astred_source_prediction_target_model_or_lang=request["target_lang"],
        astred_source_reference=True,
    )
    report = score_analysis_summary(result["summary"])
    report.update(
        {
            key: request[key]
            for key in (
                "step",
                "development_fingerprint",
                "development_examples",
                "target_lang",
            )
        }
    )
    report["analysis_summary"] = result["summary"]
    Path(request["output_path"]).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    run_request(args.request)


if __name__ == "__main__":
    main()
