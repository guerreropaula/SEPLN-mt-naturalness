"""Select GRPO checkpoints on development set"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

from src.utils.errors import PipelineError

NATURALNESS_METRICS = ("mtld", "inverse_simpson", "ptf", "sacr")
QUALITY_METRICS = ("cometkiwi", "metricx")
SELECTION_METRIC = "selection_score"


def _number(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(f"Checkpoint selection requires a numeric {name}; got {value!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise PipelineError(f"Checkpoint selection requires a finite {name}; got {value!r}.")
    return value


def compute_selection_score(
    predictions: Mapping[str, Any], references: Mapping[str, Any]
) -> dict[str, Any]:

    raw = {name: _number(predictions, name) for name in (*NATURALNESS_METRICS, *QUALITY_METRICS)}
    human = {name: _number(references, name) for name in NATURALNESS_METRICS}
    components = {}
    for name in NATURALNESS_METRICS:
        if raw[name] < 0 or human[name] < 0:
            raise PipelineError(f"{name} must be non-negative for checkpoint selection.")
        relative_error = abs(raw[name] - human[name]) / max(abs(human[name]), 1e-8)
        components[name] = 1.0 / (1.0 + relative_error)
    components["cometkiwi"] = min(1.0, max(0.0, raw["cometkiwi"]))
    if not 0.0 <= raw["metricx"] <= 25.0:
        raise PipelineError("MetricX-24 must be in [0, 25] for checkpoint selection.")
    components["metricx"] = 1.0 - raw["metricx"] / 25.0
    return {
        "selection_score": sum(components.values()) / 6.0,
        "naturalness_mean": sum(components[name] for name in NATURALNESS_METRICS) / 4.0,
        "quality_mean": sum(components[name] for name in QUALITY_METRICS) / 2.0,
        "components": components,
        "prediction_metrics": raw,
        "reference_metrics": human,
        "normalization": "inverse_relative_error; clipped_cometkiwi; 1-metricx/25",
        "aggregation": "equal_weight_mean_of_six_metrics",
    }


def validate_selection_training(training: Mapping[str, Any]) -> None:
    """Keep evaluation, saving and best-model restoration in sync."""
    if training["eval_strategy"] != "steps" or training["save_strategy"] != "steps":
        raise PipelineError("GRPO checkpoint selection requires step-based evaluation and saving.")
    eval_steps = int(training["eval_steps"])
    if eval_steps <= 0 or int(training["save_steps"]) != eval_steps:
        raise PipelineError("GRPO checkpoint selection requires equal positive eval/save steps.")
    if not training.get("load_best_model_at_end", False):
        raise PipelineError("GRPO checkpoint selection requires load_best_model_at_end=true.")


def generate_development_predictions(
    trainer: Any, records: list[dict[str, str]], settings: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Decode dev sources greedily on all ranks for FSDP/ZeRO; rank zero saves and scores."""
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from trl.models import unwrap_model_for_generation

    tokenizer = trainer.processing_class
    batch_size = int(settings["generation_batch_size"])
    generated = []
    with unwrap_model_for_generation(
        trainer.model_wrapped,
        trainer.accelerator,
        gather_deepspeed3_params=trainer.args.ds3_gather_for_generation,
    ) as model:
       
        generation_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_new_tokens": 0,
            "max_new_tokens": int(settings["max_new_tokens"]),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": False,
        }
        old_cache = model.config.use_cache
        model.config.use_cache = True
        try:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                inputs = tokenizer(
                    [record["prompt"] for record in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(settings["max_input_length"]),
                )
                inputs = {
                    key: value.to(trainer.accelerator.device) for key, value in inputs.items()
                }
                with (
                    torch.inference_mode(),
                    trainer.accelerator.autocast(),
                    FSDP.summon_full_params(trainer.model_wrapped, recurse=False)
                    if trainer.is_fsdp_enabled
                    else nullcontext(),
                ):
                    output_ids = model.generate(
                        **inputs,
                        **generation_kwargs,
                        synced_gpus=trainer.accelerator.num_processes > 1,
                    )
                outputs = tokenizer.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                for record, output in zip(batch, outputs, strict=True):
                    generated.append(
                        {
                            "sentence_id": record["sentence_id"],
                            "source": record["source"],
                            "reference": record["reference"],
                            "prediction": output.strip(),
                        }
                    )
        finally:
            model.config.use_cache = old_cache
    return generated


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


class DevelopmentCheckpointEvaluator:
    def __init__(
        self,
        records: list[dict[str, str]],
        target_lang: str,
        settings: Mapping[str, Any],
        output_dir: str | Path,
    ) -> None:
        if not records:
            raise PipelineError("GRPO checkpoint selection requires a non-empty development set.")
        ids = [record["sentence_id"] for record in records]
        if len(ids) != len(set(ids)):
            raise PipelineError("GRPO development sentence IDs must be unique.")
        self.records = records
        self.target_lang = target_lang
        self.settings = dict(settings)
        for name in ("generation_batch_size", "max_input_length", "max_new_tokens"):
            if int(self.settings[name]) <= 0:
                raise PipelineError(f"checkpoint_selection.{name} must be positive.")
        self.output_dir = Path(output_dir)
        self.fingerprint = hashlib.sha256(
            json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def score_predictions(self, predictions: list[dict[str, str]], step: int) -> dict[str, Any]:
        if len(predictions) != len(self.records) or any(
            any(
                prediction.get(key) != record[key] for key in ("sentence_id", "source", "reference")
            )
            for prediction, record in zip(predictions, self.records)
        ):
            raise PipelineError(
                "Checkpoint predictions must match the full ordered development set."
            )
        directory = self.output_dir / f"step-{step}"
        directory.mkdir(parents=True, exist_ok=True)
        protocol_path = self.output_dir / "protocol.json"
        protocol = {
            "development_fingerprint": self.fingerprint,
            "settings": self.settings,
            "normalization": "inverse_relative_error; clipped_cometkiwi; 1-metricx/25",
            "aggregation": "equal_weight_mean_of_six_metrics",
        }
        if protocol_path.exists():
            if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
                raise PipelineError("Cannot change the checkpoint selection protocol within a run.")
        else:
            _write_json(protocol_path, protocol)
        predictions_path = directory / "predictions.jsonl"
        predictions_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
        request_path = directory / "request.json"
        result_path = directory / "selection.json"
        result_path.unlink(missing_ok=True)
        _write_json(
            request_path,
            {
                "predictions_path": str(predictions_path.resolve()),
                "output_path": str(result_path.resolve()),
                "target_lang": self.target_lang,
                "settings": self.settings,
                "development_fingerprint": self.fingerprint,
                "development_examples": len(self.records),
                "step": step,
            },
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(self.settings.get("cuda_visible_devices", ""))
        for name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "TORCHELASTIC_RUN_ID",
        ):
            environment.pop(name, None)
        executable = (
            self.settings.get("python_executable")
            or environment.get("GRPO_EVAL_PYTHON")
            or sys.executable
        )
        try:
            with (directory / "metrics.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        str(executable),
                        "-m",
                        "src.grpo.checkpoint_metrics",
                        "--request",
                        str(request_path.resolve()),
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PipelineError(
                f"Development checkpoint metrics failed at step {step}; see "
                f"{directory / 'metrics.log'}. All six metrics are required."
            ) from exc
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validated = compute_selection_score(
            result["prediction_metrics"], result["reference_metrics"]
        )
        if result.get("development_fingerprint") != self.fingerprint or result.get("step") != step:
            raise PipelineError(
                "Checkpoint metric results do not match this development evaluation."
            )
        return validated

    def __call__(self, trainer: Any, metric_key_prefix: str = "eval") -> dict[str, float]:
        predictions = generate_development_predictions(trainer, self.records, self.settings)
        payload = [None]
        if trainer.is_world_process_zero():
            try:
                payload[0] = {
                    "result": self.score_predictions(predictions, trainer.state.global_step)
                }
            except Exception as exc:
                payload[0] = {"error": str(exc)}
        if trainer.accelerator.num_processes > 1:
            from accelerate.utils import broadcast_object_list

            broadcast_object_list(payload, from_process=0)
        result = payload[0]
        if result is None or "error" in result:
            raise PipelineError((result or {}).get("error", "No development metrics received."))
        report = result["result"]
        metrics = {
            name: report[name] for name in (SELECTION_METRIC, "quality_mean", "naturalness_mean")
        }
        metrics.update(report["prediction_metrics"])
        metrics.update(
            {f"reference_{name}": value for name, value in report["reference_metrics"].items()}
        )
        metrics.update(
            {f"normalized_{name}": value for name, value in report["components"].items()}
        )
        return {f"{metric_key_prefix}_{name}": value for name, value in metrics.items()}


class CheckpointSelectionTrainerMixin:
    checkpoint_evaluator: DevelopmentCheckpointEvaluator

    def evaluate(
        self, eval_dataset: Any = None, ignore_keys: Any = None, metric_key_prefix: str = "eval"
    ) -> dict[str, float]:
        if eval_dataset is not None and eval_dataset is not self.eval_dataset:
            raise PipelineError(
                "Checkpoint selection uses only the configured GRPO development set."
            )
        was_training = self.model.training
        self.model.eval()
        try:
            metrics = self.checkpoint_evaluator(self, metric_key_prefix=metric_key_prefix)
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )
            return metrics
        finally:
            self.model.train(was_training)
