"""Critic-only static prefixes and response-aligned DataProto views.

Template parsing has no GPU-stack dependencies. Tensor/verl imports are lazy;
neither this module nor its preview command starts Ray or a model engine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from string import Formatter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VARIABLE_PATHS = {
    "ground_truth": "reward_model.ground_truth",
    "reference_solution": "extra_info.answer",
    "question": "extra_info.question",
    "data_source": "data_source",
}


def lookup_field(row: Mapping, path: str) -> Any:
    """Read a dotted metadata path, never Python attributes or expressions."""
    if not isinstance(path, str) or not path or not all(part.isidentifier() for part in path.split(".")):
        raise ValueError(f"Invalid critic prefix field path: {path!r}")
    value = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Missing critic prefix field: {path}")
        value = value[part]
    if value is None:
        raise ValueError(f"Missing critic prefix field: {path}")
    return value


def validate_asymmetric_critic(config) -> None:
    """Reject unsupported training paths before any worker/model initialization."""
    asymmetric = config.get("asymmetric_critic", {})
    enabled = asymmetric.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("asymmetric_critic.enabled must be a boolean")
    if not enabled:
        return
    if config.algorithm.adv_estimator != "gae" or config.critic.enable is False:
        raise ValueError("asymmetric_critic requires algorithm.adv_estimator=gae and an enabled critic")
    if config.critic.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("asymmetric_critic currently supports only fsdp/fsdp2 critics")
    if config.trainer.get("use_legacy_worker_impl", "auto") not in ("auto", "enable"):
        raise ValueError("asymmetric_critic requires trainer.use_legacy_worker_impl=enable (or auto)")
    actor_path = config.actor_rollout_ref.model.path
    tokenizer_path = config.critic.model.get("tokenizer_path") or actor_path
    if tokenizer_path != actor_path:
        raise ValueError("asymmetric_critic requires critic.model.tokenizer_path=actor_rollout_ref.model.path")


class CriticPrefixBuilder:
    """Freeze one static prefix per source UID and build isolated critic inputs.

    Call freeze_batch once before computing old values, then build_batch for both
    inference and updates. Only this batch's UID cache is retained. No prefix
    fields/tensors are added to the actor batch.
    """

    def __init__(self, config, tokenizer, project_root: Path = PROJECT_ROOT):
        self.enabled = config.get("enabled", False)
        if not isinstance(self.enabled, bool):
            raise ValueError("asymmetric_critic.enabled must be a boolean")
        self.tokenizer = tokenizer
        self.prefix_ids: dict[str, tuple[int, ...]] = {}
        self.template = None
        self.variables: set[str] = set()
        if not self.enabled:
            return
        prefix = config.get("prefix", {})
        unknown = set(prefix) - {"source", "template", "template_path", "field", "max_tokens", "overflow"}
        if unknown:
            raise ValueError(f"Unknown critic prefix options: {sorted(unknown)}")
        self.source = prefix.get("source", "template")
        self.max_tokens = prefix.get("max_tokens", 256)
        self.field = prefix.get("field")
        template_path = prefix.get("template_path")
        template = prefix.get("template")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("critic prefix max_tokens must be a nonnegative integer")
        if prefix.get("overflow", "error") != "error":
            raise ValueError("critic prefix currently supports only overflow=error; never silently truncate")
        if self.source == "template":
            if (template is None) == (template_path is None) or self.field is not None:
                raise ValueError("Template prefix requires exactly one of template/template_path, and no field")
            if template_path is not None:
                if not isinstance(template_path, str) or not template_path:
                    raise ValueError("critic prefix template_path must be a nonempty string")
                path = Path(template_path)
                if not path.is_absolute():
                    path = Path(project_root) / path
                template = path.read_text(encoding="utf-8")
            if not isinstance(template, str):
                raise ValueError("critic prefix template must be a string (empty is allowed for controls)")
            self.template = template
            for _, variable, format_spec, conversion in Formatter().parse(template):
                if variable is None:
                    continue
                if variable not in VARIABLE_PATHS or format_spec or conversion is not None:
                    raise ValueError(f"Unsupported critic prefix template variable/expression: {variable!r}")
                self.variables.add(variable)
        elif self.source == "field":
            if template is not None or template_path is not None:
                raise ValueError("Field prefix cannot also specify template/template_path")
            if not isinstance(self.field, str) or not self.field:
                raise ValueError("Field prefix requires field, e.g. extra_info.critic_prefix")
            if not all(part.isidentifier() for part in self.field.split(".")):
                raise ValueError(f"Invalid critic prefix field path: {self.field!r}")
        else:
            raise ValueError(f"Unsupported critic prefix source: {self.source!r}; use template or field")

    def render(self, row: Mapping) -> str:
        if not self.enabled:
            return ""
        if self.source == "field":
            text = lookup_field(row, self.field)
            if not isinstance(text, str):
                raise ValueError(f"Critic prefix field {self.field} must contain a string")
            return text
        context = {}
        for variable in self.variables:
            value = lookup_field(row, VARIABLE_PATHS[variable])
            if isinstance(value, (Mapping, list, tuple)) or getattr(value, "ndim", 0) > 0 or not str(value).strip():
                raise ValueError(f"Critic prefix variable {variable} must be a nonempty scalar")
            context[variable] = str(value)
        return self.template.format_map(context)

    def encode(self, row: Mapping) -> tuple[int, ...]:
        text = self.render(row)
        ids = tuple(self.tokenizer.encode(text, add_special_tokens=False)) if text else ()
        if self.enabled and len(ids) > self.max_tokens:
            raise ValueError(f"Critic prefix has {len(ids)} tokens, exceeding max_tokens={self.max_tokens}")
        return ids

    @staticmethod
    def _uids(batch) -> list[str]:
        fields = batch.non_tensor_batch
        key = "source_uid" if "source_uid" in fields else "uid"
        if key not in fields or len(fields[key]) != len(batch):
            raise ValueError("Critic prefix requires one source_uid (or uid) per row")
        uids = list(fields[key])
        if any(not isinstance(uid, str) or not uid for uid in uids):
            raise ValueError("Critic prefix source UIDs must be nonempty strings")
        return uids

    def freeze_batch(self, batch) -> None:
        if not self.enabled:
            return
        self.prefix_ids = {}
        frozen = {}
        for i, uid in enumerate(self._uids(batch)):
            if uid in frozen:
                continue
            row = {key: values[i] for key, values in batch.non_tensor_batch.items()}
            try:
                frozen[uid] = self.encode(row)
            except ValueError as error:
                raise ValueError(f"Critic prefix for source_uid={uid!r}: {error}") from error
        self.prefix_ids = frozen

    def build_batch(self, batch):
        if not self.enabled:
            return batch
        import torch

        from verl import DataProto

        tensors = batch.batch
        required = ("prompts", "responses", "input_ids", "attention_mask", "position_ids", "response_mask")
        if any(key not in tensors for key in required):
            raise ValueError(f"Critic prefix requires tensor fields {required}")
        non_tensors = dict(batch.non_tensor_batch)
        multimodal = non_tensors.pop("multi_modal_inputs", None)
        # Text rollouts carry per-row empty placeholders. Drop them only from
        # the critic view; reject actual multimodal data and keep actor data intact.
        has_multimodal = multimodal is not None and any(
            value is not None and (not isinstance(value, Mapping) or len(value) != 0) for value in multimodal
        )
        if has_multimodal or tensors["position_ids"].ndim != 2:
            raise ValueError("asymmetric_critic currently supports text-only inputs, not multimodal/mRoPE")
        prompts, responses = tensors["prompts"], tensors["responses"]
        if prompts.ndim != 2 or responses.ndim != 2 or len(batch) == 0 or prompts.shape[1] == 0:
            raise ValueError("Critic prefix requires a nonempty 2D prompt/response batch")
        prompt_length, response_length = prompts.shape[1], responses.shape[1]
        ids, attention = tensors["input_ids"], tensors["attention_mask"]
        if response_length == 0 or ids.shape != (len(batch), prompt_length + response_length):
            raise ValueError("Actor input_ids must be prompt block followed by a fixed-width response block")
        if attention.shape != ids.shape or tensors["position_ids"].shape != ids.shape:
            raise ValueError("Actor attention_mask/position_ids must match input_ids")
        if tensors["response_mask"].shape != responses.shape:
            raise ValueError("Actor response_mask must match responses")
        if not bool(((tensors["response_mask"] == 0) | (tensors["response_mask"] == 1)).all()):
            raise ValueError("Actor response_mask must be binary")
        if not torch.equal(ids[:, :prompt_length], prompts) or not torch.equal(ids[:, prompt_length:], responses):
            raise ValueError("Actor input_ids must preserve exact prompt and response token IDs")
        if not bool(((attention == 0) | (attention == 1)).all()):
            raise ValueError("Actor attention_mask must be binary")
        prompt_mask = attention[:, :prompt_length].bool()
        response_attention = attention[:, prompt_length:]
        if bool((prompt_mask[:, :-1] & ~prompt_mask[:, 1:]).any()):
            raise ValueError("Actor prompts must be left-padded, with no interior padding")
        if not bool(prompt_mask[:, -1].all()):
            raise ValueError("Each critic input needs a valid prompt-last state position")
        if bool((~response_attention[:, :-1].bool() & response_attention[:, 1:].bool()).any()):
            raise ValueError("Actor response attention must be right-padded")
        if bool((tensors["response_mask"].bool() & ~response_attention.bool()).any()):
            raise ValueError("Action response_mask cannot include padded response positions")

        uids = self._uids(batch)
        missing = set(uids) - self.prefix_ids.keys()
        if missing:
            raise ValueError(f"Prefixes must be frozen during value inference before critic update: {sorted(missing)}")
        rows = []
        for i, uid in enumerate(uids):
            prefix = prompts.new_tensor(self.prefix_ids[uid])
            rows.append(torch.cat((prefix, prompts[i][prompt_mask[i]])))
        width = max(row.numel() for row in rows)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Critic prefix tokenizer must have a pad_token_id")
        critic_prompts = prompts.new_full((len(batch), width), pad_id)
        critic_prompt_mask = attention.new_zeros((len(batch), width))
        for i, row in enumerate(rows):
            critic_prompts[i, -row.numel():] = row
            critic_prompt_mask[i, -row.numel():] = 1
        critic_attention = torch.cat((critic_prompt_mask, response_attention), dim=1)
        critic_positions = (critic_attention.long().cumsum(dim=1) - 1).clamp_min(0)
        critic_positions.masked_fill_(critic_attention == 0, 0)

        # Shallow-copy untouched tensors; all replaced inputs/masks are independent.
        # Update mini-batching uses index selection, which copies response loss masks.
        critic_tensors = dict(tensors.items())
        critic_tensors.update(
            prompts=critic_prompts,
            input_ids=torch.cat((critic_prompts, responses), dim=1),
            attention_mask=critic_attention,
            position_ids=critic_positions,
            response_mask=tensors["response_mask"].clone(),
        )
        metadata = dict(batch.meta_info)
        metadata["global_token_num"] = critic_attention.sum(dim=-1).tolist()
        metadata["critic_prefix_token_counts"] = [len(self.prefix_ids[uid]) for uid in uids]
        return DataProto.from_dict(
            tensors=critic_tensors, non_tensors=non_tensors, meta_info=metadata,
        )

    def save_snapshot(self, directory) -> None:
        """Save the loaded template, not per-task answers; never overwrite a different run snapshot."""
        if not self.enabled:
            return
        snapshot = {
            "source": self.source, "field": self.field, "template": self.template,
            "max_tokens": self.max_tokens, "overflow": "error",
            "template_sha256": hashlib.sha256(self.template.encode("utf-8")).hexdigest()
            if self.template is not None else None,
        }
        path = Path(directory) / "asymmetric_critic_prefix.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != snapshot:
                raise ValueError(
                    f"Critic prefix snapshot differs from this run: {path}; use a new output directory"
                ) from None
