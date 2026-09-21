"""Dependency-light greedy rollout controls and trajectory-level reward pairing.

These helpers do not compute advantages or update a policy. Keeping pairing
independent of tensors makes its invariants testable without a GPU stack.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


def normalize_source_uid(value: Any) -> str:
    if value is None or not str(value).strip():
        raise ValueError("A rollout source uid must be non-empty")
    return str(value)


def build_sampling_params(config: Any, meta_info: dict) -> dict:
    """Build fresh parameters; greedy overrides validation without changing config.

    ``rollout.n`` is a task repetition count in the trainer, not a backend
    sampling parameter. Every individual generation request returns one response.
    """
    mode = meta_info.get("rollout_mode", "sample")
    if mode not in ("sample", "greedy"):
        raise ValueError(f"Unsupported rollout_mode: {mode!r}")
    if config.response_length <= 0:
        raise ValueError("rollout.response_length must be positive")
    params = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repetition_penalty": 1.0,
        "logprobs": config.calculate_log_probs,
        "max_tokens": config.response_length,
    }
    if meta_info.get("validate", False):
        params.update(temperature=config.val_kwargs.temperature, top_p=config.val_kwargs.top_p)
    if mode == "greedy":
        params.update(temperature=0.0, top_p=1.0, logprobs=False)
    return params


def generation_metadata(output: Any, response_length: int, eos_token_id: Any = None) -> dict:
    """Reject incomplete requests and describe response-budget termination.

    Some verl versions merge backend ``stop`` and ``length`` into ``completed``.
    Without a raw finish reason, a full-budget response without EOS is marked
    conservatively as truncated. This is not a guarantee of backend determinism.
    """
    stop_reason = getattr(output, "stop_reason", None)
    finish_reason = getattr(output, "finish_reason", None)
    if stop_reason in ("abort", "aborted", "error", "failed", "cancelled", "timeout") or finish_reason == "abort":
        raise RuntimeError(f"Generation did not complete: {finish_reason or stop_reason}")
    token_ids = output.token_ids
    if not token_ids:
        raise RuntimeError("Generation returned an empty response")
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id])
    at_limit = len(token_ids) >= response_length
    ends_in_eos = token_ids[-1] in eos_ids
    truncated = len(token_ids) > response_length or finish_reason == "length" or stop_reason == "length"
    if finish_reason is None and stop_reason not in ("stop", "length"):
        truncated = truncated or (at_limit and not ends_in_eos)
    return {
        "generation_stop_reason": finish_reason or stop_reason or "unknown",
        "response_at_limit": at_limit,
        "response_truncated": truncated,
    }


def terminal_status(metadata: dict, reason: str = "final_answer") -> dict:
    """Classify a flow which stops after a model response, not an environment done."""
    truncated = bool(metadata["response_truncated"])
    return {
        "terminated": not truncated,
        "truncated": truncated,
        "termination_reason": "response_length" if truncated else reason,
    }


@dataclass(frozen=True)
class GreedyBaseline:
    source_uid: str
    trajectory_uid: str
    reward: float
    num_steps: int
    terminated: bool
    truncated: bool
    termination_reason: str


@dataclass
class ReMaxRolloutCollection:
    """Collection only: neither output has advantages or optimizer updates."""

    sampled: Any
    greedy: Any
    baselines: dict[str, GreedyBaseline]


def summarize_greedy_baselines(rows: list[dict], expected_source_uids: list[str]) -> dict[str, GreedyBaseline]:
    """Sum immediate rewards over each whole trajectory, independent of row order.

    Require one greedy trajectory per source, contiguous steps and an explicit
    terminal/truncation marker. Do not invent a zero reward for missing trajectories.
    """
    expected = [normalize_source_uid(uid) for uid in expected_source_uids]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected one unique source uid per original task")
    trajectories = defaultdict(list)
    for row in rows:
        if row["rollout_mode"] != "greedy":
            raise ValueError("Baseline input contains a non-greedy trajectory")
        trajectories[row["trajectory_uid"]].append(row)

    baselines = {}
    for trajectory_uid, steps in trajectories.items():
        steps.sort(key=lambda row: row["step_index"])
        if [row["step_index"] for row in steps] != list(range(len(steps))):
            raise ValueError(f"Missing or duplicate steps in trajectory {trajectory_uid}")
        source_uid = normalize_source_uid(steps[0]["source_uid"])
        if any(normalize_source_uid(row["source_uid"]) != source_uid for row in steps):
            raise ValueError(f"Mixed source uids in trajectory {trajectory_uid}")
        if source_uid in baselines:
            raise ValueError(f"Multiple greedy trajectories for source uid {source_uid}")
        if any(row["terminated"] or row["truncated"] for row in steps[:-1]):
            raise ValueError(f"Trajectory {trajectory_uid} continues after its end marker")
        last = steps[-1]
        if bool(last["terminated"]) == bool(last["truncated"]):
            raise ValueError(f"Trajectory {trajectory_uid} needs exactly one terminal/truncation marker")
        if last["termination_reason"] in ("unknown", "ongoing", ""):
            raise ValueError(f"Trajectory {trajectory_uid} has no explicit termination reason")
        rewards = [float(row["reward"]) for row in steps]
        if not all(math.isfinite(value) for value in rewards):
            raise ValueError(f"Non-finite reward in trajectory {trajectory_uid}")
        try:
            reward = math.fsum(rewards)
        except OverflowError as error:
            raise ValueError(f"Reward overflow in trajectory {trajectory_uid}") from error
        if not math.isfinite(reward):
            raise ValueError(f"Non-finite reward in trajectory {trajectory_uid}")
        baselines[source_uid] = GreedyBaseline(
            source_uid=source_uid,
            trajectory_uid=trajectory_uid,
            reward=reward,
            num_steps=len(steps),
            terminated=bool(last["terminated"]),
            truncated=bool(last["truncated"]),
            termination_reason=last["termination_reason"],
        )
    if set(baselines) != set(expected):
        missing = set(expected) - set(baselines)
        unexpected = set(baselines) - set(expected)
        raise ValueError(f"Greedy/source uid mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    return baselines


def pair_baseline_rewards(source_uids: list[str], baselines: dict[str, GreedyBaseline]) -> list[float]:
    """Broadcast each task's greedy reward to its sampled step rows."""
    rewards = []
    for value in source_uids:
        uid = normalize_source_uid(value)
        if uid not in baselines:
            raise ValueError(f"No greedy baseline for sampled source uid {uid}")
        rewards.append(baselines[uid].reward)
    return rewards
