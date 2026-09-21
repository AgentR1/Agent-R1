"""CPU-only prefix/DataProto preview. No Ray cluster or neural model is started."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="recipes/gsm8k/asymmetric_critic.yaml")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--model-path", required=True, help="Tokenizer path; no model weights are loaded")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--show-text", action="store_true", help="Explicitly display privileged prefix text")
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("--rows must be positive")

    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import OmegaConf

    from agent_r1.trainer.ppo.critic_prefix import CriticPrefixBuilder
    from verl import DataProto
    from verl.utils import hf_tokenizer

    config = OmegaConf.load(args.config)
    tokenizer = hf_tokenizer(args.model_path)
    builder = CriticPrefixBuilder(config.get("asymmetric_critic", {}), tokenizer)
    rows = pd.read_parquet(args.data_file).head(args.rows).to_dict("records")
    if not rows:
        raise ValueError("Prefix preview requires a nonempty dataset")
    chat_kwargs = config.get("data", {}).get("apply_chat_template_kwargs", {"enable_thinking": False})
    prompt_ids = [tokenizer.apply_chat_template(
        list(row["prompt"]), tokenize=True, add_generation_prompt=True, **chat_kwargs,
    ) for row in rows]
    width = max(len(ids) for ids in prompt_ids)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Preview tokenizer requires pad_token_id and eos_token_id")
    prompts = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros_like(prompts)
    for i, ids in enumerate(prompt_ids):
        prompts[i, -len(ids):] = torch.tensor(ids)
        prompt_mask[i, -len(ids):] = 1
    # Synthetic response only exercises layout; this is not generation/value prediction.
    responses = torch.tensor([[tokenizer.eos_token_id, tokenizer.pad_token_id]] * len(rows))
    response_mask = torch.tensor([[1, 0]] * len(rows))
    attention = torch.cat((prompt_mask, response_mask), dim=1)
    positions = (attention.cumsum(-1) - 1).clamp_min(0).masked_fill(attention == 0, 0)
    fields = {key: np.array([row.get(key) for row in rows], dtype=object) for key in rows[0]}
    fields["source_uid"] = np.array([f"prefix-preview-{i}" for i in range(len(rows))], dtype=object)
    fields["multi_modal_inputs"] = np.array([{} for _ in rows], dtype=object)
    batch = DataProto.from_dict(tensors={
        "prompts": prompts, "responses": responses, "input_ids": torch.cat((prompts, responses), dim=1),
        "attention_mask": attention, "position_ids": positions, "response_mask": response_mask,
    }, non_tensors=fields)
    original = {key: value.clone() for key, value in batch.batch.items()}
    builder.freeze_batch(batch)
    critic_batch = builder.build_batch(batch)
    if any(not torch.equal(batch.batch[key], value) for key, value in original.items()):
        raise AssertionError("Prefix preview changed actor tensors")
    if not torch.equal(critic_batch.batch["responses"], batch.batch["responses"]):
        raise AssertionError("Prefix preview changed actor responses")
    if batch.non_tensor_batch["multi_modal_inputs"] is not fields["multi_modal_inputs"]:
        raise AssertionError("Prefix preview changed actor multimodal placeholders")
    if builder.enabled and "multi_modal_inputs" in critic_batch.non_tensor_batch:
        raise AssertionError("Text critic view must not contain empty multimodal placeholders")
    report = {
        "kind": "cpu_critic_prefix_preview", "enabled": builder.enabled,
        "ray_started": False, "neural_models_loaded": False, "synthetic_responses": True,
        "actor_tensors_unchanged": True, "response_ids_unchanged": True,
        "text_rollout_placeholder_checked": True,
        "rows": len(rows), "prefix_token_counts": critic_batch.meta_info.get("critic_prefix_token_counts", []),
        "actor_prompt_token_counts": prompt_mask.sum(-1).tolist(),
        "critic_prompt_token_counts": critic_batch.batch["attention_mask"][:, :-2].sum(-1).tolist(),
        "critic_input_shape": list(critic_batch.batch["input_ids"].shape),
        "chat_template_kwargs": dict(chat_kwargs),
    }
    if args.show_text:
        report["prefix_texts"] = [builder.render(row) for row in rows]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
