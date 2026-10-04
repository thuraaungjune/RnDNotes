#!/usr/bin/env python3
"""
smoke_self_distillation.py - end-to-end smoke test for run_self_distillation_pass(),
the newly-wired self-distillation training step in train_active_learning_extended.py.

No GPU locally means this couldn't be unit-tested before landing -- run this on the
box BEFORE trusting a full diva_selfdistill launch. It:
  1. Builds a tiny self-distillation pool (~8 items) the same way
     evaluate_and_select_lines() does -- real generate() calls, real token ids,
     real per-token confidences -- not synthetic data.
  2. Runs run_self_distillation_pass() once and checks it completes without error,
     with a finite, sane loss.
  3. Runs it a SECOND time on the SAME pool and checks the loss went down --
     the one check that actually proves gradients are flowing correctly through the
     weighted loss, not just that the tensor shapes happen to line up.

Usage (from the codes/ directory):
  CUDA_VISIBLE_DEVICES=0 python smoke_self_distillation.py \
      --input_dir /dest/thura/data/Teklia_Belfort-line_labeled_10 \
      --prompt_path ../eval/prompt_Teklia_Belfort-line.txt
"""
import argparse
import sys

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_active_learning_extended import (
    load_data,
    run_self_distillation_pass,
    to_pil_image,
)


def build_pool(model, processor, imgs, prompt_text, eos_token_id, pad_token_id, max_new_tokens=64):
    """Mirrors the scoring loop in evaluate_and_select_lines(): real generate() calls,
    real per-token confidences, real truncation to the EOS step -- not synthetic."""
    from qwen_vl_utils import process_vision_info

    items = []
    for idx, img in enumerate(imgs):
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text_prompt], images=[image_inputs[0]], padding="longest", return_tensors="pt").to(model.device)
        inputs = {k: (v.to(model.dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=pad_token_id, eos_token_id=eos_token_id,
            )
        gen_ids_full = out.sequences[0][inputs["input_ids"].shape[1]:]
        confidences = []
        for step_idx, step_logits in enumerate(out.scores):
            probs = torch.softmax(step_logits, dim=-1)
            max_prob = torch.max(probs, dim=-1).values[0].item()
            cur_tokens = gen_ids_full[:step_idx]
            if eos_token_id is not None and eos_token_id in cur_tokens:
                continue
            confidences.append(max_prob)
        token_ids = gen_ids_full[:len(confidences)].tolist()
        text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  item {idx}: {len(token_ids)} tokens, mean confidence {np.mean(confidences) if confidences else 0:.4f}, text: {text[:60]!r}")
        items.append({
            "line_id": f"smoke_{idx}",
            "predicted_text": text,
            "uncertainty": 1.0 - float(np.mean(confidences)) if confidences else 1.0,
            "predicted_token_ids": token_ids,
            "token_confidences": confidences,
        })
    return items


def eval_pool_loss(model, processor, items, full_dataset, line_dict, prompt_text, seed):
    """Runs run_self_distillation_pass with lr=0 (no optimizer step, just forward +
    loss) to read the pool's current loss without changing the model -- used to
    compare before/after without contaminating the "after" measurement."""
    return run_self_distillation_pass(
        model=model, processor=processor, self_distill_items=items,
        full_dataset=full_dataset, unlabeled_line_dict=line_dict,
        prompt_text=prompt_text, lr=0.0, batch_size=len(items), epochs=1, seed=seed,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--prompt_path", required=True)
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_lora", action="store_true")
    args = p.parse_args()

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    ds = load_data(args.input_dir)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.num_images, len(ds)), replace=False)
    imgs = [to_pil_image(ds[int(i)]["Image"]) for i in idxs]
    print(f"Loaded {len(imgs)} images from {args.input_dir}")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )
    if not args.no_lora:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
    print(f"Model wrapped as: {type(model).__name__}")

    eos_token_id = processor.tokenizer.eos_token_id
    pad_token_id = processor.tokenizer.pad_token_id

    print("\nBuilding self-distillation pool (real generate() calls)...")
    model.eval()
    items = build_pool(model, processor, imgs, prompt_text, eos_token_id, pad_token_id)

    failures = []
    for it in items:
        if len(it["predicted_token_ids"]) != len(it["token_confidences"]):
            failures.append(f"{it['line_id']}: token_ids/confidences length mismatch")
        if not (0.0 <= it["uncertainty"] <= 1.0):
            failures.append(f"{it['line_id']}: uncertainty out of [0,1]: {it['uncertainty']}")
    if failures:
        print("\nFAIL (pool construction):")
        for m in failures:
            print(f"  - {m}")
        sys.exit(1)

    # Dummy line_dict: every item's "image" is just itself, at row index == its position
    # in a 1-row-per-item fake dataset, mirroring the real unlabeled_line_dict shape.
    full_dataset = ds.select([int(i) for i in idxs])
    line_dict = {it["line_id"]: [row] for row, it in enumerate(items)}

    print("\nLoss BEFORE training (lr=0, measurement only):")
    loss_before = eval_pool_loss(model, processor, items, full_dataset, line_dict, prompt_text, args.seed)
    if not np.isfinite(loss_before) or loss_before <= 0:
        print(f"FAIL: loss_before is not a sane positive finite number: {loss_before}")
        sys.exit(1)

    print("\nRunning one real self-distillation pass (lr=1e-4)...")
    run_self_distillation_pass(
        model=model, processor=processor, self_distill_items=items,
        full_dataset=full_dataset, unlabeled_line_dict=line_dict,
        prompt_text=prompt_text, lr=1e-4, batch_size=len(items), epochs=3, seed=args.seed,
    )

    print("\nLoss AFTER training (lr=0, measurement only):")
    loss_after = eval_pool_loss(model, processor, items, full_dataset, line_dict, prompt_text, args.seed)

    print(f"\nloss_before={loss_before:.4f}  loss_after={loss_after:.4f}")
    if loss_after >= loss_before:
        print("FAIL: loss did not decrease after training on the exact same pool it was "
              "just trained on -- gradients aren't flowing correctly through the weighted loss.")
        sys.exit(1)

    print("\nPASS: self-distillation pass runs end-to-end and demonstrably reduces loss "
          "on its own training pool.")


if __name__ == "__main__":
    main()
