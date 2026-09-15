#!/usr/bin/env python3
"""
train_al_baselines.py - Fixed Active Learning Baselines (Random, Entropy, KMeans Center)

Applies all fixes:
- Line-level budget accounting (tracks unique original line annotations).
- Exact EOS masking in sequence uncertainty.
- Attention-mask weighted feature pooling + float32 casting.
- Full seed control via --seed.
- Unified CER evaluation (raw, clean, per-line, corpus).
- Saves acquired line manifests to JSON for 100% auditability.
"""

import os
import sys
import gc
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk, Dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from evaluate_metrics import compute_comprehensive_cer, clean_text_symmetric


def to_pil_image(image_value, scale=1.0):
    if isinstance(image_value, Image.Image):
        img = image_value.convert("RGB")
    elif isinstance(image_value, str) and os.path.exists(image_value):
        img = Image.open(image_value).convert("RGB")
    elif isinstance(image_value, dict):
        if image_value.get("path") and os.path.exists(image_value["path"]):
            img = Image.open(image_value["path"]).convert("RGB")
        elif image_value.get("bytes") is not None:
            import io
            img = Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
        else:
            raise ValueError(f"Cannot parse image dict: {image_value.keys()}")
    else:
        img = image_value

    if scale != 1.0 and isinstance(img, Image.Image):
        w, h = img.size
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return img


def load_data(split_path):
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Requested dataset path does not exist: {split_path}")
    if os.path.isdir(split_path) and os.path.exists(os.path.join(split_path, "state.json")):
        return load_from_disk(split_path)
    if os.path.isdir(os.path.join(split_path, "train")):
        return load_from_disk(os.path.join(split_path, "train"))
    return load_from_disk(split_path)


def extract_base_line_id(row: Dict[str, Any], idx: int) -> str:
    ident = str(row.get("Identifier", "")).strip()
    attack = str(row.get("attack_type", "")).strip()
    if ident and attack and attack != "none" and ident.endswith(f"_{attack}"):
        return ident[: -(len(attack) + 1)]
    if ident:
        return ident
    return f"line_{idx}"


def extract_visual_diversity_features(model, inputs: Dict[str, Any]) -> np.ndarray:
    """
    Pools one feature vector per image straight from the vision encoder/projector output
    (`model.visual`), NOT from the LLM decoder's hidden state -- see the identical helper
    in train_active_learning_extended.py for the full rationale. Kept consistent between
    the two scripts so KMeans-center is a fair diversity-only ablation against DIVA.
    """
    # Unwrap LoRA/PEFT to reach the underlying HF model, since the vision tower lives
    # on the base architecture, not on the PeftModel/LoraModel wrapper.
    base_model = model.get_base_model() if isinstance(model, PeftModel) else model

    # Qwen2-VL / Qwen2.5-VL expose the vision tower at `.visual` directly; Qwen3-VL
    # nests it one level deeper at `.model.visual` (under the composite Qwen3VLModel).
    vision_module = getattr(base_model, "visual", None)
    if vision_module is None:
        vision_module = getattr(getattr(base_model, "model", None), "visual", None)
    if vision_module is None:
        raise AttributeError(
            "Model has no `.visual` or `.model.visual` submodule; cannot extract vision-encoder "
            "features for diversity clustering. Update extract_visual_diversity_features for this architecture."
        )

    pixel_values = inputs.get("pixel_values")
    grid_thw = inputs.get("image_grid_thw")
    if pixel_values is None or grid_thw is None:
        raise KeyError(
            "Expected 'pixel_values' and 'image_grid_thw' in processor output for "
            "vision-encoder feature extraction."
        )

    with torch.no_grad():
        vision_output = vision_module(pixel_values, grid_thw=grid_thw)

    # Qwen2-VL / Qwen2.5-VL vision towers return the merged per-image tokens directly as a
    # plain Tensor. Qwen3-VL's vision tower instead returns a BaseModelOutputWithDeepstackFeatures,
    # whose `.pooler_output` (post spatial-merger) is the equivalent merged-token tensor -- its
    # `.last_hidden_state` is the PRE-merge raw patch sequence and would break the merge_ratio
    # math below (it stays at the raw patch count, so merge_ratio would come out as 1).
    if isinstance(vision_output, torch.Tensor):
        visual_tokens = vision_output
    elif getattr(vision_output, "pooler_output", None) is not None:
        visual_tokens = vision_output.pooler_output
    elif hasattr(vision_output, "last_hidden_state"):
        visual_tokens = vision_output.last_hidden_state
    else:
        raise AttributeError(
            f"Vision module output of type {type(vision_output)} has neither `.pooler_output` nor "
            "`.last_hidden_state` and is not a plain Tensor; update extract_visual_diversity_features."
        )

    grid_thw_cpu = grid_thw.detach().cpu()
    raw_patch_counts = [int(t) * int(h) * int(w) for t, h, w in grid_thw_cpu.tolist()]
    total_raw = sum(raw_patch_counts)
    total_out = visual_tokens.shape[0]
    merge_ratio = max(1, total_raw // total_out) if total_out > 0 else 1
    per_image_counts = [max(1, count // merge_ratio) for count in raw_patch_counts]

    drift = total_out - sum(per_image_counts)
    per_image_counts[-1] += drift

    visual_tokens_f32 = visual_tokens.float()
    pooled = []
    cursor = 0
    for count in per_image_counts:
        segment = visual_tokens_f32[cursor: cursor + count]
        cursor += count
        pooled.append(segment.mean(dim=0))
    return torch.stack(pooled).cpu().numpy()


def select_baseline_lines(
    model,
    processor,
    unlabeled_line_dict: Dict[str, List[int]],
    full_dataset: Dataset,
    prompt_text: str,
    num_lines_to_select: int,
    subset_size: int,
    batch_size: int,
    strategy: str,
    al_iter: int,
    seed: int,
    diversity_embedding_type: str = "vision_encoder"
) -> List[str]:
    """Selects unique lines using Random, Entropy, or KMeans Center."""
    unlabeled_line_ids = list(unlabeled_line_dict.keys())

    if strategy == "random":
        rng = np.random.default_rng(seed + al_iter)
        selected = list(rng.choice(unlabeled_line_ids, size=min(num_lines_to_select, len(unlabeled_line_ids)), replace=False))
        print(f"[AL Baseline: Random] Randomly selected {len(selected)} unique lines.")
        return selected

    # Subsample candidates for evaluation if unlabeled pool is huge
    if len(unlabeled_line_ids) > subset_size:
        rng = np.random.default_rng(seed + al_iter)
        eval_line_ids = list(rng.choice(unlabeled_line_ids, size=subset_size, replace=False))
    else:
        eval_line_ids = list(unlabeled_line_ids)

    candidate_rows = []
    for lid in eval_line_ids:
        row_indices = unlabeled_line_dict[lid]
        chosen_idx = row_indices[0]
        for r_idx in row_indices:
            row_data = full_dataset[r_idx]
            if row_data.get("attack_type", "clean") in ["clean", "none", None]:
                chosen_idx = r_idx
                break
        candidate_rows.append((lid, chosen_idx))

    eos_token_id = processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None
    model.eval()
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    scored_candidates = []
    features_list = []

    print(f"[AL Baseline: {strategy}] Extracting representations for {len(candidate_rows)} candidate lines...")
    for i in tqdm(range(0, len(candidate_rows), batch_size), desc=f"Scoring {strategy}"):
        batch_slice = candidate_rows[i : i + batch_size]
        batch_line_ids = [item[0] for item in batch_slice]
        batch_imgs = [to_pil_image(full_dataset[item[1]]["Image"]) for item in batch_slice]

        from qwen_vl_utils import process_vision_info
        batch_inputs_structs = []
        for img in batch_imgs:
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            batch_inputs_structs.append((text_prompt, image_inputs, video_inputs))

        text_prompts = [x[0] for x in batch_inputs_structs]
        all_images = [x[1][0] for x in batch_inputs_structs if x[1]]

        inputs = processor(text=text_prompts, images=all_images, padding="longest", return_tensors="pt").to(model.device)
        if hasattr(model, "dtype"):
            inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        if strategy == "kmeans_center" and diversity_embedding_type == "vision_encoder":
            # Diversity features pooled straight from the vision encoder/projector output,
            # not the LLM decoder -- no generation needed at all for this strategy.
            pooled = extract_visual_diversity_features(model, inputs)
            for b in range(len(batch_imgs)):
                features_list.append((batch_line_ids[b], pooled[b]))
            continue

        with torch.no_grad():
            generated_outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                output_scores=(strategy == "entropy"),
                return_dict_in_generate=True,
                output_hidden_states=(strategy == "kmeans_center"),
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None,
                eos_token_id=eos_token_id
            )

            if strategy == "entropy":
                gen_sequences = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_outputs.sequences)]
                batch_confidences = [[] for _ in range(len(batch_imgs))]
                for step_idx, step_logits in enumerate(generated_outputs.scores):
                    probs = torch.softmax(step_logits, dim=-1)
                    max_probs, _ = torch.max(probs, dim=-1)
                    for b in range(len(batch_imgs)):
                        cur_tokens = gen_sequences[b][:step_idx]
                        if eos_token_id is not None and eos_token_id in cur_tokens:
                            continue
                        batch_confidences[b].append(max_probs[b].item())

                for b in range(len(batch_imgs)):
                    u = 1.0 - float(np.mean(batch_confidences[b])) if len(batch_confidences[b]) > 0 else 1.0
                    scored_candidates.append((batch_line_ids[b], u))

            elif strategy == "kmeans_center":
                last_hidden_state = generated_outputs.hidden_states[0][-1]
                attn_mask = inputs.get("attention_mask", None)
                if attn_mask is not None:
                    mask = attn_mask.unsqueeze(-1).float()
                    pooled = ((last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).float().cpu().numpy()
                else:
                    pooled = torch.mean(last_hidden_state, dim=1).float().cpu().numpy()

                for b in range(len(batch_imgs)):
                    features_list.append((batch_line_ids[b], pooled[b]))

    if strategy == "entropy":
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [x[0] for x in scored_candidates[:num_lines_to_select]]
        print(f"[AL Baseline: Entropy] Selected {len(selected)} lines with top uncertainty.")
        return selected

    elif strategy == "kmeans_center":
        k = min(num_lines_to_select, len(features_list))
        X = np.array([x[1] for x in features_list], dtype=np.float32)
        kmeans = KMeans(n_clusters=k, random_state=seed + al_iter, n_init="auto").fit(X)
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        selected = [features_list[idx][0] for idx in closest_indices]
        print(f"[AL Baseline: KMeans Center] Selected {len(selected)} lines closest to cluster centroids.")
        return selected

    raise ValueError(f"Unknown baseline strategy: {strategy}")


def evaluate_test_set(model, processor, test_dataset, prompt_text, batch_size, output_dir, al_iter, is_latin):
    total_records = len(test_dataset)
    all_preds, all_refs, all_attacks, all_ids = [], [], [], []

    model.eval()
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    for i in tqdm(range(0, total_records, batch_size), desc=f"Eval Iter {al_iter}"):
        batch_slice = [test_dataset[j] for j in range(i, min(i + batch_size, total_records))]
        batch_imgs = [to_pil_image(row["Image"]) for row in batch_slice]

        from qwen_vl_utils import process_vision_info
        batch_inputs_structs = []
        for img in batch_imgs:
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            batch_inputs_structs.append((text_prompt, image_inputs, video_inputs))

        text_prompts = [x[0] for x in batch_inputs_structs]
        all_images = [x[1][0] for x in batch_inputs_structs if x[1]]

        inputs = processor(text=text_prompts, images=all_images, padding="longest", return_tensors="pt").to(model.device)
        if hasattr(model, "dtype"):
            inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None,
                eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None
            )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        batch_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for sub_idx, row in enumerate(batch_slice):
            all_preds.append(batch_preds[sub_idx].strip())
            all_refs.append(str(row["Text"]).strip())
            all_attacks.append(row.get("attack_type", "clean"))
            all_ids.append(row.get("Identifier", f"sample_{i+sub_idx}"))

    cer_results = compute_comprehensive_cer(
        references=all_refs,
        predictions=all_preds,
        is_latin=is_latin,
        attack_types=all_attacks,
        identifiers=all_ids
    )
    per_sample_details = cer_results.pop("per_sample_details")
    preds_csv = os.path.join(output_dir, f"iter_{al_iter}_predictions.csv")
    pd.DataFrame(per_sample_details).to_csv(preds_csv, index=False)
    return cer_results


def main():
    parser = argparse.ArgumentParser(description="AL Baselines Runner (Fixed)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--unlabeled_input_dir", type=str, default=None)
    parser.add_argument("--aug_test_dir", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/baseline-run")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--initial_pool_size", type=float, default=10)
    parser.add_argument("--al_iterations", type=int, default=5)
    parser.add_argument("--samples_per_iter", type=int, default=200)
    parser.add_argument("--al_eval_subset", type=int, default=3000)
    parser.add_argument("--al_strategy", type=str, required=True, choices=["random", "entropy", "kmeans_center"])
    parser.add_argument("--diversity_embedding_type", type=str, default="vision_encoder", choices=["vision_encoder", "decoder"], help="Diversity feature space for kmeans_center")
    parser.add_argument("--eval_all_errors", action="store_true")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--tuning_mode", type=str, default="lora", choices=["lora", "qlora", "full"])
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--freeze_vision_encoder", action="store_true")
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
            for k, v in yaml_cfg.items():
                if f"--{k}" not in sys.argv:
                    setattr(args, k, v)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    strategy_folder = f"{args.al_strategy}_results"
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", strategy_folder)
    os.makedirs(results_dir, exist_ok=True)

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    is_latin = any(x in args.input_dir.lower() for x in ["teklia", "esposalles", "himanis", "newseye", "norhand", "belfort", "alcar"])
    initial_labeled_dataset = load_data(args.input_dir)

    # When a pre-split unlabeled pool is supplied, it must live in the SAME index space as
    # the initial labeled data. Concatenating up front guarantees that any line acquired from
    # the unlabeled pool later resolves to real rows during training, instead of silently
    # contributing zero rows (see line_to_rows lookup below).
    if args.unlabeled_input_dir and os.path.exists(args.unlabeled_input_dir):
        unlabeled_only_dataset = load_data(args.unlabeled_input_dir)
        full_train_dataset = concatenate_datasets([initial_labeled_dataset, unlabeled_only_dataset])
        initial_labeled_size = len(initial_labeled_dataset)
    else:
        full_train_dataset = initial_labeled_dataset
        initial_labeled_size = None

    line_to_rows: Dict[str, List[int]] = defaultdict(list)
    for idx in range(len(full_train_dataset)):
        lid = extract_base_line_id(full_train_dataset[idx], idx)
        line_to_rows[lid].append(idx)

    unique_line_ids = sorted(list(line_to_rows.keys()))
    print(f"Total Rows: {len(full_train_dataset)} | Unique Original Lines: {len(unique_line_ids)}")

    rng = np.random.default_rng(args.seed)
    shuffled_line_ids = list(unique_line_ids)
    rng.shuffle(shuffled_line_ids)

    if initial_labeled_size is not None:
        labeled_line_ids = set()
        for idx in range(initial_labeled_size):
            labeled_line_ids.add(extract_base_line_id(full_train_dataset[idx], idx))
        unlabeled_line_to_rows = {lid: rows for lid, rows in line_to_rows.items() if lid not in labeled_line_ids}
    else:
        if args.initial_pool_size <= 100:
            init_k = max(1, int((args.initial_pool_size / 100.0) * len(unique_line_ids)))
        else:
            init_k = min(len(unique_line_ids), int(args.initial_pool_size))
        labeled_line_ids = set(shuffled_line_ids[:init_k])
        unlabeled_line_to_rows = {lid: line_to_rows[lid] for lid in shuffled_line_ids[init_k:]}

    unlabeled_dataset = full_train_dataset

    test_ds = None
    if args.aug_test_dir:
        test_path = os.path.join(args.aug_test_dir, "test") if os.path.isdir(os.path.join(args.aug_test_dir, "test")) else args.aug_test_dir
        test_ds = load_data(test_path)
        if not args.eval_all_errors and "attack_type" in test_ds.column_names:
            test_ds = test_ds.filter(lambda x: x.get("attack_type", "clean") in ["clean", "none", None, "motion_blur"])

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    def collate_fn(examples):
        images = [to_pil_image(ex["Image"]) for ex in examples]
        texts = [str(ex["Text"]).strip() for ex in examples]
        prompt_prefixes = []
        texts_input = []
        for text in texts:
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            prefix = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            prompt_prefixes.append(prefix)
            texts_input.append(prefix + text + processor.tokenizer.eos_token)

        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "right"

        batch = processor(text=texts_input, images=images, return_tensors="pt", padding=True, max_pixels=286720)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        for idx, (prefix, img) in enumerate(zip(prompt_prefixes, images)):
            p_inputs = processor(text=[prefix], images=[img], return_tensors="pt", max_pixels=286720)
            p_len = p_inputs["input_ids"].shape[1]
            labels[idx, :p_len] = -100

        batch["labels"] = labels
        return batch

    run_name = os.path.basename(os.path.normpath(args.output_dir))
    metrics_csv = os.path.join(results_dir, f"{run_name}_metrics.csv")
    metrics_json = os.path.join(results_dir, f"{run_name}_metrics.json")
    all_metrics = []

    for al_iter in range(args.al_iterations + 1):
        print(f"\n{'='*60}\nACTIVE LEARNING ({args.al_strategy.upper()}) ITERATION {al_iter}/{args.al_iterations}\n{'='*60}")
        cur_labeled_rows = []
        for lid in labeled_line_ids:
            if lid not in line_to_rows:
                raise KeyError(
                    f"Labeled line ID '{lid}' has no corresponding rows in the training dataset index. "
                    "This would silently drop acquired lines from training; fix the indexing instead of ignoring it."
                )
            cur_labeled_rows.extend(line_to_rows[lid])
        cur_train_data = full_train_dataset.select(cur_labeled_rows)
        print(f"Training on {len(labeled_line_ids)} unique lines ({len(cur_train_data)} total images)...")

        iter_save_path = os.path.join(args.output_dir, f"iter_{al_iter}_model")
        os.makedirs(iter_save_path, exist_ok=True)

        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )

        if args.tuning_mode == "lora":
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)

        if args.freeze_vision_encoder:
            # Same `.visual` vs `.model.visual` architecture split as extract_visual_diversity_features:
            # a bare hasattr(model, "visual") silently no-ops on Qwen3-VL, leaving the vision
            # encoder trainable despite --freeze_vision_encoder being set.
            _vlm = model.get_base_model() if isinstance(model, PeftModel) else model
            _vision_module = getattr(_vlm, "visual", None) or getattr(getattr(_vlm, "model", None), "visual", None)
            if _vision_module is not None:
                for param in _vision_module.parameters():
                    param.requires_grad = False
            else:
                raise AttributeError(
                    "--freeze_vision_encoder was set but no `.visual` or `.model.visual` submodule "
                    "was found; refusing to silently continue with an unfrozen vision encoder."
                )

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "trainer_tmp"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            remove_unused_columns=False,
            seed=args.seed + al_iter
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=cur_train_data,
            data_collator=collate_fn
        )
        trainer.train()
        trainer.save_model(iter_save_path)
        processor.save_pretrained(iter_save_path)

        iter_metrics: Dict[str, Any] = {
            "iteration": al_iter,
            "strategy": args.al_strategy,
            "annotated_lines": len(labeled_line_ids),
            "total_train_rows": len(cur_train_data)
        }

        if test_ds is not None:
            eval_results = evaluate_test_set(
                model=model,
                processor=processor,
                test_dataset=test_ds,
                prompt_text=prompt_text,
                batch_size=args.batch_size * 2,
                output_dir=iter_save_path,
                al_iter=al_iter,
                is_latin=is_latin
            )
            iter_metrics.update(eval_results)

        if al_iter < args.al_iterations:
            selected_lines = select_baseline_lines(
                model=model,
                processor=processor,
                unlabeled_line_dict=unlabeled_line_to_rows,
                full_dataset=unlabeled_dataset,
                prompt_text=prompt_text,
                num_lines_to_select=args.samples_per_iter,
                subset_size=args.al_eval_subset,
                batch_size=args.batch_size,
                strategy=args.al_strategy,
                al_iter=al_iter,
                seed=args.seed,
                diversity_embedding_type=args.diversity_embedding_type
            )

            manifest_file = os.path.join(results_dir, f"{run_name}_iter_{al_iter}_acquired_ids.json")
            with open(manifest_file, "w", encoding="utf-8") as mf:
                json.dump({"iteration": al_iter, "strategy": args.al_strategy, "acquired_lines": selected_lines}, mf, indent=2)

            for lid in selected_lines:
                labeled_line_ids.add(lid)
                unlabeled_line_to_rows.pop(lid, None)

        all_metrics.append(iter_metrics)
        with open(metrics_json, "w", encoding="utf-8") as jf:
            json.dump(all_metrics, jf, indent=4)
        pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)

        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n[Done] Baseline {args.al_strategy} complete! Stored at:\nCSV : {metrics_csv}\nJSON: {metrics_json}")


if __name__ == "__main__":
    main()
