#!/usr/bin/env python3
"""
train_diva_selfdistill.py - standalone DIVA + self-distillation training script.

A separate entry point from train_active_learning_extended.py (per request), rather
than reaching it via `--al_strategy vis_div --self_distill_ratio 1.0` on the shared
script. It imports the genuinely-shared, already-tested pieces instead of
re-implementing them -- the correctness-sensitive logic (line-ID grouping, the
oracle/self-distill pool split, the confidence-weighted distillation loss) lives in
ONE place either way; duplicating it here would just create a second copy to keep in
sync and a second place for the same bug to hide in.

Imported from train_active_learning_extended.py, unchanged:
  - load_data, to_pil_image           (dataset loading)
  - extract_base_line_id              (Fix 03: groups augmented rows under one line id)
  - evaluate_and_select_lines         (DIVA selection + the self-distill pool split)
  - run_self_distillation_pass        (the confidence-weighted CE pass -- see its
                                        docstring for why that loss form, not full KL)
  - evaluate_test_set                 (CER evaluation)

Reimplemented here (the AL loop / Trainer plumbing, same shape as main() in
train_active_learning_extended.py, trimmed to just the DIVA + self-distillation path --
no --al_strategy branch, no vanilla/random/entropy/kmeans code paths):
  - argument parsing (DIVA-only subset of flags, self_distill_ratio defaults to 1.0
    here since running this script at all means you want self-distillation on)
  - the active-learning loop itself (dataset setup, per-iteration Trainer, the
    self-distillation pass, checkpoint/metrics writing)

Before a full run, smoke-test with smoke_self_distillation.py first (no local GPU to
verify this against during development -- see its docstring).

Usage:
  python train_diva_selfdistill.py \
      --model_id Qwen/Qwen3-VL-4B-Instruct \
      --input_dir /dest/thura/data/Teklia_Belfort-line_labeled_10 \
      --unlabeled_input_dir /dest/thura/data/Teklia_Belfort-line_unlabeled_90 \
      --aug_test_dir /dest/thura/data/Teklia_Belfort-line \
      --prompt_path ../eval/prompt_Teklia_Belfort-line.txt \
      --output_dir models/Teklia_Belfort-line_al_diva_selfdistill_alpha20_seed42_standalone \
      --alpha 20 --beta 3 --al_eval_subset 3000 --dynamic_quota \
      --diversity_embedding_type vision_encoder --self_distill_ratio 1.0 \
      --freeze_vision_encoder --al_iterations 5 --samples_per_iter 200
"""
import os
import sys
import gc
import json
import random
import argparse
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import concatenate_datasets
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel

from train_active_learning_extended import (
    load_data,
    to_pil_image,
    extract_base_line_id,
    evaluate_and_select_lines,
    run_self_distillation_pass,
    evaluate_test_set,
)


def main():
    parser = argparse.ArgumentParser(description="DIVA Active Learning + self-distillation (standalone)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--unlabeled_input_dir", type=str, default=None)
    parser.add_argument("--aug_test_dir", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/diva-selfdistill-run")
    parser.add_argument("--seed", type=int, default=42)

    # Active learning / DIVA
    parser.add_argument("--initial_pool_size", type=float, default=10)
    parser.add_argument("--al_iterations", type=int, default=5)
    parser.add_argument("--samples_per_iter", type=int, default=200)
    parser.add_argument("--al_eval_subset", type=int, default=3000)
    parser.add_argument("--alpha", type=int, default=20)
    parser.add_argument("--beta", type=int, default=2)
    parser.add_argument("--dynamic_quota", action="store_true")
    parser.add_argument("--eval_all_errors", action="store_true")
    parser.add_argument("--diversity_embedding_type", type=str, default="vision_encoder", choices=["vision_encoder", "decoder"])

    # Self-distillation (on by default here -- this script's whole point)
    parser.add_argument("--self_distill_ratio", type=float, default=1.0, help="0.0 disables it, same as the shared script; this script defaults to 1.0 since that's why you'd run it separately.")
    parser.add_argument("--self_distill_lr", type=float, default=None, help="Default: fall back to --lr.")
    parser.add_argument("--self_distill_batch_size", type=int, default=4)
    parser.add_argument("--self_distill_epochs", type=int, default=1)

    # Training
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
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "vis_div_results")
    os.makedirs(results_dir, exist_ok=True)

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    is_latin = any(x in args.input_dir.lower() for x in ["teklia", "esposalles", "himanis", "newseye", "norhand", "belfort", "alcar", "iam"])

    print("Loading datasets...")
    initial_labeled_dataset = load_data(args.input_dir)

    # Fix 03 (follow-up): same pre-split concatenation as the shared script -- acquired
    # lines (oracle or self-distill) must resolve to real rows, never silently drop.
    if args.unlabeled_input_dir and os.path.exists(args.unlabeled_input_dir):
        print(f"Using pre-split unlabeled directory: {args.unlabeled_input_dir}")
        unlabeled_only_dataset = load_data(args.unlabeled_input_dir)
        full_train_dataset = concatenate_datasets([initial_labeled_dataset, unlabeled_only_dataset])
        initial_labeled_size = len(initial_labeled_dataset)
    else:
        full_train_dataset = initial_labeled_dataset
        initial_labeled_size = None

    print("Mapping dataset rows to unique original line IDs...")
    line_to_rows: Dict[str, List[int]] = defaultdict(list)
    for idx in range(len(full_train_dataset)):
        lid = extract_base_line_id(full_train_dataset[idx], idx)
        line_to_rows[lid].append(idx)

    unique_line_ids = sorted(line_to_rows.keys())
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
    print(f"Initial Labeled Lines: {len(labeled_line_ids)} | Initial Unlabeled Lines: {len(unlabeled_line_to_rows)}")

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

        # Fix 15: Mask prompt tokens
        for idx, (prefix, img) in enumerate(zip(prompt_prefixes, images)):
            p_inputs = processor(text=[prefix], images=[img], return_tensors="pt", max_pixels=286720)
            p_len = p_inputs["input_ids"].shape[1]
            labels[idx, :p_len] = -100

        batch["labels"] = labels
        return batch

    run_name = os.path.basename(os.path.normpath(args.output_dir))
    metrics_csv = os.path.join(results_dir, f"{run_name}_metrics.csv")
    metrics_json = os.path.join(results_dir, f"{run_name}_metrics.json")
    model_metrics_csv = os.path.join(args.output_dir, "metrics.csv")
    model_metrics_json = os.path.join(args.output_dir, "metrics.json")
    all_metrics = []
    # Self-distillation pool identified by iteration N's model, consumed at the start
    # of iteration N+1's training -- see run_self_distillation_pass()'s docstring.
    pending_self_distill_items: List[Dict[str, Any]] = []

    for al_iter in range(args.al_iterations + 1):
        print(f"\n{'='*60}\nACTIVE LEARNING ITERATION {al_iter}/{args.al_iterations}\n{'='*60}")
        cur_labeled_rows = []
        for lid in labeled_line_ids:
            if lid not in line_to_rows:
                raise KeyError(
                    f"Labeled line ID '{lid}' has no corresponding rows in the training dataset index. "
                    "This would silently drop acquired lines from training; fix the indexing instead of ignoring it."
                )
            cur_labeled_rows.extend(line_to_rows[lid])
        cur_train_data = full_train_dataset.select(cur_labeled_rows)
        print(f"Training on {len(labeled_line_ids)} unique lines ({len(cur_train_data)} total augmented images)...")

        iter_save_path = os.path.join(args.output_dir, f"iter_{al_iter}_model")
        os.makedirs(iter_save_path, exist_ok=True)

        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
        )

        if args.tuning_mode == "lora":
            peft_config = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.target_modules,
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

        if args.freeze_vision_encoder:
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
            seed=args.seed + al_iter,
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=cur_train_data, data_collator=collate_fn)
        trainer.train()

        if pending_self_distill_items:
            run_self_distillation_pass(
                model=model,
                processor=processor,
                self_distill_items=pending_self_distill_items,
                full_dataset=unlabeled_dataset,
                unlabeled_line_dict=unlabeled_line_to_rows,
                prompt_text=prompt_text,
                lr=args.self_distill_lr if args.self_distill_lr is not None else args.lr,
                batch_size=args.self_distill_batch_size,
                epochs=args.self_distill_epochs,
                seed=args.seed + al_iter,
            )
            pending_self_distill_items = []

        trainer.save_model(iter_save_path)
        processor.save_pretrained(iter_save_path)

        iter_metrics: Dict[str, Any] = {
            "iteration": al_iter,
            "annotated_lines": len(labeled_line_ids),
            "total_train_rows": len(cur_train_data),
        }

        if test_ds is not None:
            eval_results = evaluate_test_set(
                model=model, processor=processor, test_dataset=test_ds, prompt_text=prompt_text,
                batch_size=args.batch_size * 2, output_dir=iter_save_path, al_iter=al_iter, is_latin=is_latin,
            )
            iter_metrics.update(eval_results)

        if al_iter < args.al_iterations:
            selected_lines, avg_u, cluster_info, self_distill_items = evaluate_and_select_lines(
                model=model,
                processor=processor,
                unlabeled_line_dict=unlabeled_line_to_rows,
                full_dataset=unlabeled_dataset,
                prompt_text=prompt_text,
                num_lines_to_select=args.samples_per_iter,
                subset_size=args.al_eval_subset,
                batch_size=args.batch_size,
                strategy="vis_div",
                alpha=args.alpha,
                beta=args.beta,
                al_iter=al_iter,
                seed=args.seed,
                dynamic_quota=args.dynamic_quota,
                is_latin=is_latin,
                diversity_embedding_type=args.diversity_embedding_type,
                self_distill_ratio=args.self_distill_ratio,
            )
            iter_metrics["avg_selected_uncertainty"] = avg_u
            iter_metrics.update(cluster_info)
            iter_metrics["self_distill_pool_size"] = len(self_distill_items)

            manifest_file = os.path.join(results_dir, f"{run_name}_iter_{al_iter}_acquired_ids.json")
            with open(manifest_file, "w", encoding="utf-8") as mf:
                json.dump({"iteration": al_iter, "acquired_lines": selected_lines}, mf, indent=2)

            if self_distill_items:
                sd_manifest_file = os.path.join(results_dir, f"{run_name}_iter_{al_iter}_self_distill_pool.json")
                with open(sd_manifest_file, "w", encoding="utf-8") as sdf:
                    json.dump({"iteration": al_iter, "self_distill_items": self_distill_items}, sdf, indent=2)
            pending_self_distill_items = self_distill_items

            for lid in selected_lines:
                labeled_line_ids.add(lid)
                unlabeled_line_to_rows.pop(lid, None)

        all_metrics.append(iter_metrics)
        with open(metrics_json, "w", encoding="utf-8") as jf:
            json.dump(all_metrics, jf, indent=4)
        pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
        with open(model_metrics_json, "w", encoding="utf-8") as jf:
            json.dump(all_metrics, jf, indent=4)
        pd.DataFrame(all_metrics).to_csv(model_metrics_csv, index=False)

        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n[Done] DIVA + self-distillation run complete! Results stored at:\nCSV : {metrics_csv}\nJSON: {metrics_json}")


if __name__ == "__main__":
    main()
