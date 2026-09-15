#!/usr/bin/env python3
"""
train_ocr.py - Full Dataset Supervised Fine-Tuning (100% Data)

Includes:
- Prompt token loss masking with -100.
- Configurable random seed.
- Unified comprehensive CER evaluation (raw, clean, per-line, corpus).
"""

import os
import sys
import gc
import json
import random
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, PeftModel

from evaluate_metrics import compute_comprehensive_cer


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


def main():
    parser = argparse.ArgumentParser(description="Full Dataset Fine-Tuning")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/full-run")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--tuning_mode", type=str, default="lora", choices=["lora", "full"])
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--freeze_vision_encoder", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--eval_all_errors", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    final_model_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_model_dir, exist_ok=True)

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    is_latin = any(x in args.input_dir.lower() for x in ["teklia", "esposalles", "himanis", "newseye", "norhand", "belfort", "alcar"])

    train_ds = load_data(os.path.join(args.input_dir, "train") if os.path.isdir(os.path.join(args.input_dir, "train")) else args.input_dir)
    test_ds = None
    test_path = os.path.join(args.input_dir, "test")
    if os.path.exists(test_path):
        test_ds = load_data(test_path)
        if not args.eval_all_errors and "attack_type" in test_ds.column_names:
            test_ds = test_ds.filter(lambda x: x.get("attack_type", "clean") in ["clean", "none", None, "motion_blur"])

    print(f"Training on 100% data: {len(train_ds)} samples...")
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

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )

    if args.tuning_mode == "lora":
        target_mods = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_mods,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    if args.freeze_vision_encoder:
        # Qwen2-VL/2.5-VL expose `.visual` directly; Qwen3-VL nests it at `.model.visual`
        # under the composite Qwen3VLModel. A bare hasattr(model, "visual") silently no-ops
        # on the latter, leaving the vision encoder trainable despite the flag being set.
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
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    print(f"Model saved to: {final_model_dir}")

    # Evaluate on test set
    if test_ds is not None:
        total_records = len(test_ds)
        all_preds, all_refs, all_attacks, all_ids = [], [], [], []
        model.eval()
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"

        print(f"Evaluating {total_records} test samples...")
        for i in tqdm(range(0, total_records, args.batch_size * 2), desc="Test Inference"):
            batch_slice = [test_ds[j] for j in range(i, min(i + args.batch_size * 2, total_records))]
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
        preds_csv = os.path.join(final_model_dir, "test_predictions.csv")
        pd.DataFrame(per_sample_details).to_csv(preds_csv, index=False)

        metrics_json = os.path.join(final_model_dir, "test_metrics.json")
        with open(metrics_json, "w", encoding="utf-8") as jf:
            json.dump(cer_results, jf, indent=4)

        print("\n" + "="*60)
        print("FULL FINE-TUNING EVALUATION RESULTS:")
        print(f"Raw CER (Per-Line Mean) : {cer_results['raw_cer_per_line']:.4%}")
        print(f"Raw CER (Corpus Level)  : {cer_results['raw_cer_corpus']:.4%}")
        print(f"Clean CER (Per-Line)    : {cer_results['clean_cer_per_line']:.4%}")
        print(f"Clean CER (Corpus Level): {cer_results['clean_cer_corpus']:.4%}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
