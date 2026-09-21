#!/usr/bin/env python3
"""
Fast smoke test of the vision-encoder diversity path (no training, no CER eval).

Loads the model (optionally LoRA-wrapped, like the AL loop), pools vision-tower
features for N real images via extract_visual_diversity_features, then checks the
features are usable for clustering: right shape, no NaN/inf, non-degenerate spread,
and KMeans/silhouette at a few k values.

Run from the codes/ directory:
  CUDA_VISIBLE_DEVICES=0 python smoke_visual_features.py \
      --input_dir /dest/thura/data/Teklia_Belfort-line_labeled_10 \
      --prompt_path ../eval/prompt_Teklia_Belfort-line.txt
"""
import argparse
import sys

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_active_learning_extended import (
    extract_visual_diversity_features,
    load_data,
    to_pil_image,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--prompt_path", required=True)
    p.add_argument("--num_images", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_lora", action="store_true", help="skip the PEFT wrap (tests the bare-model path)")
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
    model.eval()
    print(f"Model wrapped as: {type(model).__name__}")

    from qwen_vl_utils import process_vision_info

    feats = []
    for i in range(0, len(imgs), args.batch_size):
        batch = imgs[i:i + args.batch_size]
        texts, images = [], []
        for img in batch:
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            image_inputs, _ = process_vision_info(messages)
            images.append(image_inputs[0])
        inputs = processor(text=texts, images=images, padding="longest", return_tensors="pt").to(model.device)
        inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        with torch.no_grad():
            feats.append(extract_visual_diversity_features(model, inputs))
        print(f"  batch {i // args.batch_size + 1}: pooled {feats[-1].shape}")

    X = np.concatenate(feats, axis=0).astype(np.float32)
    failures = []

    print(f"\nFeature matrix: {X.shape}")
    if X.shape[0] != len(imgs):
        failures.append(f"expected {len(imgs)} vectors, got {X.shape[0]}")
    if not np.isfinite(X).all():
        failures.append("features contain NaN/inf")
    spread = float(X.std(axis=0).mean())
    print(f"Mean per-dim std across images: {spread:.6f}")
    if spread < 1e-6:
        failures.append("features are (near-)constant across images -- pooling is degenerate")

    if not failures:
        print("\nKMeans / silhouette (higher = clusters actually separate):")
        for k in (2, 5, 10):
            if k < len(X):
                labels = KMeans(n_clusters=k, random_state=args.seed, n_init="auto").fit_predict(X)
                print(f"  k={k:2d}  silhouette={silhouette_score(X, labels):.4f}")

    if failures:
        print("\nFAIL:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)
    print("\nPASS: vision-encoder features extracted and usable for clustering")


if __name__ == "__main__":
    main()
