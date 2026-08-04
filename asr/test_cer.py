import os
import argparse
import sys
import json
import re
import string
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
from datasets import load_from_disk
import io
from tqdm import tqdm

try:
    from sacrebleu.metrics import CHRF
    from jiwer import cer
except ImportError:
    print("The 'jiwer' and 'sacrebleu' libraries are required to calculate CER/CHRF.")
    print("Please install using: pip install jiwer sacrebleu")
    exit(1)

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "eval"))
from model_runners.common import calculate_cer_components, clean_ocr_text

# Regex patterns for cleaning
DIACRITICS_PATTERN = re.compile(r'[\u064b-\u0652\u0640\u0670]')
SPECIAL_TOKENS_PATTERN = re.compile(r'<[^>]+>')
PUNCTUATION_PATTERN = re.compile(f'[{re.escape(string.punctuation)}]')

def clean_text(text: str, is_latin: bool = False) -> str:
    """
    Cleans OCR text by:
    1. Standardizing via clean_ocr_text
    2. Handling explicit <space> tokens vs character spacing
    3. Removing diacritics & punctuation
    4. Normalizing whitespace
    """
    if not text:
        return ""
    
    # 1. Base normalization
    cleaned = clean_ocr_text(text, is_latin=is_latin)
    
    # 2. Handle character-spaced ground truths with explicit <space> tokens
    if '<space>' in cleaned:
        # Convert explicit word boundaries (<space>) to a unique placeholder marker
        cleaned = cleaned.replace('<space>', '___WORD_BOUND___')
        
        # Remove any remaining special tokens (<unk>, etc.)
        cleaned = SPECIAL_TOKENS_PATTERN.sub('', cleaned)
        
        # Remove single spaces between characters
        cleaned = cleaned.replace(' ', '')
        
        # Restore actual word boundaries as standard spaces
        cleaned = cleaned.replace('___WORD_BOUND___', ' ')
    else:
        # Standard token cleanup for model predictions or standard texts
        cleaned = SPECIAL_TOKENS_PATTERN.sub(' ', cleaned)
    
    # 3. Remove diacritics
    cleaned = DIACRITICS_PATTERN.sub('', cleaned)
    
    # 4. Remove punctuation
    cleaned = PUNCTUATION_PATTERN.sub('', cleaned)
    
    # 5. Collapse multiple whitespace characters into a single space and strip leading/trailing spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def to_pil_image(image_value, scale=1.0):
    if isinstance(image_value, Image.Image):
        img = image_value.convert("RGB")
    elif isinstance(image_value, str) and os.path.exists(image_value):
        img = Image.open(image_value).convert("RGB")
    elif isinstance(image_value, dict):
        if image_value.get("path"):
            img = Image.open(image_value["path"]).convert("RGB")
        elif image_value.get("bytes") is not None:
            img = Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
    else:
        img = image_value
        
    if scale != 1.0 and isinstance(img, Image.Image):
        w, h = img.size
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        
    return img

def load_data(split_path, scale=1.0):
    if os.path.isdir(split_path) and os.path.exists(os.path.join(split_path, "state.json")):
        print(f"Detected Hugging Face dataset format at {split_path}")
        ds = load_from_disk(split_path)
        df = ds.to_pandas()
    else:
        print(f"Loading raw parquet dataset from {split_path}")
        df = pd.read_parquet(split_path)
        
    df['Image'] = df['Image'].apply(lambda x: to_pil_image(x, scale))
    return df

def main():
    parser = argparse.ArgumentParser(description="Test CER for Jawi & Latin OCR")
    parser.add_argument("--base_model_id", type=str, default="aisingapore/Qwen-SEA-LION-v4-8B-VL", help="Base model ID")
    parser.add_argument("--lora_model_dir", type=str, default="models/my-custom-run/final", help="Path to trained LoRA adapter")
    parser.add_argument("--input_dir", type=str, default="Jawi-OCR-data-v4", help="Path to input data directory containing the 'test' file")
    parser.add_argument("--prompt_path", type=str, default="eval/vanilla_prompt.txt", help="Path to prompt text file")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to test")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default="eval/results", help="Directory to save the evaluation results")
    parser.add_argument("--output_name", type=str, default=None, help="Output name prefix")
    parser.add_argument("--resolution_scale", type=float, default=1.0, help="Scale to resize image for resolution robustness testing")
    
    args = parser.parse_args()
    
    def resolve_path(path_arg, *subpaths):
        if os.path.isabs(path_arg):
            return os.path.join(path_arg, *subpaths) if subpaths else path_arg
        else:
            return os.path.join(base_dir, path_arg, *subpaths) if subpaths else os.path.join(base_dir, path_arg)

    test_path = resolve_path(args.input_dir, "test")
    if not os.path.exists(test_path):
        val_path = resolve_path(args.input_dir, "validation")
        if os.path.exists(val_path):
            print(f"Test split not found at {test_path}. Falling back to validation split at {val_path}")
            test_path = val_path
        else:
            print(f"Neither test nor validation found at subdirectories of {args.input_dir}. Using {args.input_dir} directly.")
            test_path = resolve_path(args.input_dir)
            
    prompt_path = resolve_path(args.prompt_path)
    lora_model_dir = resolve_path(args.lora_model_dir)
    
    with open(prompt_path, "r") as f:
        prompt_text = f.read().strip()
        
    print(f"Loading test data from {test_path} with scale {args.resolution_scale}...")
    df_test = load_data(test_path, args.resolution_scale)
    if args.max_samples:
        df_test = df_test.head(args.max_samples)
    
    print(f"Loading base model {args.base_model_id}...")
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id, 
        device_map="auto", 
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    if os.path.exists(lora_model_dir) and lora_model_dir.lower() != "none" and lora_model_dir != "none":
        if os.path.exists(os.path.join(lora_model_dir, "adapter_config.json")):
            print(f"Loading LoRA weights from {lora_model_dir}...")
            model = PeftModel.from_pretrained(base_model, lora_model_dir)
        elif os.path.exists(os.path.join(lora_model_dir, "config.json")):
            print(f"Loading full finetuned model from {lora_model_dir}...")
            del base_model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            model = AutoModelForImageTextToText.from_pretrained(
                lora_model_dir,
                device_map="auto",
                dtype=torch.bfloat16,
                trust_remote_code=True
            )
        else:
            print(f"Warning: Model directory '{lora_model_dir}' found but lacks adapter_config.json or config.json. Evaluating with BASE MODEL only.")
            model = base_model
    else:
        print(f"Evaluating with BASE MODEL only.")
        model = base_model
        
    model.eval()
    
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    
    chrf_metric = CHRF()
    predictions = []
    references = []
    results_records = []
    
    dataset_records = df_test.to_dict('records')
    total_records = len(dataset_records)
    
    print(f"Starting batched evaluation on {total_records} samples (Batch Size: {args.batch_size})...")
    for i in tqdm(range(0, total_records, args.batch_size), desc="Processing Batches"):
        batch_slice = dataset_records[i : i + args.batch_size]
        batch_imgs = [row["Image"] for row in batch_slice]
        
        try:
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
                    **inputs, 
                    max_new_tokens=256, 
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None,
                    eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None
                )
                
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            batch_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            batch_predictions = [p.strip() for p in batch_preds]
        except Exception as e:
            print(f"Batch generation failed: {e}. Falling back to individual generation...")
            batch_predictions = []
            for img in batch_imgs:
                messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
                text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text_input], images=[img], return_tensors="pt", padding=True).to(model.device)
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                batch_predictions.append(output_text.strip())

        for sub_idx, row in enumerate(batch_slice):
            ref = str(row["Text"]).strip()
            pred = batch_predictions[sub_idx]
            identifier = row["Identifier"] if "Identifier" in row else f"idx_{i+sub_idx}"
            attack_type = f"resolution_{args.resolution_scale}" if args.resolution_scale != 1.0 else "clean"
            
            if not ref: ref = " "
            if not pred: pred = " "
            
            predictions.append(pred)
            references.append(ref)

            input_ds_path = getattr(args, 'dataset_path', args.input_dir)
            is_latin_ds = input_ds_path and any(x in input_ds_path.lower() for x in ["teklia", "esposalles", "himanis", "newseye", "norhand", "belfort", "alcar"])
            
            # --- Raw Evaluation ---
            cer_data_raw = calculate_cer_components(ref, pred)
            try: chrf_raw = chrf_metric.sentence_score(pred, [ref]).score / 100.0
            except: chrf_raw = 0.0

            # --- Clean Evaluation (Preserves normal spaces) ---
            clean_gt = clean_text(ref, is_latin=is_latin_ds)
            clean_pred = clean_text(pred, is_latin=is_latin_ds)

            cer_data_clean = calculate_cer_components(clean_gt, clean_pred)
            try: chrf_clean = chrf_metric.sentence_score(clean_pred, [clean_gt]).score / 100.0
            except: chrf_clean = 0.0

            results_records.append({
                "Identifier": identifier, 
                "attack_type": attack_type,
                "original_text": ref, 
                "raw_prediction": pred, 
                "clean_groundtruth": clean_gt,
                "clean_prediction": clean_pred, 
                "cer_raw_pred": cer_data_raw["cer"], 
                "substitute_cer_raw_pred": cer_data_raw["sub_rate"], 
                "delete_cer_raw_pred": cer_data_raw["del_rate"], 
                "insertion_cer_raw_pred": cer_data_raw["ins_rate"], 
                "chrF_raw_pred": chrf_raw,
                "cer_clean": cer_data_clean["cer"], 
                "substitute_cer_clean": cer_data_clean["sub_rate"], 
                "delete_cer_clean": cer_data_clean["del_rate"], 
                "insertion_cer_clean": cer_data_clean["ins_rate"], 
                "chrF_clean": chrf_clean
            })
            
    print("\nComputing Overall Raw CER...")
    cer_score = cer(references, predictions)
    
    df_results = pd.DataFrame(results_records)
    
    print("\n" + "="*40)
    print("EVALUATION COMPLETE")
    print("="*40)
    print(f"Total Samples Tested : {total_records}")
    print(f"Global Raw CER       : {cer_score:.4f} ({(cer_score*100):.2f}%)")
    print(f"Average CER (raw_pred) : {df_results['cer_raw_pred'].mean():.4f}")
    print(f"Average CER (clean)    : {df_results['cer_clean'].mean():.4f}")
    print("="*40)
    
    # Extract model name
    if args.output_name:
        model_name = args.output_name
    elif args.lora_model_dir and args.lora_model_dir.lower() != "none":
        model_name = os.path.basename(os.path.normpath(args.lora_model_dir))
        if model_name in ["final"] or model_name.startswith("checkpoint") or model_name.startswith("iter_"):
            model_name = os.path.basename(os.path.dirname(os.path.normpath(args.lora_model_dir)))
    else:
        model_name = os.path.basename(os.path.normpath(args.base_model_id))
        if model_name in ["final"] or model_name.startswith("checkpoint") or model_name.startswith("iter_"):
            model_name = os.path.basename(os.path.dirname(os.path.normpath(args.base_model_id)))

    dataset_name = os.path.basename(os.path.normpath(args.input_dir))
    
    eval_results_dir = resolve_path(args.output_dir)
    os.makedirs(eval_results_dir, exist_ok=True)
    
    scale_suffix = f"_res{args.resolution_scale}" if args.resolution_scale != 1.0 else ""
    results_path = os.path.join(eval_results_dir, f"{model_name}_{dataset_name}{scale_suffix}.csv")
    df_results.to_csv(results_path, index=False)
    
    json_output_path = os.path.join(eval_results_dir, f"{model_name}_{dataset_name}{scale_suffix}.json")
    agg_json_metrics = {}
    
    if "attack_type" in df_results.columns:
        grouped = df_results.groupby("attack_type")
        for attack_name, group in grouped:
            agg_json_metrics[attack_name] = {
                "sample_count": int(len(group)),
                "raw_pred": {
                    "avg_cer": float(group["cer_raw_pred"].mean()), 
                    "avg_substitute_cer": float(group["substitute_cer_raw_pred"].mean()), 
                    "avg_delete_cer": float(group["delete_cer_raw_pred"].mean()), 
                    "avg_insertion_cer": float(group["insertion_cer_raw_pred"].mean()), 
                    "avg_chrF": float(group["chrF_raw_pred"].mean())
                },
                "clean": {
                    "avg_cer": float(group["cer_clean"].mean()), 
                    "avg_substitute_cer": float(group["substitute_cer_clean"].mean()), 
                    "avg_delete_cer": float(group["delete_cer_clean"].mean()), 
                    "avg_insertion_cer": float(group["insertion_cer_clean"].mean()), 
                    "avg_chrF": float(group["chrF_clean"].mean())
                }
            }

    agg_json_metrics["GLOBAL_TOTAL_AVERAGE"] = {
        "sample_count": int(len(df_results)),
        "raw_pred": {
            "avg_cer": float(df_results["cer_raw_pred"].mean()), 
            "avg_substitute_cer": float(df_results["substitute_cer_raw_pred"].mean()), 
            "avg_delete_cer": float(df_results["delete_cer_raw_pred"].mean()), 
            "avg_insertion_cer": float(df_results["insertion_cer_raw_pred"].mean()), 
            "avg_chrF": float(df_results["chrF_raw_pred"].mean())
        },
        "clean": {
            "avg_cer": float(df_results["cer_clean"].mean()), 
            "avg_substitute_cer": float(df_results["substitute_cer_clean"].mean()), 
            "avg_delete_cer": float(df_results["delete_cer_clean"].mean()), 
            "avg_insertion_cer": float(df_results["insertion_cer_clean"].mean()), 
            "avg_chrF": float(df_results["chrF_clean"].mean())
        }
    }
        
    with open(json_output_path, "w", encoding="utf-8") as json_file: 
        json.dump(agg_json_metrics, json_file, indent=4, ensure_ascii=False)

    print(f"Detailed predictions saved to {results_path} and {json_output_path}")

if __name__ == "__main__":
    main()
