#!/bin/bash

export PYTHONUNBUFFERED=1

# Configuration
MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
BATCH_SIZE=1
GRAD_ACCUM=16
EPOCHS=3

DATASETS=(
    "Teklia_Esposalles-line"
    "Teklia_Himanis-line"
    "Teklia_Belfort-line"
    "Jawi-OCR-data-v4"
    # "Teklia_HOME-Alcar-line"
    # "Teklia_NewsEye-Austrian-line"
)

echo "Preparing Original Datasets..."
python prepare_original_data.py
python prepare_al_pools_50.py

echo "Generating Dataset-Specific Prompts..."
mkdir -p eval
for DS in "${DATASETS[@]}"; do
    case "$DS" in
        "Jawi-OCR-data-v4") SCRIPT="Jawi"; LANG="Malay" ;;
        "Teklia_Esposalles-line") SCRIPT="Latin"; LANG="Spanish" ;;
        "Teklia_HOME-Alcar-line") SCRIPT="Latin"; LANG="Spanish" ;;
        "Teklia_NewsEye-Austrian-line") SCRIPT="Latin"; LANG="German" ;;
        "Teklia_NorHand-v3-line") SCRIPT="Latin"; LANG="Norwegian" ;;
        "Teklia_Himanis-line") SCRIPT="Latin"; LANG="French" ;;
        "Teklia_Belfort-line") SCRIPT="Latin"; LANG="French" ;;
        *) SCRIPT="Latin"; LANG="unknown" ;;
    esac
    echo "Transcribe the ${SCRIPT} script in this ${LANG} text image into ${LANG} text" > "eval/prompt_${DS}.txt"
done

echo "Starting Experiments on Original other_hist_bench data"
echo "Running sequentially (one dataset/strategy at a time) on GPUs 0,1."
echo "========================================================================="

mkdir -p logs_hist_bench
mkdir -p other_hist_bench/finetune_models
mkdir -p eval/results/hist_bench_zeroshot

# 1. Full-Dataset Full Finetuning
echo "--- Running Full-Dataset Full Finetuning ---"
for DS in "${DATASETS[@]}"; do
    case "$DS" in
        "Jawi-OCR-data-v4")             DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        "Teklia_Esposalles-line")      DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        # "Teklia_NorHand-v3-line")      DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        "Teklia_Belfort-line")          DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        # "Teklia_HOME-Alcar-line")      DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        # "Teklia_NewsEye-Austrian-line") DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        "Teklia_Himanis-line")          DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        *)                              DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
    esac

    if [ "$DS" == "Jawi-OCR-data-v4" ]; then
        INPUT_DIR="Jawi-OCR-data-v4"
    else
        INPUT_DIR="/dest/thura/data/${DS}"
    fi
    OUTPUT_DIR="other_hist_bench/finetune_models/${DS}_full"
    LOG_FILE="logs_hist_bench/train_${DS}_full.log"
    EVAL_OUT="eval/results/hist_bench_full/${DS}"
    EVAL_LOG="logs_hist_bench/eval_${DS}_full.log"
    
    # Skip if both training and eval are complete
    if [ -d "${OUTPUT_DIR}/final" ] && [ -f "${EVAL_LOG}" ] && grep -q 'EVALUATION COMPLETE' "${EVAL_LOG}"; then
        echo "Skipping completed Full-Dataset Full Finetuning for ${DS}."
        continue
    fi
    
    echo "Running Full-Dataset Full Finetuning on GPUs 0,1 for ${DS} (Batch: ${DS_BATCH_SIZE}, Accum: ${DS_GRAD_ACCUM})"
    
    if [ ! -d "${OUTPUT_DIR}/final" ]; then
        CUDA_VISIBLE_DEVICES="0,1" python train_ocr.py \
            --model_id "${MODEL_ID}" \
            --input_dir "${INPUT_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --prompt_path "eval/prompt_${DS}.txt" \
            --tuning_mode "full" \
            --lr 5e-6 \
            --batch_size ${DS_BATCH_SIZE} \
            --gradient_accumulation_steps ${DS_GRAD_ACCUM} \
            --epochs ${EPOCHS} \
            --resume_from_checkpoint \
            > "${LOG_FILE}" 2>&1
    fi
    
    if [ ! -f "${EVAL_LOG}" ] || ! grep -q 'EVALUATION COMPLETE' "${EVAL_LOG}"; then
        echo "Running evaluation for ${DS}..."
        CUDA_VISIBLE_DEVICES="0,1" python test_cer.py \
            --base_model_id "${MODEL_ID}" \
            --lora_model_dir "./${OUTPUT_DIR}/final" \
            --input_dir "${INPUT_DIR}" \
            --prompt_path "eval/prompt_${DS}.txt" \
            --batch_size ${DS_BATCH_SIZE} \
            --output_dir "${EVAL_OUT}" \
            > "${EVAL_LOG}" 2>&1
    fi
done

# 2. Active Learning Baselines and DIVA
AL_ITERATIONS=5

echo "--- Running Active Learning ---"
for DS in "${DATASETS[@]}"; do
    case "$DS" in
        "Jawi-OCR-data-v4")             DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        "Teklia_Esposalles-line")      DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        # "Teklia_NorHand-v3-line")      DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        "Teklia_Belfort-line")          DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        # "Teklia_HOME-Alcar-line")      DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        # "Teklia_NewsEye-Austrian-line") DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        "Teklia_Himanis-line")          DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
        *)                              DS_BATCH_SIZE=1; DS_GRAD_ACCUM=128 ;;
    esac

    # Dynamically set budget per iteration based on dataset size
    case "$DS" in
        "Teklia_NorHand-v3-line")
            SAMPLES_PER_ITER=1000
            ;;
        "Teklia_HOME-Alcar-line" | "Teklia_NewsEye-Austrian-line")
            SAMPLES_PER_ITER=500
            ;;
        "Teklia_Belfort-line" | "Teklia_Himanis-line")
            SAMPLES_PER_ITER=200
            ;;
        "Teklia_Esposalles-line" | "Jawi-OCR-data-v4")
            SAMPLES_PER_ITER=100
            ;;
        *)
            SAMPLES_PER_ITER=100
            ;;
    esac

    # Set pre-split dataset paths for active learning
    if [ "$DS" == "Jawi-OCR-data-v4" ]; then
        INPUT_DIR="Jawi-OCR-data-v4_10"
        UNLABELED_DIR="Jawi-OCR-data-v4_90_unlabeled"
        TEST_DIR="Jawi-OCR-data-v4"
    else
        INPUT_DIR="/dest/thura/data/${DS}_labeled_10"
        UNLABELED_DIR="/dest/thura/data/${DS}_unlabeled_90"
        TEST_DIR="/dest/thura/data/${DS}"
    fi
    
    # Baselines
    for STRATEGY in "random" "entropy" "kmeans_center"; do
        OUTPUT_DIR="other_hist_bench/finetune_models/${DS}_al_${STRATEGY}_full"
        LOG_FILE="logs_hist_bench/train_${DS}_al_${STRATEGY}_full.log"
        EVAL_LOG_FINAL="logs_hist_bench/eval_${DS}_${STRATEGY}_full_iter_${AL_ITERATIONS}.log"
        
        # Skip if both training and final eval are complete
        if [ -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ] && [ -f "${EVAL_LOG_FINAL}" ] && grep -q 'EVALUATION COMPLETE' "${EVAL_LOG_FINAL}"; then
            echo "Skipping AL baseline (${STRATEGY}) for ${DS} (already complete)."
            continue
        fi
        
        echo "Running AL baseline (${STRATEGY}) on GPUs 0,1 for ${DS} (Batch: ${DS_BATCH_SIZE}, Accum: ${DS_GRAD_ACCUM})"
        
        if [ ! -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ]; then
            CUDA_VISIBLE_DEVICES="0,1" python train_al_baselines.py \
                --model_id "${MODEL_ID}" \
                --input_dir "${INPUT_DIR}" \
                --unlabeled_input_dir "${UNLABELED_DIR}" \
                --output_dir "${OUTPUT_DIR}" \
                --prompt_path "eval/prompt_${DS}.txt" \
                --aug_test_dir "${TEST_DIR}" \
                --al_strategy "${STRATEGY}" \
                --al_iterations ${AL_ITERATIONS} \
                --samples_per_iter ${SAMPLES_PER_ITER} \
                --tuning_mode "full" \
                --lr 5e-6 \
                --batch_size ${DS_BATCH_SIZE} \
                --gradient_accumulation_steps ${DS_GRAD_ACCUM} \
                --epochs ${EPOCHS} \
                > "${LOG_FILE}" 2>&1
        fi
        
        for al_iter in $(seq 0 ${AL_ITERATIONS}); do
            EVAL_OUT="eval/results/hist_bench_al_full/${DS}_${STRATEGY}/iter_${al_iter}"
            EVAL_LOG="logs_hist_bench/eval_${DS}_${STRATEGY}_full_iter_${al_iter}.log"
            
            if [ ! -d "./${OUTPUT_DIR}/iter_${al_iter}_model" ]; then
                echo "Skipping evaluation for ${DS} ${STRATEGY} iter_${al_iter} (model not found)."
                continue
            fi
            
            if [ ! -f "${EVAL_LOG}" ] || ! grep -q 'EVALUATION COMPLETE' "${EVAL_LOG}"; then
                echo "Evaluating iteration ${al_iter}..."
                CUDA_VISIBLE_DEVICES="0,1" python test_cer.py \
                    --base_model_id "${MODEL_ID}" \
                    --lora_model_dir "./${OUTPUT_DIR}/iter_${al_iter}_model" \
                    --input_dir "${TEST_DIR}" \
                    --prompt_path "eval/prompt_${DS}.txt" \
                    --batch_size ${DS_BATCH_SIZE} \
                    --output_dir "${EVAL_OUT}" \
                    > "${EVAL_LOG}" 2>&1
            fi
        done
    done
    
    # DIVA alpha=10
    OUTPUT_DIR="other_hist_bench/finetune_models/${DS}_al_diva_alpha10_full"
    LOG_FILE="logs_hist_bench/train_${DS}_al_diva_alpha10_full.log"
    EVAL_LOG_FINAL="logs_hist_bench/eval_${DS}_diva_alpha10_full_iter_${AL_ITERATIONS}.log"
    
    # Skip if both training and final eval are complete
    if [ -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ] && [ -f "${EVAL_LOG_FINAL}" ] && grep -q 'EVALUATION COMPLETE' "${EVAL_LOG_FINAL}"; then
        echo "Skipping DIVA (alpha=10) for ${DS} (already complete)."
        continue
    fi
    
    echo "Running DIVA (alpha=10) on GPUs 0,1 for ${DS} (Batch: ${DS_BATCH_SIZE}, Accum: ${DS_GRAD_ACCUM})"
    
    if [ ! -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ]; then
        CUDA_VISIBLE_DEVICES="0,1" python train_active_learning_extended.py \
            --model_id "${MODEL_ID}" \
            --input_dir "${INPUT_DIR}" \
            --unlabeled_input_dir "${UNLABELED_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --prompt_path "eval/prompt_${DS}.txt" \
            --aug_test_dir "${TEST_DIR}" \
            --al_strategy "vis_div" \
            --alpha 10 \
            --al_iterations ${AL_ITERATIONS} \
            --samples_per_iter ${SAMPLES_PER_ITER} \
            --tuning_mode "full" \
            --lr 5e-6 \
            --batch_size ${DS_BATCH_SIZE} \
            --gradient_accumulation_steps ${DS_GRAD_ACCUM} \
            --epochs ${EPOCHS} \
            > "${LOG_FILE}" 2>&1
    fi
    
    for al_iter in $(seq 0 ${AL_ITERATIONS}); do
        EVAL_OUT="eval/results/hist_bench_al_full/${DS}_diva_alpha10/iter_${al_iter}"
        EVAL_LOG="logs_hist_bench/eval_${DS}_diva_alpha10_full_iter_${al_iter}.log"
        
        if [ ! -d "./${OUTPUT_DIR}/iter_${al_iter}_model" ]; then
            echo "Skipping evaluation for ${DS} diva_alpha10 iter_${al_iter} (model not found)."
            continue
        fi
        
        if [ ! -f "${EVAL_LOG}" ] || ! grep -q 'EVALUATION COMPLETE' "${EVAL_LOG}"; then
            echo "Evaluating iteration ${al_iter}..."
            CUDA_VISIBLE_DEVICES="0,1" python test_cer.py \
                --base_model_id "${MODEL_ID}" \
                --lora_model_dir "./${OUTPUT_DIR}/iter_${al_iter}_model" \
                --input_dir "${TEST_DIR}" \
                --prompt_path "eval/prompt_${DS}.txt" \
                --batch_size ${DS_BATCH_SIZE} \
                --output_dir "${EVAL_OUT}" \
                > "${EVAL_LOG}" 2>&1
        fi
    done
done

echo "========================================================================="
echo "All experiments finished!"
