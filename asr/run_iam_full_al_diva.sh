#!/bin/bash
# run_iam_full_al_diva.sh - standalone launcher for IAM: full-data fine-tune, the 3 AL
# baselines, and both DIVA embedding variants (decoder, vision_encoder).
#
# No new Python needed -- train_ocr.py, train_al_baselines.py, and
# train_active_learning_extended.py are already dataset-agnostic (is_latin detection
# already covers "iam", see each script's is_latin line). This just calls them
# directly for IAM, standalone from run_al_and_full_lorafixed_fixed_run2_run3.sh (the
# same "iam"/"iam_full"/"iam_random"/"iam_diva_visenc"/etc. modes already exist there
# too, via resolve_dataset_config() -- this script is the separated version, same
# convention as run_diva_selfdistill.sh).
#
# Prerequisite: prepare_iam_dataset.py has already been run (IAM-line, IAM-line/train,
# IAM-line/test, IAM-line_labeled_10, IAM-line_unlabeled_90 all exist under /dest/thura/data).
#
# Usage:
#   ./run_iam_full_al_diva.sh <task> [gpu_id]
#   task: full | random | entropy | kmeans | diva_decoder | diva_visenc | all
#   ./run_iam_full_al_diva.sh all 0     # all 6 tasks, sequential, cheapest-ish first
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:/dest/thura/code/FYP_jonpwk:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_PATH="${PYTHON_PATH:-/dest/thura/conda_envs/jawi_ocr_eval/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
SEED="${SEED:-42}"
LR="${LR:-2e-4}"
AL_EPOCHS="${AL_EPOCHS:-8}"
FULL_EPOCHS="${FULL_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAMPLES_PER_ITER="${SAMPLES_PER_ITER:-200}"
AL_ITERATIONS="${AL_ITERATIONS:-5}"
TAIL="${TAIL:-standalone}"
LORA_TARGETS="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
LORA_TARGETS_COMMA="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Fixed IAM paths (matches resolve_dataset_config() in the main pipeline script, and
# prepare_iam_dataset.py's output layout -- update both places if these ever move)
INPUT_DIR="/dest/thura/data/IAM-line_labeled_10"
UNLABELED_DIR="/dest/thura/data/IAM-line_unlabeled_90"
TEST_DIR="/dest/thura/data/IAM-line"
PROMPT_PATH="${SCRIPT_DIR}/../eval/prompt_IAM-line.txt"
PROMPT_TEXT="Transcribe the English handwritten text in this image into English text"
ALPHA=20
BETA=2
SUBSET=3000

if [ ! -f "${PROMPT_PATH}" ]; then
    echo "Creating prompt file at ${PROMPT_PATH}..."
    echo "${PROMPT_TEXT}" > "${PROMPT_PATH}"
fi

TASK="${1:?Usage: $0 <full|random|entropy|kmeans|diva_decoder|diva_visenc|all> [gpu_id]}"
GPU_ID="${2:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/models" \
    "${SCRIPT_DIR}/results/full_results" "${SCRIPT_DIR}/results/random_results" \
    "${SCRIPT_DIR}/results/entropy_results" "${SCRIPT_DIR}/results/kmeans_center_results" \
    "${SCRIPT_DIR}/results/vis_div_results" "${SCRIPT_DIR}/eval_results"

run_full() {
    local output_dir="${SCRIPT_DIR}/models/IAM-line_full_seed${SEED}_${TAIL}"
    local train_log="${SCRIPT_DIR}/logs/train_IAM-line_full_seed${SEED}_${TAIL}.log"
    local eval_out="${SCRIPT_DIR}/eval_results/IAM-line_full_seed${SEED}_${TAIL}"

    if [ -d "${output_dir}/final" ] && [ -f "${output_dir}/final/adapter_config.json" ]; then
        echo "Full model already exists at ${output_dir}/final. Skipping."
        return 0
    fi
    rm -rf "${output_dir}"

    echo "=== IAM full-data fine-tune (ceiling model) on GPU ${GPU_ID} ==="
    "${PYTHON_PATH}" "${SCRIPT_DIR}/train_ocr.py" \
        --model_id "${MODEL_ID}" \
        --input_dir "${TEST_DIR}" \
        --output_dir "${output_dir}" \
        --prompt_path "${PROMPT_PATH}" \
        --tuning_mode lora \
        --lora_target_modules "${LORA_TARGETS_COMMA}" \
        --lr "${LR}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --epochs "${FULL_EPOCHS}" \
        --freeze_vision_encoder \
        --seed "${SEED}" \
        > "${train_log}" 2>&1 || { echo "ERROR: full fine-tune failed! Check: ${train_log}"; return 1; }
    rm -rf "${output_dir}/trainer_tmp" 2>/dev/null

    echo "Evaluating full model on test set..."
    "${PYTHON_PATH}" "${SCRIPT_DIR}/test_cer.py" \
        --base_model_id "${MODEL_ID}" \
        --lora_model_dir "${output_dir}/final" \
        --input_dir "${TEST_DIR}" \
        --prompt_path "${PROMPT_PATH}" \
        --batch_size $((BATCH_SIZE * 2)) \
        --output_dir "${eval_out}" \
        >> "${train_log}" 2>&1
    echo "Full fine-tune complete."
}

run_baseline() {
    local strategy="$1"
    local results_subdir="$2"
    local output_dir="${SCRIPT_DIR}/models/IAM-line_al_${strategy}_seed${SEED}_${TAIL}"
    local train_log="${SCRIPT_DIR}/logs/train_IAM-line_al_${strategy}_seed${SEED}_${TAIL}.log"

    if [ -d "${output_dir}/iter_${AL_ITERATIONS}_model" ]; then
        echo "${strategy} already complete at ${output_dir}. Skipping."
        return 0
    fi
    rm -rf "${output_dir}"

    echo "=== IAM AL baseline: ${strategy} on GPU ${GPU_ID} ==="
    "${PYTHON_PATH}" "${SCRIPT_DIR}/train_al_baselines.py" \
        --model_id "${MODEL_ID}" \
        --input_dir "${INPUT_DIR}" \
        --unlabeled_input_dir "${UNLABELED_DIR}" \
        --output_dir "${output_dir}" \
        --prompt_path "${PROMPT_PATH}" \
        --aug_test_dir "${TEST_DIR}" \
        --al_strategy "${strategy}" \
        --al_iterations "${AL_ITERATIONS}" \
        --samples_per_iter "${SAMPLES_PER_ITER}" \
        --al_eval_subset "${SUBSET}" \
        --tuning_mode lora \
        --target_modules ${LORA_TARGETS} \
        --diversity_embedding_type vision_encoder \
        --lr "${LR}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --epochs "${AL_EPOCHS}" \
        --freeze_vision_encoder \
        --seed "${SEED}" \
        > "${train_log}" 2>&1 || { echo "ERROR: ${strategy} failed! Check: ${train_log}"; return 1; }
    rm -rf "${output_dir}/trainer_tmp" 2>/dev/null
    echo "${strategy} complete."
}

run_diva() {
    local embed_type="$1"       # "decoder" | "vision_encoder"
    local variant_label="$2"    # "decoder" | "visenc" -- for the output dir name
    local output_dir="${SCRIPT_DIR}/models/IAM-line_al_diva_${variant_label}_alpha${ALPHA}_seed${SEED}_${TAIL}"
    local train_log="${SCRIPT_DIR}/logs/train_IAM-line_al_diva_${variant_label}_alpha${ALPHA}_seed${SEED}_${TAIL}.log"

    if [ -d "${output_dir}/iter_${AL_ITERATIONS}_model" ]; then
        echo "DIVA (${embed_type}) already complete at ${output_dir}. Skipping."
        return 0
    fi
    rm -rf "${output_dir}"

    echo "=== IAM DIVA (${embed_type}, alpha=${ALPHA}) on GPU ${GPU_ID} ==="
    "${PYTHON_PATH}" "${SCRIPT_DIR}/train_active_learning_extended.py" \
        --model_id "${MODEL_ID}" \
        --input_dir "${INPUT_DIR}" \
        --unlabeled_input_dir "${UNLABELED_DIR}" \
        --output_dir "${output_dir}" \
        --prompt_path "${PROMPT_PATH}" \
        --aug_test_dir "${TEST_DIR}" \
        --al_strategy vis_div \
        --alpha "${ALPHA}" \
        --beta "${BETA}" \
        --al_eval_subset "${SUBSET}" \
        --dynamic_quota \
        --al_iterations "${AL_ITERATIONS}" \
        --samples_per_iter "${SAMPLES_PER_ITER}" \
        --tuning_mode lora \
        --target_modules ${LORA_TARGETS} \
        --diversity_embedding_type "${embed_type}" \
        --lr "${LR}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --epochs "${AL_EPOCHS}" \
        --freeze_vision_encoder \
        --seed "${SEED}" \
        > "${train_log}" 2>&1 || { echo "ERROR: DIVA (${embed_type}) failed! Check: ${train_log}"; return 1; }
    rm -rf "${output_dir}/trainer_tmp" 2>/dev/null
    echo "DIVA (${embed_type}) complete."
}

case "${TASK}" in
    full) run_full ;;
    random) run_baseline "random" "random_results" ;;
    entropy) run_baseline "entropy" "entropy_results" ;;
    kmeans) run_baseline "kmeans_center" "kmeans_center_results" ;;
    diva_decoder) run_diva "decoder" "decoder" ;;
    diva_visenc) run_diva "vision_encoder" "visenc" ;;
    all)
        run_full
        run_baseline "random" "random_results"
        run_baseline "entropy" "entropy_results"
        run_baseline "kmeans_center" "kmeans_center_results"
        run_diva "decoder" "decoder"
        run_diva "vision_encoder" "visenc"
        ;;
    *)
        echo "Usage: $0 <full|random|entropy|kmeans|diva_decoder|diva_visenc|all> [gpu_id]" >&2
        exit 1
        ;;
esac

echo "Done. Metrics in ${SCRIPT_DIR}/results/{full,random,entropy,kmeans_center,vis_div}_results/IAM-line_*_${TAIL}*"
