#!/bin/bash
# run_diva_selfdistill.sh - standalone launcher for train_diva_selfdistill.py.
#
# Separate from run_al_and_full_lorafixed_fixed_run2_run3.sh (per request) -- doesn't
# touch or depend on that script. Dataset paths/prompts/alpha/beta are duplicated from
# its resolve_dataset_config() (Belfort/Himanis/Esposalles/IAM only -- the datasets
# this experiment applies to), trimmed to just what this script needs. If those paths
# ever change in the main script, update them here too; nothing shares that config.
#
# NOT independently verified on a GPU during development (no local GPU). Run
# smoke_self_distillation.py first and confirm it reports PASS before trusting a full
# launch here.
#
# Usage:
#   ./run_diva_selfdistill.sh <dataset> [gpu_id]
#   ./run_diva_selfdistill.sh all [gpu_id]       # all 4 datasets, sequential, cheapest first
#
#   dataset: belfort | himanis | esposalles | iam | all
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
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAMPLES_PER_ITER="${SAMPLES_PER_ITER:-200}"
AL_ITERATIONS="${AL_ITERATIONS:-5}"
SELF_DISTILL_RATIO="${SELF_DISTILL_RATIO:-1.0}"
TAIL="${TAIL:-selfdistill_standalone}"
LORA_TARGETS="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

DATASET="${1:?Usage: $0 <belfort|himanis|esposalles|iam|all> [gpu_id]}"
GPU_ID="${2:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/models" "${SCRIPT_DIR}/results/vis_div_results"

run_one() {
    local ds_name="$1"
    local input_dir="$2"
    local unlabeled_dir="$3"
    local test_dir="$4"
    local prompt_path="$5"
    local alpha="$6"
    local beta="$7"
    local subset="$8"

    local output_dir="${SCRIPT_DIR}/models/${ds_name}_al_diva_selfdistill_alpha${alpha}_seed${SEED}_${TAIL}"
    local train_log="${SCRIPT_DIR}/logs/train_${ds_name}_al_diva_selfdistill_alpha${alpha}_seed${SEED}_${TAIL}.log"

    if [ -d "${output_dir}/iter_${AL_ITERATIONS}_model" ]; then
        echo "Already complete at ${output_dir}. Skipping (delete it to force a rerun)."
        return 0
    fi
    if [ -d "${output_dir}" ]; then
        echo "Cleaning incomplete previous run at ${output_dir}..."
        rm -rf "${output_dir}"
    fi

    local extra_args=()
    if [ -n "${unlabeled_dir}" ] && [ -d "${unlabeled_dir}" ]; then
        extra_args+=(--unlabeled_input_dir "${unlabeled_dir}")
    else
        extra_args+=(--initial_pool_size 10)
    fi

    echo "=== ${ds_name}: DIVA + self-distillation (alpha=${alpha}, beta=${beta}, ratio=${SELF_DISTILL_RATIO}) on GPU ${GPU_ID} ==="
    "${PYTHON_PATH}" "${SCRIPT_DIR}/train_diva_selfdistill.py" \
        --model_id "${MODEL_ID}" \
        --input_dir "${input_dir}" \
        "${extra_args[@]}" \
        --output_dir "${output_dir}" \
        --prompt_path "${prompt_path}" \
        --aug_test_dir "${test_dir}" \
        --alpha "${alpha}" \
        --beta "${beta}" \
        --al_eval_subset "${subset}" \
        --dynamic_quota \
        --al_iterations "${AL_ITERATIONS}" \
        --samples_per_iter "${SAMPLES_PER_ITER}" \
        --tuning_mode lora \
        --target_modules ${LORA_TARGETS} \
        --diversity_embedding_type vision_encoder \
        --self_distill_ratio "${SELF_DISTILL_RATIO}" \
        --lr "${LR}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --epochs "${AL_EPOCHS}" \
        --freeze_vision_encoder \
        --seed "${SEED}" \
        > "${train_log}" 2>&1 || { echo "ERROR: ${ds_name} failed! Check: ${train_log}"; return 1; }
    rm -rf "${output_dir}/trainer_tmp" 2>/dev/null
    echo "${ds_name} complete."
}

# dataset_name input_dir unlabeled_dir test_dir prompt_path alpha beta subset
run_belfort() {
    run_one "Teklia_Belfort-line" \
        "/dest/thura/data/Teklia_Belfort-line_labeled_10" \
        "/dest/thura/data/Teklia_Belfort-line_unlabeled_90" \
        "/dest/thura/data/Teklia_Belfort-line" \
        "${SCRIPT_DIR}/../eval/prompt_Teklia_Belfort-line.txt" \
        20 3 3000
}
run_himanis() {
    run_one "Teklia_Himanis-line" \
        "/dest/thura/data/Teklia_Himanis-line_labeled_10" \
        "/dest/thura/data/Teklia_Himanis-line_unlabeled_90" \
        "/dest/thura/data/Teklia_Himanis-line" \
        "${SCRIPT_DIR}/../eval/prompt_Teklia_Himanis-line.txt" \
        20 2 3000
}
run_esposalles() {
    run_one "Teklia_Esposalles-line" \
        "/dest/thura/data/Teklia_Esposalles-line_labeled_10" \
        "/dest/thura/data/Teklia_Esposalles-line_unlabeled_90" \
        "/dest/thura/data/Teklia_Esposalles-line" \
        "${SCRIPT_DIR}/../eval/prompt_Teklia_Esposalles-line.txt" \
        20 2 2000
}
run_iam() {
    run_one "IAM-line" \
        "/dest/thura/data/IAM-line_labeled_10" \
        "/dest/thura/data/IAM-line_unlabeled_90" \
        "/dest/thura/data/IAM-line" \
        "${SCRIPT_DIR}/../eval/prompt_IAM-line.txt" \
        20 2 3000
}

case "${DATASET}" in
    belfort) run_belfort ;;
    himanis) run_himanis ;;
    esposalles) run_esposalles ;;
    iam) run_iam ;;
    all)
        # Cheapest first, same ordering convention as diva_visenc/diva_selfdistill in
        # the main pipeline script, so you get the fastest end-to-end signal first.
        run_esposalles
        run_himanis
        run_belfort
        run_iam
        ;;
    *)
        echo "Usage: $0 <belfort|himanis|esposalles|iam|all> [gpu_id]" >&2
        exit 1
        ;;
esac

echo "Done. Metrics in ${SCRIPT_DIR}/results/vis_div_results/*_al_diva_selfdistill_alpha*_seed${SEED}_${TAIL}_metrics.csv"
