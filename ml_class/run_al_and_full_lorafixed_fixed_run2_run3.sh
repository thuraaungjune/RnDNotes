#!/bin/bash

# =========================================================================
# Multi-Dataset Active Learning & LoRA Pipeline (Run 2 - aug_experiments_fixed)
# 
# Target Datasets (rerun with Fix 03 follow-up pre-split concatenation fix):
#   1. Teklia_Himanis-line     (himanis)
#   2. Teklia_Belfort-line     (belfort)
#   3. Teklia_Esposalles-line  (esposalles)
#
# Fixes Verified:
#   - Pre-split concatenation: full_train_dataset combines labeled + unlabeled
#     pools so acquired lines always map to valid row indices (no frozen training!).
#   - Finding 01: Unified CER evaluation (Raw CER, Clean CER, per-line & corpus).
#   - Finding 02: Matched baseline execution for all datasets.
#   - Finding 03: Original-line budget accounting (unique original lines).
#   - Finding 04: Float32 casting on pooled embeddings & strict error checks.
#   - Finding 07: Uncertainty pre-filtering correctly dimensioned.
#   - Finding 08: Exact EOS masking in sequence uncertainty calculation.
#   - Finding 09: Attention-mask weighted feature pooling (no padding bias).
#   - Finding 10: Multi-seed support ($SEED environment variable, default: 42).
#   - Finding 13: Acquired sample IDs manifest persisted per iteration to JSON.
#   - Finding 15: Prompt tokens masked (-100) during loss calculation.
#   - DIVA cluster de-fragmentation: default alpha bumped 1-2 -> 20 for all datasets.
#     At alpha=1-2, num_clusters (samples_per_iter/alpha) vastly outnumbered the
#     candidate pool (~2-6 candidates/cluster), so KMeans found no real structure
#     (silhouette_score ~0-0.03 across every DIVA run2 iteration). alpha=20 was the
#     best-performing value in the existing retrain_diva_alphas.sh sweep data
#     (silhouette climbs monotonically with alpha on every dataset tested).
#
# Output Tail / Tag:
#   Default: "vis_embed" (directories, logs, results suffixed with _vis_embed)
# =========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:/dest/thura/code/FYP_jonpwk:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Handle Ctrl+C cleanly
trap "echo -e '\n[INT] Script interrupted by user. Exiting...'; exit 1" INT TERM

# Run Tail Identifier (default: vis_embed)
TAIL=${TAIL:-"vis_embed"}
DIVERSITY_EMBED=${DIVERSITY_EMBED:-"vision_encoder"}

# Command Line Arguments:
#   $1: Execution mode [parallel | parallel_datasets | himanis | belfort | esposalles | all | clean] (default: "parallel")
#   $2: GPU IDs (default: "0,1,2,3" for parallel, "0" for single-gpu)
#   $3: Dataset name override / Alpha override
MODE=${1:-"parallel"}
if [[ "${MODE}" =~ ^parallel ]]; then
    GPU_IDS=${2:-"0,1,2,3"}
else
    GPU_IDS=${2:-"0"}
fi
DS_OVERRIDE=${3:-""}

if [[ ! "${MODE}" =~ ^parallel ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

# Environment & Model
PYTHON_PATH="/dest/thura/conda_envs/jawi_ocr_eval/bin/python"
MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"

# Training & Active Learning Hyperparameters
SEED=${SEED:-42}
TUNING_MODE="lora"
LR=2e-4
AL_EPOCHS=8
FULL_EPOCHS=3
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
SAMPLES_PER_ITER=${SAMPLES_PER_ITER:-200}
AL_ITERATIONS=${AL_ITERATIONS:-5}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
FORCE=${FORCE:-0}
SKIP_FULL=${SKIP_FULL:-0}
PRUNE_INTERMEDIATE=${PRUNE_INTERMEDIATE:-0}

# LoRA Target Modules: Fixed Embeddings (vision encoder frozen)
LORA_TARGETS="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
LORA_TARGETS_COMMA="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Directory Structure
mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/models"
mkdir -p "${SCRIPT_DIR}/results/full_results"
mkdir -p "${SCRIPT_DIR}/results/random_results"
mkdir -p "${SCRIPT_DIR}/results/entropy_results"
mkdir -p "${SCRIPT_DIR}/results/kmeans_center_results"
mkdir -p "${SCRIPT_DIR}/results/vis_div_results"
mkdir -p "${SCRIPT_DIR}/eval_results"

# Space check helper
check_disk_space() {
    local avail_kb
    avail_kb=$(df -k /dest | awk 'NR==2 {print $4}')
    local avail_gb=$((avail_kb / 1024 / 1024))
    echo "[Disk Check] Available space on /dest: ${avail_gb} GB"
    if [ "${avail_gb}" -lt 15 ]; then
        echo "WARNING: Low disk space (< 15 GB free)! Consider removing older models."
    fi
}

# Cleanup previous models helper
cleanup_previous_models() {
    echo "========================================================================="
    echo "CLEANUP: Removing previous Teklia models to reclaim disk space..."
    echo "========================================================================="
    local count=0
    for dir in "${SCRIPT_DIR}/models"/Teklia_*; do
        if [ -d "${dir}" ] && [[ ! "${dir}" =~ _${TAIL}$ ]]; then
            echo "Removing: ${dir}"
            rm -rf "${dir}"
            count=$((count + 1))
        fi
    done
    echo "Cleanup complete. Removed ${count} previous model directorie(s)."
    check_disk_space
}

# Helper: Resolve dataset paths and hyperparameters
resolve_dataset_config() {
    local target_ds="$1"
    case "${target_ds}" in
        "Teklia_Himanis-line"|"himanis")
            CUR_DS="Teklia_Himanis-line"
            INPUT_DIR="/dest/thura/data/Teklia_Himanis-line_labeled_10"
            UNLABELED_DIR="/dest/thura/data/Teklia_Himanis-line_unlabeled_90"
            TEST_DIR="/dest/thura/data/Teklia_Himanis-line"
            PROMPT_PATH="${SCRIPT_DIR}/../eval/prompt_Teklia_Himanis-line.txt"
            PROMPT_TEXT="Transcribe the Latin script in this French text image into French text"
            # Alpha=20 (was 1): alpha=1 gives num_clusters = samples_per_iter/alpha = 200
            # clusters from a ~400-candidate pool (~2 candidates/cluster) -- KMeans can't
            # find real structure at that ratio. Confirmed via retrain_diva_alphas.sh's
            # alpha sweep: silhouette_score climbs monotonically with alpha across every
            # dataset (Himanis peaks ~0.071 at alpha=20 vs ~0.002-0.03 at alpha=1).
            DEFAULT_ALPHA=20
            DEFAULT_BETA=2
            DEFAULT_SUBSET=3000
            ;;
        "Teklia_Belfort-line"|"belfort")
            CUR_DS="Teklia_Belfort-line"
            INPUT_DIR="/dest/thura/data/Teklia_Belfort-line_labeled_10"
            UNLABELED_DIR="/dest/thura/data/Teklia_Belfort-line_unlabeled_90"
            TEST_DIR="/dest/thura/data/Teklia_Belfort-line"
            PROMPT_PATH="${SCRIPT_DIR}/../eval/prompt_Teklia_Belfort-line.txt"
            PROMPT_TEXT="Transcribe the Latin script in this French text image into French text"
            # Alpha=20 (was 2): same cluster-fragmentation fix as Himanis above --
            # silhouette_score at alpha=20 peaks ~0.072 vs ~0.025 at alpha=2.
            DEFAULT_ALPHA=20
            DEFAULT_BETA=3
            DEFAULT_SUBSET=3000
            ;;
        "Teklia_Esposalles-line"|"esposalles")
            CUR_DS="Teklia_Esposalles-line"
            INPUT_DIR="/dest/thura/data/Teklia_Esposalles-line_labeled_10"
            UNLABELED_DIR="/dest/thura/data/Teklia_Esposalles-line_unlabeled_90"
            TEST_DIR="/dest/thura/data/Teklia_Esposalles-line"
            PROMPT_PATH="${SCRIPT_DIR}/../eval/prompt_Teklia_Esposalles-line.txt"
            PROMPT_TEXT="Transcribe the Latin script in this Spanish text image into Spanish text"
            # Alpha=20 (was 2): same cluster-fragmentation fix -- silhouette_score at
            # alpha=20 peaks ~0.054 vs ~0.025 at alpha=2.
            DEFAULT_ALPHA=20
            DEFAULT_BETA=2
            DEFAULT_SUBSET=2000
            ;;
        "Jawi-OCR-data-v4-augmented"|"jawi")
            CUR_DS="Jawi-OCR-data-v4-augmented"
            INPUT_DIR="/dest/thura/data/Jawi-OCR-data-v4-augmented"
            UNLABELED_DIR=""
            TEST_DIR="/dest/thura/data/Jawi-OCR-data-v4-augmented"
            PROMPT_PATH="${SCRIPT_DIR}/../eval/prompt_Jawi-OCR-data-v4.txt"
            PROMPT_TEXT="Transcribe the Jawi script in this Malay text image into Malay text"
            # Alpha=20 (was 1): same cluster-fragmentation fix -- confirmed via the
            # retrain_diva_alphas.sh alpha sweep on this dataset too (silhouette
            # improves as alpha increases, same monotonic pattern as the other 3).
            DEFAULT_ALPHA=20
            DEFAULT_BETA=2
            DEFAULT_SUBSET=3000
            ;;
        *)
            CUR_DS="${target_ds}"
            INPUT_DIR="/dest/thura/data/${target_ds}"
            UNLABELED_DIR=""
            TEST_DIR="/dest/thura/data/${target_ds}"
            PROMPT_PATH="${SCRIPT_DIR}/../eval/prompt_${target_ds}.txt"
            PROMPT_TEXT="Transcribe the text in this image"
            # Alpha=20: no dedicated sweep evidence for arbitrary/unlisted datasets,
            # but defaulting to the same de-fragmented value is safer than alpha=2
            # (which we now know fragments badly) until this dataset gets its own sweep.
            DEFAULT_ALPHA=20
            DEFAULT_BETA=2
            DEFAULT_SUBSET=2000
            ;;
    esac

    if [ ! -f "${PROMPT_PATH}" ]; then
        echo "Creating prompt file at ${PROMPT_PATH}..."
        echo "${PROMPT_TEXT}" > "${PROMPT_PATH}"
    fi
}

echo "========================================================================="
echo "AUG_EXPERIMENTS_FIXED PIPELINE (RERUN - TAIL: ${TAIL})"
echo "Mode           : ${MODE}"
echo "GPUs           : ${GPU_IDS}"
echo "Seed           : ${SEED}"
echo "Tail Suffix    : _${TAIL}"
echo "LR / AL Epochs : ${LR} | ${AL_EPOCHS} epochs | Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} (Eff: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Budget         : ${SAMPLES_PER_ITER} lines/iter | Iterations: ${AL_ITERATIONS}"
check_disk_space
echo "========================================================================="

# -------------------------------------------------------------------------
# Phase 1: Full Fine-Tuning (100% Data)
# -------------------------------------------------------------------------
run_full_finetuning() {
    local target_ds="$1"
    resolve_dataset_config "${target_ds}"

    local OUTPUT_DIR="${SCRIPT_DIR}/models/${CUR_DS}_full_seed${SEED}_${TAIL}"
    local TRAIN_LOG="${SCRIPT_DIR}/logs/train_${CUR_DS}_full_seed${SEED}_${TAIL}.log"
    local EVAL_OUT="${SCRIPT_DIR}/eval_results/${CUR_DS}_full_seed${SEED}_${TAIL}"

    echo ""
    echo "#########################################################################"
    echo "Full Dataset Fine-Tuning (100% Data): ${CUR_DS} [Seed ${SEED}, Tail ${TAIL}]"
    echo "#########################################################################"

    if [ "${SKIP_FULL}" -eq 1 ]; then
        echo "SKIP_FULL=1: Skipping full fine-tuning for ${CUR_DS}."
        return 0
    fi

    if [ "${FORCE}" -ne 1 ] && [ -d "${OUTPUT_DIR}/final" ] && [ -f "${OUTPUT_DIR}/final/adapter_config.json" ]; then
        echo "Full model already exists at ${OUTPUT_DIR}/final. Skipping training."
    else
        # Remove partial output dir if restarting to save space
        if [ -d "${OUTPUT_DIR}" ]; then
            echo "Removing existing/partial output directory: ${OUTPUT_DIR}"
            rm -rf "${OUTPUT_DIR}"
        fi

        echo "Launching Full LoRA Training for ${CUR_DS}..."
        "$PYTHON_PATH" "${SCRIPT_DIR}/train_ocr.py" \
            --model_id "${MODEL_ID}" \
            --input_dir "${TEST_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --prompt_path "${PROMPT_PATH}" \
            --tuning_mode "${TUNING_MODE}" \
            --lora_target_modules "${LORA_TARGETS_COMMA}" \
            --lr ${LR} \
            --batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRAD_ACCUM} \
            --epochs ${FULL_EPOCHS} \
            --freeze_vision_encoder \
            --seed ${SEED} \
            > "${TRAIN_LOG}" 2>&1

        if [ $? -ne 0 ]; then
            echo "ERROR: Full training failed for ${CUR_DS}! Check: ${TRAIN_LOG}"
            return 1
        fi
    fi

    # Clean trainer temporary checkpoints to save space
    rm -rf "${OUTPUT_DIR}/trainer_tmp" 2>/dev/null

    # Standalone comprehensive evaluation
    echo "Evaluating Full Model on Test Set..."
    "$PYTHON_PATH" "${SCRIPT_DIR}/test_cer.py" \
        --base_model_id "${MODEL_ID}" \
        --lora_model_dir "${OUTPUT_DIR}/final" \
        --input_dir "${TEST_DIR}" \
        --prompt_path "${PROMPT_PATH}" \
        --batch_size ${EVAL_BATCH_SIZE} \
        --output_dir "${EVAL_OUT}" \
        >> "${TRAIN_LOG}" 2>&1
    echo "Full model training & evaluation complete for ${CUR_DS}."
}

# -------------------------------------------------------------------------
# Phase 2: Active Learning Baselines (Random, Entropy, KMeans Center)
# -------------------------------------------------------------------------
run_al_baseline() {
    local target_ds="$1"
    local STRATEGY="$2"
    resolve_dataset_config "${target_ds}"

    local OUTPUT_DIR="${SCRIPT_DIR}/models/${CUR_DS}_al_${STRATEGY}_seed${SEED}_${TAIL}"
    local TRAIN_LOG="${SCRIPT_DIR}/logs/train_${CUR_DS}_al_${STRATEGY}_seed${SEED}_${TAIL}.log"

    echo ""
    echo "#########################################################################"
    echo "Active Learning Baseline: ${STRATEGY} on ${CUR_DS} [Seed ${SEED}, Tail ${TAIL}]"
    echo "#########################################################################"

    if [ "${FORCE}" -ne 1 ] && [ -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ]; then
        echo "Active Learning loop for ${STRATEGY} already complete at ${OUTPUT_DIR}. Skipping."
    else
        # Remove partial run if incomplete
        if [ -d "${OUTPUT_DIR}" ] && [ ! -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ]; then
            echo "Cleaning incomplete previous run at ${OUTPUT_DIR} for space..."
            rm -rf "${OUTPUT_DIR}"
        fi

        local EXTRA_ARGS=()
        if [ -n "${UNLABELED_DIR}" ] && [ -d "${UNLABELED_DIR}" ]; then
            EXTRA_ARGS+=(--unlabeled_input_dir "${UNLABELED_DIR}")
        else
            EXTRA_ARGS+=(--initial_pool_size 10)
        fi

        echo "Launching AL Baseline (${STRATEGY})..."
        "$PYTHON_PATH" "${SCRIPT_DIR}/train_al_baselines.py" \
            --model_id "${MODEL_ID}" \
            --input_dir "${INPUT_DIR}" \
            "${EXTRA_ARGS[@]}" \
            --output_dir "${OUTPUT_DIR}" \
            --prompt_path "${PROMPT_PATH}" \
            --aug_test_dir "${TEST_DIR}" \
            --al_strategy "${STRATEGY}" \
            --al_iterations ${AL_ITERATIONS} \
            --samples_per_iter ${SAMPLES_PER_ITER} \
            --al_eval_subset ${DEFAULT_SUBSET} \
            --tuning_mode "${TUNING_MODE}" \
            --target_modules ${LORA_TARGETS} \
            --diversity_embedding_type "${DIVERSITY_EMBED}" \
            --lr ${LR} \
            --batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRAD_ACCUM} \
            --epochs ${AL_EPOCHS} \
            --freeze_vision_encoder \
            --seed ${SEED} \
            > "${TRAIN_LOG}" 2>&1

        local exit_code=$?
        if [ ${exit_code} -ne 0 ]; then
            echo "ERROR: Training failed for baseline ${STRATEGY}! Check: ${TRAIN_LOG}"
            return 1
        fi

        # Clean trainer_tmp to save space
        rm -rf "${OUTPUT_DIR}/trainer_tmp" 2>/dev/null

        # Optional pruning of intermediate iterations if disk space is critical
        if [ "${PRUNE_INTERMEDIATE}" -eq 1 ]; then
            echo "Pruning intermediate checkpoints (iter 0..$((AL_ITERATIONS - 1))) to save space..."
            for ((it=0; it<AL_ITERATIONS; it++)); do
                rm -rf "${OUTPUT_DIR}/iter_${it}_model" 2>/dev/null
            done
        fi

        echo "AL baseline ${STRATEGY} completed successfully on ${CUR_DS}."
    fi
}

# -------------------------------------------------------------------------
# Phase 3: Active Learning DIVA (Visual Diversity + Uncertainty)
# -------------------------------------------------------------------------
run_al_diva() {
    local target_ds="$1"
    local opt_alpha="$2"
    local opt_beta="$3"
    local opt_subset="$4"
    resolve_dataset_config "${target_ds}"

    local alpha_val=${opt_alpha:-${ALPHA:-${DEFAULT_ALPHA}}}
    local beta_val=${opt_beta:-${BETA:-${DEFAULT_BETA}}}
    local subset_val=${opt_subset:-${DEFAULT_SUBSET}}

    local OUTPUT_DIR="${SCRIPT_DIR}/models/${CUR_DS}_al_diva_alpha${alpha_val}_seed${SEED}_${TAIL}"
    local TRAIN_LOG="${SCRIPT_DIR}/logs/train_${CUR_DS}_al_diva_alpha${alpha_val}_seed${SEED}_${TAIL}.log"

    echo ""
    echo "#########################################################################"
    echo "DIVA Active Learning: ${CUR_DS} (Alpha=${alpha_val}, Beta=${beta_val}) [Seed ${SEED}, Tail ${TAIL}]"
    echo "#########################################################################"

    if [ "${FORCE}" -ne 1 ] && [ -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ]; then
        echo "DIVA loop for ${CUR_DS} already complete at ${OUTPUT_DIR}. Skipping."
    else
        # Remove partial run if incomplete
        if [ -d "${OUTPUT_DIR}" ] && [ ! -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ]; then
            echo "Cleaning incomplete previous run at ${OUTPUT_DIR} for space..."
            rm -rf "${OUTPUT_DIR}"
        fi

        local EXTRA_ARGS=()
        if [ -n "${UNLABELED_DIR}" ] && [ -d "${UNLABELED_DIR}" ]; then
            EXTRA_ARGS+=(--unlabeled_input_dir "${UNLABELED_DIR}")
        else
            EXTRA_ARGS+=(--initial_pool_size 10)
        fi

        echo "Launching DIVA Training on ${CUR_DS}..."
        "$PYTHON_PATH" "${SCRIPT_DIR}/train_active_learning_extended.py" \
            --model_id "${MODEL_ID}" \
            --input_dir "${INPUT_DIR}" \
            "${EXTRA_ARGS[@]}" \
            --output_dir "${OUTPUT_DIR}" \
            --prompt_path "${PROMPT_PATH}" \
            --aug_test_dir "${TEST_DIR}" \
            --al_strategy "vis_div" \
            --alpha ${alpha_val} \
            --beta ${beta_val} \
            --al_eval_subset ${subset_val} \
            --dynamic_quota \
            --al_iterations ${AL_ITERATIONS} \
            --samples_per_iter ${SAMPLES_PER_ITER} \
            --tuning_mode "${TUNING_MODE}" \
            --target_modules ${LORA_TARGETS} \
            --diversity_embedding_type "${DIVERSITY_EMBED}" \
            --lr ${LR} \
            --batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRAD_ACCUM} \
            --epochs ${AL_EPOCHS} \
            --freeze_vision_encoder \
            --seed ${SEED} \
            > "${TRAIN_LOG}" 2>&1

        local exit_code=$?
        if [ ${exit_code} -ne 0 ]; then
            echo "ERROR: DIVA training failed for ${CUR_DS}! Check: ${TRAIN_LOG}"
            return 1
        fi

        # Clean trainer_tmp to save space
        rm -rf "${OUTPUT_DIR}/trainer_tmp" 2>/dev/null

        # Optional pruning of intermediate iterations if disk space is critical
        if [ "${PRUNE_INTERMEDIATE}" -eq 1 ]; then
            echo "Pruning intermediate checkpoints (iter 0..$((AL_ITERATIONS - 1))) to save space..."
            for ((it=0; it<AL_ITERATIONS; it++)); do
                rm -rf "${OUTPUT_DIR}/iter_${it}_model" 2>/dev/null
            done
        fi

        echo "DIVA training on ${CUR_DS} completed successfully."
    fi
}

# -------------------------------------------------------------------------
# Phase 3b: DIVA Variant Sweep -- isolates each design lever (embedding space,
# cluster granularity, budget-allocation rule) as its own tagged run, so they
# can be compared against each other and against random/entropy/kmeans_center
# on equal footing (same AL_ITERATIONS/SAMPLES_PER_ITER/epochs/seed).
#
# variant_label   what it changes vs. the dataset's default DIVA config
#   visenc          diversity_embedding_type: vision_encoder (default DIVA uses
#                   whatever $DIVERSITY_EMBED resolves to, usually "decoder")
#   fixedquota      dynamic_quota OFF -- round-robin `alpha` picks per cluster,
#                   instead of budget proportional to each cluster's mean uncertainty
#   alpha40         alpha=40 -- even coarser clustering than the new default (20), to
#                   check whether fewer/bigger clusters keep helping or start hurting
#   widebeta        beta doubled -- larger uncertain candidate pool feeds the
#                   diversity clustering, instead of a tightly uncertainty-filtered one
# -------------------------------------------------------------------------
run_al_diva_variant() {
    local target_ds="$1"
    local variant_label="$2"
    local embed_override="$3"     # "" = use $DIVERSITY_EMBED
    local alpha_override="$4"     # "" = use dataset default
    local beta_override="$5"      # "" = use dataset default
    local use_dynamic_quota="$6"  # "1" or "0"
    resolve_dataset_config "${target_ds}"

    local embed_val=${embed_override:-${DIVERSITY_EMBED}}
    local alpha_val=${alpha_override:-${DEFAULT_ALPHA}}
    local beta_val=${beta_override:-${DEFAULT_BETA}}
    local subset_val=${DEFAULT_SUBSET}

    local OUTPUT_DIR="${SCRIPT_DIR}/models/${CUR_DS}_al_diva_${variant_label}_alpha${alpha_val}_seed${SEED}_${TAIL}"
    local TRAIN_LOG="${SCRIPT_DIR}/logs/train_${CUR_DS}_al_diva_${variant_label}_alpha${alpha_val}_seed${SEED}_${TAIL}.log"

    echo ""
    echo "#########################################################################"
    echo "DIVA Variant [${variant_label}]: ${CUR_DS} (Alpha=${alpha_val}, Beta=${beta_val}, Embed=${embed_val}, DynQuota=${use_dynamic_quota}) [Seed ${SEED}, Tail ${TAIL}]"
    echo "#########################################################################"

    if [ "${FORCE}" -ne 1 ] && [ -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ]; then
        echo "DIVA variant [${variant_label}] for ${CUR_DS} already complete at ${OUTPUT_DIR}. Skipping."
    else
        if [ -d "${OUTPUT_DIR}" ] && [ ! -d "${OUTPUT_DIR}/iter_${AL_ITERATIONS}_model" ]; then
            echo "Cleaning incomplete previous run at ${OUTPUT_DIR} for space..."
            rm -rf "${OUTPUT_DIR}"
        fi

        local EXTRA_ARGS=()
        if [ -n "${UNLABELED_DIR}" ] && [ -d "${UNLABELED_DIR}" ]; then
            EXTRA_ARGS+=(--unlabeled_input_dir "${UNLABELED_DIR}")
        else
            EXTRA_ARGS+=(--initial_pool_size 10)
        fi
        if [ "${use_dynamic_quota}" -eq 1 ]; then
            EXTRA_ARGS+=(--dynamic_quota)
        fi

        echo "Launching DIVA Variant [${variant_label}] on ${CUR_DS}..."
        "$PYTHON_PATH" "${SCRIPT_DIR}/train_active_learning_extended.py" \
            --model_id "${MODEL_ID}" \
            --input_dir "${INPUT_DIR}" \
            "${EXTRA_ARGS[@]}" \
            --output_dir "${OUTPUT_DIR}" \
            --prompt_path "${PROMPT_PATH}" \
            --aug_test_dir "${TEST_DIR}" \
            --al_strategy "vis_div" \
            --alpha ${alpha_val} \
            --beta ${beta_val} \
            --al_eval_subset ${subset_val} \
            --al_iterations ${AL_ITERATIONS} \
            --samples_per_iter ${SAMPLES_PER_ITER} \
            --tuning_mode "${TUNING_MODE}" \
            --target_modules ${LORA_TARGETS} \
            --diversity_embedding_type "${embed_val}" \
            --lr ${LR} \
            --batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRAD_ACCUM} \
            --epochs ${AL_EPOCHS} \
            --freeze_vision_encoder \
            --seed ${SEED} \
            > "${TRAIN_LOG}" 2>&1

        local exit_code=$?
        if [ ${exit_code} -ne 0 ]; then
            echo "ERROR: DIVA variant [${variant_label}] failed for ${CUR_DS}! Check: ${TRAIN_LOG}"
            return 1
        fi

        rm -rf "${OUTPUT_DIR}/trainer_tmp" 2>/dev/null

        if [ "${PRUNE_INTERMEDIATE}" -eq 1 ]; then
            for ((it=0; it<AL_ITERATIONS; it++)); do
                rm -rf "${OUTPUT_DIR}/iter_${it}_model" 2>/dev/null
            done
        fi

        echo "DIVA variant [${variant_label}] on ${CUR_DS} completed successfully."
    fi
}

# -------------------------------------------------------------------------
# Multi-GPU Parallel Task Queue Runner
# -------------------------------------------------------------------------
run_parallel_task_queue() {
    local task_list=("$@")
    IFS=',' read -r -a GPUS <<< "${GPU_IDS}"

    # OOM resilience knobs (env-overridable):
    #   MAX_TASK_RETRIES   how many times a failed task gets requeued before being
    #                      given up on for good (default 2 -> 3 attempts total)
    #   RETRY_DELAY_SECONDS   pause before a requeued task's next attempt, to give a
    #                      contending external process (on a shared/multi-tenant GPU)
    #                      a chance to finish and free memory
    #   MIN_FREE_GB        a worker won't start a new task on its GPU until at least
    #                      this much VRAM is free; 0 disables the check
    MAX_TASK_RETRIES=${MAX_TASK_RETRIES:-2}
    RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-60}
    MIN_FREE_GB=${MIN_FREE_GB:-15}

    echo "========================================================================="
    echo "LAUNCHING DYNAMIC PARALLEL QUEUE"
    echo "GPUs Assigned : ${GPUS[*]}"
    echo "Tasks (${#task_list[@]})     : ${task_list[*]}"
    echo "OOM Handling  : retries=${MAX_TASK_RETRIES} delay=${RETRY_DELAY_SECONDS}s min_free=${MIN_FREE_GB}GB"
    echo "========================================================================="

    QUEUE_FILE=$(mktemp /tmp/aug_fixed_queue_${TAIL}.XXXXXX)
    LOCK_DIR="/tmp/aug_fixed_queue_${TAIL}.lock"
    FAILURES_FILE=$(mktemp /tmp/aug_fixed_failures_${TAIL}.XXXXXX)
    : > "${FAILURES_FILE}"
    rm -rf "${LOCK_DIR}"

    # Each queue line is "task_name::retry_count" -- retry_count starts at 0.
    for t in "${task_list[@]}"; do
        echo "${t}::0" >> "${QUEUE_FILE}"
    done

    WORKER_PIDS=()
    cleanup_parallel() {
        echo -e "\n[INT] Terminating all workers..."
        for pid in "${WORKER_PIDS[@]}"; do
            kill -TERM "${pid}" 2>/dev/null
        done
        rm -f "${QUEUE_FILE}" "${FAILURES_FILE}"
        rm -rf "${LOCK_DIR}"
        exit 1
    }
    trap cleanup_parallel INT TERM

    # Blocks until GPU ${1} reports at least MIN_FREE_GB free, or gives up after a
    # bounded wait and proceeds anyway (so a stuck/misreporting GPU can't hang the
    # queue forever). Skips entirely if MIN_FREE_GB=0 or nvidia-smi isn't available.
    wait_for_gpu_headroom() {
        local gpu_id="$1"
        local worker_log="$2"
        [ "${MIN_FREE_GB}" -le 0 ] && return 0
        command -v nvidia-smi >/dev/null 2>&1 || return 0

        local max_checks=10
        local check_interval=30
        for ((i = 0; i < max_checks; i++)); do
            local free_mb
            free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | tr -d ' ')
            [ -z "${free_mb}" ] && return 0
            local free_gb=$((free_mb / 1024))
            if [ "${free_gb}" -ge "${MIN_FREE_GB}" ]; then
                return 0
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id} only ${free_gb}GB free (< ${MIN_FREE_GB}GB) -- likely another process using it. Waiting ${check_interval}s..." >> "${worker_log}"
            sleep "${check_interval}"
        done
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu_id} still below ${MIN_FREE_GB}GB free after $((max_checks * check_interval))s -- proceeding anyway." >> "${worker_log}"
    }

    run_gpu_worker() {
        local worker_gpu="$1"
        local worker_log="${SCRIPT_DIR}/logs/worker_gpu${worker_gpu}_${TAIL}.log"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker started on GPU ${worker_gpu}" > "${worker_log}"

        while true; do
            local raw_task=""
            while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
                sleep 0.1
            done

            if [ -s "${QUEUE_FILE}" ]; then
                raw_task=$(head -n 1 "${QUEUE_FILE}")
                sed -i '1d' "${QUEUE_FILE}"
            fi
            rmdir "${LOCK_DIR}" 2>/dev/null

            if [ -z "${raw_task}" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queue empty. Worker GPU ${worker_gpu} exiting." >> "${worker_log}"
                break
            fi

            local task="${raw_task%%::*}"
            local retry_count="${raw_task##*::}"

            wait_for_gpu_headroom "${worker_gpu}" "${worker_log}"

            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${worker_gpu} >>> Starting Task: ${task} (attempt $((retry_count + 1))/$((MAX_TASK_RETRIES + 1)))" >> "${worker_log}"
            "$0" "${task}" "${worker_gpu}" >> "${worker_log}" 2>&1
            local exit_code=$?

            if [ ${exit_code} -ne 0 ]; then
                if [ "${retry_count}" -lt "${MAX_TASK_RETRIES}" ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task ${task} FAILED (attempt $((retry_count + 1))) on GPU ${worker_gpu} -- requeueing after ${RETRY_DELAY_SECONDS}s." >> "${worker_log}"
                    sleep "${RETRY_DELAY_SECONDS}"
                    while ! mkdir "${LOCK_DIR}" 2>/dev/null; do sleep 0.1; done
                    echo "${task}::$((retry_count + 1))" >> "${QUEUE_FILE}"
                    rmdir "${LOCK_DIR}" 2>/dev/null
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task ${task} PERMANENTLY FAILED after $((MAX_TASK_RETRIES + 1)) attempts on GPU ${worker_gpu}!" >> "${worker_log}"
                    while ! mkdir "${LOCK_DIR}" 2>/dev/null; do sleep 0.1; done
                    echo "${task}" >> "${FAILURES_FILE}"
                    rmdir "${LOCK_DIR}" 2>/dev/null
                fi
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: Task ${task} completed on GPU ${worker_gpu}." >> "${worker_log}"
            fi
            # Never exit the loop on failure -- one bad task must not bench this GPU
            # for the rest of the queue; it either got requeued above or was logged
            # as a permanent failure, and either way this worker keeps pulling tasks.
        done
    }

    for gpu in "${GPUS[@]}"; do
        run_gpu_worker "${gpu}" &
        pid=$!
        WORKER_PIDS+=("${pid}")
        echo "Worker launched on GPU ${gpu} (PID: ${pid})"
    done

    for pid in "${WORKER_PIDS[@]}"; do
        wait "${pid}"
    done

    rm -f "${QUEUE_FILE}"
    rm -rf "${LOCK_DIR}"

    if [ -s "${FAILURES_FILE}" ]; then
        local n_failed
        n_failed=$(wc -l < "${FAILURES_FILE}" | tr -d ' ')
        echo "ERROR: ${n_failed} task(s) permanently failed after $((MAX_TASK_RETRIES + 1)) attempts each:"
        cat "${FAILURES_FILE}"
        echo "Check ${SCRIPT_DIR}/logs/worker_gpu*_${TAIL}.log and the individual train_*.log files for details."
        rm -f "${FAILURES_FILE}"
        exit 1
    fi
    rm -f "${FAILURES_FILE}"
}

# -------------------------------------------------------------------------
# Execution Dispatcher
# -------------------------------------------------------------------------
# Check for task-level embedding type suffix
if [[ "${MODE}" =~ _run3$ ]]; then
    export TAIL="run3"
    export DIVERSITY_EMBED="decoder"
    MODE="${MODE%_run3}"
elif [[ "${MODE}" =~ _run2$ ]]; then
    export TAIL="run2"
    export DIVERSITY_EMBED="decoder"
    MODE="${MODE%_run2}"
elif [[ "${MODE}" =~ _vis_embed$ ]]; then
    export TAIL="vis_embed"
    export DIVERSITY_EMBED="vision_encoder"
    MODE="${MODE%_vis_embed}"
fi

case "${MODE}" in
    "clean")
        cleanup_previous_models
        exit 0
        ;;
    "parallel_both")
        # Runs BOTH normal embed (decoder) and vis embed (vision encoder)
        run_parallel_task_queue \
            "himanis_random_run2" "himanis_entropy_run2" "himanis_kmeans_run2" "himanis_diva_run2" \
            "belfort_random_run2" "belfort_entropy_run2" "belfort_kmeans_run2" "belfort_diva_run2" \
            "esposalles_random_run2" "esposalles_entropy_run2" "esposalles_kmeans_run2" "esposalles_diva_run2" \
            "himanis_kmeans_vis_embed" "himanis_diva_vis_embed" \
            "belfort_kmeans_vis_embed" "belfort_diva_vis_embed" \
            "esposalles_kmeans_vis_embed" "esposalles_diva_vis_embed"
        ;;
    "parallel_normal")
        # Runs normal embed (decoder hidden state) only
        export DIVERSITY_EMBED="decoder"
        export TAIL="run2"
        run_parallel_task_queue \
            "himanis_random" "himanis_entropy" "himanis_kmeans" "himanis_diva" \
            "belfort_random" "belfort_entropy" "belfort_kmeans" "belfort_diva" \
            "esposalles_random" "esposalles_entropy" "esposalles_kmeans" "esposalles_diva"
        ;;
    "parallel_normal_run3")
        # Same task set as parallel_normal, tagged run3 so models/logs/results
        # land in separate _run3 paths instead of overwriting the run2 outputs.
        export DIVERSITY_EMBED="decoder"
        export TAIL="run3"
        run_parallel_task_queue \
            "himanis_random" "himanis_entropy" "himanis_kmeans" "himanis_diva" \
            "belfort_random" "belfort_entropy" "belfort_kmeans" "belfort_diva" \
            "esposalles_random" "esposalles_entropy" "esposalles_kmeans" "esposalles_diva"
        ;;
    "parallel"|"parallel_vis_embed"|"parallel_queue"|"parallel4")
        # Dynamic queue of all AL tasks with vision encoder embeddings (_vis_embed)
        export DIVERSITY_EMBED="vision_encoder"
        export TAIL="vis_embed"
        run_parallel_task_queue \
            "himanis_random" "himanis_entropy" "himanis_kmeans" "himanis_diva" \
            "belfort_random" "belfort_entropy" "belfort_kmeans" "belfort_diva" \
            "esposalles_random" "esposalles_entropy" "esposalles_kmeans" "esposalles_diva"
        ;;
    "parallel_with_full")
        # Dynamic queue including full fine-tuning + AL tasks
        run_parallel_task_queue \
            "himanis_full" "belfort_full" "esposalles_full" \
            "himanis_random" "himanis_entropy" "himanis_kmeans" "himanis_diva" \
            "belfort_random" "belfort_entropy" "belfort_kmeans" "belfort_diva" \
            "esposalles_random" "esposalles_entropy" "esposalles_kmeans" "esposalles_diva"
        ;;
    "parallel_datasets")
        # 1 dataset per GPU concurrently (Himanis on G0, Belfort on G1, Esposalles on G2)
        run_parallel_task_queue "himanis" "belfort" "esposalles"
        ;;
    "parallel_diva")
        run_parallel_task_queue "himanis_diva" "belfort_diva" "esposalles_diva"
        ;;
    "resume_run3")
        # One-off: the 3 tasks that OOM'd from external GPU contention in the last
        # run2/run3 launch (belfort_entropy, belfort_kmeans, himanis_random -- died
        # before completing any/all 5 AL iterations) plus the 3 default DIVA tasks
        # (now alpha=20 instead of the fragmented 1/2). Queued together so the OOM
        # retry/requeue/headroom-check logic in run_parallel_task_queue covers all 6 --
        # running them as separate standalone commands would skip that protection
        # for exactly the tasks that need it most.
        run_parallel_task_queue \
            "belfort_entropy" "belfort_kmeans" "himanis_random" \
            "himanis_diva" "belfort_diva" "esposalles_diva"
        ;;
    "diva_sweep")
        # Runs the default DIVA config (now alpha=20, de-fragmented) plus 4 variants
        # (visenc/fixedquota/alpha40/widebeta)
        # per dataset -- 15 tasks total. Combine with random/entropy/kmeans_center results
        # from the same TAIL to compare every DIVA variant against the baselines on equal
        # footing (same AL_ITERATIONS/SAMPLES_PER_ITER/epochs/seed).
        run_parallel_task_queue \
            "himanis_diva" "himanis_diva_visenc" "himanis_diva_fixedquota" "himanis_diva_alpha40" "himanis_diva_widebeta" \
            "belfort_diva" "belfort_diva_visenc" "belfort_diva_fixedquota" "belfort_diva_alpha40" "belfort_diva_widebeta" \
            "esposalles_diva" "esposalles_diva_visenc" "esposalles_diva_fixedquota" "esposalles_diva_alpha40" "esposalles_diva_widebeta"
        ;;
    "parallel_baselines")
        run_parallel_task_queue \
            "himanis_random" "himanis_entropy" "himanis_kmeans" \
            "belfort_random" "belfort_entropy" "belfort_kmeans" \
            "esposalles_random" "esposalles_entropy" "esposalles_kmeans"
        ;;
    "himanis")
        run_full_finetuning "Teklia_Himanis-line" || exit 1
        run_al_baseline "Teklia_Himanis-line" "random" || exit 1
        run_al_baseline "Teklia_Himanis-line" "entropy" || exit 1
        run_al_baseline "Teklia_Himanis-line" "kmeans_center" || exit 1
        run_al_diva "Teklia_Himanis-line" 20 2 3000 || exit 1
        ;;
    "himanis_full")
        run_full_finetuning "Teklia_Himanis-line" || exit 1
        ;;
    "himanis_random")
        run_al_baseline "Teklia_Himanis-line" "random" || exit 1
        ;;
    "himanis_entropy")
        run_al_baseline "Teklia_Himanis-line" "entropy" || exit 1
        ;;
    "himanis_kmeans")
        run_al_baseline "Teklia_Himanis-line" "kmeans_center" || exit 1
        ;;
    "himanis_diva")
        run_al_diva "Teklia_Himanis-line" 20 2 3000 || exit 1
        ;;
    "himanis_diva_visenc")
        run_al_diva_variant "Teklia_Himanis-line" "visenc" "vision_encoder" "" "" 1 || exit 1
        ;;
    "himanis_diva_fixedquota")
        run_al_diva_variant "Teklia_Himanis-line" "fixedquota" "" "" "" 0 || exit 1
        ;;
    "himanis_diva_alpha40")
        run_al_diva_variant "Teklia_Himanis-line" "alpha40" "" 40 "" 1 || exit 1
        ;;
    "himanis_diva_widebeta")
        run_al_diva_variant "Teklia_Himanis-line" "widebeta" "" "" 4 1 || exit 1
        ;;
    "belfort")
        run_full_finetuning "Teklia_Belfort-line" || exit 1
        run_al_baseline "Teklia_Belfort-line" "random" || exit 1
        run_al_baseline "Teklia_Belfort-line" "entropy" || exit 1
        run_al_baseline "Teklia_Belfort-line" "kmeans_center" || exit 1
        run_al_diva "Teklia_Belfort-line" 20 3 3000 || exit 1
        ;;
    "belfort_full")
        run_full_finetuning "Teklia_Belfort-line" || exit 1
        ;;
    "belfort_random")
        run_al_baseline "Teklia_Belfort-line" "random" || exit 1
        ;;
    "belfort_entropy")
        run_al_baseline "Teklia_Belfort-line" "entropy" || exit 1
        ;;
    "belfort_kmeans")
        run_al_baseline "Teklia_Belfort-line" "kmeans_center" || exit 1
        ;;
    "belfort_diva")
        run_al_diva "Teklia_Belfort-line" 20 3 3000 || exit 1
        ;;
    "belfort_diva_visenc")
        run_al_diva_variant "Teklia_Belfort-line" "visenc" "vision_encoder" "" "" 1 || exit 1
        ;;
    "belfort_diva_fixedquota")
        run_al_diva_variant "Teklia_Belfort-line" "fixedquota" "" "" "" 0 || exit 1
        ;;
    "belfort_diva_alpha40")
        run_al_diva_variant "Teklia_Belfort-line" "alpha40" "" 40 "" 1 || exit 1
        ;;
    "belfort_diva_widebeta")
        run_al_diva_variant "Teklia_Belfort-line" "widebeta" "" "" 6 1 || exit 1
        ;;
    "esposalles")
        run_full_finetuning "Teklia_Esposalles-line" || exit 1
        run_al_baseline "Teklia_Esposalles-line" "random" || exit 1
        run_al_baseline "Teklia_Esposalles-line" "entropy" || exit 1
        run_al_baseline "Teklia_Esposalles-line" "kmeans_center" || exit 1
        run_al_diva "Teklia_Esposalles-line" 20 2 2000 || exit 1
        ;;
    "esposalles_full")
        run_full_finetuning "Teklia_Esposalles-line" || exit 1
        ;;
    "esposalles_random")
        run_al_baseline "Teklia_Esposalles-line" "random" || exit 1
        ;;
    "esposalles_entropy")
        run_al_baseline "Teklia_Esposalles-line" "entropy" || exit 1
        ;;
    "esposalles_kmeans")
        run_al_baseline "Teklia_Esposalles-line" "kmeans_center" || exit 1
        ;;
    "esposalles_diva")
        run_al_diva "Teklia_Esposalles-line" 20 2 2000 || exit 1
        ;;
    "esposalles_diva_visenc")
        run_al_diva_variant "Teklia_Esposalles-line" "visenc" "vision_encoder" "" "" 1 || exit 1
        ;;
    "esposalles_diva_fixedquota")
        run_al_diva_variant "Teklia_Esposalles-line" "fixedquota" "" "" "" 0 || exit 1
        ;;
    "esposalles_diva_alpha40")
        run_al_diva_variant "Teklia_Esposalles-line" "alpha40" "" 40 "" 1 || exit 1
        ;;
    "esposalles_diva_widebeta")
        run_al_diva_variant "Teklia_Esposalles-line" "widebeta" "" "" 4 1 || exit 1
        ;;
    "diva")
        TARGET=${DS_OVERRIDE:-"Teklia_Himanis-line"}
        run_al_diva "${TARGET}"
        ;;
    "random")
        TARGET=${DS_OVERRIDE:-"Teklia_Himanis-line"}
        run_al_baseline "${TARGET}" "random"
        ;;
    "entropy")
        TARGET=${DS_OVERRIDE:-"Teklia_Himanis-line"}
        run_al_baseline "${TARGET}" "entropy"
        ;;
    "kmeans"|"kmeans_center")
        TARGET=${DS_OVERRIDE:-"Teklia_Himanis-line"}
        run_al_baseline "${TARGET}" "kmeans_center"
        ;;
    "baselines")
        TARGET=${DS_OVERRIDE:-"Teklia_Himanis-line"}
        run_al_baseline "${TARGET}" "random"
        run_al_baseline "${TARGET}" "entropy"
        run_al_baseline "${TARGET}" "kmeans_center"
        ;;
    "full")
        TARGET=${DS_OVERRIDE:-"Teklia_Himanis-line"}
        run_full_finetuning "${TARGET}"
        ;;
    "all")
        # Sequential run across the 3 datasets
        for ds in "Teklia_Himanis-line" "Teklia_Belfort-line" "Teklia_Esposalles-line"; do
            run_full_finetuning "${ds}"
            run_al_baseline "${ds}" "random"
            run_al_baseline "${ds}" "entropy"
            run_al_baseline "${ds}" "kmeans_center"
            run_al_diva "${ds}"
        done
        ;;
    *)
        echo "Usage: $0 [parallel | parallel_with_full | parallel_datasets | parallel_diva | parallel_baselines | parallel_normal | parallel_normal_run3 | diva_sweep | diva_sweep_run3 | resume_run3 | himanis | belfort | esposalles | all | clean] [gpu_ids] [dataset_override]"
        echo "DIVA variant tasks (per dataset): <ds>_diva_visenc | <ds>_diva_fixedquota | <ds>_diva_alpha40 | <ds>_diva_widebeta"
        exit 1
        ;;
esac

echo ""
echo "========================================================================="
echo "REQUESTED EXPERIMENTS COMPLETED!"
echo "Checkpoints : ${SCRIPT_DIR}/models/*_${TAIL}"
echo "Logs        : ${SCRIPT_DIR}/logs/*_${TAIL}.log"
echo "Metrics     : ${SCRIPT_DIR}/results/*/*_${TAIL}*"
check_disk_space
echo "========================================================================="
