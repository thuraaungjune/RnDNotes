#!/bin/bash
# Rerun the 3 baseline tasks that OOM'd from external GPU contention in the run2
# launch (see slides_notes.md section 7/9):
#   - Teklia_Belfort-line  entropy        (died right after iteration 0)
#   - Teklia_Belfort-line  kmeans_center  (died right after iteration 0)
#   - Teklia_Himanis-line  random         (died after iteration 2)
#
# This calls the "resume_ooms" mode in the main pipeline script, which:
#   - routes all 3 tasks through the retry/requeue/GPU-headroom-check queue built
#     for this exact failure mode (shared GPU, invisible external contention --
#     not a pipeline bug, so blind retry-without-checking would just OOM again)
#   - restarts each task's full 5-iteration AL loop from scratch (there is no
#     mid-run resume/checkpoint flag) and tags the output _run3, so a fresh,
#     complete run never collides with the partial _run2 CSVs mid-write
#
# Usage:
#   ./rerun_ooms.sh [gpu_id]      # default gpu_id: 0
#
# After it finishes, check:
#   results/entropy_results/Teklia_Belfort-line_al_entropy_seed42_run3_metrics.csv
#   results/kmeans_center_results/Teklia_Belfort-line_al_kmeans_center_seed42_run3_metrics.csv
#   results/random_results/Teklia_Himanis-line_al_random_seed42_run3_metrics.csv
# each should have 6 rows (iteration 0-5). Use those _run3 rows in place of the
# partial _run2 ones for Belfort entropy/kmeans_center and Himanis random in any
# comparison table or figure -- everything else stays on _run2 as before.
set -euo pipefail

GPU_ID="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== rerunning OOM'd baselines (belfort_entropy, belfort_kmeans, himanis_random) on GPU ${GPU_ID} ==="
"${SCRIPT_DIR}/run_al_and_full_lorafixed_fixed_run2_run3.sh" resume_ooms "${GPU_ID}"
