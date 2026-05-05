#!/usr/bin/env bash
set -u -o pipefail

# -------------------------------------------------------------------------
# Segger comprehensive ablation study (auto-detect GPUs, N-way parallel)
# -------------------------------------------------------------------------
# Systematically removes or swaps individual components (loss terms,
# architecture choices, features) to measure their contribution.
#
# Usage:
#   bash scripts/run_ablation_study.sh
#
# Optional overrides (environment variables):
#   INPUT_DIR=data/xe_pancreas_mossi/
#   OUTPUT_ROOT=./results/mossi_ablation_study
#   NUM_GPUS=<auto>            # Override auto-detected GPU count
#   N_EPOCHS=20
#   RESUME_IF_EXISTS=1
#   DRY_RUN=0
#   SEGMENT_TIMEOUT_MIN=90
#   ALIGNMENT_SCRNA_REFERENCE_PATH=data/ref_pancreas.h5ad
#   ALIGNMENT_SCRNA_CELLTYPE_COLUMN=cell_type
#   SEGMENT_NUM_WORKERS=8
#   SEGMENT_ANC_RETRY_WORKERS=0
#   TORCH_SHARING_STRATEGY=file_system
#   RUN_VALIDATION_TABLE=1
#   VALIDATION_SCRIPT=scripts/build_benchmark_validation_table.sh
#
# Block toggles (set to 0 to skip):
#   RUN_LOSS_ABLATION=1
#   RUN_SGLOSS_ABLATION=1
#   RUN_ALIGNMENT_SWEEP=1
#   RUN_ARCH_ABLATION=1
#   RUN_PREDICTION_ABLATION=1
#   RUN_LR_ABLATION=1
# -------------------------------------------------------------------------

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

# -------------------------------------------------------------------------
# GPU detection
# -------------------------------------------------------------------------
detect_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo $(( $(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l) ))
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus 2>/dev/null | wc -l
  else
    echo 1
  fi
}

NUM_GPUS="${NUM_GPUS:-$(detect_gpus)}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  NUM_GPUS=1
fi

# Build array of GPU IDs (0..N-1, or from CUDA_VISIBLE_DEVICES).
GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  for ((g = 0; g < NUM_GPUS; g++)); do
    GPU_IDS+=("${g}")
  done
fi

# -------------------------------------------------------------------------
# Paths and defaults
# -------------------------------------------------------------------------
DEFAULT_INPUT_DIR="data/xe_pancreas_mossi/"
INPUT_DIR="${INPUT_DIR:-${DEFAULT_INPUT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/mossi_ablation_study}"

if [[ "${INPUT_DIR}" == "${DEFAULT_INPUT_DIR}" ]] && \
   [[ ! -d "${INPUT_DIR}" ]] && \
   [[ -d "../data/xe_pancreas_mossi/" ]]; then
  INPUT_DIR="../data/xe_pancreas_mossi/"
fi

N_EPOCHS="${N_EPOCHS:-20}"
PREDICTION_MODE="${PREDICTION_MODE:-nucleus}"

BOUNDARY_METHOD="${BOUNDARY_METHOD:-convex_hull}"
BOUNDARY_VOXEL_SIZE="${BOUNDARY_VOXEL_SIZE:-5}"
XENIUM_NUM_WORKERS="${XENIUM_NUM_WORKERS:-8}"

RESUME_IF_EXISTS="${RESUME_IF_EXISTS:-1}"
DRY_RUN="${DRY_RUN:-0}"
PREDICT_FALLBACK_ON_OOM="${PREDICT_FALLBACK_ON_OOM:-1}"
SEGMENT_TIMEOUT_MIN="${SEGMENT_TIMEOUT_MIN:-90}"
SEGMENT_TIMEOUT_SEC=$((SEGMENT_TIMEOUT_MIN * 60))
SEGMENT_NUM_WORKERS="${SEGMENT_NUM_WORKERS:-8}"
SEGMENT_ANC_RETRY_WORKERS="${SEGMENT_ANC_RETRY_WORKERS:-0}"
TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"

# Alignment defaults (needed by anchor and alignment sweep).
ALIGNMENT_LOSS_WEIGHT_START="${ALIGNMENT_LOSS_WEIGHT_START:-0.0}"
ALIGNMENT_ME_GENE_PAIRS_PATH="${ALIGNMENT_ME_GENE_PAIRS_PATH:-}"
ALIGNMENT_SCRNA_REFERENCE_PATH="${ALIGNMENT_SCRNA_REFERENCE_PATH:-data/ref_pancreas.h5ad}"
ALIGNMENT_SCRNA_CELLTYPE_COLUMN="${ALIGNMENT_SCRNA_CELLTYPE_COLUMN:-cell_type}"

if [[ "${ALIGNMENT_SCRNA_REFERENCE_PATH}" == "data/ref_pancreas.h5ad" ]] && \
   [[ ! -f "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]] && \
   [[ -f "../data/ref_pancreas.h5ad" ]]; then
  ALIGNMENT_SCRNA_REFERENCE_PATH="../data/ref_pancreas.h5ad"
fi

# -------------------------------------------------------------------------
# Anchor configuration (defaults matching current best config)
# -------------------------------------------------------------------------
ANCHOR_USE_3D="${ANCHOR_USE_3D:-true}"
ANCHOR_EXPANSION="${ANCHOR_EXPANSION:-2.5}"
ANCHOR_TX_K="${ANCHOR_TX_K:-5}"
ANCHOR_TX_DIST="${ANCHOR_TX_DIST:-20}"
ANCHOR_N_LAYERS="${ANCHOR_N_LAYERS:-2}"
ANCHOR_N_HEADS="${ANCHOR_N_HEADS:-4}"
ANCHOR_CELLS_MIN="${ANCHOR_CELLS_MIN:-5}"
ANCHOR_MIN_QV="${ANCHOR_MIN_QV:-0}"
ANCHOR_ALIGNMENT="${ANCHOR_ALIGNMENT:-true}"
ANCHOR_SG_LOSS="${ANCHOR_SG_LOSS:-triplet}"
ANCHOR_HIDDEN="${ANCHOR_HIDDEN:-64}"
ANCHOR_OUT="${ANCHOR_OUT:-64}"
ANCHOR_TX_WEIGHT="${ANCHOR_TX_WEIGHT:-1.0}"
ANCHOR_BD_WEIGHT="${ANCHOR_BD_WEIGHT:-1.0}"
ANCHOR_SG_WEIGHT="${ANCHOR_SG_WEIGHT:-0.5}"
ANCHOR_ALIGN_WEIGHT="${ANCHOR_ALIGN_WEIGHT:-0.03}"
ANCHOR_POS_EMB="${ANCHOR_POS_EMB:-true}"
ANCHOR_NORM_EMB="${ANCHOR_NORM_EMB:-true}"
ANCHOR_CELLS_REP="${ANCHOR_CELLS_REP:-pca}"
ANCHOR_LR="${ANCHOR_LR:-1e-3}"

# -------------------------------------------------------------------------
# Block toggles
# -------------------------------------------------------------------------
RUN_LOSS_ABLATION="${RUN_LOSS_ABLATION:-1}"
RUN_SGLOSS_ABLATION="${RUN_SGLOSS_ABLATION:-1}"
RUN_ALIGNMENT_SWEEP="${RUN_ALIGNMENT_SWEEP:-1}"
RUN_ARCH_ABLATION="${RUN_ARCH_ABLATION:-1}"
RUN_PREDICTION_ABLATION="${RUN_PREDICTION_ABLATION:-1}"
RUN_LR_ABLATION="${RUN_LR_ABLATION:-1}"

# -------------------------------------------------------------------------
# Directories
# -------------------------------------------------------------------------
RUNS_DIR="${OUTPUT_ROOT}/runs"
EXPORTS_DIR="${OUTPUT_ROOT}/exports"
LOGS_DIR="${OUTPUT_ROOT}/logs"
SUMMARY_DIR="${OUTPUT_ROOT}/summaries"
PLAN_FILE="${OUTPUT_ROOT}/job_plan.tsv"
RUN_VALIDATION_TABLE="${RUN_VALIDATION_TABLE:-1}"
VALIDATION_SCRIPT="${VALIDATION_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_benchmark_validation_table.sh}"
VALIDATION_INCLUDE_DEFAULT_10X="${VALIDATION_INCLUDE_DEFAULT_10X:-true}"

mkdir -p "${RUNS_DIR}" "${EXPORTS_DIR}" "${LOGS_DIR}" "${SUMMARY_DIR}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "WARN: INPUT_DIR does not exist (dry run only): ${INPUT_DIR}"
  else
    echo "ERROR: INPUT_DIR does not exist: ${INPUT_DIR}"
    exit 1
  fi
fi

if [[ "${DRY_RUN}" != "1" ]] && ! command -v segger >/dev/null 2>&1; then
  echo "ERROR: 'segger' command not found in PATH."
  exit 1
fi

# Check alignment inputs (needed if any ablation uses alignment).
need_alignment_inputs=0
if [[ "${RUN_LOSS_ABLATION}" == "1" ]] || \
   [[ "${RUN_ALIGNMENT_SWEEP}" == "1" ]] || \
   [[ "${ANCHOR_ALIGNMENT}" == "true" ]]; then
  need_alignment_inputs=1
fi

if [[ "${need_alignment_inputs}" == "1" ]]; then
  if [[ -z "${ALIGNMENT_ME_GENE_PAIRS_PATH}" ]] && [[ -z "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]]; then
    echo "ERROR: Alignment ablation requires ALIGNMENT_ME_GENE_PAIRS_PATH or ALIGNMENT_SCRNA_REFERENCE_PATH."
    exit 1
  fi
  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ -n "${ALIGNMENT_ME_GENE_PAIRS_PATH}" ]] && [[ ! -f "${ALIGNMENT_ME_GENE_PAIRS_PATH}" ]]; then
      echo "ERROR: ALIGNMENT_ME_GENE_PAIRS_PATH not found: ${ALIGNMENT_ME_GENE_PAIRS_PATH}"
      exit 1
    fi
    if [[ -n "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]] && [[ ! -f "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]]; then
      echo "ERROR: ALIGNMENT_SCRNA_REFERENCE_PATH not found: ${ALIGNMENT_SCRNA_REFERENCE_PATH}"
      exit 1
    fi
  fi
fi

# =========================================================================
# Job spec: extended 21-field pipe-delimited format
# =========================================================================
# Fields:
#  1  job_name
#  2  use_3d
#  3  expansion
#  4  tx_k
#  5  tx_dist
#  6  n_layers
#  7  n_heads
#  8  cells_min_counts
#  9  min_qv
# 10  alignment_loss       (true/false)
# 11  sg_loss_type         (triplet/bce)
# 12  hidden_channels
# 13  out_channels
# 14  tx_weight_end
# 15  bd_weight_end
# 16  sg_weight_end
# 17  alignment_weight_end
# 18  positional_embeddings (true/false)
# 19  normalize_embeddings  (true/false)
# 20  cells_representation  (pca/morphology)
# 21  learning_rate
# =========================================================================

JOB_SPECS=()

add_job() {
  local job_name="$1"
  local use_3d="${2:-${ANCHOR_USE_3D}}"
  local expansion="${3:-${ANCHOR_EXPANSION}}"
  local tx_k="${4:-${ANCHOR_TX_K}}"
  local tx_dist="${5:-${ANCHOR_TX_DIST}}"
  local n_layers="${6:-${ANCHOR_N_LAYERS}}"
  local n_heads="${7:-${ANCHOR_N_HEADS}}"
  local cells_min="${8:-${ANCHOR_CELLS_MIN}}"
  local min_qv="${9:-${ANCHOR_MIN_QV}}"
  local align="${10:-${ANCHOR_ALIGNMENT}}"
  local sg_loss="${11:-${ANCHOR_SG_LOSS}}"
  local hidden="${12:-${ANCHOR_HIDDEN}}"
  local out="${13:-${ANCHOR_OUT}}"
  local tx_w="${14:-${ANCHOR_TX_WEIGHT}}"
  local bd_w="${15:-${ANCHOR_BD_WEIGHT}}"
  local sg_w="${16:-${ANCHOR_SG_WEIGHT}}"
  local align_w="${17:-${ANCHOR_ALIGN_WEIGHT}}"
  local pos_emb="${18:-${ANCHOR_POS_EMB}}"
  local norm_emb="${19:-${ANCHOR_NORM_EMB}}"
  local cells_rep="${20:-${ANCHOR_CELLS_REP}}"
  local lr="${21:-${ANCHOR_LR}}"

  JOB_SPECS+=("${job_name}|${use_3d}|${expansion}|${tx_k}|${tx_dist}|${n_layers}|${n_heads}|${cells_min}|${min_qv}|${align}|${sg_loss}|${hidden}|${out}|${tx_w}|${bd_w}|${sg_w}|${align_w}|${pos_emb}|${norm_emb}|${cells_rep}|${lr}")
}

# Helper: add_job with only overridden fields (positional anchor defaults).
# Usage: add_ablation_job <name> [field=value ...]
# This is a convenience wrapper; for clarity each block calls add_job directly.

job_block() {
  local job_name="$1"
  case "${job_name}" in
    abl_sg_*|abl_full) echo "loss_decomposition" ;;
    abl_sgloss_*) echo "sg_loss_type" ;;
    abl_aw_*) echo "alignment_sweep" ;;
    abl_depth_*|abl_width_*|abl_heads_*|abl_no_pos|abl_no_norm|abl_morph) echo "architecture" ;;
    abl_pred_*) echo "prediction_mode" ;;
    abl_lr_*) echo "learning_rate" ;;
    *) echo "other" ;;
  esac
}

build_jobs() {
  # -------------------------------------------------------------------
  # Block A: Loss decomposition (6 jobs)
  # -------------------------------------------------------------------
  if [[ "${RUN_LOSS_ABLATION}" == "1" ]]; then
    # sg only: tx=0, bd=0, no alignment
    add_job "abl_sg_only" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "false" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "0" "0" "${ANCHOR_SG_WEIGHT}" "0" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"

    # sg + transcript triplet: bd=0, no alignment
    add_job "abl_sg_tx" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "false" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "0" "${ANCHOR_SG_WEIGHT}" "0" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"

    # sg + boundary metric: tx=0, no alignment
    add_job "abl_sg_bd" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "false" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "0" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "0" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"

    # sg + both clustering: no alignment
    add_job "abl_sg_tx_bd" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "false" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "0" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"

    # sg + alignment only: tx=0, bd=0
    add_job "abl_sg_align" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "true" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "0" "0" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"

    # full (anchor baseline)
    add_job "abl_full" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"
  fi

  # -------------------------------------------------------------------
  # Block B: Segmentation loss type (2 jobs)
  # -------------------------------------------------------------------
  if [[ "${RUN_SGLOSS_ABLATION}" == "1" ]]; then
    add_job "abl_sgloss_triplet" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "triplet" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"

    add_job "abl_sgloss_bce" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "bce" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"
  fi

  # -------------------------------------------------------------------
  # Block C: Alignment weight sweep (5 jobs)
  # -------------------------------------------------------------------
  if [[ "${RUN_ALIGNMENT_SWEEP}" == "1" ]]; then
    local aw_values=(0 0.01 0.03 0.1 0.3)
    local aw_tags=(0 001 003 01 03)
    local aw_i
    for aw_i in "${!aw_values[@]}"; do
      local aw="${aw_values[$aw_i]}"
      local aw_tag="${aw_tags[$aw_i]}"
      local aw_align="true"
      if [[ "${aw}" == "0" ]]; then
        aw_align="false"
      fi
      add_job "abl_aw_${aw_tag}" \
        "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
        "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
        "${aw_align}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
        "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${aw}" \
        "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"
    done
  fi

  # -------------------------------------------------------------------
  # Block D: Architecture ablation (10 jobs)
  # -------------------------------------------------------------------
  if [[ "${RUN_ARCH_ABLATION}" == "1" ]]; then
    # Depth: 0, 1, 3 mid layers
    for depth in 0 1 3; do
      add_job "abl_depth_${depth}" \
        "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
        "${depth}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
        "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
        "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
        "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"
    done

    # Width: 32/32 and 128/128
    for width in 32 128; do
      add_job "abl_width_${width}" \
        "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
        "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
        "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${width}" "${width}" \
        "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
        "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"
    done

    # Heads: 1 and 8
    for heads in 1 8; do
      add_job "abl_heads_${heads}" \
        "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
        "${ANCHOR_N_LAYERS}" "${heads}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
        "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
        "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
        "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"
    done

    # No positional embeddings
    add_job "abl_no_pos" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "false" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"

    # No embedding normalization
    add_job "abl_no_norm" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "false" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"

    # Morphology instead of PCA
    add_job "abl_morph" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "morphology" "${ANCHOR_LR}"
  fi

  # -------------------------------------------------------------------
  # Block E: Prediction mode (2 jobs)
  # -------------------------------------------------------------------
  if [[ "${RUN_PREDICTION_ABLATION}" == "1" ]]; then
    add_job "abl_pred_cell" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"

    add_job "abl_pred_uniform" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "${ANCHOR_LR}"
  fi

  # -------------------------------------------------------------------
  # Block F: Learning rate (3 jobs)
  # -------------------------------------------------------------------
  if [[ "${RUN_LR_ABLATION}" == "1" ]]; then
    add_job "abl_lr_3e4" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "3e-4"

    add_job "abl_lr_3e3" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "3e-3"

    add_job "abl_lr_1e2" \
      "${ANCHOR_USE_3D}" "${ANCHOR_EXPANSION}" "${ANCHOR_TX_K}" "${ANCHOR_TX_DIST}" \
      "${ANCHOR_N_LAYERS}" "${ANCHOR_N_HEADS}" "${ANCHOR_CELLS_MIN}" "${ANCHOR_MIN_QV}" \
      "${ANCHOR_ALIGNMENT}" "${ANCHOR_SG_LOSS}" "${ANCHOR_HIDDEN}" "${ANCHOR_OUT}" \
      "${ANCHOR_TX_WEIGHT}" "${ANCHOR_BD_WEIGHT}" "${ANCHOR_SG_WEIGHT}" "${ANCHOR_ALIGN_WEIGHT}" \
      "${ANCHOR_POS_EMB}" "${ANCHOR_NORM_EMB}" "${ANCHOR_CELLS_REP}" "1e-2"
  fi
}

# =========================================================================
# Helper functions (identical to run_robustness_ablation_2gpu.sh)
# =========================================================================

run_cmd() {
  local log_file="$1"
  shift
  local -a cmd=("$@")

  {
    printf '[%s] CMD:' "$(timestamp)"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } >> "${log_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}" >> "${log_file}" 2>&1
}

run_cmd_with_timeout() {
  local log_file="$1"
  local timeout_seconds="$2"
  shift 2
  local -a cmd=("$@")

  {
    printf '[%s] CMD(timeout=%ss):' "$(timestamp)" "${timeout_seconds}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } >> "${log_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if [[ "${timeout_seconds}" -le 0 ]]; then
    "${cmd[@]}" >> "${log_file}" 2>&1
    return $?
  fi

  local start_ts now elapsed
  local cmd_pid timed_out rc
  timed_out=0
  start_ts="$(date +%s)"

  "${cmd[@]}" >> "${log_file}" 2>&1 &
  cmd_pid=$!

  while kill -0 "${cmd_pid}" 2>/dev/null; do
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if (( elapsed >= timeout_seconds )); then
      timed_out=1
      echo "[$(timestamp)] OOT: command exceeded ${timeout_seconds}s; terminating PID=${cmd_pid}" >> "${log_file}"
      kill -TERM "${cmd_pid}" 2>/dev/null || true
      pkill -TERM -P "${cmd_pid}" 2>/dev/null || true
      sleep 5
      kill -KILL "${cmd_pid}" 2>/dev/null || true
      pkill -KILL -P "${cmd_pid}" 2>/dev/null || true
      break
    fi
    sleep 10
  done

  wait "${cmd_pid}"
  rc=$?
  if (( timed_out == 1 )); then
    return 124
  fi
  return "${rc}"
}

is_oom_failure() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    return 1
  fi
  local pattern="out of memory|cuda error: out of memory|cublas status alloc failed|cuda driver error.*memory"
  if command -v rg >/dev/null 2>&1; then
    rg -qi "${pattern}" "${log_file}"
  else
    grep -Eiq "${pattern}" "${log_file}"
  fi
}

is_ancdata_failure() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    return 1
  fi
  local pattern="received [0-9]+ items of ancdata|multiprocessing/resource_sharer\\.py"
  if command -v rg >/dev/null 2>&1; then
    rg -qi "${pattern}" "${log_file}"
  else
    grep -Eiq "${pattern}" "${log_file}"
  fi
}

LAST_EXPORT_STATUS="ok"

run_exports_for_job() {
  local job_name="$1"
  local seg_dir="$2"
  local log_file="$3"

  local seg_file="${seg_dir}/segger_segmentation.parquet"
  local anndata_dir="${EXPORTS_DIR}/${job_name}/anndata"
  local anndata_file="${anndata_dir}/segger_segmentation.h5ad"
  local xenium_dir="${EXPORTS_DIR}/${job_name}/xenium_explorer"
  local xenium_file="${xenium_dir}/seg_experiment.xenium"

  mkdir -p "${anndata_dir}" "${xenium_dir}"

  if [[ ! -f "${seg_file}" ]] && [[ "${DRY_RUN}" != "1" ]]; then
    LAST_EXPORT_STATUS="missing_segmentation"
    return 1
  fi

  if [[ ! -f "${anndata_file}" ]]; then
    local -a anndata_cmd=(
      segger export
      -s "${seg_file}"
      -i "${INPUT_DIR}"
      -o "${anndata_dir}"
      --format anndata
    )
    if ! run_cmd "${log_file}" "${anndata_cmd[@]}"; then
      LAST_EXPORT_STATUS="anndata_export_failed"
      return 1
    fi
  else
    echo "[$(timestamp)] SKIP anndata export (existing): ${anndata_file}" >> "${log_file}"
  fi

  if [[ ! -f "${xenium_file}" ]]; then
    local -a xenium_cmd=(
      segger export
      -s "${seg_file}"
      -i "${INPUT_DIR}"
      -o "${xenium_dir}"
      --format xenium_explorer
      --boundary-method "${BOUNDARY_METHOD}"
      --boundary-voxel-size "${BOUNDARY_VOXEL_SIZE}"
      --num-workers "${XENIUM_NUM_WORKERS}"
    )
    if ! run_cmd "${log_file}" "${xenium_cmd[@]}"; then
      LAST_EXPORT_STATUS="xenium_export_failed"
      return 1
    fi
  else
    echo "[$(timestamp)] SKIP xenium export (existing): ${xenium_file}" >> "${log_file}"
  fi

  LAST_EXPORT_STATUS="ok"
  return 0
}

# =========================================================================
# run_job — extended to handle 21-field spec
# =========================================================================

LAST_JOB_STATUS="unknown"

run_job() {
  local gpu="$1"
  local spec="$2"

  local job_name use_3d expansion tx_k tx_dist n_layers n_heads cells_min_counts min_qv
  local alignment_loss sg_loss_type hidden_channels out_channels
  local tx_weight_end bd_weight_end sg_weight_end alignment_weight_end
  local positional_embeddings normalize_embeddings cells_representation learning_rate
  IFS='|' read -r \
    job_name use_3d expansion tx_k tx_dist n_layers n_heads cells_min_counts min_qv \
    alignment_loss sg_loss_type hidden_channels out_channels \
    tx_weight_end bd_weight_end sg_weight_end alignment_weight_end \
    positional_embeddings normalize_embeddings cells_representation learning_rate \
    <<< "${spec}"

  # Resolve prediction mode from job name (Block E override).
  local job_prediction_mode="${PREDICTION_MODE}"
  case "${job_name}" in
    abl_pred_cell) job_prediction_mode="cell" ;;
    abl_pred_uniform) job_prediction_mode="uniform" ;;
  esac

  local seg_dir="${RUNS_DIR}/${job_name}"
  local seg_file="${seg_dir}/segger_segmentation.parquet"
  local anndata_dir="${EXPORTS_DIR}/${job_name}/anndata"
  local anndata_file="${anndata_dir}/segger_segmentation.h5ad"
  local xenium_dir="${EXPORTS_DIR}/${job_name}/xenium_explorer"
  local xenium_file="${xenium_dir}/seg_experiment.xenium"
  local log_file="${LOGS_DIR}/${job_name}.gpu${gpu}.log"

  mkdir -p "${seg_dir}" "${anndata_dir}" "${xenium_dir}"

  {
    echo "=================================================================="
    echo "[$(timestamp)] START job=${job_name} gpu=${gpu}"
    echo "params: use3d=${use_3d} expansion=${expansion} tx_k=${tx_k} tx_dist=${tx_dist} layers=${n_layers} heads=${n_heads} cells_min=${cells_min_counts} min_qv=${min_qv} align=${alignment_loss} sg_loss=${sg_loss_type} hidden=${hidden_channels} out=${out_channels} tx_w=${tx_weight_end} bd_w=${bd_weight_end} sg_w=${sg_weight_end} align_w=${alignment_weight_end} pos_emb=${positional_embeddings} norm_emb=${normalize_embeddings} cells_rep=${cells_representation} lr=${learning_rate} pred_mode=${job_prediction_mode} timeout_min=${SEGMENT_TIMEOUT_MIN}"
  } | tee -a "${log_file}" >/dev/null

  if [[ "${RESUME_IF_EXISTS}" == "1" ]] && \
     [[ -f "${seg_file}" ]] && \
     [[ -f "${anndata_file}" ]] && \
     [[ -f "${xenium_file}" ]]; then
    echo "[$(timestamp)] SKIP job=${job_name} (all outputs already present)" | tee -a "${log_file}" >/dev/null
    LAST_JOB_STATUS="skipped_existing"
    return 0
  fi

  if [[ ! -f "${seg_file}" ]]; then
    # Build positional/normalize flags for cyclopts booleans.
    local pos_flag="--use-positional-embeddings"
    if [[ "${positional_embeddings}" == "false" ]]; then
      pos_flag="--no-use-positional-embeddings"
    fi
    local norm_flag="--normalize-embeddings"
    if [[ "${normalize_embeddings}" == "false" ]]; then
      norm_flag="--no-normalize-embeddings"
    fi

    local -a seg_cmd=(
      env CUDA_VISIBLE_DEVICES="${gpu}"
      PYTORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY}"
      SEGGER_NUM_WORKERS="${SEGMENT_NUM_WORKERS}"
      segger segment
      -i "${INPUT_DIR}"
      -o "${seg_dir}"
      --n-epochs "${N_EPOCHS}"
      --prediction-mode "${job_prediction_mode}"
      --prediction-expansion-ratio "${expansion}"
      --cells-min-counts "${cells_min_counts}"
      --min-qv "${min_qv}"
      --use-3d "${use_3d}"
      --transcripts-max-k "${tx_k}"
      --transcripts-max-dist "${tx_dist}"
      --n-mid-layers "${n_layers}"
      --n-heads "${n_heads}"
      --segmentation-loss "${sg_loss_type}"
      --hidden-channels "${hidden_channels}"
      --out-channels "${out_channels}"
      --transcripts-loss-weight-end "${tx_weight_end}"
      --cells-loss-weight-end "${bd_weight_end}"
      --segmentation-loss-weight-end "${sg_weight_end}"
      --learning-rate "${learning_rate}"
      --cells-representation "${cells_representation}"
      "${pos_flag}"
      "${norm_flag}"
    )

    if [[ "${alignment_loss}" == "true" ]]; then
      seg_cmd+=(
        --alignment-loss
        --alignment-loss-weight-start "${ALIGNMENT_LOSS_WEIGHT_START}"
        --alignment-loss-weight-end "${alignment_weight_end}"
      )
      if [[ -n "${ALIGNMENT_ME_GENE_PAIRS_PATH}" ]]; then
        seg_cmd+=(--alignment-me-gene-pairs-path "${ALIGNMENT_ME_GENE_PAIRS_PATH}")
      fi
      if [[ -n "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]]; then
        seg_cmd+=(
          --scrna-reference-path "${ALIGNMENT_SCRNA_REFERENCE_PATH}"
          --scrna-celltype-column "${ALIGNMENT_SCRNA_CELLTYPE_COLUMN}"
        )
      fi
    fi

    run_cmd_with_timeout "${log_file}" "${SEGMENT_TIMEOUT_SEC}" "${seg_cmd[@]}"
    local seg_rc=$?
    if [[ "${seg_rc}" -ne 0 ]]; then
      if [[ "${seg_rc}" -eq 124 ]]; then
        echo "[$(timestamp)] FAIL job=${job_name} step=segment (OOT ${SEGMENT_TIMEOUT_MIN}m)" | tee -a "${log_file}" >/dev/null
        LAST_JOB_STATUS="segment_oot"
        return 1
      fi

      if [[ "${SEGMENT_ANC_RETRY_WORKERS}" != "${SEGMENT_NUM_WORKERS}" ]] && is_ancdata_failure "${log_file}"; then
        echo "[$(timestamp)] WARN job=${job_name} segment failed with ancdata; retrying with SEGGER_NUM_WORKERS=${SEGMENT_ANC_RETRY_WORKERS}" | tee -a "${log_file}" >/dev/null
        local -a seg_retry_cmd=("${seg_cmd[@]}")
        local i
        for i in "${!seg_retry_cmd[@]}"; do
          if [[ "${seg_retry_cmd[$i]}" == SEGGER_NUM_WORKERS=* ]]; then
            seg_retry_cmd[$i]="SEGGER_NUM_WORKERS=${SEGMENT_ANC_RETRY_WORKERS}"
            break
          fi
        done
        run_cmd_with_timeout "${log_file}" "${SEGMENT_TIMEOUT_SEC}" "${seg_retry_cmd[@]}"
        seg_rc=$?
        if [[ "${seg_rc}" -eq 0 ]]; then
          echo "[$(timestamp)] OK job=${job_name} segment retry succeeded with SEGGER_NUM_WORKERS=${SEGMENT_ANC_RETRY_WORKERS}" | tee -a "${log_file}" >/dev/null
        elif [[ "${seg_rc}" -eq 124 ]]; then
          echo "[$(timestamp)] FAIL job=${job_name} step=segment_retry (OOT ${SEGMENT_TIMEOUT_MIN}m)" | tee -a "${log_file}" >/dev/null
          LAST_JOB_STATUS="segment_oot"
          return 1
        fi
      fi

      if [[ "${seg_rc}" -eq 0 ]]; then
        :
      else
      local last_ckpt="${seg_dir}/checkpoints/last.ckpt"
      if [[ "${PREDICT_FALLBACK_ON_OOM}" == "1" ]] && is_oom_failure "${log_file}" && [[ -f "${last_ckpt}" ]]; then
        echo "[$(timestamp)] WARN job=${job_name} segment OOM; trying checkpoint predict fallback (${last_ckpt})" | tee -a "${log_file}" >/dev/null
        local -a predict_cmd=(
          env CUDA_VISIBLE_DEVICES="${gpu}"
          PYTORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY}"
          SEGGER_NUM_WORKERS="${SEGMENT_NUM_WORKERS}"
          segger predict
          -c "${last_ckpt}"
          -i "${INPUT_DIR}"
          -o "${seg_dir}"
        )
        if run_cmd "${log_file}" "${predict_cmd[@]}"; then
          echo "[$(timestamp)] OK job=${job_name} predict fallback succeeded after OOM" | tee -a "${log_file}" >/dev/null
        else
          echo "[$(timestamp)] FAIL job=${job_name} step=predict_fallback_after_oom" | tee -a "${log_file}" >/dev/null
          LAST_JOB_STATUS="predict_fallback_failed"
          return 1
        fi
      else
        if is_ancdata_failure "${log_file}"; then
          echo "[$(timestamp)] FAIL job=${job_name} step=segment (ancdata)" | tee -a "${log_file}" >/dev/null
          LAST_JOB_STATUS="segment_ancdata"
        elif is_oom_failure "${log_file}"; then
          echo "[$(timestamp)] FAIL job=${job_name} step=segment (oom)" | tee -a "${log_file}" >/dev/null
          LAST_JOB_STATUS="segment_oom"
        else
          echo "[$(timestamp)] FAIL job=${job_name} step=segment" | tee -a "${log_file}" >/dev/null
          LAST_JOB_STATUS="segment_failed"
        fi
        return 1
      fi
      fi
    fi
  else
    echo "[$(timestamp)] SKIP segmentation (existing): ${seg_file}" | tee -a "${log_file}" >/dev/null
  fi

  if ! run_exports_for_job "${job_name}" "${seg_dir}" "${log_file}"; then
    echo "[$(timestamp)] FAIL job=${job_name} step=${LAST_EXPORT_STATUS}" | tee -a "${log_file}" >/dev/null
    LAST_JOB_STATUS="${LAST_EXPORT_STATUS}"
    return 1
  fi

  echo "[$(timestamp)] DONE job=${job_name}" | tee -a "${log_file}" >/dev/null
  LAST_JOB_STATUS="ok"
  return 0
}

# =========================================================================
# GPU group runner
# =========================================================================

run_gpu_group() {
  local gpu="$1"
  shift
  local -a indices=("$@")
  local summary_file="${SUMMARY_DIR}/gpu${gpu}.tsv"

  printf "job\tgpu\tstatus\telapsed_s\tseg_dir\tlog_file\n" > "${summary_file}"

  local idx spec job_name start_ts end_ts elapsed_s
  for idx in "${indices[@]}"; do
    spec="${JOB_SPECS[$idx]}"
    IFS='|' read -r job_name _ <<< "${spec}"

    start_ts="$(date +%s)"
    run_job "${gpu}" "${spec}"
    end_ts="$(date +%s)"
    elapsed_s=$((end_ts - start_ts))

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job_name}" \
      "${gpu}" \
      "${LAST_JOB_STATUS}" \
      "${elapsed_s}" \
      "${RUNS_DIR}/${job_name}" \
      "${LOGS_DIR}/${job_name}.gpu${gpu}.log" \
      >> "${summary_file}"
  done
}

# =========================================================================
# Post-run recovery (predict-only from checkpoints)
# =========================================================================

run_post_recovery_predict_only_group() {
  local gpu="$1"
  local out_file="$2"
  shift 2
  local -a indices=("$@")

  printf "job\tgpu\tstatus\telapsed_s\tnote\tseg_dir\tlog_file\n" > "${out_file}"

  local idx spec job_name
  local seg_dir seg_file last_ckpt log_file note status
  local start_ts end_ts elapsed_s

  for idx in "${indices[@]}"; do
    spec="${JOB_SPECS[$idx]}"
    IFS='|' read -r job_name _ <<< "${spec}"

    seg_dir="${RUNS_DIR}/${job_name}"
    seg_file="${seg_dir}/segger_segmentation.parquet"
    last_ckpt="${seg_dir}/checkpoints/last.ckpt"
    log_file="${LOGS_DIR}/${job_name}.gpu${gpu}.log"
    mkdir -p "${seg_dir}"

    start_ts="$(date +%s)"
    note=""
    status="ok"

    if [[ -f "${seg_file}" ]]; then
      note="segmentation_exists"
      if ! run_exports_for_job "${job_name}" "${seg_dir}" "${log_file}"; then
        status="${LAST_EXPORT_STATUS}"
        note="exports_failed_after_existing_seg"
      fi
    else
      if [[ -f "${last_ckpt}" ]]; then
        echo "[$(timestamp)] RECOVERY job=${job_name}: running predict-only from ${last_ckpt}" | tee -a "${log_file}" >/dev/null
        local -a predict_cmd=(
          env CUDA_VISIBLE_DEVICES="${gpu}"
          PYTORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY}"
          SEGGER_NUM_WORKERS="${SEGMENT_NUM_WORKERS}"
          segger predict
          -c "${last_ckpt}"
          -i "${INPUT_DIR}"
          -o "${seg_dir}"
        )
        if run_cmd "${log_file}" "${predict_cmd[@]}"; then
          if run_exports_for_job "${job_name}" "${seg_dir}" "${log_file}"; then
            status="recovered_predict_ok"
            note="predict_only_from_last_ckpt"
          else
            status="${LAST_EXPORT_STATUS}"
            note="predict_recovered_but_exports_failed"
          fi
        else
          status="recovered_predict_failed"
          note="predict_only_failed"
        fi
      else
        status="recovery_no_checkpoint"
        note="missing_seg_and_last_ckpt"
      fi
    fi

    end_ts="$(date +%s)"
    elapsed_s=$((end_ts - start_ts))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job_name}" \
      "${gpu}" \
      "${status}" \
      "${elapsed_s}" \
      "${note}" \
      "${seg_dir}" \
      "${log_file}" \
      >> "${out_file}"
  done
}

run_post_recovery_predict_only() {
  local recovery_file="${SUMMARY_DIR}/recovery.tsv"
  local pids=()
  local g gpu recovery_per_gpu

  # Single GPU: run sequentially with all indices.
  if [[ "${NUM_GPUS}" -eq 1 ]]; then
    local all_indices=()
    for g in $(seq 0 $((NUM_GPUS - 1))); do
      local -n arr="GPU_${g}_INDICES"
      all_indices+=("${arr[@]}")
    done
    recovery_per_gpu="${SUMMARY_DIR}/recovery.gpu${GPU_IDS[0]}.tsv"
    run_post_recovery_predict_only_group "${GPU_IDS[0]}" "${recovery_per_gpu}" "${all_indices[@]}"
    cp "${recovery_per_gpu}" "${recovery_file}"
    return
  fi

  # Multi-GPU: run recovery groups in parallel.
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    gpu="${GPU_IDS[$g]}"
    recovery_per_gpu="${SUMMARY_DIR}/recovery.gpu${gpu}.tsv"
    local -n arr="GPU_${g}_INDICES"
    if [[ "${#arr[@]}" -gt 0 ]]; then
      run_post_recovery_predict_only_group "${gpu}" "${recovery_per_gpu}" "${arr[@]}" &
      pids+=($!)
    fi
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  # Merge recovery files.
  local first=1
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    gpu="${GPU_IDS[$g]}"
    recovery_per_gpu="${SUMMARY_DIR}/recovery.gpu${gpu}.tsv"
    if [[ -f "${recovery_per_gpu}" ]]; then
      if [[ "${first}" -eq 1 ]]; then
        cat "${recovery_per_gpu}" > "${recovery_file}"
        first=0
      else
        tail -n +2 "${recovery_per_gpu}" >> "${recovery_file}"
      fi
    fi
  done
}

# =========================================================================
# Build jobs and distribute across GPUs
# =========================================================================

build_jobs

if [[ "${#JOB_SPECS[@]}" -eq 0 ]]; then
  echo "ERROR: No ablation jobs were generated. Check block toggles (RUN_LOSS_ABLATION, etc.)."
  exit 1
fi

# Create per-GPU index arrays with round-robin distribution.
for g in $(seq 0 $((NUM_GPUS - 1))); do
  declare -a "GPU_${g}_INDICES=()"
done

idx=0
for spec in "${JOB_SPECS[@]}"; do
  g=$((idx % NUM_GPUS))
  eval "GPU_${g}_INDICES+=(${idx})"
  idx=$((idx + 1))
done

# =========================================================================
# Write job plan TSV
# =========================================================================
{
  printf "job\tstudy_block\tgpu_group\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\tsg_loss_type\thidden_channels\tout_channels\ttx_weight_end\tbd_weight_end\tsg_weight_end\talignment_weight_end\tpositional_embeddings\tnormalize_embeddings\tcells_representation\tlearning_rate\n"
  for idx in "${!JOB_SPECS[@]}"; do
    local_group=$((idx % NUM_GPUS))
    IFS='|' read -r \
      job_name use_3d expansion tx_k tx_dist n_layers n_heads cells_min_counts min_qv \
      alignment_loss sg_loss_type hidden_channels out_channels \
      tx_weight_end bd_weight_end sg_weight_end alignment_weight_end \
      positional_embeddings normalize_embeddings cells_representation learning_rate \
      <<< "${JOB_SPECS[$idx]}"
    local_block="$(job_block "${job_name}")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job_name}" "${local_block}" "${local_group}" "${use_3d}" "${expansion}" "${tx_k}" "${tx_dist}" \
      "${n_layers}" "${n_heads}" "${cells_min_counts}" "${min_qv}" "${alignment_loss}" \
      "${sg_loss_type}" "${hidden_channels}" "${out_channels}" \
      "${tx_weight_end}" "${bd_weight_end}" "${sg_weight_end}" "${alignment_weight_end}" \
      "${positional_embeddings}" "${normalize_embeddings}" "${cells_representation}" "${learning_rate}"
  done
} > "${PLAN_FILE}"

echo "[$(timestamp)] Prepared ${#JOB_SPECS[@]} ablation jobs across ${NUM_GPUS} GPU(s)."
echo "[$(timestamp)] GPUs: ${GPU_IDS[*]}"
for g in $(seq 0 $((NUM_GPUS - 1))); do
  eval "_count=\${#GPU_${g}_INDICES[@]}"
  echo "[$(timestamp)] GPU ${GPU_IDS[$g]}: ${_count} jobs"
done
echo "[$(timestamp)] Job plan: ${PLAN_FILE}"
echo "[$(timestamp)] Logs: ${LOGS_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[$(timestamp)] DRY_RUN=1 — exiting without running jobs."
  echo ""
  echo "Job plan:"
  column -t -s $'\t' "${PLAN_FILE}" 2>/dev/null || cat "${PLAN_FILE}"
  exit 0
fi

# =========================================================================
# Launch GPU groups in parallel
# =========================================================================

PIDS=()
for g in $(seq 0 $((NUM_GPUS - 1))); do
  gpu="${GPU_IDS[$g]}"
  eval "_arr=(\"\${GPU_${g}_INDICES[@]}\")"
  if [[ "${#_arr[@]}" -gt 0 ]]; then
    run_gpu_group "${gpu}" "${_arr[@]}" &
    PIDS+=($!)
  fi
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

# =========================================================================
# Post-run recovery pass
# =========================================================================

echo "[$(timestamp)] Starting post-run predict-only recovery pass..."
run_post_recovery_predict_only

# =========================================================================
# Combine summaries
# =========================================================================

COMBINED_SUMMARY="${SUMMARY_DIR}/all_jobs.tsv"
if [[ -f "${SUMMARY_DIR}/recovery.tsv" ]]; then
  awk 'FNR==1 && NR!=1 {next} {print}' "${SUMMARY_DIR}"/gpu*.tsv "${SUMMARY_DIR}/recovery.tsv" > "${COMBINED_SUMMARY}"
  FAILED_COUNT=$(
    awk -F'\t' 'NR>1 && $3!="ok" && $3!="recovered_predict_ok" {c++} END{print c+0}' "${SUMMARY_DIR}/recovery.tsv"
  )
else
  awk 'FNR==1 && NR!=1 {next} {print}' "${SUMMARY_DIR}"/gpu*.tsv > "${COMBINED_SUMMARY}"
  FAILED_COUNT=$(
    awk -F'\t' 'NR>1 && $3!="ok" && $3!="skipped_existing" {c++} END{print c+0}' "${COMBINED_SUMMARY}"
  )
fi

echo "[$(timestamp)] Combined summary: ${COMBINED_SUMMARY}"
if [[ -f "${SUMMARY_DIR}/recovery.tsv" ]]; then
  echo "[$(timestamp)] Recovery summary: ${SUMMARY_DIR}/recovery.tsv"
fi

# =========================================================================
# Validation table
# =========================================================================

if [[ "${RUN_VALIDATION_TABLE}" == "1" ]]; then
  if [[ -f "${VALIDATION_SCRIPT}" ]]; then
    echo "[$(timestamp)] Building validation metrics table..."
    validation_log="${SUMMARY_DIR}/validation_metrics.log"
    validation_cmd=(
      bash "${VALIDATION_SCRIPT}"
      --root "${OUTPUT_ROOT}"
      --input-dir "${INPUT_DIR}"
      --include-default-10x "${VALIDATION_INCLUDE_DEFAULT_10X}"
    )
    # Pass first two GPU IDs for compatibility with validation script.
    if [[ "${NUM_GPUS}" -ge 2 ]]; then
      validation_cmd+=(--gpu-a "${GPU_IDS[0]}" --gpu-b "${GPU_IDS[1]}")
    else
      validation_cmd+=(--gpu-a "${GPU_IDS[0]}" --gpu-b "${GPU_IDS[0]}")
    fi
    if [[ -n "${ALIGNMENT_ME_GENE_PAIRS_PATH}" ]]; then
      validation_cmd+=(--me-gene-pairs-path "${ALIGNMENT_ME_GENE_PAIRS_PATH}")
    fi
    if [[ -n "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]]; then
      validation_cmd+=(
        --scrna-reference-path "${ALIGNMENT_SCRNA_REFERENCE_PATH}"
        --scrna-celltype-column "${ALIGNMENT_SCRNA_CELLTYPE_COLUMN}"
      )
    fi
    if "${validation_cmd[@]}" >> "${validation_log}" 2>&1; then
      echo "[$(timestamp)] Validation table updated: ${OUTPUT_ROOT}/summaries/validation_metrics.tsv"
    else
      echo "[$(timestamp)] WARN: validation table build failed (see ${validation_log})"
    fi
  else
    echo "[$(timestamp)] WARN: VALIDATION_SCRIPT not found: ${VALIDATION_SCRIPT}"
  fi
fi

echo "[$(timestamp)] Failed jobs: ${FAILED_COUNT}"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
  exit 1
fi
