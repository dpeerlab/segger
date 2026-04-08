#!/usr/bin/env bash
set -u -o pipefail

# -------------------------------------------------------------------------
# Segger focused Mossi feature ablation runner
# -------------------------------------------------------------------------
# This script runs a default-centric feature ablation for the Mossi pancreas
# dataset with one shared baseline and a small set of targeted variants.
#
# It reuses the same execution model as the focused CRC runner:
# - skip segmentation if segger_segmentation.parquet already exists
# - optionally force-refresh exports by removing only exports/<job>/
# - rebuild validation outputs and emit a compact delta summary
#
# "No multi-head attention" is approximated by n_heads=1 because the CLI
# currently enforces n_heads > 0.
#
# Usage:
#   bash scripts/run_mossi_feature_ablation.sh
#
# Optional overrides (environment variables):
#   INPUT_DIR=data/xe_pancreas_mossi/
#   OUTPUT_ROOT=./results/mossi_feature_ablation
#   NUM_GPUS=<auto>
#   N_EPOCHS=20
#   RESUME_IF_EXISTS=1
#   DRY_RUN=0
#   SEGMENT_TIMEOUT_MIN=90
#   SEGMENT_NUM_WORKERS=8
#   SEGMENT_ANC_RETRY_WORKERS=0
#   TORCH_SHARING_STRATEGY=file_system
#   FORCE_EXPORTS=1
#   RUN_VALIDATION_TABLE=1
#   VALIDATION_RECOMPUTE=1
#   BACKUP_EXISTING_PLAN=1
#   BACKUP_EXISTING_VALIDATION=1
#   VALIDATION_SCRIPT=scripts/build_benchmark_validation_table.sh
#   ALIGNMENT_ME_GENE_PAIRS_PATH=<optional>
#   ALIGNMENT_SCRNA_REFERENCE_PATH=data/ref_pancreas.h5ad
#   ALIGNMENT_SCRNA_CELLTYPE_COLUMN=cell_type
# -------------------------------------------------------------------------

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

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

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  for ((g = 0; g < NUM_GPUS; g++)); do
    GPU_IDS+=("${g}")
  done
fi

DEFAULT_INPUT_DIR="data/xe_pancreas_mossi/"
INPUT_DIR="${INPUT_DIR:-${DEFAULT_INPUT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/mossi_feature_ablation}"

if [[ "${INPUT_DIR}" == "${DEFAULT_INPUT_DIR}" ]] && \
   [[ ! -d "${INPUT_DIR}" ]] && \
   [[ -d "../data/xe_pancreas_mossi/" ]]; then
  INPUT_DIR="../data/xe_pancreas_mossi/"
fi

N_EPOCHS="${N_EPOCHS:-20}"
RESUME_IF_EXISTS="${RESUME_IF_EXISTS:-1}"
DRY_RUN="${DRY_RUN:-0}"
PREDICT_FALLBACK_ON_OOM="${PREDICT_FALLBACK_ON_OOM:-1}"
SEGMENT_TIMEOUT_MIN="${SEGMENT_TIMEOUT_MIN:-90}"
SEGMENT_TIMEOUT_SEC=$((SEGMENT_TIMEOUT_MIN * 60))
SEGMENT_NUM_WORKERS="${SEGMENT_NUM_WORKERS:-8}"
SEGMENT_ANC_RETRY_WORKERS="${SEGMENT_ANC_RETRY_WORKERS:-0}"
TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"
FORCE_EXPORTS="${FORCE_EXPORTS:-1}"
RUN_VALIDATION_TABLE="${RUN_VALIDATION_TABLE:-1}"
VALIDATION_RECOMPUTE="${VALIDATION_RECOMPUTE:-1}"
BACKUP_EXISTING_PLAN="${BACKUP_EXISTING_PLAN:-1}"
BACKUP_EXISTING_VALIDATION="${BACKUP_EXISTING_VALIDATION:-1}"
VALIDATION_INCLUDE_DEFAULT_10X="${VALIDATION_INCLUDE_DEFAULT_10X:-true}"

BOUNDARY_METHOD="${BOUNDARY_METHOD:-convex_hull}"
BOUNDARY_VOXEL_SIZE="${BOUNDARY_VOXEL_SIZE:-5}"
XENIUM_NUM_WORKERS="${XENIUM_NUM_WORKERS:-8}"

ALIGNMENT_LOSS_WEIGHT_START="${ALIGNMENT_LOSS_WEIGHT_START:-0.0}"
ALIGNMENT_ME_GENE_PAIRS_PATH="${ALIGNMENT_ME_GENE_PAIRS_PATH:-}"
ALIGNMENT_SCRNA_REFERENCE_PATH="${ALIGNMENT_SCRNA_REFERENCE_PATH:-data/ref_pancreas.h5ad}"
ALIGNMENT_SCRNA_CELLTYPE_COLUMN="${ALIGNMENT_SCRNA_CELLTYPE_COLUMN:-cell_type}"

if [[ "${ALIGNMENT_SCRNA_REFERENCE_PATH}" == "data/ref_pancreas.h5ad" ]] && \
   [[ ! -f "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]] && \
   [[ -f "../data/ref_pancreas.h5ad" ]]; then
  ALIGNMENT_SCRNA_REFERENCE_PATH="../data/ref_pancreas.h5ad"
fi

RUNS_DIR="${OUTPUT_ROOT}/runs"
EXPORTS_DIR="${OUTPUT_ROOT}/exports"
LOGS_DIR="${OUTPUT_ROOT}/logs"
SUMMARY_DIR="${OUTPUT_ROOT}/summaries"
PLAN_FILE="${OUTPUT_ROOT}/job_plan.tsv"
DELTA_FILE="${SUMMARY_DIR}/mossi_feature_ablation_deltas.tsv"
VALIDATION_FILE="${SUMMARY_DIR}/validation_metrics.tsv"
VALIDATION_SCRIPT="${VALIDATION_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_benchmark_validation_table.sh}"
BACKUP_STAMP="$(date '+%Y%m%d_%H%M%S')"

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

if [[ -z "${ALIGNMENT_ME_GENE_PAIRS_PATH}" ]] && [[ -z "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]]; then
  echo "ERROR: Alignment ablations require ALIGNMENT_ME_GENE_PAIRS_PATH or ALIGNMENT_SCRNA_REFERENCE_PATH."
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

# -------------------------------------------------------------------------
# Hard-coded baseline configuration for the Mossi pancreas default run
# -------------------------------------------------------------------------
# This matches the existing parameter benchmark baseline.
BASE_USE_3D="true"
BASE_PREDICTION_MODE="nucleus"
BASE_EXPANSION="2.2"
BASE_TX_K="5"
BASE_TX_DIST="5"
BASE_N_LAYERS="2"
BASE_N_HEADS="2"
BASE_CELLS_MIN="5"
BASE_MIN_QV="0"
BASE_ALIGNMENT="true"
BASE_SG_LOSS="triplet"
BASE_HIDDEN="64"
BASE_OUT="64"
BASE_TX_WEIGHT="1.0"
BASE_BD_WEIGHT="1.0"
BASE_SG_WEIGHT="0.5"
BASE_ALIGN_WEIGHT="0.03"
BASE_POS_EMB="true"
BASE_NORM_EMB="true"
BASE_CELLS_REP="pca"
BASE_LR="1e-3"
BASE_FRAGMENT_MODE="false"

echo "[$(timestamp)] Expansion mode: scale_factor"
echo "[$(timestamp)] Baseline control: baseline"
echo "[$(timestamp)] Baseline scale=${BASE_EXPANSION}"

JOB_SPECS=()

add_job() {
  local job_name="$1"
  local study_block="$2"
  local use_3d="${3:-${BASE_USE_3D}}"
  local expansion="${4:-${BASE_EXPANSION}}"
  local tx_k="${5:-${BASE_TX_K}}"
  local tx_dist="${6:-${BASE_TX_DIST}}"
  local n_layers="${7:-${BASE_N_LAYERS}}"
  local n_heads="${8:-${BASE_N_HEADS}}"
  local cells_min="${9:-${BASE_CELLS_MIN}}"
  local min_qv="${10:-${BASE_MIN_QV}}"
  local alignment_loss="${11:-${BASE_ALIGNMENT}}"
  local sg_loss_type="${12:-${BASE_SG_LOSS}}"
  local hidden_channels="${13:-${BASE_HIDDEN}}"
  local out_channels="${14:-${BASE_OUT}}"
  local tx_weight_end="${15:-${BASE_TX_WEIGHT}}"
  local bd_weight_end="${16:-${BASE_BD_WEIGHT}}"
  local sg_weight_end="${17:-${BASE_SG_WEIGHT}}"
  local alignment_weight_end="${18:-${BASE_ALIGN_WEIGHT}}"
  local positional_embeddings="${19:-${BASE_POS_EMB}}"
  local normalize_embeddings="${20:-${BASE_NORM_EMB}}"
  local cells_representation="${21:-${BASE_CELLS_REP}}"
  local learning_rate="${22:-${BASE_LR}}"
  local prediction_mode="${23:-${BASE_PREDICTION_MODE}}"
  local fragment_mode="${24:-${BASE_FRAGMENT_MODE}}"

  JOB_SPECS+=("${job_name}|${study_block}|${use_3d}|${expansion}|${tx_k}|${tx_dist}|${n_layers}|${n_heads}|${cells_min}|${min_qv}|${alignment_loss}|${sg_loss_type}|${hidden_channels}|${out_channels}|${tx_weight_end}|${bd_weight_end}|${sg_weight_end}|${alignment_weight_end}|${positional_embeddings}|${normalize_embeddings}|${cells_representation}|${learning_rate}|${prediction_mode}|${fragment_mode}")
}

build_jobs() {
  add_job "baseline" "baseline"
  add_job "align_w0" "alignment" \
    "${BASE_USE_3D}" "${BASE_EXPANSION}" "${BASE_TX_K}" "${BASE_TX_DIST}" \
    "${BASE_N_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN}" "${BASE_MIN_QV}" \
    "false" "${BASE_SG_LOSS}" "${BASE_HIDDEN}" "${BASE_OUT}" \
    "${BASE_TX_WEIGHT}" "${BASE_BD_WEIGHT}" "${BASE_SG_WEIGHT}" "0" \
    "${BASE_POS_EMB}" "${BASE_NORM_EMB}" "${BASE_CELLS_REP}" "${BASE_LR}" \
    "${BASE_PREDICTION_MODE}" "${BASE_FRAGMENT_MODE}"
  add_job "align_w001" "alignment" \
    "${BASE_USE_3D}" "${BASE_EXPANSION}" "${BASE_TX_K}" "${BASE_TX_DIST}" \
    "${BASE_N_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN}" "${BASE_MIN_QV}" \
    "true" "${BASE_SG_LOSS}" "${BASE_HIDDEN}" "${BASE_OUT}" \
    "${BASE_TX_WEIGHT}" "${BASE_BD_WEIGHT}" "${BASE_SG_WEIGHT}" "0.01" \
    "${BASE_POS_EMB}" "${BASE_NORM_EMB}" "${BASE_CELLS_REP}" "${BASE_LR}" \
    "${BASE_PREDICTION_MODE}" "${BASE_FRAGMENT_MODE}"
  add_job "align_w01" "alignment" \
    "${BASE_USE_3D}" "${BASE_EXPANSION}" "${BASE_TX_K}" "${BASE_TX_DIST}" \
    "${BASE_N_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN}" "${BASE_MIN_QV}" \
    "true" "${BASE_SG_LOSS}" "${BASE_HIDDEN}" "${BASE_OUT}" \
    "${BASE_TX_WEIGHT}" "${BASE_BD_WEIGHT}" "${BASE_SG_WEIGHT}" "0.1" \
    "${BASE_POS_EMB}" "${BASE_NORM_EMB}" "${BASE_CELLS_REP}" "${BASE_LR}" \
    "${BASE_PREDICTION_MODE}" "${BASE_FRAGMENT_MODE}"
  add_job "frag_on" "fragments" \
    "${BASE_USE_3D}" "${BASE_EXPANSION}" "${BASE_TX_K}" "${BASE_TX_DIST}" \
    "${BASE_N_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN}" "${BASE_MIN_QV}" \
    "${BASE_ALIGNMENT}" "${BASE_SG_LOSS}" "${BASE_HIDDEN}" "${BASE_OUT}" \
    "${BASE_TX_WEIGHT}" "${BASE_BD_WEIGHT}" "${BASE_SG_WEIGHT}" "${BASE_ALIGN_WEIGHT}" \
    "${BASE_POS_EMB}" "${BASE_NORM_EMB}" "${BASE_CELLS_REP}" "${BASE_LR}" \
    "${BASE_PREDICTION_MODE}" "true"
  add_job "heads_1" "attention_proxy" \
    "${BASE_USE_3D}" "${BASE_EXPANSION}" "${BASE_TX_K}" "${BASE_TX_DIST}" \
    "${BASE_N_LAYERS}" "1" "${BASE_CELLS_MIN}" "${BASE_MIN_QV}" \
    "${BASE_ALIGNMENT}" "${BASE_SG_LOSS}" "${BASE_HIDDEN}" "${BASE_OUT}" \
    "${BASE_TX_WEIGHT}" "${BASE_BD_WEIGHT}" "${BASE_SG_WEIGHT}" "${BASE_ALIGN_WEIGHT}" \
    "${BASE_POS_EMB}" "${BASE_NORM_EMB}" "${BASE_CELLS_REP}" "${BASE_LR}" \
    "${BASE_PREDICTION_MODE}" "${BASE_FRAGMENT_MODE}"
  add_job "use3d_false" "use_3d" \
    "false" "${BASE_EXPANSION}" "${BASE_TX_K}" "${BASE_TX_DIST}" \
    "${BASE_N_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN}" "${BASE_MIN_QV}" \
    "${BASE_ALIGNMENT}" "${BASE_SG_LOSS}" "${BASE_HIDDEN}" "${BASE_OUT}" \
    "${BASE_TX_WEIGHT}" "${BASE_BD_WEIGHT}" "${BASE_SG_WEIGHT}" "${BASE_ALIGN_WEIGHT}" \
    "${BASE_POS_EMB}" "${BASE_NORM_EMB}" "${BASE_CELLS_REP}" "${BASE_LR}" \
    "${BASE_PREDICTION_MODE}" "${BASE_FRAGMENT_MODE}"
  add_job "scale_1p5" "scale_factor" \
    "${BASE_USE_3D}" "1.5" "${BASE_TX_K}" "${BASE_TX_DIST}" \
    "${BASE_N_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN}" "${BASE_MIN_QV}" \
    "${BASE_ALIGNMENT}" "${BASE_SG_LOSS}" "${BASE_HIDDEN}" "${BASE_OUT}" \
    "${BASE_TX_WEIGHT}" "${BASE_BD_WEIGHT}" "${BASE_SG_WEIGHT}" "${BASE_ALIGN_WEIGHT}" \
    "${BASE_POS_EMB}" "${BASE_NORM_EMB}" "${BASE_CELLS_REP}" "${BASE_LR}" \
    "${BASE_PREDICTION_MODE}" "${BASE_FRAGMENT_MODE}"
  add_job "scale_3p0" "scale_factor" \
    "${BASE_USE_3D}" "3.0" "${BASE_TX_K}" "${BASE_TX_DIST}" \
    "${BASE_N_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN}" "${BASE_MIN_QV}" \
    "${BASE_ALIGNMENT}" "${BASE_SG_LOSS}" "${BASE_HIDDEN}" "${BASE_OUT}" \
    "${BASE_TX_WEIGHT}" "${BASE_BD_WEIGHT}" "${BASE_SG_WEIGHT}" "${BASE_ALIGN_WEIGHT}" \
    "${BASE_POS_EMB}" "${BASE_NORM_EMB}" "${BASE_CELLS_REP}" "${BASE_LR}" \
    "${BASE_PREDICTION_MODE}" "${BASE_FRAGMENT_MODE}"
}

SPEC_JOB_NAME=""
SPEC_STUDY_BLOCK=""
SPEC_USE_3D=""
SPEC_EXPANSION=""
SPEC_TX_K=""
SPEC_TX_DIST=""
SPEC_N_LAYERS=""
SPEC_N_HEADS=""
SPEC_CELLS_MIN=""
SPEC_MIN_QV=""
SPEC_ALIGNMENT=""
SPEC_SG_LOSS=""
SPEC_HIDDEN=""
SPEC_OUT=""
SPEC_TX_W=""
SPEC_BD_W=""
SPEC_SG_W=""
SPEC_ALIGN_W=""
SPEC_POS_EMB=""
SPEC_NORM_EMB=""
SPEC_CELLS_REP=""
SPEC_LR=""
SPEC_PRED_MODE=""
SPEC_FRAGMENT_MODE=""

parse_spec() {
  local spec="$1"
  IFS='|' read -r \
    SPEC_JOB_NAME SPEC_STUDY_BLOCK SPEC_USE_3D SPEC_EXPANSION SPEC_TX_K SPEC_TX_DIST \
    SPEC_N_LAYERS SPEC_N_HEADS SPEC_CELLS_MIN SPEC_MIN_QV SPEC_ALIGNMENT \
    SPEC_SG_LOSS SPEC_HIDDEN SPEC_OUT SPEC_TX_W SPEC_BD_W SPEC_SG_W SPEC_ALIGN_W \
    SPEC_POS_EMB SPEC_NORM_EMB SPEC_CELLS_REP SPEC_LR SPEC_PRED_MODE SPEC_FRAGMENT_MODE \
    <<< "${spec}"
}

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
  local pattern="out of memory|cuda error: out of memory|cublas status alloc failed|cuda driver error.*memory"
  if [[ ! -f "${log_file}" ]]; then
    return 1
  fi
  if command -v rg >/dev/null 2>&1; then
    rg -qi "${pattern}" "${log_file}"
  else
    grep -Eiq "${pattern}" "${log_file}"
  fi
}

is_ancdata_failure() {
  local log_file="$1"
  local pattern="received [0-9]+ items of ancdata|multiprocessing/resource_sharer\\.py"
  if [[ ! -f "${log_file}" ]]; then
    return 1
  fi
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
  local export_root="${EXPORTS_DIR}/${job_name}"
  local seg_file="${seg_dir}/segger_segmentation.parquet"
  local anndata_dir="${export_root}/anndata"
  local xenium_dir="${export_root}/xenium_explorer"

  if [[ ! -f "${seg_file}" ]] && [[ "${DRY_RUN}" != "1" ]]; then
    LAST_EXPORT_STATUS="missing_segmentation"
    return 1
  fi

  if [[ "${FORCE_EXPORTS}" == "1" ]] && [[ -d "${export_root}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[$(timestamp)] DRY_RUN: would remove export dir ${export_root}" >> "${log_file}"
    else
      echo "[$(timestamp)] FORCE_EXPORTS=1 removing export dir: ${export_root}" >> "${log_file}"
      rm -rf "${export_root}"
    fi
  fi

  mkdir -p "${anndata_dir}" "${xenium_dir}"

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

  LAST_EXPORT_STATUS="ok"
  return 0
}

job_outputs_complete() {
  local spec="$1"
  parse_spec "${spec}"
  [[ -f "${RUNS_DIR}/${SPEC_JOB_NAME}/segger_segmentation.parquet" ]] || return 1
  [[ -f "${EXPORTS_DIR}/${SPEC_JOB_NAME}/anndata/segger_segmentation.h5ad" ]] || return 1
  [[ -f "${EXPORTS_DIR}/${SPEC_JOB_NAME}/xenium_explorer/seg_experiment.xenium" ]] || return 1
  return 0
}

write_skipped_existing_summary() {
  local out_file="$1"
  shift
  local -a indices=("$@")
  local idx spec

  if [[ "${#indices[@]}" -eq 0 ]]; then
    rm -f "${out_file}"
    return 0
  fi

  printf "job\tgpu\tstatus\telapsed_s\tseg_dir\tlog_file\n" > "${out_file}"
  for idx in "${indices[@]}"; do
    spec="${JOB_SPECS[$idx]}"
    parse_spec "${spec}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${SPEC_JOB_NAME}" \
      "-" \
      "skipped_existing" \
      "0" \
      "${RUNS_DIR}/${SPEC_JOB_NAME}" \
      "${LOGS_DIR}/${SPEC_JOB_NAME}.gpu?.log" \
      >> "${out_file}"
  done
}

run_predict_from_checkpoint_for_spec() {
  local gpu="$1"
  local checkpoint_path="$2"
  local seg_dir="$3"
  local log_file="$4"
  local fragment_flag="--no-fragment-mode"
  local -a predict_cmd

  if [[ "${SPEC_FRAGMENT_MODE}" == "true" ]]; then
    fragment_flag="--fragment-mode"
  fi

  predict_cmd=(
    env CUDA_VISIBLE_DEVICES="${gpu}"
    PYTORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY}"
    SEGGER_NUM_WORKERS="${SEGMENT_NUM_WORKERS}"
    segger predict
    -c "${checkpoint_path}"
    -i "${INPUT_DIR}"
    -o "${seg_dir}"
    --use-3d "${SPEC_USE_3D}"
    "${fragment_flag}"
  )

  run_cmd "${log_file}" "${predict_cmd[@]}"
}

LAST_JOB_STATUS="unknown"

run_job() {
  local gpu="$1"
  local spec="$2"

  parse_spec "${spec}"

  local seg_dir="${RUNS_DIR}/${SPEC_JOB_NAME}"
  local seg_file="${seg_dir}/segger_segmentation.parquet"
  local anndata_file="${EXPORTS_DIR}/${SPEC_JOB_NAME}/anndata/segger_segmentation.h5ad"
  local xenium_file="${EXPORTS_DIR}/${SPEC_JOB_NAME}/xenium_explorer/seg_experiment.xenium"
  local log_file="${LOGS_DIR}/${SPEC_JOB_NAME}.gpu${gpu}.log"

  mkdir -p "${seg_dir}"

  {
    echo "=================================================================="
    echo "[$(timestamp)] START job=${SPEC_JOB_NAME} gpu=${gpu}"
    echo "params: block=${SPEC_STUDY_BLOCK} use3d=${SPEC_USE_3D} expansion_scale=${SPEC_EXPANSION} tx_k=${SPEC_TX_K} tx_dist=${SPEC_TX_DIST} layers=${SPEC_N_LAYERS} heads=${SPEC_N_HEADS} cells_min=${SPEC_CELLS_MIN} min_qv=${SPEC_MIN_QV} align=${SPEC_ALIGNMENT} sg_loss=${SPEC_SG_LOSS} hidden=${SPEC_HIDDEN} out=${SPEC_OUT} tx_w=${SPEC_TX_W} bd_w=${SPEC_BD_W} sg_w=${SPEC_SG_W} align_w=${SPEC_ALIGN_W} pos_emb=${SPEC_POS_EMB} norm_emb=${SPEC_NORM_EMB} cells_rep=${SPEC_CELLS_REP} lr=${SPEC_LR} pred_mode=${SPEC_PRED_MODE} fragment_mode=${SPEC_FRAGMENT_MODE} timeout_min=${SEGMENT_TIMEOUT_MIN}"
  } | tee -a "${log_file}" >/dev/null

  if [[ "${FORCE_EXPORTS}" != "1" ]] && \
     [[ "${RESUME_IF_EXISTS}" == "1" ]] && \
     [[ -f "${seg_file}" ]] && \
     [[ -f "${anndata_file}" ]] && \
     [[ -f "${xenium_file}" ]]; then
    echo "[$(timestamp)] SKIP job=${SPEC_JOB_NAME} (all outputs already present)" | tee -a "${log_file}" >/dev/null
    LAST_JOB_STATUS="skipped_existing"
    return 0
  fi

  if [[ -f "${seg_file}" ]]; then
    echo "[$(timestamp)] SKIP segmentation (existing): ${seg_file}" | tee -a "${log_file}" >/dev/null
  else
    local pos_flag="--use-positional-embeddings"
    local norm_flag="--normalize-embeddings"
    local fragment_flag="--no-fragment-mode"
    local -a seg_cmd
    local seg_rc
    local last_ckpt

    if [[ "${SPEC_POS_EMB}" == "false" ]]; then
      pos_flag="--no-use-positional-embeddings"
    fi
    if [[ "${SPEC_NORM_EMB}" == "false" ]]; then
      norm_flag="--no-normalize-embeddings"
    fi
    if [[ "${SPEC_FRAGMENT_MODE}" == "true" ]]; then
      fragment_flag="--fragment-mode"
    fi

    seg_cmd=(
      env CUDA_VISIBLE_DEVICES="${gpu}"
      PYTORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY}"
      SEGGER_NUM_WORKERS="${SEGMENT_NUM_WORKERS}"
      segger segment
      -i "${INPUT_DIR}"
      -o "${seg_dir}"
      --n-epochs "${N_EPOCHS}"
      --prediction-mode "${SPEC_PRED_MODE}"
      --prediction-scale-factor "${SPEC_EXPANSION}"
      --cells-min-counts "${SPEC_CELLS_MIN}"
      --min-qv "${SPEC_MIN_QV}"
      --use-3d "${SPEC_USE_3D}"
      --transcripts-max-k "${SPEC_TX_K}"
      --transcripts-max-dist "${SPEC_TX_DIST}"
      --n-mid-layers "${SPEC_N_LAYERS}"
      --n-heads "${SPEC_N_HEADS}"
      --segmentation-loss "${SPEC_SG_LOSS}"
      --hidden-channels "${SPEC_HIDDEN}"
      --out-channels "${SPEC_OUT}"
      --transcripts-loss-weight-end "${SPEC_TX_W}"
      --cells-loss-weight-end "${SPEC_BD_W}"
      --segmentation-loss-weight-end "${SPEC_SG_W}"
      --learning-rate "${SPEC_LR}"
      --cells-representation "${SPEC_CELLS_REP}"
      "${pos_flag}"
      "${norm_flag}"
      "${fragment_flag}"
    )

    if [[ "${SPEC_ALIGNMENT}" == "true" ]]; then
      seg_cmd+=(
        --alignment-loss
        --alignment-loss-weight-start "${ALIGNMENT_LOSS_WEIGHT_START}"
        --alignment-loss-weight-end "${SPEC_ALIGN_W}"
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
    seg_rc=$?

    if [[ "${seg_rc}" -ne 0 ]] && \
       [[ "${seg_rc}" -ne 124 ]] && \
       [[ "${SEGMENT_ANC_RETRY_WORKERS}" != "${SEGMENT_NUM_WORKERS}" ]] && \
       is_ancdata_failure "${log_file}"; then
      echo "[$(timestamp)] WARN job=${SPEC_JOB_NAME} segment failed with ancdata; retrying with SEGGER_NUM_WORKERS=${SEGMENT_ANC_RETRY_WORKERS}" | tee -a "${log_file}" >/dev/null
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
        echo "[$(timestamp)] OK job=${SPEC_JOB_NAME} segment retry succeeded with SEGGER_NUM_WORKERS=${SEGMENT_ANC_RETRY_WORKERS}" | tee -a "${log_file}" >/dev/null
      fi
    fi

    if [[ "${seg_rc}" -eq 124 ]]; then
      echo "[$(timestamp)] FAIL job=${SPEC_JOB_NAME} step=segment (OOT ${SEGMENT_TIMEOUT_MIN}m)" | tee -a "${log_file}" >/dev/null
      LAST_JOB_STATUS="segment_oot"
      return 1
    fi

    if [[ "${seg_rc}" -ne 0 ]]; then
      last_ckpt="${seg_dir}/checkpoints/last.ckpt"
      if [[ "${PREDICT_FALLBACK_ON_OOM}" == "1" ]] && is_oom_failure "${log_file}" && [[ -f "${last_ckpt}" ]]; then
        echo "[$(timestamp)] WARN job=${SPEC_JOB_NAME} segment OOM; trying checkpoint predict fallback (${last_ckpt})" | tee -a "${log_file}" >/dev/null
        if run_predict_from_checkpoint_for_spec "${gpu}" "${last_ckpt}" "${seg_dir}" "${log_file}"; then
          echo "[$(timestamp)] OK job=${SPEC_JOB_NAME} predict fallback succeeded after OOM" | tee -a "${log_file}" >/dev/null
        else
          echo "[$(timestamp)] FAIL job=${SPEC_JOB_NAME} step=predict_fallback_after_oom" | tee -a "${log_file}" >/dev/null
          LAST_JOB_STATUS="predict_fallback_failed"
          return 1
        fi
      else
        if is_ancdata_failure "${log_file}"; then
          echo "[$(timestamp)] FAIL job=${SPEC_JOB_NAME} step=segment (ancdata)" | tee -a "${log_file}" >/dev/null
          LAST_JOB_STATUS="segment_ancdata"
        elif is_oom_failure "${log_file}"; then
          echo "[$(timestamp)] FAIL job=${SPEC_JOB_NAME} step=segment (oom)" | tee -a "${log_file}" >/dev/null
          LAST_JOB_STATUS="segment_oom"
        else
          echo "[$(timestamp)] FAIL job=${SPEC_JOB_NAME} step=segment" | tee -a "${log_file}" >/dev/null
          LAST_JOB_STATUS="segment_failed"
        fi
        return 1
      fi
    fi
  fi

  if ! run_exports_for_job "${SPEC_JOB_NAME}" "${seg_dir}" "${log_file}"; then
    echo "[$(timestamp)] FAIL job=${SPEC_JOB_NAME} step=${LAST_EXPORT_STATUS}" | tee -a "${log_file}" >/dev/null
    LAST_JOB_STATUS="${LAST_EXPORT_STATUS}"
    return 1
  fi

  echo "[$(timestamp)] DONE job=${SPEC_JOB_NAME}" | tee -a "${log_file}" >/dev/null
  LAST_JOB_STATUS="ok"
  return 0
}

run_gpu_group() {
  local gpu="$1"
  shift
  local -a indices=("$@")
  local summary_file="${SUMMARY_DIR}/gpu${gpu}.tsv"
  local idx spec start_ts end_ts elapsed_s

  printf "job\tgpu\tstatus\telapsed_s\tseg_dir\tlog_file\n" > "${summary_file}"

  for idx in "${indices[@]}"; do
    spec="${JOB_SPECS[$idx]}"
    parse_spec "${spec}"

    start_ts="$(date +%s)"
    run_job "${gpu}" "${spec}"
    end_ts="$(date +%s)"
    elapsed_s=$((end_ts - start_ts))

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${SPEC_JOB_NAME}" \
      "${gpu}" \
      "${LAST_JOB_STATUS}" \
      "${elapsed_s}" \
      "${RUNS_DIR}/${SPEC_JOB_NAME}" \
      "${LOGS_DIR}/${SPEC_JOB_NAME}.gpu${gpu}.log" \
      >> "${summary_file}"
  done
}

run_post_recovery_predict_only_group() {
  local gpu="$1"
  local out_file="$2"
  shift 2
  local -a indices=("$@")
  local idx spec
  local seg_dir seg_file last_ckpt log_file note status
  local start_ts end_ts elapsed_s

  printf "job\tgpu\tstatus\telapsed_s\tnote\tseg_dir\tlog_file\n" > "${out_file}"

  for idx in "${indices[@]}"; do
    spec="${JOB_SPECS[$idx]}"
    parse_spec "${spec}"

    seg_dir="${RUNS_DIR}/${SPEC_JOB_NAME}"
    seg_file="${seg_dir}/segger_segmentation.parquet"
    last_ckpt="${seg_dir}/checkpoints/last.ckpt"
    log_file="${LOGS_DIR}/${SPEC_JOB_NAME}.gpu${gpu}.log"
    note=""
    status="ok"

    start_ts="$(date +%s)"

    if [[ -f "${seg_file}" ]]; then
      note="segmentation_exists"
      if ! run_exports_for_job "${SPEC_JOB_NAME}" "${seg_dir}" "${log_file}"; then
        status="${LAST_EXPORT_STATUS}"
        note="exports_failed_after_existing_seg"
      fi
    elif [[ -f "${last_ckpt}" ]]; then
      echo "[$(timestamp)] RECOVERY job=${SPEC_JOB_NAME}: running predict-only from ${last_ckpt}" | tee -a "${log_file}" >/dev/null
      if run_predict_from_checkpoint_for_spec "${gpu}" "${last_ckpt}" "${seg_dir}" "${log_file}"; then
        if run_exports_for_job "${SPEC_JOB_NAME}" "${seg_dir}" "${log_file}"; then
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

    end_ts="$(date +%s)"
    elapsed_s=$((end_ts - start_ts))

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${SPEC_JOB_NAME}" \
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
  local recovery_per_gpu
  local pids=()
  local g gpu first

  if [[ "${NUM_GPUS}" -eq 1 ]]; then
    local all_indices=()
    for g in $(seq 0 $((NUM_GPUS - 1))); do
      local -n arr="GPU_${g}_INDICES"
      all_indices+=("${arr[@]}")
    done
    recovery_per_gpu="${SUMMARY_DIR}/recovery.gpu${GPU_IDS[0]}.tsv"
    run_post_recovery_predict_only_group "${GPU_IDS[0]}" "${recovery_per_gpu}" "${all_indices[@]}"
    cp "${recovery_per_gpu}" "${recovery_file}"
    return 0
  fi

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

  first=1
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

build_delta_summary() {
  local validation_tsv="$1"
  local out_file="$2"

  if [[ ! -f "${validation_tsv}" ]]; then
    echo "[$(timestamp)] WARN: validation TSV not found for delta summary: ${validation_tsv}"
    return 1
  fi

  if ! awk -F'\t' -v OFS='\t' '
    function is_num(v) {
      return v ~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/
    }
    function delta_value(v, base) {
      if (is_num(v) && is_num(base)) {
        return sprintf("%.6f", v - base)
      }
      return "nan"
    }
    function emit(block, job,   a, f, m, r, t, d, s) {
      if (job in seen) {
        a = assigned[job]
        f = fragments[job]
        m = mecr[job]
        r = resolvi[job]
        t = tco[job]
        d = doublet[job]
        s = status[job]
      } else {
        a = "nan"
        f = "nan"
        m = "nan"
        r = "nan"
        t = "nan"
        d = "nan"
        s = "missing_validation_row"
      }
      print block, job, \
        a, delta_value(a, base_assigned), \
        f, delta_value(f, base_fragments), \
        m, delta_value(m, base_mecr), \
        r, delta_value(r, base_resolvi), \
        t, delta_value(t, base_tco), \
        d, delta_value(d, base_doublet), \
        s
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        col[$i] = i
      }
      next
    }
    {
      job = $1
      seen[job] = 1
      assigned[job] = $(col["assigned_pct"])
      fragments[job] = $(col["fragments"])
      mecr[job] = $(col["mecr"])
      resolvi[job] = $(col["contamination_pct"])
      tco[job] = $(col["tco"])
      doublet[job] = $(col["doublet_pct"])
      status[job] = $(col["validate_status"])
    }
    END {
      base = "baseline"
      if (!(base in seen)) {
        exit 2
      }
      base_assigned = assigned[base]
      base_fragments = fragments[base]
      base_mecr = mecr[base]
      base_resolvi = resolvi[base]
      base_tco = tco[base]
      base_doublet = doublet[base]

      print "block", "job", \
        "assigned_pct", "assigned_delta_vs_base", \
        "fragments", "fragments_delta_vs_base", \
        "mecr", "mecr_delta_vs_base", \
        "contamination_pct", "resolvi_delta_vs_base", \
        "tco", "tco_delta_vs_base", \
        "doublet_pct", "doublet_delta_vs_base", \
        "validate_status"

      emit("alignment", "align_w0")
      emit("alignment", "align_w001")
      emit("alignment", "align_w01")
      emit("fragments", "frag_on")
      emit("attention_proxy", "heads_1")
      emit("use_3d", "use3d_false")
      emit("scale_factor", "scale_1p5")
      emit("scale_factor", "scale_3p0")
    }
  ' "${validation_tsv}" > "${out_file}"; then
    echo "[$(timestamp)] WARN: failed to build delta summary from ${validation_tsv}"
    return 1
  fi

  echo "[$(timestamp)] Delta summary updated: ${out_file}"
  return 0
}

build_jobs

if [[ "${#JOB_SPECS[@]}" -ne 9 ]]; then
  echo "ERROR: Expected 9 jobs, got ${#JOB_SPECS[@]}"
  exit 1
fi

for g in $(seq 0 $((NUM_GPUS - 1))); do
  declare -a "GPU_${g}_INDICES=()"
done

SKIPPED_EXISTING_FILE="${SUMMARY_DIR}/skipped_existing.tsv"
declare -a PENDING_JOB_INDICES=()
declare -a SKIPPED_JOB_INDICES=()

for idx in "${!JOB_SPECS[@]}"; do
  spec="${JOB_SPECS[$idx]}"
  if [[ "${FORCE_EXPORTS}" != "1" ]] && \
     [[ "${RESUME_IF_EXISTS}" == "1" ]] && \
     job_outputs_complete "${spec}"; then
    SKIPPED_JOB_INDICES+=("${idx}")
  else
    PENDING_JOB_INDICES+=("${idx}")
  fi
done

if [[ "${#SKIPPED_JOB_INDICES[@]}" -gt 0 ]]; then
  write_skipped_existing_summary "${SKIPPED_EXISTING_FILE}" "${SKIPPED_JOB_INDICES[@]}"
else
  rm -f "${SKIPPED_EXISTING_FILE}"
fi

idx=0
for pending_idx in "${PENDING_JOB_INDICES[@]}"; do
  g=$((idx % NUM_GPUS))
  eval "GPU_${g}_INDICES+=(${pending_idx})"
  idx=$((idx + 1))
done

if [[ "${BACKUP_EXISTING_PLAN}" == "1" ]] && [[ -f "${PLAN_FILE}" ]]; then
  PLAN_BACKUP="${OUTPUT_ROOT}/job_plan.pre_mossi_feature_ablation.${BACKUP_STAMP}.tsv"
  cp -f "${PLAN_FILE}" "${PLAN_BACKUP}"
  echo "[$(timestamp)] Backed up prior job plan: ${PLAN_BACKUP}"
fi

{
  printf "job\tstudy_block\tgpu_group\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\tsg_loss_type\thidden_channels\tout_channels\ttx_weight_end\tbd_weight_end\tsg_weight_end\talignment_weight_end\tpositional_embeddings\tnormalize_embeddings\tcells_representation\tlearning_rate\tprediction_mode\tfragment_mode\n"
  for idx in "${!JOB_SPECS[@]}"; do
    spec="${JOB_SPECS[$idx]}"
    parse_spec "${spec}"
    gpu_group=$((idx % NUM_GPUS))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${SPEC_JOB_NAME}" "${SPEC_STUDY_BLOCK}" "${gpu_group}" "${SPEC_USE_3D}" "${SPEC_EXPANSION}" \
      "${SPEC_TX_K}" "${SPEC_TX_DIST}" "${SPEC_N_LAYERS}" "${SPEC_N_HEADS}" "${SPEC_CELLS_MIN}" \
      "${SPEC_MIN_QV}" "${SPEC_ALIGNMENT}" "${SPEC_SG_LOSS}" "${SPEC_HIDDEN}" "${SPEC_OUT}" \
      "${SPEC_TX_W}" "${SPEC_BD_W}" "${SPEC_SG_W}" "${SPEC_ALIGN_W}" "${SPEC_POS_EMB}" \
      "${SPEC_NORM_EMB}" "${SPEC_CELLS_REP}" "${SPEC_LR}" "${SPEC_PRED_MODE}" "${SPEC_FRAGMENT_MODE}"
  done
} > "${PLAN_FILE}"

echo "[$(timestamp)] Prepared ${#JOB_SPECS[@]} Mossi feature ablation jobs across ${NUM_GPUS} GPU(s)."
echo "[$(timestamp)] Pending jobs: ${#PENDING_JOB_INDICES[@]} | Pre-skipped existing: ${#SKIPPED_JOB_INDICES[@]}"
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

if [[ "${#PENDING_JOB_INDICES[@]}" -gt 0 ]]; then
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
else
  echo "[$(timestamp)] No pending jobs after pre-skip scan; skipping GPU execution."
fi

if [[ "${#PENDING_JOB_INDICES[@]}" -gt 0 ]]; then
  echo "[$(timestamp)] Starting post-run predict-only recovery pass..."
  run_post_recovery_predict_only
else
  echo "[$(timestamp)] No pending jobs; skipping recovery pass."
fi

COMBINED_SUMMARY="${SUMMARY_DIR}/all_jobs.tsv"
declare -a COMBINED_INPUTS=()
if [[ -f "${SKIPPED_EXISTING_FILE}" ]]; then
  COMBINED_INPUTS+=("${SKIPPED_EXISTING_FILE}")
fi
for summary_file in "${SUMMARY_DIR}"/gpu*.tsv; do
  [[ -f "${summary_file}" ]] || continue
  COMBINED_INPUTS+=("${summary_file}")
done
if [[ -f "${SUMMARY_DIR}/recovery.tsv" ]]; then
  COMBINED_INPUTS+=("${SUMMARY_DIR}/recovery.tsv")
fi

if [[ "${#COMBINED_INPUTS[@]}" -eq 0 ]]; then
  printf "job\tgpu\tstatus\telapsed_s\tseg_dir\tlog_file\n" > "${COMBINED_SUMMARY}"
else
  awk 'FNR==1 && NR!=1 {next} {print}' "${COMBINED_INPUTS[@]}" > "${COMBINED_SUMMARY}"
fi

FAILED_COUNT=$(
  awk -F'\t' 'NR>1 && $3!="ok" && $3!="skipped_existing" && $3!="recovered_predict_ok" {c++} END{print c+0}' "${COMBINED_SUMMARY}"
)

echo "[$(timestamp)] Combined summary: ${COMBINED_SUMMARY}"
if [[ -f "${SUMMARY_DIR}/recovery.tsv" ]]; then
  echo "[$(timestamp)] Recovery summary: ${SUMMARY_DIR}/recovery.tsv"
fi

if [[ "${RUN_VALIDATION_TABLE}" == "1" ]]; then
  if [[ -f "${VALIDATION_SCRIPT}" ]]; then
    if [[ "${BACKUP_EXISTING_VALIDATION}" == "1" ]] && [[ -f "${VALIDATION_FILE}" ]]; then
      VALIDATION_BACKUP="${SUMMARY_DIR}/validation_metrics.pre_mossi_feature_ablation.${BACKUP_STAMP}.tsv"
      cp -f "${VALIDATION_FILE}" "${VALIDATION_BACKUP}"
      echo "[$(timestamp)] Backed up prior validation table: ${VALIDATION_BACKUP}"
    fi

    echo "[$(timestamp)] Building validation metrics table..."
    validation_log="${SUMMARY_DIR}/validation_metrics.log"
    validation_cmd=(
      bash "${VALIDATION_SCRIPT}"
      --root "${OUTPUT_ROOT}"
      --input-dir "${INPUT_DIR}"
      --include-default-10x "${VALIDATION_INCLUDE_DEFAULT_10X}"
    )
    if [[ "${VALIDATION_RECOMPUTE}" == "1" ]]; then
      validation_cmd+=(--recompute)
    fi
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
      echo "[$(timestamp)] Validation table updated: ${VALIDATION_FILE}"
      build_delta_summary "${VALIDATION_FILE}" "${DELTA_FILE}" || true
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
