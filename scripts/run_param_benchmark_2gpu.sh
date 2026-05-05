#!/usr/bin/env bash
set -u -o pipefail

# -------------------------------------------------------------------------
# Segger overnight benchmark runner (2 GPUs, 1 job per GPU at a time)
# -------------------------------------------------------------------------
# Usage:
#   bash scripts/run_param_benchmark_2gpu.sh
#
# Optional overrides (environment variables):
#   INPUT_DIR=data/xe_pancreas_mossi/
#   OUTPUT_ROOT=./results/mossi_main_big_benchmark_nightly
#   GPU_A=0
#   GPU_B=1
#   N_EPOCHS=20
#   INCLUDE_EXTRA_SWEEPS=1
#   RESUME_IF_EXISTS=1
#   DRY_RUN=0
#   SEGMENT_TIMEOUT_MIN=90
#   ALIGNMENT_LOSS=true
#   ALIGNMENT_SCRNA_REFERENCE_PATH=data/ref_pancreas.h5ad
#   ALIGNMENT_SCRNA_CELLTYPE_COLUMN=cell_type
#   SEGMENT_NUM_WORKERS=8
#   SEGMENT_ANC_RETRY_WORKERS=0
#   TORCH_SHARING_STRATEGY=file_system
# -------------------------------------------------------------------------

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

DEFAULT_INPUT_DIR="data/xe_pancreas_mossi/"
INPUT_DIR="${INPUT_DIR:-${DEFAULT_INPUT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/mossi_main_big_benchmark_nightly}"

# Common layout fallback when running from segger-0.2.0 with data one level up.
if [[ "${INPUT_DIR}" == "${DEFAULT_INPUT_DIR}" ]] && \
   [[ ! -d "${INPUT_DIR}" ]] && \
   [[ -d "../data/xe_pancreas_mossi/" ]]; then
  INPUT_DIR="../data/xe_pancreas_mossi/"
fi

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

N_EPOCHS="${N_EPOCHS:-20}"
PREDICTION_MODE="${PREDICTION_MODE:-nucleus}"

BOUNDARY_METHOD="${BOUNDARY_METHOD:-convex_hull}"
BOUNDARY_VOXEL_SIZE="${BOUNDARY_VOXEL_SIZE:-5}"
XENIUM_NUM_WORKERS="${XENIUM_NUM_WORKERS:-8}"

INCLUDE_EXTRA_SWEEPS="${INCLUDE_EXTRA_SWEEPS:-1}"
RESUME_IF_EXISTS="${RESUME_IF_EXISTS:-1}"
DRY_RUN="${DRY_RUN:-0}"
PREDICT_FALLBACK_ON_OOM="${PREDICT_FALLBACK_ON_OOM:-1}"
SEGMENT_TIMEOUT_MIN="${SEGMENT_TIMEOUT_MIN:-90}"
SEGMENT_TIMEOUT_SEC=$((SEGMENT_TIMEOUT_MIN * 60))
SEGMENT_NUM_WORKERS="${SEGMENT_NUM_WORKERS:-8}"
SEGMENT_ANC_RETRY_WORKERS="${SEGMENT_ANC_RETRY_WORKERS:-0}"
TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"

ALIGNMENT_LOSS="${ALIGNMENT_LOSS:-true}"
ALIGNMENT_LOSS_WEIGHT_START="${ALIGNMENT_LOSS_WEIGHT_START:-0.0}"
ALIGNMENT_LOSS_WEIGHT_END="${ALIGNMENT_LOSS_WEIGHT_END:-0.03}"
ALIGNMENT_ME_GENE_PAIRS_PATH="${ALIGNMENT_ME_GENE_PAIRS_PATH:-}"
ALIGNMENT_SCRNA_REFERENCE_PATH="${ALIGNMENT_SCRNA_REFERENCE_PATH:-data/ref_pancreas.h5ad}"
ALIGNMENT_SCRNA_CELLTYPE_COLUMN="${ALIGNMENT_SCRNA_CELLTYPE_COLUMN:-cell_type}"

# Common layout fallback when running from segger-0.2.0 with data one level up.
if [[ "${ALIGNMENT_SCRNA_REFERENCE_PATH}" == "data/ref_pancreas.h5ad" ]] && \
   [[ ! -f "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]] && \
   [[ -f "../data/ref_pancreas.h5ad" ]]; then
  ALIGNMENT_SCRNA_REFERENCE_PATH="../data/ref_pancreas.h5ad"
fi

# Baseline values (matches the command you provided).
BASE_USE_3D="${BASE_USE_3D:-true}"
BASE_EXPANSION_RATIO="${BASE_EXPANSION_RATIO:-2.0}"
BASE_TX_MAX_K="${BASE_TX_MAX_K:-5}"
BASE_TX_MAX_DIST="${BASE_TX_MAX_DIST:-5}"
BASE_N_MID_LAYERS="${BASE_N_MID_LAYERS:-2}"
BASE_N_HEADS="${BASE_N_HEADS:-2}"
BASE_CELLS_MIN_COUNTS="${BASE_CELLS_MIN_COUNTS:-5}"
BASE_MIN_QV="${BASE_MIN_QV:-0}"

# One-factor-at-a-time sweep values around baseline.
USE_3D_VALUES=(false true)
EXPANSION_VALUES=(1 1.5 2.0 2.5 3.0)
TX_MAX_K_VALUES=(5 10 20)
TX_MAX_DIST_VALUES=(3 5 10 20)
N_MID_LAYER_VALUES=(1 2 3)
N_HEAD_VALUES=(2 4 8)
CELLS_MIN_COUNTS_VALUES=(3 5 10)
ALIGNMENT_VALUES=(false true)

RUNS_DIR="${OUTPUT_ROOT}/runs"
EXPORTS_DIR="${OUTPUT_ROOT}/exports"
LOGS_DIR="${OUTPUT_ROOT}/logs"
SUMMARY_DIR="${OUTPUT_ROOT}/summaries"
PLAN_FILE="${OUTPUT_ROOT}/job_plan.tsv"

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

need_alignment_inputs=0
if [[ "${ALIGNMENT_LOSS}" == "true" ]]; then
  need_alignment_inputs=1
elif [[ "${INCLUDE_EXTRA_SWEEPS}" == "1" ]]; then
  for v in "${ALIGNMENT_VALUES[@]}"; do
    if [[ "${v}" == "true" ]]; then
      need_alignment_inputs=1
      break
    fi
  done
fi

if [[ "${need_alignment_inputs}" == "1" ]]; then
  if [[ -z "${ALIGNMENT_ME_GENE_PAIRS_PATH}" ]] && [[ -z "${ALIGNMENT_SCRNA_REFERENCE_PATH}" ]]; then
    echo "ERROR: ALIGNMENT_LOSS=true requires ALIGNMENT_ME_GENE_PAIRS_PATH or ALIGNMENT_SCRNA_REFERENCE_PATH."
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

JOB_SPECS=()

add_job() {
  local job_name="$1"
  local use_3d="$2"
  local expansion="$3"
  local tx_k="$4"
  local tx_dist="$5"
  local n_layers="$6"
  local n_heads="$7"
  local cells_min_counts="$8"
  local min_qv="$9"
  local alignment_loss="${10}"
  JOB_SPECS+=("${job_name}|${use_3d}|${expansion}|${tx_k}|${tx_dist}|${n_layers}|${n_heads}|${cells_min_counts}|${min_qv}|${alignment_loss}")
}

build_jobs() {
  local v tag

  add_job \
    "baseline" \
    "${BASE_USE_3D}" \
    "${BASE_EXPANSION_RATIO}" \
    "${BASE_TX_MAX_K}" \
    "${BASE_TX_MAX_DIST}" \
    "${BASE_N_MID_LAYERS}" \
    "${BASE_N_HEADS}" \
    "${BASE_CELLS_MIN_COUNTS}" \
    "${BASE_MIN_QV}" \
    "${ALIGNMENT_LOSS}"

  for v in "${USE_3D_VALUES[@]}"; do
    [[ "${v}" == "${BASE_USE_3D}" ]] && continue
    add_job "use3d_${v}" \
      "${v}" "${BASE_EXPANSION_RATIO}" "${BASE_TX_MAX_K}" "${BASE_TX_MAX_DIST}" \
      "${BASE_N_MID_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN_COUNTS}" \
      "${BASE_MIN_QV}" "${ALIGNMENT_LOSS}"
  done

  for v in "${EXPANSION_VALUES[@]}"; do
    [[ "${v}" == "${BASE_EXPANSION_RATIO}" ]] && continue
    tag="${v//./p}"
    add_job "expansion_${tag}" \
      "${BASE_USE_3D}" "${v}" "${BASE_TX_MAX_K}" "${BASE_TX_MAX_DIST}" \
      "${BASE_N_MID_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN_COUNTS}" \
      "${BASE_MIN_QV}" "${ALIGNMENT_LOSS}"
  done

  for v in "${TX_MAX_K_VALUES[@]}"; do
    [[ "${v}" == "${BASE_TX_MAX_K}" ]] && continue
    add_job "txk_${v}" \
      "${BASE_USE_3D}" "${BASE_EXPANSION_RATIO}" "${v}" "${BASE_TX_MAX_DIST}" \
      "${BASE_N_MID_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN_COUNTS}" \
      "${BASE_MIN_QV}" "${ALIGNMENT_LOSS}"
  done

  for v in "${TX_MAX_DIST_VALUES[@]}"; do
    [[ "${v}" == "${BASE_TX_MAX_DIST}" ]] && continue
    tag="${v//./p}"
    add_job "txdist_${tag}" \
      "${BASE_USE_3D}" "${BASE_EXPANSION_RATIO}" "${BASE_TX_MAX_K}" "${v}" \
      "${BASE_N_MID_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN_COUNTS}" \
      "${BASE_MIN_QV}" "${ALIGNMENT_LOSS}"
  done

  for v in "${N_MID_LAYER_VALUES[@]}"; do
    [[ "${v}" == "${BASE_N_MID_LAYERS}" ]] && continue
    add_job "layers_${v}" \
      "${BASE_USE_3D}" "${BASE_EXPANSION_RATIO}" "${BASE_TX_MAX_K}" "${BASE_TX_MAX_DIST}" \
      "${v}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN_COUNTS}" \
      "${BASE_MIN_QV}" "${ALIGNMENT_LOSS}"
  done

  for v in "${N_HEAD_VALUES[@]}"; do
    [[ "${v}" == "${BASE_N_HEADS}" ]] && continue
    add_job "heads_${v}" \
      "${BASE_USE_3D}" "${BASE_EXPANSION_RATIO}" "${BASE_TX_MAX_K}" "${BASE_TX_MAX_DIST}" \
      "${BASE_N_MID_LAYERS}" "${v}" "${BASE_CELLS_MIN_COUNTS}" \
      "${BASE_MIN_QV}" "${ALIGNMENT_LOSS}"
  done

  for v in "${CELLS_MIN_COUNTS_VALUES[@]}"; do
    [[ "${v}" == "${BASE_CELLS_MIN_COUNTS}" ]] && continue
    add_job "cellsmin_${v}" \
      "${BASE_USE_3D}" "${BASE_EXPANSION_RATIO}" "${BASE_TX_MAX_K}" "${BASE_TX_MAX_DIST}" \
      "${BASE_N_MID_LAYERS}" "${BASE_N_HEADS}" "${v}" \
      "${BASE_MIN_QV}" "${ALIGNMENT_LOSS}"
  done

  if [[ "${INCLUDE_EXTRA_SWEEPS}" == "1" ]]; then
    for v in "${ALIGNMENT_VALUES[@]}"; do
      [[ "${v}" == "${ALIGNMENT_LOSS}" ]] && continue
      add_job "align_${v}" \
        "${BASE_USE_3D}" "${BASE_EXPANSION_RATIO}" "${BASE_TX_MAX_K}" "${BASE_TX_MAX_DIST}" \
        "${BASE_N_MID_LAYERS}" "${BASE_N_HEADS}" "${BASE_CELLS_MIN_COUNTS}" \
        "${BASE_MIN_QV}" "${v}"
    done
  fi
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

LAST_JOB_STATUS="unknown"

run_job() {
  local gpu="$1"
  local spec="$2"

  local job_name use_3d expansion tx_k tx_dist n_layers n_heads cells_min_counts min_qv alignment_loss
  IFS='|' read -r \
    job_name use_3d expansion tx_k tx_dist n_layers n_heads cells_min_counts min_qv alignment_loss \
    <<< "${spec}"

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
    echo "params: use3d=${use_3d} expansion=${expansion} tx_k=${tx_k} tx_dist=${tx_dist} layers=${n_layers} heads=${n_heads} cells_min=${cells_min_counts} min_qv=${min_qv} align=${alignment_loss} timeout_min=${SEGMENT_TIMEOUT_MIN} dl_workers=${SEGMENT_NUM_WORKERS} anc_retry_workers=${SEGMENT_ANC_RETRY_WORKERS} sharing=${TORCH_SHARING_STRATEGY}"
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
    local -a seg_cmd=(
      env CUDA_VISIBLE_DEVICES="${gpu}"
      PYTORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY}"
      SEGGER_NUM_WORKERS="${SEGMENT_NUM_WORKERS}"
      segger segment
      -i "${INPUT_DIR}"
      -o "${seg_dir}"
      --n-epochs "${N_EPOCHS}"
      --prediction-mode "${PREDICTION_MODE}"
      --prediction-expansion-ratio "${expansion}"
      --cells-min-counts "${cells_min_counts}"
      --min-qv "${min_qv}"
      --use-3d "${use_3d}"
      --transcripts-max-k "${tx_k}"
      --transcripts-max-dist "${tx_dist}"
      --n-mid-layers "${n_layers}"
      --n-heads "${n_heads}"
    )
    if [[ "${alignment_loss}" == "true" ]]; then
      seg_cmd+=(
        --alignment-loss
        --alignment-loss-weight-start "${ALIGNMENT_LOSS_WEIGHT_START}"
        --alignment-loss-weight-end "${ALIGNMENT_LOSS_WEIGHT_END}"
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

run_post_recovery_predict_only_group() {
  local gpu="$1"
  local out_file="$2"
  shift 2
  local -a indices=("$@")

  printf "job\tgpu\tstatus\telapsed_s\tnote\tseg_dir\tlog_file\n" > "${out_file}"

  local idx spec job_name use_3d expansion tx_k tx_dist n_layers n_heads cells_min_counts min_qv alignment_loss
  local seg_dir seg_file last_ckpt log_file note status
  local start_ts end_ts elapsed_s

  for idx in "${indices[@]}"; do
    spec="${JOB_SPECS[$idx]}"
    IFS='|' read -r \
      job_name use_3d expansion tx_k tx_dist n_layers n_heads cells_min_counts min_qv alignment_loss \
      <<< "${spec}"

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
  local recovery_a="${SUMMARY_DIR}/recovery.gpu${GPU_A}.tsv"
  local recovery_b="${SUMMARY_DIR}/recovery.gpu${GPU_B}.tsv"
  local pid_a pid_b

  if [[ "${GPU_A}" == "${GPU_B}" ]]; then
    run_post_recovery_predict_only_group "${GPU_A}" "${recovery_a}" "${GPU_A_INDICES[@]}" "${GPU_B_INDICES[@]}"
    cp "${recovery_a}" "${recovery_file}"
    return
  fi

  run_post_recovery_predict_only_group "${GPU_A}" "${recovery_a}" "${GPU_A_INDICES[@]}" &
  pid_a=$!
  run_post_recovery_predict_only_group "${GPU_B}" "${recovery_b}" "${GPU_B_INDICES[@]}" &
  pid_b=$!

  wait "${pid_a}"
  wait "${pid_b}"

  awk 'FNR==1 && NR!=1 {next} {print}' "${recovery_a}" "${recovery_b}" > "${recovery_file}"
}

build_jobs

if [[ "${#JOB_SPECS[@]}" -eq 0 ]]; then
  echo "ERROR: No benchmark jobs were generated."
  exit 1
fi

GPU_A_INDICES=()
GPU_B_INDICES=()

idx=0
for spec in "${JOB_SPECS[@]}"; do
  if (( idx % 2 == 0 )); then
    GPU_A_INDICES+=("${idx}")
  else
    GPU_B_INDICES+=("${idx}")
  fi
  idx=$((idx + 1))
done

{
  printf "job\tgroup\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\n"
  for idx in "${!JOB_SPECS[@]}"; do
    local_group="A"
    if (( idx % 2 == 1 )); then
      local_group="B"
    fi
    IFS='|' read -r \
      job_name use_3d expansion tx_k tx_dist n_layers n_heads cells_min_counts min_qv alignment_loss \
      <<< "${JOB_SPECS[$idx]}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job_name}" "${local_group}" "${use_3d}" "${expansion}" "${tx_k}" "${tx_dist}" \
      "${n_layers}" "${n_heads}" "${cells_min_counts}" "${min_qv}" "${alignment_loss}"
  done
} > "${PLAN_FILE}"

echo "[$(timestamp)] Prepared ${#JOB_SPECS[@]} jobs."
echo "[$(timestamp)] Group A (GPU ${GPU_A}): ${#GPU_A_INDICES[@]} jobs"
echo "[$(timestamp)] Group B (GPU ${GPU_B}): ${#GPU_B_INDICES[@]} jobs"
echo "[$(timestamp)] Job plan: ${PLAN_FILE}"
echo "[$(timestamp)] Logs: ${LOGS_DIR}"

run_gpu_group "${GPU_A}" "${GPU_A_INDICES[@]}" &
PID_A=$!
run_gpu_group "${GPU_B}" "${GPU_B_INDICES[@]}" &
PID_B=$!

wait "${PID_A}"
wait "${PID_B}"

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "[$(timestamp)] Starting post-run predict-only recovery pass..."
  run_post_recovery_predict_only
fi

COMBINED_SUMMARY="${SUMMARY_DIR}/all_jobs.tsv"
if [[ "${DRY_RUN}" != "1" ]] && [[ -f "${SUMMARY_DIR}/recovery.tsv" ]]; then
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
echo "[$(timestamp)] Failed jobs: ${FAILED_COUNT}"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
  exit 1
fi
