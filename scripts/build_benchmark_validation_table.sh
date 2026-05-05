#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build per-job validation metrics table for a benchmark root.

Usage:
  bash scripts/build_benchmark_validation_table.sh [options]

Options:
  --root <dir>                    Benchmark root (default: ./results/mossi_main_big_benchmark_nightly)
  --input-dir <dir>               Source dataset path for contamination/geometry/doublet metrics (optional)
  --out-tsv <file>                Output TSV (default: <root>/summaries/validation_metrics.tsv)
  --recompute                     Recompute all jobs even if already present in output TSV
  --segger-bin <cmd>              Segger executable/command (default: segger)
  --me-gene-pairs-path <file>     Optional ME-gene pair file passed to segger validate
  --scrna-reference-path <file>   Optional scRNA h5ad passed to segger validate
  --scrna-celltype-column <name>  scRNA cell type column (default: cell_type)
  --max-me-gene-pairs <int>       Max sampled ME-gene pairs (default: 500)
  --gpu-a <id>                    GPU id used for group A labels (default: env GPU_A or 0)
  --gpu-b <id>                    GPU id used for group B labels (default: env GPU_B or 1)
  --include-default-10x <bool>    Include ref_10x_cell/ref_10x_nucleus rows (default: true)
  --reference-universe-seg <file> Canonical Segger universe segmentation override
  -h, --help                      Show this help
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

sanitize_tsv_field() {
  local value="${1:-}"
  value="${value//$'\t'/ }"
  value="${value//$'\r'/ }"
  value="${value//$'\n'/ }"
  printf '%s' "${value}"
}

normalize_token() {
  local value="${1:-}"
  value="$(sanitize_tsv_field "${value}")"
  if [[ -z "${value}" ]]; then
    printf '%s' "-"
  else
    printf '%s' "${value}"
  fi
}

normalize_bool() {
  local v="${1:-}"
  local lc
  lc="$(printf '%s' "${v}" | tr '[:upper:]' '[:lower:]')"
  case "${lc}" in
    1|true|t|yes|y|on)
      printf 'true'
      ;;
    0|false|f|no|n|off)
      printf 'false'
      ;;
    *)
      return 1
      ;;
  esac
}

ROOT="./results/mossi_main_big_benchmark_nightly"
INPUT_DIR=""
OUT_TSV=""
SEGGER_BIN="segger"
ME_GENE_PAIRS_PATH=""
SCRNA_REFERENCE_PATH=""
SCRNA_CELLTYPE_COLUMN="cell_type"
MAX_ME_GENE_PAIRS=500
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
INCLUDE_DEFAULT_10X="true"
REFERENCE_UNIVERSE_SEG=""
RECOMPUTE=0

require_value() {
  local opt="$1"
  if [[ $# -lt 2 ]] || [[ -z "${2}" ]] || [[ "${2}" == -* ]]; then
    echo "ERROR: ${opt} requires a value." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      require_value "$1" "${2-}"
      ROOT="$2"
      shift 2
      ;;
    --input-dir)
      require_value "$1" "${2-}"
      INPUT_DIR="$2"
      shift 2
      ;;
    --out-tsv)
      require_value "$1" "${2-}"
      OUT_TSV="$2"
      shift 2
      ;;
    --recompute)
      RECOMPUTE=1
      shift
      ;;
    --segger-bin)
      require_value "$1" "${2-}"
      SEGGER_BIN="$2"
      shift 2
      ;;
    --me-gene-pairs-path)
      require_value "$1" "${2-}"
      ME_GENE_PAIRS_PATH="$2"
      shift 2
      ;;
    --scrna-reference-path)
      require_value "$1" "${2-}"
      SCRNA_REFERENCE_PATH="$2"
      shift 2
      ;;
    --scrna-celltype-column)
      require_value "$1" "${2-}"
      SCRNA_CELLTYPE_COLUMN="$2"
      shift 2
      ;;
    --max-me-gene-pairs)
      require_value "$1" "${2-}"
      MAX_ME_GENE_PAIRS="$2"
      shift 2
      ;;
    --gpu-a)
      require_value "$1" "${2-}"
      GPU_A="$2"
      shift 2
      ;;
    --gpu-b)
      require_value "$1" "${2-}"
      GPU_B="$2"
      shift 2
      ;;
    --include-default-10x)
      require_value "$1" "${2-}"
      INCLUDE_DEFAULT_10X="$2"
      shift 2
      ;;
    --reference-universe-seg)
      require_value "$1" "${2-}"
      REFERENCE_UNIVERSE_SEG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${OUT_TSV}" ]]; then
  OUT_TSV="${ROOT}/summaries/validation_metrics.tsv"
fi

if ! [[ "${MAX_ME_GENE_PAIRS}" =~ ^[0-9]+$ ]] || [[ "${MAX_ME_GENE_PAIRS}" -le 0 ]]; then
  echo "ERROR: --max-me-gene-pairs must be a positive integer." >&2
  exit 1
fi

if ! INCLUDE_DEFAULT_10X="$(normalize_bool "${INCLUDE_DEFAULT_10X}")"; then
  echo "ERROR: --include-default-10x must be true/false." >&2
  exit 1
fi

if [[ -n "${INPUT_DIR}" ]] && [[ ! -e "${INPUT_DIR}" ]]; then
  echo "ERROR: --input-dir not found: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ -n "${ME_GENE_PAIRS_PATH}" ]] && [[ ! -f "${ME_GENE_PAIRS_PATH}" ]]; then
  echo "ERROR: --me-gene-pairs-path not found: ${ME_GENE_PAIRS_PATH}" >&2
  exit 1
fi

if [[ -n "${SCRNA_REFERENCE_PATH}" ]] && [[ ! -f "${SCRNA_REFERENCE_PATH}" ]]; then
  echo "ERROR: --scrna-reference-path not found: ${SCRNA_REFERENCE_PATH}" >&2
  exit 1
fi

if [[ -n "${REFERENCE_UNIVERSE_SEG}" ]] && [[ ! -f "${REFERENCE_UNIVERSE_SEG}" ]]; then
  echo "ERROR: --reference-universe-seg not found: ${REFERENCE_UNIVERSE_SEG}" >&2
  exit 1
fi

SEGGER_CMD_PATH=""
if [[ "${SEGGER_BIN}" == */* ]]; then
  if [[ ! -x "${SEGGER_BIN}" ]]; then
    echo "ERROR: --segger-bin is not executable: ${SEGGER_BIN}" >&2
    exit 1
  fi
  SEGGER_CMD_PATH="${SEGGER_BIN}"
else
  if ! SEGGER_CMD_PATH="$(command -v "${SEGGER_BIN}" 2>/dev/null)"; then
    echo "ERROR: segger command not found: ${SEGGER_BIN}" >&2
    exit 1
  fi
fi

SEGGER_PYTHON="$(dirname "${SEGGER_CMD_PATH}")/python"
if [[ ! -x "${SEGGER_PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    SEGGER_PYTHON="$(command -v python3)"
  else
    echo "ERROR: Could not resolve python interpreter for reference artifact builder." >&2
    exit 1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFERENCE_BUILDER_PY="${SCRIPT_DIR}/build_default_10x_reference_artifacts.py"
if [[ "${INCLUDE_DEFAULT_10X}" == "true" ]] && [[ ! -f "${REFERENCE_BUILDER_PY}" ]]; then
  echo "ERROR: Missing reference artifact builder: ${REFERENCE_BUILDER_PY}" >&2
  exit 1
fi

PLAN_FILE="${ROOT}/job_plan.tsv"
RUNS_DIR="${ROOT}/runs"
EXPORTS_DIR="${ROOT}/exports"
SUMMARY_DIR="${ROOT}/summaries"
LOG_FILE="${SUMMARY_DIR}/validation_metrics.log"
REFERENCE_ARTIFACTS_DIR="${SUMMARY_DIR}/reference_artifacts"

if [[ ! -f "${PLAN_FILE}" ]]; then
  echo "ERROR: Missing benchmark plan file: ${PLAN_FILE}" >&2
  exit 1
fi

mkdir -p "${SUMMARY_DIR}" "$(dirname "${OUT_TSV}")" "${REFERENCE_ARTIFACTS_DIR}"

tmp_out="$(mktemp)"
tmp_row="$(mktemp "${TMPDIR:-/tmp}/segger_validate_row.XXXXXX.tsv")"
trap 'rm -f "${tmp_out}" "${tmp_row}"' EXIT

METRIC_SCHEMA_VERSION="2026-02-25-v8"
RUN_INPUT_DIR_TOKEN="$(normalize_token "${INPUT_DIR}")"
RUN_SCRNA_REFERENCE_TOKEN="$(normalize_token "${SCRNA_REFERENCE_PATH}")"
RUN_ME_GENE_PAIRS_TOKEN="$(normalize_token "${ME_GENE_PAIRS_PATH}")"

REFERENCE_UNIVERSE_SEG_RESOLVED=""
if [[ -n "${REFERENCE_UNIVERSE_SEG}" ]]; then
  REFERENCE_UNIVERSE_SEG_RESOLVED="${REFERENCE_UNIVERSE_SEG}"
elif [[ -f "${RUNS_DIR}/baseline/segger_segmentation.parquet" ]]; then
  REFERENCE_UNIVERSE_SEG_RESOLVED="${RUNS_DIR}/baseline/segger_segmentation.parquet"
else
  first_run_seg="$(find "${RUNS_DIR}" -mindepth 2 -maxdepth 2 -type f -name 'segger_segmentation.parquet' | sort | head -n1 || true)"
  if [[ -n "${first_run_seg}" ]]; then
    REFERENCE_UNIVERSE_SEG_RESOLVED="${first_run_seg}"
  elif [[ -n "${INPUT_DIR}" ]] && [[ -f "${INPUT_DIR}/segger_segmentation.parquet" ]]; then
    REFERENCE_UNIVERSE_SEG_RESOLVED="${INPUT_DIR}/segger_segmentation.parquet"
  fi
fi
RUN_REFERENCE_UNIVERSE_TOKEN="$(normalize_token "${REFERENCE_UNIVERSE_SEG_RESOLVED}")"

OUTPUT_HEADER=$'job\tgroup\tgpu\tis_reference\treference_kind\tvalidate_status\tvalidate_error\tgpu_time_s\tcells\tassigned_pct\tassigned_ci95\tmecr\tmecr_ci95\tcontamination_pct\tcontamination_ci95\tresolvi_contamination_pct\tresolvi_contamination_ci95\ttco\ttco_ci95\tdoublet_pct\tdoublet_ci95\tsegmentation_path\tanndata_path\toutput_path\tupdated_at\tmetric_schema_version\trun_input_dir\trun_scrna_reference_path\trun_me_gene_pairs_path\trun_reference_universe_seg'
reuse_existing=0
if [[ "${RECOMPUTE}" != "1" ]] && [[ -s "${OUT_TSV}" ]]; then
  existing_header="$(head -n1 "${OUT_TSV}" || true)"
  if [[ "${existing_header}" == "${OUTPUT_HEADER}" ]]; then
    reuse_existing=1
  else
    echo "[$(timestamp)] WARN existing TSV header mismatch; recomputing all jobs" >> "${LOG_FILE}"
  fi
fi

get_field() {
  local column_name="$1"
  if [[ ! -s "${tmp_row}" ]]; then
    return 0
  fi
  awk -F'\t' -v key="${column_name}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == key) {
          idx = i
          break
        }
      }
      next
    }
    NR == 2 {
      if (idx > 0) {
        print $idx
      }
      exit
    }
  ' "${tmp_row}"
}

get_existing_field_by_job() {
  local job_name="$1"
  local column_name="$2"
  awk -F'\t' -v j="${job_name}" -v key="${column_name}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == key) {
          idx = i
          break
        }
      }
      next
    }
    $1 == j {
      if (idx > 0) {
        print $idx
      }
      exit
    }
  ' "${OUT_TSV}"
}

lookup_gpu_time() {
  local job_name="$1"
  local preferred_gpu="$2"
  local elapsed=""
  local f row

  if [[ -f "${SUMMARY_DIR}/gpu${preferred_gpu}.tsv" ]]; then
    row="$(awk -F'\t' -v j="${job_name}" '
      NR == 1 {
        for (i = 1; i <= NF; i++) {
          if ($i == "job") job_col = i
          if ($i == "elapsed_s") elapsed_col = i
        }
        next
      }
      job_col > 0 && $job_col == j {
        if (elapsed_col > 0) print $elapsed_col
        exit
      }
    ' "${SUMMARY_DIR}/gpu${preferred_gpu}.tsv")"
    if [[ -n "${row}" ]]; then
      elapsed="${row}"
    fi
  fi

  if [[ -z "${elapsed}" ]]; then
    for f in "${SUMMARY_DIR}"/gpu*.tsv "${SUMMARY_DIR}/recovery.tsv"; do
      [[ -f "${f}" ]] || continue
      row="$(awk -F'\t' -v j="${job_name}" '
        NR == 1 {
          for (i = 1; i <= NF; i++) {
            if ($i == "job") job_col = i
            if ($i == "elapsed_s") elapsed_col = i
          }
          next
        }
        job_col > 0 && $job_col == j {
          if (elapsed_col > 0) print $elapsed_col
          exit
        }
      ' "${f}")"
      if [[ -n "${row}" ]]; then
        elapsed="${row}"
        break
      fi
    done
  fi

  if [[ -z "${elapsed}" ]]; then
    elapsed="0"
  fi
  printf '%s' "${elapsed}"
}

scale_frac_to_pct() {
  local value="${1:-}"
  local lower
  lower="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "${value}" ]] || [[ "${lower}" == "nan" ]] || [[ "${lower}" == "none" ]]; then
    printf '%s' "nan"
    return 0
  fi
  awk -v v="${value}" 'BEGIN { printf "%.6f", (v + 0.0) * 100.0 }'
}

append_row() {
  local job="$1"
  local group="$2"
  local gpu="$3"
  local is_reference="$4"
  local reference_kind="$5"
  local validate_status="$6"
  local validate_error="$7"
  local gpu_time_s="$8"
  local cells="$9"
  local assigned_pct="${10}"
  local assigned_ci95="${11}"
  local mecr="${12}"
  local mecr_ci95="${13}"
  local contamination_pct="${14}"
  local contamination_ci95="${15}"
  local resolvi_contamination_pct="${16}"
  local resolvi_contamination_ci95="${17}"
  local tco="${18}"
  local tco_ci95="${19}"
  local doublet_pct="${20}"
  local doublet_ci95="${21}"
  local row_seg_path="${22}"
  local row_anndata_path="${23}"

  validate_error="$(sanitize_tsv_field "${validate_error}")"
  reference_kind="$(normalize_token "${reference_kind}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${job}" "${group}" "${gpu}" "${is_reference}" "${reference_kind}" "${validate_status}" "${validate_error}" "${gpu_time_s}" \
    "${cells}" "${assigned_pct}" "${assigned_ci95}" "${mecr}" "${mecr_ci95}" "${contamination_pct}" "${contamination_ci95}" \
    "${resolvi_contamination_pct}" "${resolvi_contamination_ci95}" "${tco}" "${tco_ci95}" "${doublet_pct}" "${doublet_ci95}" "${row_seg_path}" "${row_anndata_path}" "${OUT_TSV}" \
    "$(timestamp)" "${METRIC_SCHEMA_VERSION}" "${RUN_INPUT_DIR_TOKEN}" "${RUN_SCRNA_REFERENCE_TOKEN}" "${RUN_ME_GENE_PAIRS_TOKEN}" "${RUN_REFERENCE_UNIVERSE_TOKEN}" \
    >> "${tmp_out}"
}

should_reuse_row() {
  local job="$1"
  local is_reference="$2"
  local should_reuse=0
  local reason=""

  if [[ "${reuse_existing}" != "1" ]]; then
    printf '0\tno_reuse_mode\n'
    return 0
  fi

  existing_row="$(awk -F'\t' -v j="${job}" 'NR > 1 && $1 == j { print; exit }' "${OUT_TSV}")"
  if [[ -z "${existing_row}" ]]; then
    printf '0\tmissing_existing_row\n'
    return 0
  fi

  existing_status="$(get_existing_field_by_job "${job}" "validate_status")"
  existing_metric_schema_version="$(get_existing_field_by_job "${job}" "metric_schema_version")"
  existing_run_input_dir="$(get_existing_field_by_job "${job}" "run_input_dir")"
  existing_run_scrna_ref="$(get_existing_field_by_job "${job}" "run_scrna_reference_path")"
  existing_run_me_pairs="$(get_existing_field_by_job "${job}" "run_me_gene_pairs_path")"
  existing_run_ref_universe="$(get_existing_field_by_job "${job}" "run_reference_universe_seg")"
  existing_cells="$(get_existing_field_by_job "${job}" "cells")"
  existing_is_reference="$(get_existing_field_by_job "${job}" "is_reference")"

  should_reuse=1
  reason="ok"

  if [[ "${existing_status}" != "ok" ]]; then
    should_reuse=0
    reason="status=${existing_status}"
  elif [[ "${existing_metric_schema_version:-}" != "${METRIC_SCHEMA_VERSION}" ]]; then
    should_reuse=0
    reason="schema_mismatch"
  elif [[ "${existing_run_input_dir:-}" != "${RUN_INPUT_DIR_TOKEN}" ]] || \
       [[ "${existing_run_scrna_ref:-}" != "${RUN_SCRNA_REFERENCE_TOKEN}" ]] || \
       [[ "${existing_run_me_pairs:-}" != "${RUN_ME_GENE_PAIRS_TOKEN}" ]] || \
       [[ "${existing_run_ref_universe:-}" != "${RUN_REFERENCE_UNIVERSE_TOKEN}" ]]; then
    should_reuse=0
    reason="validation_inputs_changed"
  elif [[ "${existing_is_reference:-0}" != "${is_reference}" ]]; then
    should_reuse=0
    reason="reference_flag_changed"
  elif [[ -z "${existing_cells:-}" ]] || [[ "$(printf '%s' "${existing_cells}" | tr '[:upper:]' '[:lower:]')" == "nan" ]]; then
    should_reuse=0
    reason="missing_cells"
  elif [[ -n "${INPUT_DIR}" ]]; then
    existing_tco="$(get_existing_field_by_job "${job}" "tco")"
    existing_contam="$(get_existing_field_by_job "${job}" "contamination_pct")"
    existing_doublet="$(get_existing_field_by_job "${job}" "doublet_pct")"
    if [[ -z "${existing_tco:-}" ]] || [[ -z "${existing_contam:-}" ]] || [[ -z "${existing_doublet:-}" ]]; then
      should_reuse=0
      reason="missing_input_dependent_metrics"
    elif [[ -n "${SCRNA_REFERENCE_PATH}" ]]; then
      existing_resolvi="$(get_existing_field_by_job "${job}" "resolvi_contamination_pct")"
      if [[ -z "${existing_resolvi:-}" ]]; then
        should_reuse=0
        reason="missing_resolvi_metric"
      fi
    fi
  fi

  printf '%s\t%s\n' "${should_reuse}" "${reason}"
}

run_validate_for_row() {
  local job="$1"
  local seg_path="$2"
  local anndata_path="$3"
  local apply_elapsed_fallback="$4"

  validate_status="ok"
  validate_error=""
  cells="0"
  assigned_pct="nan"
  assigned_ci95="nan"
  mecr="nan"
  mecr_ci95="nan"
  contamination_pct="nan"
  contamination_ci95="nan"
  resolvi_contamination_pct="nan"
  resolvi_contamination_ci95="nan"
  tco="nan"
  tco_ci95="nan"
  doublet_pct="nan"
  doublet_ci95="nan"
  row_seg_path="${seg_path}"
  row_anndata_path=""

  if [[ ! -f "${seg_path}" ]]; then
    validate_status="missing_segmentation"
    validate_error="segger_segmentation.parquet not found"
    return 0
  fi

  : > "${tmp_row}"
  cmd=(
    "${SEGGER_BIN}" validate
    -s "${seg_path}"
    -o "${tmp_row}"
    --max-me-gene-pairs "${MAX_ME_GENE_PAIRS}"
  )
  if [[ -f "${anndata_path}" ]]; then
    cmd+=(-a "${anndata_path}")
  fi
  if [[ -n "${INPUT_DIR}" ]]; then
    cmd+=(-i "${INPUT_DIR}")
  fi
  if [[ -n "${ME_GENE_PAIRS_PATH}" ]]; then
    cmd+=(--me-gene-pairs-path "${ME_GENE_PAIRS_PATH}")
  fi
  if [[ -n "${SCRNA_REFERENCE_PATH}" ]]; then
    cmd+=(
      --scrna-reference-path "${SCRNA_REFERENCE_PATH}"
      --scrna-celltype-column "${SCRNA_CELLTYPE_COLUMN}"
    )
  fi

  {
    printf '[%s] job=%s CMD:' "$(timestamp)" "${job}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } >> "${LOG_FILE}"

  if "${cmd[@]}" >> "${LOG_FILE}" 2>&1; then
    parsed_status="$(get_field "validate_status")"
    [[ -n "${parsed_status}" ]] && validate_status="${parsed_status}"
    parsed_error="$(get_field "validate_error")"
    [[ -n "${parsed_error}" ]] && validate_error="${parsed_error}"

    parsed_elapsed="$(get_field "elapsed_s")"
    if [[ "${apply_elapsed_fallback}" == "1" ]] && [[ -n "${parsed_elapsed}" ]] && [[ "${gpu_time_s}" == "0" || -z "${gpu_time_s}" ]]; then
      gpu_time_s="${parsed_elapsed}"
    fi

    parsed_val="$(get_field "cells_total")"
    if [[ -n "${parsed_val}" ]] && [[ "$(printf '%s' "${parsed_val}" | tr '[:upper:]' '[:lower:]')" != "nan" ]]; then
      cells="${parsed_val}"
    else
      parsed_val="$(get_field "cells_assigned")"
      [[ -n "${parsed_val}" ]] && cells="${parsed_val}"
    fi

    parsed_val="$(get_field "transcripts_assigned_pct")"
    [[ -n "${parsed_val}" ]] && assigned_pct="${parsed_val}"
    parsed_val="$(get_field "transcripts_assigned_pct_ci95")"
    [[ -n "${parsed_val}" ]] && assigned_ci95="${parsed_val}"

    parsed_val="$(get_field "mecr_fast")"
    [[ -n "${parsed_val}" ]] && mecr="${parsed_val}"
    parsed_val="$(get_field "mecr_ci95_fast")"
    [[ -n "${parsed_val}" ]] && mecr_ci95="${parsed_val}"

    parsed_val="$(get_field "border_contaminated_cells_pct_fast")"
    [[ -n "${parsed_val}" ]] && contamination_pct="${parsed_val}"
    parsed_val="$(get_field "border_contaminated_cells_pct_ci95_fast")"
    [[ -n "${parsed_val}" ]] && contamination_ci95="${parsed_val}"

    parsed_val="$(get_field "resolvi_contamination_pct_fast")"
    [[ -n "${parsed_val}" ]] && resolvi_contamination_pct="${parsed_val}"
    parsed_val="$(get_field "resolvi_contamination_ci95_fast")"
    [[ -n "${parsed_val}" ]] && resolvi_contamination_ci95="${parsed_val}"

    parsed_val="$(get_field "transcript_centroid_offset_fast")"
    [[ -n "${parsed_val}" ]] && tco="${parsed_val}"
    parsed_val="$(get_field "transcript_centroid_offset_ci95_fast")"
    [[ -n "${parsed_val}" ]] && tco_ci95="${parsed_val}"

    parsed_val="$(get_field "signal_doublet_like_fraction_fast")"
    if [[ -n "${parsed_val}" ]]; then
      doublet_pct="$(scale_frac_to_pct "${parsed_val}")"
    fi
    parsed_val="$(get_field "signal_doublet_like_fraction_ci95_fast")"
    if [[ -n "${parsed_val}" ]]; then
      doublet_ci95="$(scale_frac_to_pct "${parsed_val}")"
    fi

    parsed_val="$(get_field "segmentation_path")"
    [[ -n "${parsed_val}" ]] && row_seg_path="${parsed_val}"
    parsed_val="$(get_field "anndata_path")"
    [[ -n "${parsed_val}" ]] && row_anndata_path="${parsed_val}"
  else
    validate_status="validate_command_failed"
    validate_error="segger validate command failed"
  fi
}

printf "%s\n" "${OUTPUT_HEADER}" > "${tmp_out}"

echo "[$(timestamp)] START benchmark validation table build" >> "${LOG_FILE}"
echo "[$(timestamp)] ROOT=${ROOT}" >> "${LOG_FILE}"
echo "[$(timestamp)] OUT_TSV=${OUT_TSV}" >> "${LOG_FILE}"
echo "[$(timestamp)] RECOMPUTE=${RECOMPUTE}" >> "${LOG_FILE}"
echo "[$(timestamp)] INCLUDE_DEFAULT_10X=${INCLUDE_DEFAULT_10X}" >> "${LOG_FILE}"
echo "[$(timestamp)] REFERENCE_UNIVERSE_SEG=${RUN_REFERENCE_UNIVERSE_TOKEN}" >> "${LOG_FILE}"

reused_count=0
computed_count=0

while IFS=$'\t' read -r \
  job group _use3d _expansion _txk _txdist _layers _heads _cellsmin _minqv _alignment; do
  if [[ -z "${job:-}" ]] || [[ "${job}" == "job" ]]; then
    continue
  fi

  reuse_info="$(should_reuse_row "${job}" "0")"
  should_reuse="${reuse_info%%$'\t'*}"
  reuse_reason="${reuse_info#*$'\t'}"
  if [[ "${should_reuse}" == "1" ]]; then
    existing_row="$(awk -F'\t' -v j="${job}" 'NR > 1 && $1 == j { print; exit }' "${OUT_TSV}")"
    printf "%s\n" "${existing_row}" >> "${tmp_out}"
    reused_count=$((reused_count + 1))
    echo "[$(timestamp)] SKIP job=${job}: existing row reused" >> "${LOG_FILE}"
    continue
  fi
  if [[ "${reuse_existing}" == "1" ]]; then
    echo "[$(timestamp)] RECOMPUTE job=${job}: ${reuse_reason}" >> "${LOG_FILE}"
  fi

  computed_count=$((computed_count + 1))

  gpu="${GPU_A}"
  if [[ "${group}" == "B" ]]; then
    gpu="${GPU_B}"
  fi

  seg_path="${RUNS_DIR}/${job}/segger_segmentation.parquet"
  anndata_path="${EXPORTS_DIR}/${job}/anndata/segger_segmentation.h5ad"
  gpu_time_s="$(lookup_gpu_time "${job}" "${gpu}")"

  start_ts="$(date +%s)"
  run_validate_for_row "${job}" "${seg_path}" "${anndata_path}" "1"
  end_ts="$(date +%s)"
  if [[ "${gpu_time_s}" == "0" || -z "${gpu_time_s}" ]]; then
    gpu_time_s="$((end_ts - start_ts))"
  fi

  append_row \
    "${job}" "${group}" "${gpu}" "0" "-" \
    "${validate_status}" "${validate_error}" "${gpu_time_s}" "${cells}" \
    "${assigned_pct}" "${assigned_ci95}" "${mecr}" "${mecr_ci95}" \
    "${contamination_pct}" "${contamination_ci95}" "${resolvi_contamination_pct}" "${resolvi_contamination_ci95}" "${tco}" "${tco_ci95}" \
    "${doublet_pct}" "${doublet_ci95}" "${row_seg_path}" "${row_anndata_path}"
done < "${PLAN_FILE}"

if [[ "${INCLUDE_DEFAULT_10X}" == "true" ]]; then
  for reference_kind in 10x_cell 10x_nucleus; do
    job="ref_${reference_kind}"
    group="R"
    gpu="-"

    reuse_info="$(should_reuse_row "${job}" "1")"
    should_reuse="${reuse_info%%$'\t'*}"
    reuse_reason="${reuse_info#*$'\t'}"
    if [[ "${should_reuse}" == "1" ]]; then
      existing_row="$(awk -F'\t' -v j="${job}" 'NR > 1 && $1 == j { print; exit }' "${OUT_TSV}")"
      printf "%s\n" "${existing_row}" >> "${tmp_out}"
      reused_count=$((reused_count + 1))
      echo "[$(timestamp)] SKIP job=${job}: existing row reused" >> "${LOG_FILE}"
      continue
    fi
    if [[ "${reuse_existing}" == "1" ]]; then
      echo "[$(timestamp)] RECOMPUTE job=${job}: ${reuse_reason}" >> "${LOG_FILE}"
    fi

    computed_count=$((computed_count + 1))
    gpu_time_s="0"
    validate_status="ok"
    validate_error=""
    cells="0"
    assigned_pct="nan"
    assigned_ci95="nan"
    mecr="nan"
    mecr_ci95="nan"
    contamination_pct="nan"
    contamination_ci95="nan"
    resolvi_contamination_pct="nan"
    resolvi_contamination_ci95="nan"
    tco="nan"
    tco_ci95="nan"
    doublet_pct="nan"
    doublet_ci95="nan"
    row_seg_path="${REFERENCE_ARTIFACTS_DIR}/${job}/segger_segmentation.parquet"
    row_anndata_path="${REFERENCE_ARTIFACTS_DIR}/${job}/segger_segmentation.h5ad"

    if [[ -z "${INPUT_DIR}" ]]; then
      validate_status="missing_input_dir"
      validate_error="--input-dir is required for default 10x references"
    elif [[ -z "${REFERENCE_UNIVERSE_SEG_RESOLVED}" ]] || [[ ! -f "${REFERENCE_UNIVERSE_SEG_RESOLVED}" ]]; then
      validate_status="missing_universe_segmentation"
      validate_error="canonical Segger universe segmentation not found"
    else
      mkdir -p "$(dirname "${row_seg_path}")"
      build_cmd=(
        "${SEGGER_PYTHON}" "${REFERENCE_BUILDER_PY}"
        --input-dir "${INPUT_DIR}"
        --canonical-seg "${REFERENCE_UNIVERSE_SEG_RESOLVED}"
        --kind "${reference_kind}"
        --out-seg "${row_seg_path}"
        --out-h5ad "${row_anndata_path}"
      )

      {
        printf '[%s] job=%s BUILD_REF_CMD:' "$(timestamp)" "${job}"
        printf ' %q' "${build_cmd[@]}"
        printf '\n'
      } >> "${LOG_FILE}"

      if "${build_cmd[@]}" >> "${LOG_FILE}" 2>&1; then
        run_validate_for_row "${job}" "${row_seg_path}" "${row_anndata_path}" "0"
      else
        validate_status="reference_artifact_failed"
        validate_error="failed to build default 10x reference artifacts"
      fi
    fi

    append_row \
      "${job}" "${group}" "${gpu}" "1" "${reference_kind}" \
      "${validate_status}" "${validate_error}" "${gpu_time_s}" "${cells}" \
      "${assigned_pct}" "${assigned_ci95}" "${mecr}" "${mecr_ci95}" \
      "${contamination_pct}" "${contamination_ci95}" "${resolvi_contamination_pct}" "${resolvi_contamination_ci95}" "${tco}" "${tco_ci95}" \
      "${doublet_pct}" "${doublet_ci95}" "${row_seg_path}" "${row_anndata_path}"
  done
fi

tmp_sorted="$(mktemp)"
{
  head -n1 "${tmp_out}"
  tail -n +2 "${tmp_out}" \
    | awk -F'\t' '
        function norm_num(v, lower) {
          lower = tolower(v)
          if (v == "" || lower == "nan" || lower == "none") return ""
          return v + 0.0
        }
        {
          assigned = norm_num($10)
          mecr = norm_num($12)
          if (assigned == "") assigned_key = -1
          else assigned_key = assigned
          if (mecr == "") mecr_key = 1e99
          else mecr_key = mecr
          printf "%.10f\t%.10f\t%s\n", assigned_key, mecr_key, $0
        }
      ' \
    | sort -t $'\t' -k1,1gr -k2,2g -k3,3 \
    | cut -f3-
} > "${tmp_sorted}"
mv "${tmp_sorted}" "${tmp_out}"

mv "${tmp_out}" "${OUT_TSV}"
echo "[$(timestamp)] WROTE validation table: ${OUT_TSV}" >> "${LOG_FILE}"
echo "[$(timestamp)] SUMMARY reused=${reused_count} computed=${computed_count}" >> "${LOG_FILE}"
echo "Wrote validation table: ${OUT_TSV} (reused=${reused_count}, computed=${computed_count})"
