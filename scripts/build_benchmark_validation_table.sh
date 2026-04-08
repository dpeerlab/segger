#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build per-job validation metrics table for a benchmark root.

This script intentionally runs a fast validation subset by default:
assigned + mecr only.

Usage:
  bash scripts/build_benchmark_validation_table.sh [options]

Options:
  --root <dir>                    Benchmark root (default: ./results/mossi_main_big_benchmark_nightly)
  --input-dir <dir>               Source dataset path for contamination/geometry/doublet metrics (optional)
  --out-tsv <file>                Output TSV (default: <root>/summaries/validation_metrics_.tsv)
  --recompute                     Recompute all jobs even if already present in output TSV
  --segger-bin <cmd>              Segger executable/command (default: segger)
  --me-gene-pairs-path <file>     Optional ME-gene pair file passed to segger validate
  --tissue-type <name>            Disabled in this benchmark workflow (use --scrna-reference-path)
  --scrna-reference-path <file>   Optional scRNA h5ad passed to segger validate
  --scrna-celltype-column <name>  scRNA cell type column (default: cell_type)
  --max-me-gene-pairs <int>       Max sampled ME-gene pairs (default: 500)
  --gpu-a <id>                    GPU id used for group A labels (default: env GPU_A or 0)
  --gpu-b <id>                    GPU id used for group B labels (default: env GPU_B or 1)
  --include-default-10x <bool>    Include ref_10x_cell/ref_10x_nucleus rows (default: true)
  --exclude-fragment-jobs <bool>  Skip fragment-mode jobs (name contains "fragon") (default: false)
  --progressive-write <bool>      Update output TSV incrementally while running (default: true)
  --include-positive-marker-recall <bool>
                                  Include positive-marker-recall metric (default: false)
  --include-center-border-similarity <bool>
                                  Include center-border-similarity metric (default: false)
  --include-resolvi <bool>        Include resolvi contamination metric (default: false)
  --include-sce <bool>            Include spurious-coexpression metric (default: false)
  --include-mm <bool>             Include morphological-match metric (default: false)
  --include-aes <bool>            Include angular-expression-symmetry metric (default: false)
  --include-vsi <bool>            Include VSI metric (default: false)
  --reference-universe-seg <file> Canonical Segger universe segmentation override
  --only-jobs <csv>               Optional comma-separated job names to validate (e.g. baseline)
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

flag_present_in_help() {
  local help_text="${1:-}"
  local flag="${2:-}"
  local escaped=""
  if [[ -z "${help_text}" ]] || [[ -z "${flag}" ]]; then
    return 1
  fi
  escaped="$(printf '%s' "${flag}" | sed 's/[][(){}.^$*+?|\\/]/\\&/g')"
  # Use -a to treat help output as text even when it contains UTF-8 box drawing
  # or terminal color escape bytes from wrappers.
  printf '%s\n' "${help_text}" | grep -aEiq "(^|[^A-Za-z0-9-])${escaped}([^A-Za-z0-9-]|$)"
}

pick_validate_flag() {
  local help_text="${1:-}"
  shift || true
  local flag
  for flag in "$@"; do
    if flag_present_in_help "${help_text}" "${flag}"; then
      printf '%s' "${flag}"
      return 0
    fi
  done
  return 1
}

trim_spaces() {
  local s="${1:-}"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

csv_contains_exact() {
  local needle="$1"
  local csv="$2"
  local item trimmed
  IFS=',' read -r -a _csv_items <<< "${csv}"
  for item in "${_csv_items[@]}"; do
    trimmed="$(trim_spaces "${item}")"
    if [[ -n "${trimmed}" ]] && [[ "${trimmed}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

metric_missing() {
  local value="${1:-}"
  local lc
  lc="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  [[ -z "${value}" ]] || [[ "${lc}" == "nan" ]] || [[ "${lc}" == "none" ]] || [[ "${lc}" == "null" ]] || [[ "${value}" == "-" ]]
}

ROOT="./results/mossi_main_big_benchmark_nightly"
INPUT_DIR=""
OUT_TSV=""
SEGGER_BIN="segger"
ME_GENE_PAIRS_PATH=""
TISSUE_TYPE=""
SCRNA_REFERENCE_PATH=""
SCRNA_CELLTYPE_COLUMN="cell_type"
MAX_ME_GENE_PAIRS=500
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
INCLUDE_DEFAULT_10X="true"
EXCLUDE_FRAGMENT_JOBS="false"
PROGRESSIVE_WRITE="true"
INCLUDE_POSITIVE_MARKER_RECALL="false"
INCLUDE_BEI="false"
INCLUDE_CTM="false"
INCLUDE_SCE="false"
INCLUDE_MM="false"
INCLUDE_EAU="false"
INCLUDE_VD="false"
REFERENCE_UNIVERSE_SEG=""
ONLY_JOBS_CSV=""
RECOMPUTE=0
VALIDATE_FAST_FLAGS=()
NEED_SOURCE_INPUT="false"
NEED_SCRNA_ALIGNMENT="false"

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
    --tissue-type)
      require_value "$1" "${2-}"
      TISSUE_TYPE="$2"
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
    --exclude-fragment-jobs)
      require_value "$1" "${2-}"
      EXCLUDE_FRAGMENT_JOBS="$2"
      shift 2
      ;;
    --progressive-write)
      require_value "$1" "${2-}"
      PROGRESSIVE_WRITE="$2"
      shift 2
      ;;
    --include-positive-marker-recall)
      require_value "$1" "${2-}"
      INCLUDE_POSITIVE_MARKER_RECALL="$2"
      shift 2
      ;;
    --include-center-border-similarity|--include-center-border-ncv|--include-bei)
      require_value "$1" "${2-}"
      INCLUDE_BEI="$2"
      shift 2
      ;;
    --include-resolvi|--include-border-contamination|--include-ctm)
      require_value "$1" "${2-}"
      INCLUDE_CTM="$2"
      shift 2
      ;;
    --include-spurious-coexpression|--include-sce)
      require_value "$1" "${2-}"
      INCLUDE_SCE="$2"
      shift 2
      ;;
    --include-morphological-match|--include-mm)
      require_value "$1" "${2-}"
      INCLUDE_MM="$2"
      shift 2
      ;;
    --include-aes|--include-tco|--include-eau)
      require_value "$1" "${2-}"
      INCLUDE_EAU="$2"
      shift 2
      ;;
    --include-vsi|--include-vd)
      require_value "$1" "${2-}"
      INCLUDE_VD="$2"
      shift 2
      ;;
    --reference-universe-seg)
      require_value "$1" "${2-}"
      REFERENCE_UNIVERSE_SEG="$2"
      shift 2
      ;;
    --only-jobs)
      require_value "$1" "${2-}"
      ONLY_JOBS_CSV="$2"
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
  OUT_TSV="${ROOT}/summaries/validation_metrics_.tsv"
fi

if ! [[ "${MAX_ME_GENE_PAIRS}" =~ ^[0-9]+$ ]] || [[ "${MAX_ME_GENE_PAIRS}" -le 0 ]]; then
  echo "ERROR: --max-me-gene-pairs must be a positive integer." >&2
  exit 1
fi

if ! INCLUDE_DEFAULT_10X="$(normalize_bool "${INCLUDE_DEFAULT_10X}")"; then
  echo "ERROR: --include-default-10x must be true/false." >&2
  exit 1
fi
if ! EXCLUDE_FRAGMENT_JOBS="$(normalize_bool "${EXCLUDE_FRAGMENT_JOBS}")"; then
  echo "ERROR: --exclude-fragment-jobs must be true/false." >&2
  exit 1
fi
if ! PROGRESSIVE_WRITE="$(normalize_bool "${PROGRESSIVE_WRITE}")"; then
  echo "ERROR: --progressive-write must be true/false." >&2
  exit 1
fi
if ! INCLUDE_POSITIVE_MARKER_RECALL="$(normalize_bool "${INCLUDE_POSITIVE_MARKER_RECALL}")"; then
  echo "ERROR: --include-positive-marker-recall must be true/false." >&2
  exit 1
fi
if ! INCLUDE_BEI="$(normalize_bool "${INCLUDE_BEI}")"; then
  echo "ERROR: --include-bei must be true/false." >&2
  exit 1
fi
if ! INCLUDE_CTM="$(normalize_bool "${INCLUDE_CTM}")"; then
  echo "ERROR: --include-ctm must be true/false." >&2
  exit 1
fi
if ! INCLUDE_SCE="$(normalize_bool "${INCLUDE_SCE}")"; then
  echo "ERROR: --include-sce must be true/false." >&2
  exit 1
fi
if ! INCLUDE_MM="$(normalize_bool "${INCLUDE_MM}")"; then
  echo "ERROR: --include-mm must be true/false." >&2
  exit 1
fi
if ! INCLUDE_EAU="$(normalize_bool "${INCLUDE_EAU}")"; then
  echo "ERROR: --include-eau must be true/false." >&2
  exit 1
fi
if ! INCLUDE_VD="$(normalize_bool "${INCLUDE_VD}")"; then
  echo "ERROR: --include-vd must be true/false." >&2
  exit 1
fi

read -r -a SEGGER_BIN_PARTS <<< "${SEGGER_BIN}"
if [[ ${#SEGGER_BIN_PARTS[@]} -eq 0 ]]; then
  echo "ERROR: --segger-bin resolved to empty command." >&2
  exit 1
fi
if ! command -v "${SEGGER_BIN_PARTS[0]}" >/dev/null 2>&1; then
  echo "ERROR: segger executable not found: ${SEGGER_BIN_PARTS[0]}" >&2
  exit 1
fi

VALIDATE_HELP_TEXT="$("${SEGGER_BIN_PARTS[@]}" validate --help 2>&1 || true)"
if [[ -z "${VALIDATE_HELP_TEXT}" ]]; then
  echo "WARN: could not inspect '${SEGGER_BIN} validate --help'; using fallback validate flags." >&2
fi

coverage_flag="$(pick_validate_flag "${VALIDATE_HELP_TEXT}" --coverage --cov --assigned || true)"
mecr_flag="$(pick_validate_flag "${VALIDATE_HELP_TEXT}" --mecr || true)"
pmr_flag="$(pick_validate_flag "${VALIDATE_HELP_TEXT}" --positive-marker-recall --pmr || true)"
bei_flag="$(pick_validate_flag "${VALIDATE_HELP_TEXT}" --border-expression-integrity --bei --center-border-ncv --center-border-similarity || true)"
ctm_flag="$(pick_validate_flag "${VALIDATE_HELP_TEXT}" --contamination --ctm --resolvi --border-contamination || true)"
sce_flag="$(pick_validate_flag "${VALIDATE_HELP_TEXT}" --spurious-coexpression --sce || true)"
mm_flag="$(pick_validate_flag "${VALIDATE_HELP_TEXT}" --morphological-match --mm || true)"
eau_flag="$(pick_validate_flag "${VALIDATE_HELP_TEXT}" --expression-angular-uniformity --eau --aes --tco || true)"
vd_flag="$(pick_validate_flag "${VALIDATE_HELP_TEXT}" --vertical-doublet --vd --vsi || true)"

if [[ -z "${coverage_flag}" ]]; then
  if [[ -n "${VALIDATE_HELP_TEXT}" ]]; then
    echo "WARN: coverage flags not detected from 'segger validate --help'; falling back to --coverage." >&2
  fi
  coverage_flag="--coverage"
fi
if [[ -z "${mecr_flag}" ]]; then
  if [[ -n "${VALIDATE_HELP_TEXT}" ]]; then
    echo "WARN: MECR flag not detected from 'segger validate --help'; falling back to --mecr." >&2
  fi
  mecr_flag="--mecr"
fi

VALIDATE_FAST_FLAGS=("${coverage_flag}" "${mecr_flag}")
if [[ "${INCLUDE_POSITIVE_MARKER_RECALL}" == "true" ]]; then
  if [[ -z "${pmr_flag}" ]]; then
    echo "WARN: PMR flag not detected from 'segger validate --help'; falling back to --positive-marker-recall." >&2
    pmr_flag="--positive-marker-recall"
  fi
  VALIDATE_FAST_FLAGS+=("${pmr_flag}")
fi
if [[ "${INCLUDE_BEI}" == "true" ]]; then
  if [[ -z "${bei_flag}" ]]; then
    echo "WARN: BEI flag not detected from 'segger validate --help'; falling back to --border-expression-integrity." >&2
    bei_flag="--border-expression-integrity"
  fi
  VALIDATE_FAST_FLAGS+=("${bei_flag}")
fi
if [[ "${INCLUDE_CTM}" == "true" ]]; then
  if [[ -z "${ctm_flag}" ]]; then
    echo "WARN: CTM flag not detected from 'segger validate --help'; falling back to --contamination." >&2
    ctm_flag="--contamination"
  fi
  VALIDATE_FAST_FLAGS+=("${ctm_flag}")
fi
if [[ "${INCLUDE_SCE}" == "true" ]]; then
  if [[ -z "${sce_flag}" ]]; then
    echo "WARN: SCE flag not detected from 'segger validate --help'; falling back to --spurious-coexpression." >&2
    sce_flag="--spurious-coexpression"
  fi
  VALIDATE_FAST_FLAGS+=("${sce_flag}")
fi
if [[ "${INCLUDE_MM}" == "true" ]]; then
  if [[ -z "${mm_flag}" ]]; then
    echo "WARN: MM flag not detected from 'segger validate --help'; falling back to --morphological-match." >&2
    mm_flag="--morphological-match"
  fi
  VALIDATE_FAST_FLAGS+=("${mm_flag}")
fi
if [[ "${INCLUDE_EAU}" == "true" ]]; then
  if [[ -z "${eau_flag}" ]]; then
    echo "WARN: EAU flag not detected from 'segger validate --help'; falling back to --expression-angular-uniformity." >&2
    eau_flag="--expression-angular-uniformity"
  fi
  VALIDATE_FAST_FLAGS+=("${eau_flag}")
fi
if [[ "${INCLUDE_VD}" == "true" ]]; then
  if [[ -z "${vd_flag}" ]]; then
    echo "WARN: VD flag not detected from 'segger validate --help'; falling back to --vertical-doublet." >&2
    vd_flag="--vertical-doublet"
  fi
  VALIDATE_FAST_FLAGS+=("${vd_flag}")
fi

if [[ "${INCLUDE_POSITIVE_MARKER_RECALL}" == "true" ]] || [[ "${INCLUDE_CTM}" == "true" ]]; then
  NEED_SCRNA_ALIGNMENT="true"
fi
if [[ "${INCLUDE_POSITIVE_MARKER_RECALL}" == "true" ]] || \
   [[ "${INCLUDE_BEI}" == "true" ]] || \
   [[ "${INCLUDE_CTM}" == "true" ]] || \
   [[ "${INCLUDE_SCE}" == "true" ]] || \
   [[ "${INCLUDE_MM}" == "true" ]] || \
   [[ "${INCLUDE_EAU}" == "true" ]] || \
   [[ "${INCLUDE_VD}" == "true" ]]; then
  NEED_SOURCE_INPUT="true"
fi

if [[ -n "${INPUT_DIR}" ]] && [[ ! -e "${INPUT_DIR}" ]]; then
  echo "ERROR: --input-dir not found: ${INPUT_DIR}" >&2
  exit 1
fi
if [[ "${NEED_SOURCE_INPUT}" == "true" ]] && [[ -z "${INPUT_DIR}" ]]; then
  echo "ERROR: --input-dir is required for selected source-dependent metrics." >&2
  exit 1
fi

if [[ -n "${ME_GENE_PAIRS_PATH}" ]] && [[ ! -f "${ME_GENE_PAIRS_PATH}" ]]; then
  echo "ERROR: --me-gene-pairs-path not found: ${ME_GENE_PAIRS_PATH}" >&2
  exit 1
fi

if [[ -n "${TISSUE_TYPE}" ]]; then
  echo "ERROR: --tissue-type is disabled for this benchmark workflow; use --scrna-reference-path with local .h5ad." >&2
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
OUT_TSV_BASENAME="$(basename "${OUT_TSV}")"
OUT_TSV_STEM="${OUT_TSV_BASENAME}"
if [[ "${OUT_TSV_STEM}" == *.tsv ]]; then
  OUT_TSV_STEM="${OUT_TSV_STEM%.tsv}"
fi
LOG_FILE="${SUMMARY_DIR}/${OUT_TSV_STEM}.log"
REFERENCE_ARTIFACTS_DIR="${SUMMARY_DIR}/reference_artifacts"
VALIDATE_ARTIFACTS_DIR="${SUMMARY_DIR}/validation_artifacts"
VALIDATE_ROWS_DIR="${VALIDATE_ARTIFACTS_DIR}/rows"
VALIDATE_CMD_LOGS_DIR="${VALIDATE_ARTIFACTS_DIR}/cmd_logs"
CURRENT_VALIDATE_ROW_FILE=""
PARTIAL_OUT_TSV="${SUMMARY_DIR}/${OUT_TSV_STEM}.partial.tsv"

if [[ ! -f "${PLAN_FILE}" ]]; then
  echo "ERROR: Missing benchmark plan file: ${PLAN_FILE}" >&2
  exit 1
fi

mkdir -p "${SUMMARY_DIR}" "$(dirname "${OUT_TSV}")" "${REFERENCE_ARTIFACTS_DIR}" "${VALIDATE_ROWS_DIR}" "${VALIDATE_CMD_LOGS_DIR}"

LOCK_FILE="${SUMMARY_DIR}/${OUT_TSV_STEM}.lock"
LOCK_FD=""
LOCK_MODE=""

if command -v flock >/dev/null 2>&1; then
  exec {LOCK_FD}> "${LOCK_FILE}"
  if ! flock -n "${LOCK_FD}"; then
    echo "ERROR: Validation lock busy for ${ROOT}. Another build may still be running (${LOCK_FILE})." >&2
    exit 2
  fi
  LOCK_MODE="flock"
else
  if ( set -o noclobber; : > "${LOCK_FILE}" ) 2>/dev/null; then
    LOCK_MODE="file"
  else
    echo "ERROR: Validation lock busy for ${ROOT}. Another build may still be running (${LOCK_FILE})." >&2
    exit 2
  fi
fi

tmp_out="$(mktemp)"
backup_out=""
build_completed=0
if [[ -f "${OUT_TSV}" ]]; then
  backup_out="$(mktemp "${TMPDIR:-/tmp}/segger_validate_backup.XXXXXX.tsv")"
  cp "${OUT_TSV}" "${backup_out}"
fi

cleanup_validation_build() {
  local rc=$?
  local partial_line_count

  # Avoid leaving partially-written tables behind on interrupted/failed runs.
  if [[ "${build_completed}" != "1" ]]; then
    if [[ -f "${tmp_out}" ]]; then
      partial_line_count="$(wc -l < "${tmp_out}" | tr -d '[:space:]')"
      if [[ "${partial_line_count:-0}" -gt 1 ]]; then
        cp "${tmp_out}" "${PARTIAL_OUT_TSV}" 2>/dev/null || true
        echo "[$(timestamp)] WARN saved partial validation table: ${PARTIAL_OUT_TSV}" >> "${LOG_FILE}" 2>/dev/null || true
      fi
    fi
    if [[ -n "${backup_out}" && -f "${backup_out}" ]]; then
      cp "${backup_out}" "${OUT_TSV}" 2>/dev/null || true
    elif [[ "${PROGRESSIVE_WRITE}" == "true" && -f "${OUT_TSV}" ]]; then
      rm -f "${OUT_TSV}" || true
    fi
  else
    rm -f "${PARTIAL_OUT_TSV}" 2>/dev/null || true
  fi

  rm -f "${tmp_out}"
  if [[ -n "${backup_out}" ]]; then
    rm -f "${backup_out}"
  fi

  if [[ "${LOCK_MODE}" == "file" ]]; then
    rm -f "${LOCK_FILE}" 2>/dev/null || true
  elif [[ "${LOCK_MODE}" == "flock" && -n "${LOCK_FD}" ]]; then
    eval "exec ${LOCK_FD}>&-"
  fi

  return "${rc}"
}

trap cleanup_validation_build EXIT INT TERM

METRIC_SCHEMA_VERSION="2026-04-05-v17-cov_mecr-pmr_${INCLUDE_POSITIVE_MARKER_RECALL}-bei_${INCLUDE_BEI}-ctm_${INCLUDE_CTM}-sce_${INCLUDE_SCE}-mm_${INCLUDE_MM}-eau_${INCLUDE_EAU}-vd_${INCLUDE_VD}"
RUN_INPUT_DIR_TOKEN="$(normalize_token "${INPUT_DIR}")"
if [[ -n "${TISSUE_TYPE}" ]]; then
  RUN_SCRNA_REFERENCE_TOKEN="$(normalize_token "tissue:${TISSUE_TYPE}")"
else
  RUN_SCRNA_REFERENCE_TOKEN="$(normalize_token "${SCRNA_REFERENCE_PATH}")"
fi
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

OUTPUT_HEADER=$'job\tgroup\tgpu\tis_reference\treference_kind\tvalidate_status\tvalidate_error\tgpu_time_s\tcells\tfragments\tassigned_pct\tassigned_ci95\tpositive_marker_recall_pct\tpositive_marker_recall_ci95\tmecr\tmecr_ci95\tcontamination_pct\tcontamination_ci95\tresolvi_contamination_pct\tresolvi_contamination_ci95\tcenter_border_similarity\tcenter_border_similarity_ci95\tspurious_coexpression\tspurious_coexpression_ci95\treference_morphology_match\treference_morphology_match_ci95\taes\taes_ci95\tdoublet_pct\tdoublet_ci95\tsegmentation_path\tanndata_path\toutput_path\tupdated_at\tmetric_schema_version\trun_input_dir\trun_scrna_reference_path\trun_me_gene_pairs_path\trun_reference_universe_seg'
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
  if [[ -z "${CURRENT_VALIDATE_ROW_FILE}" ]] || [[ ! -s "${CURRENT_VALIDATE_ROW_FILE}" ]]; then
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
  ' "${CURRENT_VALIDATE_ROW_FILE}"
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

resolve_segmentation_path_for_job() {
  local job_name="$1"
  local primary_path="${RUNS_DIR}/${job_name}/segger_segmentation.parquet"
  local fallback_path="${RUNS_DIR}/${job_name}/transcripts.parquet"

  if [[ -f "${primary_path}" ]]; then
    printf '%s' "${primary_path}"
    return 0
  fi

  if [[ -f "${fallback_path}" ]]; then
    echo "[$(timestamp)] WARN job=${job_name} missing segger_segmentation.parquet; using transcripts.parquet fallback" >> "${LOG_FILE}"
    printf '%s' "${fallback_path}"
    return 0
  fi

  printf '%s' "${primary_path}"
}

resolve_anndata_path_for_job() {
  local job_name="$1"
  local primary_path="${EXPORTS_DIR}/${job_name}/anndata/segger_segmentation.h5ad"
  local cells_only_path="${EXPORTS_DIR}/${job_name}/anndata/segger_cells.h5ad"
  local flat_primary_path="${EXPORTS_DIR}/${job_name}/segger_segmentation.h5ad"
  local flat_cells_only_path="${EXPORTS_DIR}/${job_name}/segger_cells.h5ad"
  local run_primary_path="${RUNS_DIR}/${job_name}/segger_segmentation.h5ad"
  local discovered_path=""

  # Preferred canonical export path
  if [[ -f "${primary_path}" ]]; then
    printf '%s' "${primary_path}"
    return 0
  fi

  # Common alternate export layouts (older scripts/manual exports)
  for discovered_path in \
    "${flat_primary_path}" \
    "${cells_only_path}" \
    "${flat_cells_only_path}" \
    "${run_primary_path}"; do
    if [[ -f "${discovered_path}" ]]; then
      if [[ "${discovered_path}" == *segger_cells.h5ad ]]; then
        echo "[$(timestamp)] WARN job=${job_name} missing segger_segmentation.h5ad; using ${discovered_path}" >> "${LOG_FILE}"
      elif [[ "${discovered_path}" != "${primary_path}" ]]; then
        echo "[$(timestamp)] WARN job=${job_name} using non-canonical AnnData path: ${discovered_path}" >> "${LOG_FILE}"
      fi
      printf '%s' "${discovered_path}"
      return 0
    fi
  done

  # Last resort: search export subtree for a plausible h5ad artifact.
  discovered_path="$(
    find "${EXPORTS_DIR}/${job_name}" -maxdepth 3 -type f \( -name 'segger_segmentation.h5ad' -o -name 'segger_cells.h5ad' \) \
      2>/dev/null | sort | head -n1 || true
  )"
  if [[ -n "${discovered_path}" ]]; then
    echo "[$(timestamp)] WARN job=${job_name} discovered AnnData path via search: ${discovered_path}" >> "${LOG_FILE}"
    printf '%s' "${discovered_path}"
    return 0
  fi

  # Keep canonical target for traceability even when file is absent.
  printf '%s' "${primary_path}"
}

resolve_reference_anndata_path_for_job() {
  local job_name="$1"
  local reference_primary_path="${REFERENCE_ARTIFACTS_DIR}/${job_name}/segger_segmentation.h5ad"
  local reference_cells_only_path="${REFERENCE_ARTIFACTS_DIR}/${job_name}/segger_cells.h5ad"
  local discovered_path=""

  if [[ -f "${reference_primary_path}" ]]; then
    printf '%s' "${reference_primary_path}"
    return 0
  fi

  if [[ -f "${reference_cells_only_path}" ]]; then
    echo "[$(timestamp)] WARN job=${job_name} missing reference segger_segmentation.h5ad; using ${reference_cells_only_path}" >> "${LOG_FILE}"
    printf '%s' "${reference_cells_only_path}"
    return 0
  fi

  discovered_path="$(
    find "${REFERENCE_ARTIFACTS_DIR}/${job_name}" -maxdepth 3 -type f \( -name 'segger_segmentation.h5ad' -o -name 'segger_cells.h5ad' \) \
      2>/dev/null | sort | head -n1 || true
  )"
  if [[ -n "${discovered_path}" ]]; then
    echo "[$(timestamp)] WARN job=${job_name} discovered reference AnnData path via search: ${discovered_path}" >> "${LOG_FILE}"
    printf '%s' "${discovered_path}"
    return 0
  fi

  printf '%s' "${reference_primary_path}"
}

resolve_any_anndata_path_for_job() {
  local job_name="$1"
  local is_reference="${2:-0}"
  if [[ "${is_reference}" == "1" ]] || [[ "${job_name}" == ref_* ]]; then
    resolve_reference_anndata_path_for_job "${job_name}"
    return 0
  fi
  resolve_anndata_path_for_job "${job_name}"
}

get_existing_row_with_field_override() {
  local job="$1"
  local field_name="$2"
  local replacement="$3"
  awk -F'\t' -v OFS='\t' -v j="${job}" -v f="${field_name}" -v r="${replacement}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == f) {
          col = i
          break
        }
      }
      next
    }
    $1 == j {
      if (col > 0) {
        $col = r
      }
      print
      exit
    }
  ' "${OUT_TSV}"
}

scale_frac_to_pct() {
  local value="${1:-}"
  local lower
  lower="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "${value}" ]] || [[ "${lower}" == "nan" ]] || [[ "${lower}" == "none" ]]; then
    printf '%s' "nan"
    return 0
  fi
  awk -v v="${value}" '
    BEGIN {
      x = v + 0.0
      if (x >= 0.0 && x <= 1.0) {
        printf "%.6f", x * 100.0
      } else {
        printf "%.6f", x
      }
    }
  '
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
  local fragments="${10}"
  local assigned_pct="${11}"
  local assigned_ci95="${12}"
  local positive_marker_recall_pct="${13}"
  local positive_marker_recall_ci95="${14}"
  local mecr="${15}"
  local mecr_ci95="${16}"
  local contamination_pct="${17}"
  local contamination_ci95="${18}"
  local resolvi_contamination_pct="${19}"
  local resolvi_contamination_ci95="${20}"
  local center_border_similarity="${21}"
  local center_border_similarity_ci95="${22}"
  local spurious_coexpression="${23}"
  local spurious_coexpression_ci95="${24}"
  local reference_morphology_match="${25}"
  local reference_morphology_match_ci95="${26}"
  local aes="${27}"
  local aes_ci95="${28}"
  local doublet_pct="${29}"
  local doublet_ci95="${30}"
  local row_seg_path="${31}"
  local row_anndata_path="${32}"

  validate_error="$(sanitize_tsv_field "${validate_error}")"
  reference_kind="$(normalize_token "${reference_kind}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${job}" "${group}" "${gpu}" "${is_reference}" "${reference_kind}" "${validate_status}" "${validate_error}" "${gpu_time_s}" \
    "${cells}" "${fragments}" "${assigned_pct}" "${assigned_ci95}" "${positive_marker_recall_pct}" "${positive_marker_recall_ci95}" "${mecr}" "${mecr_ci95}" \
    "${contamination_pct}" "${contamination_ci95}" "${resolvi_contamination_pct}" "${resolvi_contamination_ci95}" "${center_border_similarity}" "${center_border_similarity_ci95}" \
    "${spurious_coexpression}" "${spurious_coexpression_ci95}" "${reference_morphology_match}" "${reference_morphology_match_ci95}" "${aes}" "${aes_ci95}" \
    "${doublet_pct}" "${doublet_ci95}" "${row_seg_path}" "${row_anndata_path}" "${OUT_TSV}" \
    "$(timestamp)" "${METRIC_SCHEMA_VERSION}" "${RUN_INPUT_DIR_TOKEN}" "${RUN_SCRNA_REFERENCE_TOKEN}" "${RUN_ME_GENE_PAIRS_TOKEN}" "${RUN_REFERENCE_UNIVERSE_TOKEN}" \
    >> "${tmp_out}"

  if [[ "${PROGRESSIVE_WRITE}" == "true" ]]; then
    cp "${tmp_out}" "${OUT_TSV}"
  fi
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
  elif metric_missing "${existing_cells:-}"; then
    should_reuse=0
    reason="missing_cells"
  elif [[ "${NEED_SOURCE_INPUT}" == "true" ]]; then
    if [[ "${INCLUDE_EAU}" == "true" ]]; then
      existing_eau="$(get_existing_field_by_job "${job}" "eau")"
      if metric_missing "${existing_eau:-}"; then
        should_reuse=0
        reason="missing_eau_metric"
      fi
    fi
    if [[ "${should_reuse}" == "1" && "${INCLUDE_BEI}" == "true" ]]; then
      existing_bei="$(get_existing_field_by_job "${job}" "center_border_similarity")"
      if metric_missing "${existing_bei:-}"; then
        should_reuse=0
        reason="missing_bei_metric"
      fi
    fi
    if [[ "${should_reuse}" == "1" && "${INCLUDE_VD}" == "true" ]]; then
      existing_doublet="$(get_existing_field_by_job "${job}" "doublet_pct")"
      if metric_missing "${existing_doublet:-}"; then
        should_reuse=0
        reason="missing_vd_metric"
      fi
    fi
    if [[ "${should_reuse}" == "1" && "${INCLUDE_SCE}" == "true" ]]; then
      existing_sce="$(get_existing_field_by_job "${job}" "spurious_coexpression")"
      if metric_missing "${existing_sce:-}"; then
        should_reuse=0
        reason="missing_sce_metric"
      fi
    fi
    if [[ "${should_reuse}" == "1" && "${INCLUDE_MM}" == "true" ]]; then
      existing_mm="$(get_existing_field_by_job "${job}" "reference_morphology_match")"
      if metric_missing "${existing_mm:-}"; then
        should_reuse=0
        reason="missing_mm_metric"
      fi
    fi
    if [[ "${should_reuse}" == "1" && "${NEED_SCRNA_ALIGNMENT}" == "true" && "${RUN_SCRNA_REFERENCE_TOKEN}" != "-" ]]; then
      if [[ "${INCLUDE_CTM}" == "true" ]]; then
        existing_ctm="$(get_existing_field_by_job "${job}" "contamination_pct")"
        if metric_missing "${existing_ctm:-}"; then
          should_reuse=0
          reason="missing_ctm_metric"
        fi
      fi
      if [[ "${should_reuse}" == "1" && "${INCLUDE_POSITIVE_MARKER_RECALL}" == "true" ]]; then
        existing_marker_recall="$(get_existing_field_by_job "${job}" "positive_marker_recall_pct")"
        if metric_missing "${existing_marker_recall:-}"; then
          should_reuse=0
          reason="missing_positive_marker_recall_metric"
        fi
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
  local job_tag
  local row_file
  local cmd_log_file

  validate_status="ok"
  validate_error=""
  cells="0"
  fragments="0"
  assigned_pct="nan"
  assigned_ci95="nan"
  positive_marker_recall_pct="nan"
  positive_marker_recall_ci95="nan"
  mecr="nan"
  mecr_ci95="nan"
  contamination_pct="nan"
  contamination_ci95="nan"
  resolvi_contamination_pct="nan"
  resolvi_contamination_ci95="nan"
  center_border_similarity="nan"
  center_border_similarity_ci95="nan"
  spurious_coexpression="nan"
  spurious_coexpression_ci95="nan"
  reference_morphology_match="nan"
  reference_morphology_match_ci95="nan"
  aes="nan"
  aes_ci95="nan"
  doublet_pct="nan"
  doublet_ci95="nan"
  row_seg_path="${seg_path}"
  row_anndata_path="${anndata_path}"

  if [[ ! -f "${seg_path}" ]]; then
    validate_status="missing_segmentation"
    validate_error="segger_segmentation.parquet not found"
    return 0
  fi

  job_tag="$(printf '%s' "${job}" | tr -c 'A-Za-z0-9._-' '_')"
  row_file="${VALIDATE_ROWS_DIR}/${job_tag}.tsv"
  cmd_log_file="${VALIDATE_CMD_LOGS_DIR}/${job_tag}.log"
  CURRENT_VALIDATE_ROW_FILE="${row_file}"
  : > "${row_file}"
  : > "${cmd_log_file}"
  cmd=(
    "${SEGGER_BIN_PARTS[@]}" validate
    -s "${seg_path}"
    -o "${row_file}"
    "${VALIDATE_FAST_FLAGS[@]}"
    --max-me-gene-pairs "${MAX_ME_GENE_PAIRS}"
  )
  if [[ -f "${anndata_path}" ]]; then
    cmd+=(-a "${anndata_path}")
  fi
  if [[ "${NEED_SOURCE_INPUT}" == "true" && -n "${INPUT_DIR}" ]]; then
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
    printf '[%s] job=%s ARTIFACTS: row=%s cmd_log=%s\n' "$(timestamp)" "${job}" "${row_file}" "${cmd_log_file}"
  } >> "${LOG_FILE}"

  local failure_hint

  if "${cmd[@]}" > "${cmd_log_file}" 2>&1; then
    cat "${cmd_log_file}" >> "${LOG_FILE}"
    parsed_status="$(get_field "validate_status")"
    [[ -n "${parsed_status}" ]] && validate_status="${parsed_status}" || true
    parsed_error="$(get_field "validate_error")"
    [[ -n "${parsed_error}" ]] && validate_error="${parsed_error}" || true
    # Newer validate writes per-metric runtime failures here even with rc=0.
    if [[ -z "${validate_error}" ]]; then
      parsed_error="$(get_field "validate_metric_errors")"
      [[ -n "${parsed_error}" ]] && validate_error="${parsed_error}" || true
    fi

    parsed_elapsed="$(get_field "elapsed_s")"
    if [[ "${apply_elapsed_fallback}" == "1" ]] && [[ -n "${parsed_elapsed}" ]] && [[ "${gpu_time_s}" == "0" || -z "${gpu_time_s}" ]]; then
      gpu_time_s="${parsed_elapsed}"
    fi

    parsed_val="$(get_field "cells_non_fragment_total")"
    if [[ -n "${parsed_val}" ]] && [[ "$(printf '%s' "${parsed_val}" | tr '[:upper:]' '[:lower:]')" != "nan" ]]; then
      cells="${parsed_val}"
    else
      parsed_val="$(get_field "cells_assigned")"
      if [[ -n "${parsed_val}" ]] && [[ "$(printf '%s' "${parsed_val}" | tr '[:upper:]' '[:lower:]')" != "nan" ]]; then
        cells="${parsed_val}"
      else
        parsed_val="$(get_field "cells_total")"
        [[ -n "${parsed_val}" ]] && cells="${parsed_val}" || true
      fi
    fi

    parsed_val="$(get_field "fragments_total")"
    if [[ -n "${parsed_val}" ]] && [[ "$(printf '%s' "${parsed_val}" | tr '[:upper:]' '[:lower:]')" != "nan" ]]; then
      fragments="${parsed_val}"
    else
      parsed_val="$(get_field "fragments_assigned")"
      [[ -n "${parsed_val}" ]] && fragments="${parsed_val}" || true
    fi

    parsed_val="$(get_field "coverage_pct")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "transcripts_assigned_pct")"
    fi
    [[ -n "${parsed_val}" ]] && assigned_pct="${parsed_val}" || true
    parsed_val="$(get_field "coverage_pct_ci95")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "transcripts_assigned_pct_ci95")"
    fi
    [[ -n "${parsed_val}" ]] && assigned_ci95="${parsed_val}" || true

    parsed_val="$(get_field "positive_marker_recall_fast")"
    [[ -n "${parsed_val}" ]] && positive_marker_recall_pct="${parsed_val}" || true
    parsed_val="$(get_field "positive_marker_recall_ci95_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "positive_marker_recall_ci95")"
    fi
    [[ -n "${parsed_val}" ]] && positive_marker_recall_ci95="${parsed_val}" || true

    parsed_val="$(get_field "mecr_fast")"
    [[ -n "${parsed_val}" ]] && mecr="${parsed_val}" || true
    parsed_val="$(get_field "mecr_ci95_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "mecr_ci95")"
    fi
    [[ -n "${parsed_val}" ]] && mecr_ci95="${parsed_val}" || true

    parsed_val="$(get_field "border_contaminated_cells_pct_fast")"
    [[ -n "${parsed_val}" ]] && contamination_pct="${parsed_val}" || true
    parsed_val="$(get_field "border_contaminated_cells_pct_ci95_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "contamination_ci95")"
    fi
    [[ -n "${parsed_val}" ]] && contamination_ci95="${parsed_val}" || true

    parsed_val="$(get_field "contamination_pct_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "resolvi_contamination_pct_fast")"
    fi
    [[ -n "${parsed_val}" ]] && resolvi_contamination_pct="${parsed_val}" || true
    parsed_val="$(get_field "contamination_ci95_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "resolvi_contamination_ci95_fast")"
    fi
    [[ -n "${parsed_val}" ]] && resolvi_contamination_ci95="${parsed_val}" || true

    parsed_val="$(get_field "border_expression_integrity_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "center_border_similarity_score_fast")"
    fi
    [[ -n "${parsed_val}" ]] && center_border_similarity="${parsed_val}" || true
    parsed_val="$(get_field "border_expression_integrity_ci95_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "border_expression_integrity_ci95")"
    fi
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "center_border_similarity_ci95_fast")"
    fi
    [[ -n "${parsed_val}" ]] && center_border_similarity_ci95="${parsed_val}" || true

    parsed_val="$(get_field "spurious_coexpression_fast")"
    [[ -n "${parsed_val}" ]] && spurious_coexpression="${parsed_val}" || true
    parsed_val="$(get_field "spurious_coexpression_ci95_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "spurious_coexpression_ci95")"
    fi
    [[ -n "${parsed_val}" ]] && spurious_coexpression_ci95="${parsed_val}" || true

    parsed_val="$(get_field "morphological_match_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "reference_morphology_match_fast")"
    fi
    [[ -n "${parsed_val}" ]] && reference_morphology_match="${parsed_val}" || true
    parsed_val="$(get_field "morphological_match_ci95_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "reference_morphology_match_ci95_fast")"
    fi
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "morphological_match_ci95")"
    fi
    [[ -n "${parsed_val}" ]] && reference_morphology_match_ci95="${parsed_val}" || true

    parsed_val="$(get_field "expression_angular_uniformity_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "angular_expression_symmetry_fast")"
    fi
    [[ -n "${parsed_val}" ]] && aes="${parsed_val}" || true
    parsed_val="$(get_field "expression_angular_uniformity_ci95_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "expression_angular_uniformity_ci95")"
    fi
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "angular_expression_symmetry_ci95_fast")"
    fi
    [[ -n "${parsed_val}" ]] && aes_ci95="${parsed_val}" || true

    parsed_val="$(get_field "vsi_doublet_fraction_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "vertical_doublet_pct_fast")"
    fi
    if [[ -n "${parsed_val}" ]]; then
      doublet_pct="$(scale_frac_to_pct "${parsed_val}")"
    fi
    parsed_val="$(get_field "vsi_doublet_fraction_ci95_fast")"
    if [[ -z "${parsed_val}" ]]; then
      parsed_val="$(get_field "vertical_doublet_pct_ci95_fast")"
    fi
    if [[ -n "${parsed_val}" ]]; then
      doublet_ci95="$(scale_frac_to_pct "${parsed_val}")"
    fi

    parsed_val="$(get_field "segmentation_path")"
    [[ -n "${parsed_val}" ]] && row_seg_path="${parsed_val}" || true
    parsed_val="$(get_field "anndata_path")"
    [[ -n "${parsed_val}" ]] && row_anndata_path="${parsed_val}" || true
  else
    cat "${cmd_log_file}" >> "${LOG_FILE}"
    validate_status="validate_command_failed"
    failure_hint="$(
      tail -n 8 "${cmd_log_file}" \
        | tr '\n' ' ' \
        | sed 's/[[:space:]]\+/ /g; s/^[[:space:]]*//; s/[[:space:]]*$//'
    )"
    if [[ -n "${failure_hint}" ]]; then
      validate_error="segger validate command failed: ${failure_hint}"
    else
      validate_error="segger validate command failed"
    fi
  fi

}

printf "%s\n" "${OUTPUT_HEADER}" > "${tmp_out}"
if [[ "${PROGRESSIVE_WRITE}" == "true" ]]; then
  cp "${tmp_out}" "${OUT_TSV}"
fi

echo "[$(timestamp)] START benchmark validation table build" >> "${LOG_FILE}"
echo "[$(timestamp)] ROOT=${ROOT}" >> "${LOG_FILE}"
echo "[$(timestamp)] OUT_TSV=${OUT_TSV}" >> "${LOG_FILE}"
echo "[$(timestamp)] RECOMPUTE=${RECOMPUTE}" >> "${LOG_FILE}"
echo "[$(timestamp)] INCLUDE_DEFAULT_10X=${INCLUDE_DEFAULT_10X}" >> "${LOG_FILE}"
echo "[$(timestamp)] EXCLUDE_FRAGMENT_JOBS=${EXCLUDE_FRAGMENT_JOBS}" >> "${LOG_FILE}"
echo "[$(timestamp)] PROGRESSIVE_WRITE=${PROGRESSIVE_WRITE}" >> "${LOG_FILE}"
echo "[$(timestamp)] INCLUDE_POSITIVE_MARKER_RECALL=${INCLUDE_POSITIVE_MARKER_RECALL}" >> "${LOG_FILE}"
echo "[$(timestamp)] INCLUDE_BEI=${INCLUDE_BEI}" >> "${LOG_FILE}"
echo "[$(timestamp)] INCLUDE_CTM=${INCLUDE_CTM}" >> "${LOG_FILE}"
echo "[$(timestamp)] INCLUDE_EAU=${INCLUDE_EAU}" >> "${LOG_FILE}"
echo "[$(timestamp)] INCLUDE_VD=${INCLUDE_VD}" >> "${LOG_FILE}"
echo "[$(timestamp)] SEGGER_BIN=${SEGGER_BIN}" >> "${LOG_FILE}"
echo "[$(timestamp)] VALIDATE_FAST_FLAGS=${VALIDATE_FAST_FLAGS[*]}" >> "${LOG_FILE}"
echo "[$(timestamp)] NEED_SOURCE_INPUT=${NEED_SOURCE_INPUT}" >> "${LOG_FILE}"
echo "[$(timestamp)] REFERENCE_UNIVERSE_SEG=${RUN_REFERENCE_UNIVERSE_TOKEN}" >> "${LOG_FILE}"
echo "[$(timestamp)] ONLY_JOBS=${ONLY_JOBS_CSV:-<all>}" >> "${LOG_FILE}"
echo "[$(timestamp)] LOCK=${LOCK_FILE} mode=${LOCK_MODE} pid=$$" >> "${LOG_FILE}"

reused_count=0
computed_count=0

while IFS=$'\t' read -r \
  job group _use3d _expansion _txk _txdist _layers _heads _cellsmin _minqv _alignment; do
  if [[ -z "${job:-}" ]] || [[ "${job}" == "job" ]]; then
    continue
  fi
  if [[ -n "${ONLY_JOBS_CSV}" ]] && ! csv_contains_exact "${job}" "${ONLY_JOBS_CSV}"; then
    echo "[$(timestamp)] SKIP job=${job}: filtered by --only-jobs" >> "${LOG_FILE}"
    continue
  fi
  if [[ "${EXCLUDE_FRAGMENT_JOBS}" == "true" ]] && [[ "${job}" == *fragon* ]]; then
    echo "[$(timestamp)] SKIP job=${job}: excluded fragment job" >> "${LOG_FILE}"
    continue
  fi

  reuse_info="$(should_reuse_row "${job}" "0")"
  should_reuse="${reuse_info%%$'\t'*}"
  reuse_reason="${reuse_info#*$'\t'}"
  if [[ "${should_reuse}" == "1" ]]; then
    existing_row="$(awk -F'\t' -v j="${job}" 'NR > 1 && $1 == j { print; exit }' "${OUT_TSV}")"
    existing_anndata_path="$(get_existing_field_by_job "${job}" "anndata_path")"
    if [[ -z "${existing_anndata_path:-}" ]] || [[ "${existing_anndata_path}" == "-" ]]; then
      backfilled_anndata_path="$(resolve_any_anndata_path_for_job "${job}" "0")"
      patched_row="$(get_existing_row_with_field_override "${job}" "anndata_path" "${backfilled_anndata_path}")"
      if [[ -n "${patched_row}" ]]; then
        existing_row="${patched_row}"
      fi
      echo "[$(timestamp)] SKIP job=${job}: existing row reused with anndata_path backfill (${backfilled_anndata_path})" >> "${LOG_FILE}"
    fi
    printf "%s\n" "${existing_row}" >> "${tmp_out}"
    if [[ "${PROGRESSIVE_WRITE}" == "true" ]]; then
      cp "${tmp_out}" "${OUT_TSV}"
    fi
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

  seg_path="$(resolve_segmentation_path_for_job "${job}")"
  anndata_path="$(resolve_anndata_path_for_job "${job}")"
  gpu_time_s="$(lookup_gpu_time "${job}" "${gpu}")"

  start_ts="$(date +%s)"
  run_validate_for_row "${job}" "${seg_path}" "${anndata_path}" "1"
  end_ts="$(date +%s)"
  if [[ "${gpu_time_s}" == "0" || -z "${gpu_time_s}" ]]; then
    gpu_time_s="$((end_ts - start_ts))"
  fi

  append_row \
    "${job}" "${group}" "${gpu}" "0" "-" \
    "${validate_status}" "${validate_error}" "${gpu_time_s}" "${cells}" "${fragments}" \
    "${assigned_pct}" "${assigned_ci95}" "${positive_marker_recall_pct}" "${positive_marker_recall_ci95}" "${mecr}" "${mecr_ci95}" \
    "${contamination_pct}" "${contamination_ci95}" "${resolvi_contamination_pct}" "${resolvi_contamination_ci95}" "${center_border_similarity}" "${center_border_similarity_ci95}" \
    "${spurious_coexpression}" "${spurious_coexpression_ci95}" "${reference_morphology_match}" "${reference_morphology_match_ci95}" "${aes}" "${aes_ci95}" \
    "${doublet_pct}" "${doublet_ci95}" "${row_seg_path}" "${row_anndata_path}"
done < "${PLAN_FILE}"

if [[ "${INCLUDE_DEFAULT_10X}" == "true" ]]; then
  for reference_kind in 10x_cell 10x_nucleus; do
    job="ref_${reference_kind}"
    if [[ -n "${ONLY_JOBS_CSV}" ]] && ! csv_contains_exact "${job}" "${ONLY_JOBS_CSV}"; then
      echo "[$(timestamp)] SKIP job=${job}: filtered by --only-jobs" >> "${LOG_FILE}"
      continue
    fi
    group="R"
    gpu="-"

    reuse_info="$(should_reuse_row "${job}" "1")"
    should_reuse="${reuse_info%%$'\t'*}"
    reuse_reason="${reuse_info#*$'\t'}"
    if [[ "${should_reuse}" == "1" ]]; then
      existing_row="$(awk -F'\t' -v j="${job}" 'NR > 1 && $1 == j { print; exit }' "${OUT_TSV}")"
      existing_anndata_path="$(get_existing_field_by_job "${job}" "anndata_path")"
      if [[ -z "${existing_anndata_path:-}" ]] || [[ "${existing_anndata_path}" == "-" ]]; then
        backfilled_anndata_path="$(resolve_any_anndata_path_for_job "${job}" "1")"
        patched_row="$(get_existing_row_with_field_override "${job}" "anndata_path" "${backfilled_anndata_path}")"
        if [[ -n "${patched_row}" ]]; then
          existing_row="${patched_row}"
        fi
        echo "[$(timestamp)] SKIP job=${job}: existing row reused with anndata_path backfill (${backfilled_anndata_path})" >> "${LOG_FILE}"
      fi
      printf "%s\n" "${existing_row}" >> "${tmp_out}"
      if [[ "${PROGRESSIVE_WRITE}" == "true" ]]; then
        cp "${tmp_out}" "${OUT_TSV}"
      fi
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
    fragments="0"
    assigned_pct="nan"
    assigned_ci95="nan"
    positive_marker_recall_pct="nan"
    positive_marker_recall_ci95="nan"
    mecr="nan"
    mecr_ci95="nan"
    contamination_pct="nan"
    contamination_ci95="nan"
    resolvi_contamination_pct="nan"
    resolvi_contamination_ci95="nan"
    center_border_similarity="nan"
    center_border_similarity_ci95="nan"
    spurious_coexpression="nan"
    spurious_coexpression_ci95="nan"
    reference_morphology_match="nan"
    reference_morphology_match_ci95="nan"
    aes="nan"
    aes_ci95="nan"
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
      "${validate_status}" "${validate_error}" "${gpu_time_s}" "${cells}" "${fragments}" \
      "${assigned_pct}" "${assigned_ci95}" "${positive_marker_recall_pct}" "${positive_marker_recall_ci95}" "${mecr}" "${mecr_ci95}" \
      "${contamination_pct}" "${contamination_ci95}" "${resolvi_contamination_pct}" "${resolvi_contamination_ci95}" "${center_border_similarity}" "${center_border_similarity_ci95}" \
      "${spurious_coexpression}" "${spurious_coexpression_ci95}" "${reference_morphology_match}" "${reference_morphology_match_ci95}" "${aes}" "${aes_ci95}" \
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
          assigned = norm_num($11)
          mecr = norm_num($15)
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
build_completed=1
echo "[$(timestamp)] WROTE validation table: ${OUT_TSV}" >> "${LOG_FILE}"
echo "[$(timestamp)] SUMMARY reused=${reused_count} computed=${computed_count}" >> "${LOG_FILE}"
echo "Wrote validation table: ${OUT_TSV} (reused=${reused_count}, computed=${computed_count})"
