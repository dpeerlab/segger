#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Aggregate per-dataset validation tables produced by
build_benchmark_validation_table.sh.

Usage:
  bash scripts/build_benchmark_validation_table_lsf_multi.sh [options]

Options:
  --root <dir>       Top-level LSF benchmark root
                     (default: /dkfz/cluster/gpu/checkpoints/OE0606/elihei/segger_lsf_benchmark_fixed)
  --out-tsv <file>   Output aggregate TSV path
                     (default: <root>/summaries/aggregate_validation_metrics_.tsv)
  --exclude-fragment-jobs <bool>
                     Skip fragment-mode jobs (name contains "fragon") in per-dataset and aggregate outputs
                     (default: false)
  --include-default-10x <bool>
                     Include ref_10x_cell/ref_10x_nucleus rows in each dataset validation table
                     (default: true)
  --include-positive-marker-recall <bool>
                     Include positive-marker-recall metric in per-dataset validation (default: false)
  --include-center-border-similarity <bool>
                     Include center-border-similarity metric in per-dataset validation (default: false)
  --include-resolvi <bool>
                     Include resolvi contamination metric in per-dataset validation (default: false)
  --include-sce <bool>
                     Include spurious-coexpression metric in per-dataset validation (default: false)
  --include-mm <bool>
                     Include morphological-match metric in per-dataset validation (default: false)
  --include-aes <bool>
                     Include angular-expression-symmetry metric in per-dataset validation (default: false)
  --include-vsi <bool>
                     Include VSI metric in per-dataset validation (default: false)
  --progressive-write <bool>
                     Update aggregate TSV incrementally while running (default: true)
  --jobs <int>       Number of datasets to refresh in parallel (default: 1)
  --only-jobs <csv>  Optional comma-separated job names to validate per dataset (e.g. baseline)
  --recompute        Recompute validation rows for each dataset (passes --recompute to per-dataset script)
  --no-refresh       Do not run the per-dataset validation script when a table is missing
  -h, --help         Show this help
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

ROOT="/dkfz/cluster/gpu/checkpoints/OE0606/elihei/segger_lsf_benchmark_fixed"
OUT_TSV=""
REFRESH=1
EXCLUDE_FRAGMENT_JOBS="false"
INCLUDE_DEFAULT_10X="true"
INCLUDE_POSITIVE_MARKER_RECALL="false"
INCLUDE_BEI="false"
INCLUDE_CTM="false"
INCLUDE_SCE="false"
INCLUDE_MM="false"
INCLUDE_EAU="false"
INCLUDE_VD="false"
PROGRESSIVE_WRITE="true"
JOBS="${VALIDATION_JOBS:-1}"
RECOMPUTE=0
ONLY_JOBS_CSV=""
DATASET_VALIDATION_TSV_NAME="validation_metrics_.tsv"

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

tsv_data_rows() {
  local tsv_path="$1"
  if [[ ! -f "${tsv_path}" ]]; then
    printf '%s' "-1"
    return 0
  fi
  awk 'END { if (NR > 0) print NR - 1; else print 0 }' "${tsv_path}" 2>/dev/null || printf '%s' "-1"
}

refresh_dataset_validation() {
  local dataset_dir="$1"
  local idx="$2"
  local total="$3"
  local dataset validation_tsv legacy_validation_tsv context_file refresh_log rc
  local partial_validation_tsv
  local validation_rows legacy_rows partial_rows
  local -a recompute_args
  local -a only_jobs_args

  dataset="$(basename "${dataset_dir}")"
  validation_tsv="${dataset_dir}/summaries/${DATASET_VALIDATION_TSV_NAME}"
  legacy_validation_tsv="${dataset_dir}/summaries/validation_metrics.tsv"
  partial_validation_tsv="${dataset_dir}/summaries/${DATASET_VALIDATION_TSV_NAME%.tsv}.partial.tsv"
  context_file="${dataset_dir}/dataset_context.env"
  refresh_log="${dataset_dir}/summaries/validation_refresh.log"
  rc=0

  if [[ "${REFRESH}" != "1" ]]; then
    echo "[$(timestamp)] [${idx}/${total}] REFRESH skip dataset=${dataset} reason=refresh_disabled"
    return 0
  fi
  validation_rows="$(tsv_data_rows "${validation_tsv}")"
  if [[ "${RECOMPUTE}" != "1" && "${validation_rows}" -gt 0 ]]; then
    echo "[$(timestamp)] [${idx}/${total}] REFRESH skip dataset=${dataset} reason=validation_exists"
    return 0
  fi
  if [[ ! -f "${context_file}" ]]; then
    echo "[$(timestamp)] [${idx}/${total}] REFRESH skip dataset=${dataset} reason=missing_dataset_context"
    return 0
  fi

  echo "[$(timestamp)] [${idx}/${total}] REFRESH start dataset=${dataset}"
  : > "${refresh_log}"
  echo "[$(timestamp)] REFRESH dataset=${dataset} root=${dataset_dir}" >> "${refresh_log}"

  unset DATASET_KEY INPUT_DIR SCRNA_REFERENCE_PATH TISSUE_TYPE SEGGER_BIN || true
  # shellcheck disable=SC1090
  source "${context_file}"

  recompute_args=()
  only_jobs_args=()
  if [[ "${RECOMPUTE}" == "1" ]]; then
    recompute_args+=(--recompute)
  fi
  if [[ -n "${ONLY_JOBS_CSV}" ]]; then
    only_jobs_args+=(--only-jobs "${ONLY_JOBS_CSV}")
  fi

  if [[ -z "${SCRNA_REFERENCE_PATH:-}" ]]; then
    echo "[$(timestamp)] ERROR dataset=${dataset} missing SCRNA_REFERENCE_PATH in ${context_file}; refusing tissue fallback." >> "${refresh_log}"
    echo "[$(timestamp)] [${idx}/${total}] REFRESH done dataset=${dataset} status=missing_scrna_reference"
    return 0
  fi
  if [[ ! -f "${SCRNA_REFERENCE_PATH}" ]]; then
    echo "[$(timestamp)] ERROR dataset=${dataset} configured SCRNA_REFERENCE_PATH missing; refusing tissue fallback: ${SCRNA_REFERENCE_PATH}" >> "${refresh_log}"
    echo "[$(timestamp)] [${idx}/${total}] REFRESH done dataset=${dataset} status=missing_scrna_reference"
    return 0
  fi

  if bash "${SCRIPT_DIR}/build_benchmark_validation_table.sh" \
    --root "${dataset_dir}" \
    --out-tsv "${validation_tsv}" \
    --input-dir "${INPUT_DIR}" \
    --scrna-reference-path "${SCRNA_REFERENCE_PATH}" \
    --include-default-10x "${INCLUDE_DEFAULT_10X}" \
    --include-positive-marker-recall "${INCLUDE_POSITIVE_MARKER_RECALL}" \
    --include-bei "${INCLUDE_BEI}" \
    --include-ctm "${INCLUDE_CTM}" \
    --include-sce "${INCLUDE_SCE}" \
    --include-mm "${INCLUDE_MM}" \
    --include-eau "${INCLUDE_EAU}" \
    --include-vd "${INCLUDE_VD}" \
    --exclude-fragment-jobs "${EXCLUDE_FRAGMENT_JOBS}" \
    --progressive-write "${PROGRESSIVE_WRITE}" \
    "${only_jobs_args[@]}" \
    "${recompute_args[@]}" \
    >> "${refresh_log}" 2>&1; then
    rc=0
  else
    rc=$?
  fi

  if [[ "${rc}" -ne 0 ]]; then
    echo "[$(timestamp)] [${idx}/${total}] REFRESH warn dataset=${dataset} status=validation_script_failed rc=${rc} log=${refresh_log}"
  fi

  validation_rows="$(tsv_data_rows "${validation_tsv}")"
  legacy_rows="$(tsv_data_rows "${legacy_validation_tsv}")"
  partial_rows="$(tsv_data_rows "${partial_validation_tsv}")"
  if [[ "${validation_rows}" -gt 0 ]]; then
    echo "[$(timestamp)] [${idx}/${total}] REFRESH done dataset=${dataset}"
  elif [[ "${legacy_rows}" -gt 0 ]]; then
    echo "[$(timestamp)] [${idx}/${total}] REFRESH done dataset=${dataset} status=legacy_validation_tsv"
  elif [[ "${partial_rows}" -gt 0 ]]; then
    echo "[$(timestamp)] [${idx}/${total}] REFRESH done dataset=${dataset} status=partial_validation_tsv"
  else
    echo "[$(timestamp)] [${idx}/${total}] REFRESH done dataset=${dataset} status=missing_validation_tsv"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --out-tsv)
      OUT_TSV="$2"
      shift 2
      ;;
    --exclude-fragment-jobs)
      EXCLUDE_FRAGMENT_JOBS="$2"
      shift 2
      ;;
    --include-default-10x)
      INCLUDE_DEFAULT_10X="$2"
      shift 2
      ;;
    --include-positive-marker-recall)
      INCLUDE_POSITIVE_MARKER_RECALL="$2"
      shift 2
      ;;
    --include-center-border-similarity|--include-center-border-ncv|--include-bei)
      INCLUDE_BEI="$2"
      shift 2
      ;;
    --include-resolvi|--include-border-contamination|--include-ctm)
      INCLUDE_CTM="$2"
      shift 2
      ;;
    --include-spurious-coexpression|--include-sce)
      INCLUDE_SCE="$2"
      shift 2
      ;;
    --include-morphological-match|--include-mm)
      INCLUDE_MM="$2"
      shift 2
      ;;
    --include-aes|--include-tco|--include-eau)
      INCLUDE_EAU="$2"
      shift 2
      ;;
    --include-vsi|--include-vd)
      INCLUDE_VD="$2"
      shift 2
      ;;
    --progressive-write)
      PROGRESSIVE_WRITE="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --only-jobs)
      ONLY_JOBS_CSV="$2"
      shift 2
      ;;
    --no-refresh)
      REFRESH=0
      shift
      ;;
    --recompute)
      RECOMPUTE=1
      shift
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

if ! EXCLUDE_FRAGMENT_JOBS="$(normalize_bool "${EXCLUDE_FRAGMENT_JOBS}")"; then
  echo "ERROR: --exclude-fragment-jobs must be true/false." >&2
  exit 1
fi
if ! INCLUDE_DEFAULT_10X="$(normalize_bool "${INCLUDE_DEFAULT_10X}")"; then
  echo "ERROR: --include-default-10x must be true/false." >&2
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
if ! PROGRESSIVE_WRITE="$(normalize_bool "${PROGRESSIVE_WRITE}")"; then
  echo "ERROR: --progressive-write must be true/false." >&2
  exit 1
fi
if ! [[ "${JOBS}" =~ ^[0-9]+$ ]] || [[ "${JOBS}" -lt 1 ]]; then
  echo "ERROR: --jobs must be a positive integer." >&2
  exit 1
fi

if [[ -z "${OUT_TSV}" ]]; then
  OUT_TSV="${ROOT}/summaries/aggregate_validation_metrics_.tsv"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "${OUT_TSV}")"

if [[ "${JOBS}" -gt 1 ]] && ! (help wait 2>/dev/null | grep -q -- '-n'); then
  echo "[$(timestamp)] WARN wait -n not supported by this shell; forcing --jobs 1"
  JOBS=1
fi

dataset_dirs=()
for dataset_dir in "${ROOT}"/datasets/*; do
  [[ -d "${dataset_dir}" ]] || continue
  dataset_dirs+=("${dataset_dir}")
done

found_any=0
if [[ "${#dataset_dirs[@]}" -gt 0 ]]; then
  found_any=1
fi

if [[ "${found_any}" == "1" && "${REFRESH}" == "1" ]]; then
  total_refresh="${#dataset_dirs[@]}"
  running_refresh=0
  completed_refresh=0
  launched_refresh=0
  echo "[$(timestamp)] REFRESH queueing datasets=${total_refresh} jobs=${JOBS}"

  for dataset_dir in "${dataset_dirs[@]}"; do
    launched_refresh=$((launched_refresh + 1))
    if [[ "${JOBS}" -gt 1 ]]; then
      while [[ "${running_refresh}" -ge "${JOBS}" ]]; do
        wait -n || true
        running_refresh=$((running_refresh - 1))
        completed_refresh=$((completed_refresh + 1))
        echo "[$(timestamp)] REFRESH progress ${completed_refresh}/${total_refresh}"
      done
      refresh_dataset_validation "${dataset_dir}" "${launched_refresh}" "${total_refresh}" &
      running_refresh=$((running_refresh + 1))
    else
      refresh_dataset_validation "${dataset_dir}" "${launched_refresh}" "${total_refresh}"
      completed_refresh=$((completed_refresh + 1))
      echo "[$(timestamp)] REFRESH progress ${completed_refresh}/${total_refresh}"
    fi
  done

  if [[ "${JOBS}" -gt 1 ]]; then
    while [[ "${running_refresh}" -gt 0 ]]; do
      wait -n || true
      running_refresh=$((running_refresh - 1))
      completed_refresh=$((completed_refresh + 1))
      echo "[$(timestamp)] REFRESH progress ${completed_refresh}/${total_refresh}"
    done
  fi
fi

header_written=0
tmp_out="$(mktemp)"
trap 'rm -f "${tmp_out}"' EXIT
: > "${tmp_out}"
if [[ "${PROGRESSIVE_WRITE}" == "true" ]]; then
  cp "${tmp_out}" "${OUT_TSV}"
fi

total_aggregate="${#dataset_dirs[@]}"
aggregate_idx=0
source_current_count=0
source_legacy_count=0
source_partial_count=0
source_missing_count=0

for dataset_dir in "${dataset_dirs[@]}"; do
  aggregate_idx=$((aggregate_idx + 1))
  dataset="$(basename "${dataset_dir}")"
  validation_tsv="${dataset_dir}/summaries/${DATASET_VALIDATION_TSV_NAME}"
  legacy_validation_tsv="${dataset_dir}/summaries/validation_metrics.tsv"
  partial_validation_tsv="${dataset_dir}/summaries/${DATASET_VALIDATION_TSV_NAME%.tsv}.partial.tsv"
  source_tsv=""
  source_label=""
  validation_rows="$(tsv_data_rows "${validation_tsv}")"
  legacy_rows="$(tsv_data_rows "${legacy_validation_tsv}")"
  partial_rows="$(tsv_data_rows "${partial_validation_tsv}")"

  echo "[$(timestamp)] [${aggregate_idx}/${total_aggregate}] AGGREGATE dataset=${dataset}"

  if [[ "${validation_rows}" -gt 0 ]]; then
    source_tsv="${validation_tsv}"
    source_label="validation_tsv"
    source_current_count=$((source_current_count + 1))
  elif [[ "${legacy_rows}" -gt 0 ]]; then
    source_tsv="${legacy_validation_tsv}"
    source_label="legacy_validation_tsv"
    source_legacy_count=$((source_legacy_count + 1))
    echo "[$(timestamp)] [${aggregate_idx}/${total_aggregate}] AGGREGATE dataset=${dataset} source=legacy_validation_tsv rows=${legacy_rows}"
  elif [[ "${partial_rows}" -gt 0 ]]; then
    source_tsv="${partial_validation_tsv}"
    source_label="partial_validation_tsv"
    source_partial_count=$((source_partial_count + 1))
    echo "[$(timestamp)] [${aggregate_idx}/${total_aggregate}] AGGREGATE dataset=${dataset} source=partial_validation_tsv rows=${partial_rows}"
  else
    source_missing_count=$((source_missing_count + 1))
    echo "[$(timestamp)] [${aggregate_idx}/${total_aggregate}] AGGREGATE dataset=${dataset} source=missing_validation_tsv"
    continue
  fi

  if [[ "${header_written}" == "0" ]]; then
    head -n1 "${source_tsv}" | awk -F'\t' '{ printf "dataset"; for (i = 1; i <= NF; i++) printf "\t%s", $i; printf "\n" }' > "${tmp_out}"
    header_written=1
  fi
  tail -n +2 "${source_tsv}" | awk -F'\t' -v dataset="${dataset}" -v exclude_frag="${EXCLUDE_FRAGMENT_JOBS}" '
    NF > 0 {
      if (exclude_frag == "true" && index($1, "fragon") > 0) {
        next
      }
      printf "%s", dataset
      for (i = 1; i <= NF; i++) {
        printf "\t%s", $i
      }
      printf "\n"
    }
  ' >> "${tmp_out}"
  if [[ "${PROGRESSIVE_WRITE}" == "true" ]]; then
    cp "${tmp_out}" "${OUT_TSV}"
  fi
done

echo "[$(timestamp)] AGGREGATE source_summary validation=${source_current_count} legacy=${source_legacy_count} partial=${source_partial_count} missing=${source_missing_count}"

if [[ "${header_written}" == "0" ]]; then
  printf "dataset\n" > "${tmp_out}"
fi

cp "${tmp_out}" "${OUT_TSV}"

if [[ "${found_any}" == "0" ]]; then
  echo "No dataset roots found under ${ROOT}/datasets" >&2
fi

if command -v column >/dev/null 2>&1; then
  column -ts $'\t' "${OUT_TSV}"
else
  cat "${OUT_TSV}"
fi
