#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the fixed-input Segger benchmark on LSF while preserving the current
benchmark-script layout per dataset root.

Environment overrides:
  OUTPUT_ROOT                Top-level output root
  WORK_TMP_ROOT              Temp/scratch dir for this run (default:
                             <OUTPUT_ROOT>/_workflow/tmp)
  DATASET_KEYS               "all", "root_missing" (default), "root_present",
                             or comma-separated dataset keys
  DRY_RUN                    1 to only render plans/scripts
  AUTO_SUBMIT                1 to submit all LSF jobs immediately
  RUN_PRIMARY_JOBS           1 to submit segment/predict jobs (default: 1)
  RUN_EXPORT_JOBS            1 to submit export jobs (default: 1)
  RUN_EXPORTS                Legacy toggle for both export types
  RUN_ANNDATA_EXPORT         1 to export AnnData outputs
  RUN_XENIUM_EXPORT          1 to export Xenium Explorer outputs
  SKIP_UNSUPPORTED_XENIUM_EXPORT
                             1 to skip Xenium export when cells.zarr.zip is missing
  RUN_VALIDATION_TABLE       1 to run the existing validation script per dataset
  RUN_STATUS_SNAPSHOT        1 to run the existing dashboard per dataset
  ENABLE_ALIGNMENT_JOBS      1 to include align_* training jobs (default: 1)
  RESUME_IF_EXISTS           1 to skip jobs with complete outputs
  FORCE_RERUN_FRAGMENT_JOBS  1 to rerun fragment-mode jobs even when outputs exist
  ONLY_RETRY_OOM_JOBS        1 to submit only unfinished jobs whose latest attempt
                             failed due to GPU OOM / TERM_MEMLIMIT
  RESET_DATASET_ROOT         1 to remove each dataset output dir before planning/submission
  RUN_LABEL                  Optional run label embedded into LSF job names
  SEGMENT_WALL_TIME_DEFAULT  Default wall time for segment/predict jobs
  PREDICT_FRAGMENT_GMEM_GB   Base gmem (GB) for fragment-mode predict jobs
                             (default: 36)
  EXPORT_WALL_TIME_DEFAULT   Default wall time for export jobs
  ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT
                             Optional --min-similarity-shift passed to AnnData
                             exports (0..1, default: 0.0)
  SEGGER_BIN                 Segger executable (default: segger)
  CLUSTER_CODE_ROOT          Cluster checkout used inside LSF scripts
  MAMBA_ACTIVATE_CMD         Optional shell snippet used inside LSF scripts
  MICROMAMBA_BIN            Micromamba executable used inside LSF scripts
  MICROMAMBA_ROOT_PREFIX     Micromamba root prefix used inside LSF scripts
  SEGGER_ENV_PATH            Env path activated inside LSF scripts
  PRESERVE_PYTHONPATH        1 to keep inherited PYTHONPATH inside LSF jobs
  ALIGNMENT_REFERENCE_MODE   One of: auto|path (default: path)
  ALIGNMENT_SCRNA_CELLTYPE_COLUMN
                             Cell-type column in local scRNA refs (default: cell_type)
  JOB_SPECS_FILE             Optional path to a custom pipe-delimited job-spec file
                             (same 15-column schema as job_specs()).
  LSF_EXEC_SHELL             Shell used by LSF to interpret the job script
  LSF_SUBMIT_HOST            Host used for remote bsub/bjobs (default: local)
  SCRNA_REF_ROOT             Shared root that contains .segger_references
  SCRNA_CACHE_DIR            Cache directory for h5ad refs
  AUTO_FETCH_SCRNA_REFS      1 to prefetch refs with "segger atlas fetch" when missing
  POLL_INTERVAL_SEC          LSF poll interval
  ROUTE_ELIGIBLE_TO_GPU      1 to route jobs that fit standard limits to GPU_QUEUE_DEFAULT
  GPU_QUEUE_DEFAULT          Queue for standard GPU jobs (default: gpu)
  GPU_QUEUE_PRO              Queue for high-memory GPU jobs (default: gpu-pro)
  EXPORT_QUEUE               Queue for export jobs (default: long)
  ALIGNMENT_GPU_QUEUE        Optional override queue only for align_* jobs
  GPU_QUEUE_MAX_GMEM         Max GPU memory (GB) for GPU_QUEUE_DEFAULT routing (default: 31)
  GPU_QUEUE_MAX_MEM_GB       Max RAM (GB) for GPU_QUEUE_DEFAULT routing (default: 84)
  GPU_QUEUE_MAX_WALL_H       Max wall-time hours for GPU_QUEUE_DEFAULT routing (default: 24)
  GPU_QUEUE_PRO_MAX_GMEM     Max GPU memory (GB) requested on GPU_QUEUE_PRO
                             (default: 39; cluster doc warns against 40G+)
  GPU_QUEUE_PRO_MAX_MEM_GB   Max RAM (GB) requested on GPU_QUEUE_PRO
                             (default: 84)
  FORCE_GPU_PRO_DATASETS     CSV dataset keys always forced to GPU_QUEUE_PRO
                             (default: xenium_crc,xenium_nsclc,xenium_mouse_liver,xenium_breast)
  REQUIRE_SUCCESS_STATUS_FOR_RESUME
                             1 to only reuse outputs with segment_status in
                             {segment_ok,predict_ok} (default: 1)
  ALLOW_UNVERIFIED_PRIMARY_OUTPUTS
                             1 to trust existing seg outputs when status files
                             are missing (default: 0)
  GPU_FALLBACK_ON_RETRY      1 to bump queue/VRAM on GPU-like failures in prior attempt
  GPU_FALLBACK_QUEUE         Queue used for retry fallback (default: GPU_QUEUE_PRO)
  GPU_RETRY_GMEM_BUMP_GB     gmem bump (GB) added on retry fallback (default: 20)
  GPU_FALLBACK_MIN_GMEM      Minimum gmem (GB) after retry fallback/bump (default: 36)
  GPU_FALLBACK_FRAGMENT_GMEM Minimum gmem (GB) for fragment-mode retry fallback/bump
                             (default: GPU_QUEUE_PRO_MAX_GMEM)
  GPU_FALLBACK_MIN_MEM_GB    Minimum host RAM (GB) after retry fallback (default: 84)
  GPU_FALLBACK_MIN_WALL_H    Minimum wall-clock hours after retry fallback (default: 12)
  GPU_USAGE_POLL_SEC         GPU-memory polling interval (seconds) for peak VRAM capture
                             in primary jobs (default: 15)
  PEND_FALLBACK_MIN          Minutes before moving standard jobs to gpu queue
  MAX_ACTIVE_STANDARD        Reserved for future in-flight throttling
  MAX_ACTIVE_FRAGMENT        Reserved for future in-flight throttling
  FORCE_SCRNA_REFETCH        1 to force-refresh atlas references before planning

Usage:
  bash scripts/run_lsf_segger_benchmark.sh
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

summary_header() {
  printf "job\tgpu\tstatus\telapsed_s\tnote\tseg_dir\tlog_file\tsegment_status\texport_status\tsegment_job_id\texport_job_id\texport_note\n"
}

is_remote_submit_enabled() {
  [[ -n "${LSF_SUBMIT_HOST:-}" ]]
}

normalize_bool() {
  local value="${1:-}"
  local lower
  lower="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    1|true|t|yes|y|on) printf '1' ;;
    0|false|f|no|n|off) printf '0' ;;
    *)
      echo "ERROR: Expected boolean value, got: ${value}" >&2
      exit 1
      ;;
  esac
}

normalize_alignment_reference_mode() {
  local value="${1:-}"
  local lower
  lower="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    auto|path) printf '%s' "${lower}" ;;
    *)
      echo "ERROR: ALIGNMENT_REFERENCE_MODE must be one of: auto, path (got: ${value})." >&2
      exit 1
      ;;
  esac
}

number_le() {
  local a="${1:-0}"
  local b="${2:-0}"
  awk -v a="${a}" -v b="${b}" 'BEGIN { exit !((a + 0) <= (b + 0)) }'
}

number_max() {
  local a="${1:-0}"
  local b="${2:-0}"
  awk -v a="${a}" -v b="${b}" 'BEGIN { if ((a + 0) >= (b + 0)) printf "%d", (a + 0); else printf "%d", (b + 0) }'
}

number_min() {
  local a="${1:-0}"
  local b="${2:-0}"
  awk -v a="${a}" -v b="${b}" 'BEGIN { if ((a + 0) <= (b + 0)) printf "%d", (a + 0); else printf "%d", (b + 0) }'
}

wall_to_minutes() {
  local wall="${1:-0:00}"
  local h="${wall%%:*}"
  local m="${wall##*:}"
  [[ -n "${h}" ]] || h="0"
  [[ -n "${m}" ]] || m="0"
  printf '%d' "$((10#${h} * 60 + 10#${m}))"
}

minutes_to_wall() {
  local total="${1:-0}"
  if [[ "${total}" -lt 0 ]]; then
    total=0
  fi
  local h=$((total / 60))
  local m=$((total % 60))
  printf '%d:%02d' "${h}" "${m}"
}

shell_join() {
  local out=""
  local part
  for part in "$@"; do
    local quoted
    printf -v quoted '%q' "${part}"
    if [[ -n "${out}" ]]; then
      out+=" "
    fi
    out+="${quoted}"
  done
  printf '%s' "${out}"
}

csv_contains() {
  local csv="${1:-}"
  local needle="${2:-}"
  local item
  local -a items=()

  [[ -n "${needle}" ]] || return 1
  [[ -n "${csv//[[:space:]]/}" ]] || return 1
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]-}"; do
    item="$(printf '%s' "${item}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [[ -n "${item}" ]] || continue
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

dataset_forces_gpu_pro() {
  local dataset="${1:-}"
  csv_contains "${FORCE_GPU_PRO_DATASETS}" "${dataset}"
}

read_stage_field() {
  local stage_file="$1"
  local field="$2"
  if [[ ! -f "${stage_file}" ]]; then
    printf '%s' ""
    return 0
  fi
  awk -F'=' -v key="${field}" '$1 == key { print $2; exit }' "${stage_file}" 2>/dev/null || true
}

is_primary_success_status() {
  case "${1:-}" in
    segment_ok|predict_ok)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

all_dataset_keys() {
  cat <<'EOF'
xenium_crc
xenium_nsclc
xenium_v1_colon
xenium_mouse_liver
xenium_breast
xenium_v1_breast
xenium_mouse_brain
merscope_mouse_brain
cosmx_human_pancreas
EOF
}

root_known_dataset_keys() {
  local root_dir="${OUTPUT_ROOT}/datasets"
  local path=""
  local key=""
  if [[ ! -d "${root_dir}" ]]; then
    return 0
  fi
  while IFS= read -r path; do
    key="$(basename "${path}")"
    if is_known_dataset "${key}"; then
      printf '%s\n' "${key}"
    else
      printf "[%s] WARN ignoring unknown dataset directory in root: %s\n" \
        "$(timestamp)" "${path}" >&2
    fi
  done < <(find "${root_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
}

is_known_dataset() {
  case "${1:-}" in
    xenium_crc|xenium_nsclc|xenium_v1_colon|xenium_mouse_liver|xenium_breast|xenium_v1_breast|xenium_mouse_brain|merscope_mouse_brain|cosmx_human_pancreas)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

dataset_input_dir() {
  case "${1:-}" in
    xenium_crc) printf '%s' "/dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_CRC_fixed" ;;
    xenium_nsclc) printf '%s' "/dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_nscls/xenium_fixed" ;;
    xenium_v1_colon) printf '%s' "/dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_v1_colon_fixed" ;;
    xenium_mouse_liver) printf '%s' "/dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_mouse_liver_fixed" ;;
    xenium_breast) printf '%s' "/dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_breast" ;;
    xenium_v1_breast) printf '%s' "/dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_v1_breast_fixed" ;;
    xenium_mouse_brain) printf '%s' "/dkfz/cluster/gpu/data/OE0606/fengyun/data/xenium_mouse_brain_fixed" ;;
    merscope_mouse_brain) printf '%s' "/dkfz/cluster/gpu/data/OE0606/fengyun/data/MERSCOPE_brain" ;;
    cosmx_human_pancreas) printf '%s' "/dkfz/cluster/gpu/data/OE0606/fengyun/data/CosMx_pancreas" ;;
    *)
      return 1
      ;;
  esac
}

dataset_primary_ref() {
  local cache_root="${SCRNA_REF_ROOT}"
  if [[ "${cache_root}" != */.segger_references ]]; then
    cache_root="${cache_root}/.segger_references"
  fi
  case "${1:-}" in
    xenium_crc|xenium_v1_colon)
      printf '%s' "${cache_root}/homo_sapiens/large_intestine/large_intestine.h5ad"
      ;;
    xenium_nsclc)
      printf '%s' "${cache_root}/homo_sapiens/lung/lung.h5ad"
      ;;
    xenium_mouse_liver)
      printf '%s' "${cache_root}/mus_musculus/liver/liver.h5ad"
      ;;
    xenium_breast|xenium_v1_breast)
      printf '%s' "${cache_root}/homo_sapiens/breast/breast.h5ad"
      ;;
    xenium_mouse_brain|merscope_mouse_brain)
      printf '%s' "${cache_root}/mus_musculus/brain/brain.h5ad"
      ;;
    cosmx_human_pancreas)
      printf '%s' "${cache_root}/homo_sapiens/pancreas/pancreas.h5ad"
      ;;
    *)
      return 1
      ;;
  esac
}

dataset_fallback_ref() {
  case "${1:-}" in
    xenium_crc|xenium_v1_colon)
      printf '%s' "${SCRNA_REF_ROOT}/human_crc.h5ad"
      ;;
    xenium_nsclc)
      printf '%s' "${SCRNA_REF_ROOT}/lung_cancer.h5ad"
      ;;
    xenium_mouse_liver)
      printf '%s' "${SCRNA_REF_ROOT}/mouse_liver_cellxgene_a34c8af2.h5ad"
      ;;
    xenium_breast|xenium_v1_breast)
      printf '%s' "${SCRNA_REF_ROOT}/breast_cancer_annotated.h5ad"
      ;;
    xenium_mouse_brain|merscope_mouse_brain)
      printf '%s' "${SCRNA_REF_ROOT}/mouse_brain.h5ad"
      ;;
    cosmx_human_pancreas)
      printf '%s' "${SCRNA_REF_ROOT}/human_pancreas_cellxgene_e51bae9a.h5ad"
      ;;
    *)
      printf '%s' ""
      ;;
  esac
}

dataset_tissue_type() {
  case "${1:-}" in
    xenium_crc|xenium_v1_colon) printf '%s' "colon" ;;
    xenium_nsclc) printf '%s' "lung" ;;
    xenium_mouse_liver) printf '%s' "liver" ;;
    xenium_breast|xenium_v1_breast) printf '%s' "breast" ;;
    xenium_mouse_brain|merscope_mouse_brain) printf '%s' "brain" ;;
    cosmx_human_pancreas) printf '%s' "pancreas" ;;
    *)
      printf '%s' ""
      ;;
  esac
}

dataset_organism() {
  case "${1:-}" in
    xenium_crc|xenium_nsclc|xenium_v1_colon|xenium_breast|xenium_v1_breast|cosmx_human_pancreas)
      printf '%s' "homo_sapiens"
      ;;
    xenium_mouse_liver|xenium_mouse_brain|merscope_mouse_brain)
      printf '%s' "mus_musculus"
      ;;
    *)
      printf '%s' ""
      ;;
  esac
}

alignment_reference_args() {
  local scrna_ref="$1"
  local mode="$2"
  local ref_args=""
  local with_celltype="0"

  if [[ "${CLI_IO_STYLE}" == "named" ]]; then
    if [[ "${CLI_SEGMENT_SUPPORTS_SCRNA_REFERENCE_PATH:-1}" != "1" ]]; then
      printf '%s' ""
      return 0
    fi
    if [[ "${CLI_SEGMENT_SUPPORTS_SCRNA_CELLTYPE_COLUMN:-1}" == "1" ]]; then
      with_celltype="1"
    fi
  else
    with_celltype="1"
  fi

  case "${mode}" in
    path)
      if [[ -n "${scrna_ref}" ]]; then
        if [[ "${with_celltype}" == "1" ]]; then
          ref_args="$(shell_join --scrna-reference-path "${scrna_ref}" --scrna-celltype-column "${ALIGNMENT_SCRNA_CELLTYPE_COLUMN}")"
        else
          ref_args="$(shell_join --scrna-reference-path "${scrna_ref}")"
        fi
      fi
      ;;
    auto)
      if [[ -n "${scrna_ref}" && -f "${scrna_ref}" ]]; then
        if [[ "${with_celltype}" == "1" ]]; then
          ref_args="$(shell_join --scrna-reference-path "${scrna_ref}" --scrna-celltype-column "${ALIGNMENT_SCRNA_CELLTYPE_COLUMN}")"
        else
          ref_args="$(shell_join --scrna-reference-path "${scrna_ref}")"
        fi
      fi
      ;;
  esac

  printf '%s' "${ref_args}"
}

job_specs() {
  if [[ -n "${JOB_SPECS_FILE:-}" ]]; then
    if [[ ! -f "${JOB_SPECS_FILE}" ]]; then
      echo "ERROR: JOB_SPECS_FILE not found: ${JOB_SPECS_FILE}" >&2
      exit 1
    fi
    # Allow comment/blank lines in custom spec files.
    awk '
      /^[[:space:]]*$/ { next }
      /^[[:space:]]*#/ { next }
      { print }
    ' "${JOB_SPECS_FILE}"
    return
  fi

  cat <<'EOF'
baseline|A|segment|false|2.2|5|20|2|4|5|0|false|0.0|2.2|false
use3d_true|A|segment|true|2.2|5|20|2|4|5|0|false|0.0|2.2|false
pred_sf1p2_fragoff|A|predict|checkpoint|1.2|5|20|2|4|5|0|false|0.0|1.2|false
pred_sf1p2_fragon|B|predict|checkpoint|1.2|5|20|2|4|5|0|false|0.0|1.2|true
pred_sf2p2_fragoff|A|predict|checkpoint|2.2|5|20|2|4|5|0|false|0.0|2.2|false
pred_sf2p2_fragon|B|predict|checkpoint|2.2|5|20|2|4|5|0|false|0.0|2.2|true
pred_sf3p2_fragoff|A|predict|checkpoint|3.2|5|20|2|4|5|0|false|0.0|3.2|false
pred_sf3p2_fragon|B|predict|checkpoint|3.2|5|20|2|4|5|0|false|0.0|3.2|true
EOF
  if [[ "${ENABLE_ALIGNMENT_JOBS}" == "1" ]]; then
    cat <<'EOF'
align_0p01|A|segment|false|2.2|5|20|2|4|5|0|true|0.01|2.2|false
align_0p03|A|segment|false|2.2|5|20|2|4|5|0|true|0.03|2.2|false
align_0p10|A|segment|false|2.2|5|20|2|4|5|0|true|0.1|2.2|false
EOF
  fi
}

job_specs_contains_mode() {
  local target_mode="$1"
  local line=""
  local mode=""
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    IFS='|' read -r _job _group mode _rest <<< "${line}"
    if [[ "${mode}" == "${target_mode}" ]]; then
      return 0
    fi
  done <<EOF
$(job_specs)
EOF
  return 1
}

paired_non_fragment_job() {
  local job="$1"
  case "${job}" in
    *_fragon)
      printf '%s' "${job%_fragon}_fragoff"
      ;;
    *)
      printf '%s' ""
      ;;
  esac
}

job_lane() {
  case "${1:-}" in
    A) printf '0' ;;
    B) printf '1' ;;
    *) printf '-' ;;
  esac
}

summary_file_for_group() {
  local dataset_root="$1"
  local group="$2"
  printf '%s' "${dataset_root}/summaries/gpu$(job_lane "${group}").tsv"
}

canonical_log_for_job() {
  local dataset_root="$1"
  local job="$2"
  local group="$3"
  printf '%s' "${dataset_root}/logs/${job}.gpu$(job_lane "${group}").log"
}

init_status_file() {
  local file_path="$1"
  mkdir -p "$(dirname "${file_path}")"
  summary_header > "${file_path}"
}

init_dataset_root() {
  local dataset_root="$1"
  mkdir -p \
    "${dataset_root}/runs" \
    "${dataset_root}/exports" \
    "${dataset_root}/logs" \
    "${dataset_root}/summaries" \
    "${dataset_root}/bsub"
  init_status_file "${dataset_root}/summaries/gpu0.tsv"
  init_status_file "${dataset_root}/summaries/gpu1.tsv"
  init_status_file "${dataset_root}/summaries/recovery.tsv"
  init_status_file "${dataset_root}/summaries/skipped_existing.tsv"
  init_status_file "${dataset_root}/summaries/all_jobs.tsv"
}

write_dataset_context() {
  local dataset_root="$1"
  local dataset="$2"
  local input_dir="$3"
  local scrna_ref="$4"
  local tissue_type="$5"
  local context_file="${dataset_root}/dataset_context.env"
  printf "DATASET_KEY=%q\n" "${dataset}" > "${context_file}"
  printf "INPUT_DIR=%q\n" "${input_dir}" >> "${context_file}"
  printf "SCRNA_REFERENCE_PATH=%q\n" "${scrna_ref}" >> "${context_file}"
  printf "TISSUE_TYPE=%q\n" "${tissue_type}" >> "${context_file}"
  printf "SEGGER_BIN=%q\n" "${SEGGER_BIN}" >> "${context_file}"
}

upsert_status_row() {
  local file_path="$1"
  local job="$2"
  local gpu="$3"
  local status="$4"
  local elapsed_s="$5"
  local note="$6"
  local seg_dir="$7"
  local log_file="$8"
  local segment_status="${9:-}"
  local export_status="${10:-}"
  local segment_job_id="${11:-}"
  local export_job_id="${12:-}"
  local export_note="${13:-}"
  local tmp_file
  local note_clean
  local export_note_clean

  note_clean="$(sanitize_tsv_field "${note}")"
  export_note_clean="$(sanitize_tsv_field "${export_note}")"
  tmp_file="$(mktemp)"
  {
    head -n1 "${file_path}" 2>/dev/null || summary_header
    tail -n +2 "${file_path}" 2>/dev/null | awk -F'\t' -v target="${job}" '$1 != target'
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job}" "${gpu}" "${status}" "${elapsed_s}" "${note_clean}" "${seg_dir}" "${log_file}" \
      "${segment_status}" "${export_status}" "${segment_job_id}" "${export_job_id}" "${export_note_clean}"
  } > "${tmp_file}"
  mv "${tmp_file}" "${file_path}"
}

rebuild_all_jobs_summary() {
  local dataset_root="$1"
  local out_file="${dataset_root}/summaries/all_jobs.tsv"
  awk 'FNR == 1 && NR != 1 { next } { print }' \
    "${dataset_root}/summaries/gpu0.tsv" \
    "${dataset_root}/summaries/gpu1.tsv" \
    "${dataset_root}/summaries/skipped_existing.tsv" \
    "${dataset_root}/summaries/recovery.tsv" \
    > "${out_file}"
}

append_canonical_log() {
  local dataset_root="$1"
  local job="$2"
  local group="$3"
  local message="$4"
  local log_file
  log_file="$(canonical_log_for_job "${dataset_root}" "${job}" "${group}")"
  mkdir -p "$(dirname "${log_file}")"
  printf "[%s] %s\n" "$(timestamp)" "${message}" >> "${log_file}"
}

announce_job_event() {
  local level="$1"
  local dataset="$2"
  local job="$3"
  local message="$4"
  if [[ "${level}" == "ERROR" ]]; then
    printf "[%s] %s dataset=%s job=%s %s\n" "$(timestamp)" "${level}" "${dataset}" "${job}" "${message}" >&2
  else
    printf "[%s] %s dataset=%s job=%s %s\n" "$(timestamp)" "${level}" "${dataset}" "${job}" "${message}"
  fi
}

write_dataset_plan() {
  local dataset_root="$1"
  local dataset="$2"
  local plan_file="${dataset_root}/job_plan.tsv"
  local line
  local mode=""
  local fragment_mode=""
  local queue_name=""
  local gmem=""
  local mem_gb=""
  local wall_time=""

  printf "job\tgroup\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\tmode\tfragment_mode\tqueue\tgmem_gb\tmem_gb\twall_time\n" > "${plan_file}"
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    IFS='|' read -r \
      job group mode use_3d expansion txk txdist layers heads cellsmin minqv alignment _align_weight _pred_scale fragment_mode \
      <<< "${line}"
    IFS=$'\t' read -r queue_name gmem mem_gb wall_time <<< "$(resolve_primary_resources "${dataset}" "${mode}" "${fragment_mode}" "${alignment}")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job}" "${group}" "${use_3d}" "${expansion}" "${txk}" "${txdist}" "${layers}" "${heads}" "${cellsmin}" "${minqv}" "${alignment}" \
      "${mode}" "${fragment_mode}" "${queue_name}" "${gmem}" "${mem_gb}" "${wall_time}" \
      >> "${plan_file}"
    : > "$(canonical_log_for_job "${dataset_root}" "${job}" "${group}")"
  done <<EOF
$(job_specs)
EOF
}

resolve_requested_datasets() {
  local token
  local old_ifs
  local dataset
  local dataset_root
  local resolved_any=0
  if [[ "${DATASET_KEYS}" == "all" ]]; then
    all_dataset_keys
    return 0
  fi

  case "${DATASET_KEYS}" in
    root_present)
      root_known_dataset_keys
      return 0
      ;;
    root_missing|missing|auto_missing)
      while IFS= read -r dataset; do
        [[ -z "${dataset}" ]] && continue
        dataset_root="${OUTPUT_ROOT}/datasets/${dataset}"
        if dataset_missing_primary_results "${dataset}" "${dataset_root}"; then
          printf '%s\n' "${dataset}"
          resolved_any=1
        fi
      done <<EOF
$(all_dataset_keys)
EOF
      if [[ "${resolved_any}" -eq 0 ]]; then
        if [[ -d "${OUTPUT_ROOT}/datasets" ]] && find "${OUTPUT_ROOT}/datasets" -mindepth 1 -maxdepth 1 -type d | read -r _; then
          return 0
        fi
        all_dataset_keys
      fi
      return 0
      ;;
  esac

  old_ifs="${IFS}"
  IFS=','
  set -- ${DATASET_KEYS}
  IFS="${old_ifs}"
  for token in "$@"; do
    token="$(printf '%s' "${token}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [[ -z "${token}" ]] && continue
    if ! is_known_dataset "${token}"; then
      echo "ERROR: Unknown dataset key: ${token}" >&2
      exit 1
    fi
    printf '%s\n' "${token}"
  done
}

resolve_scrna_reference() {
  local dataset="$1"
  local primary
  local fallback

  primary="$(dataset_primary_ref "${dataset}")"
  fallback="$(dataset_fallback_ref "${dataset}")"
  if [[ -n "${primary}" && -f "${primary}" ]]; then
    printf '%s' "${primary}"
    return 0
  fi
  if [[ -n "${fallback}" && -f "${fallback}" ]]; then
    printf '%s' "${fallback}"
    return 0
  fi
  if [[ -n "${primary}" ]]; then
    printf '%s' "${primary}"
    return 0
  fi
  if [[ -n "${fallback}" ]]; then
    printf '%s' "${fallback}"
    return 0
  fi

  printf '%s' ""
}

job_primary_output_exists() {
  local dataset_root="$1"
  local job="$2"
  local seg_dir="${dataset_root}/runs/${job}"
  local seg_file="${seg_dir}/segger_segmentation.parquet"
  local transcripts_file="${seg_dir}/transcripts.parquet"
  local segment_status_file="${seg_dir}/.segment_status"
  local segment_status=""

  if [[ ! -s "${seg_file}" && ! -s "${transcripts_file}" ]]; then
    return 1
  fi

  if [[ "${REQUIRE_SUCCESS_STATUS_FOR_RESUME}" != "1" ]]; then
    return 0
  fi

  segment_status="$(read_stage_field "${segment_status_file}" "segment_status")"
  if [[ -z "${segment_status}" ]]; then
    if [[ "${ALLOW_UNVERIFIED_PRIMARY_OUTPUTS}" == "1" ]]; then
      return 0
    fi
    return 1
  fi
  if is_primary_success_status "${segment_status}"; then
    return 0
  fi
  return 1
}

job_export_outputs_complete() {
  local dataset_root="$1"
  local input_dir="$2"
  local job="$3"
  local anndata_file="${dataset_root}/exports/${job}/anndata/segger_segmentation.h5ad"
  local xenium_file="${dataset_root}/exports/${job}/xenium_explorer/seg_experiment.xenium"

  if ! job_exports_requested; then
    return 0
  fi

  if [[ "${RUN_ANNDATA_EXPORT}" == "1" && ! -f "${anndata_file}" ]]; then
    return 1
  fi

  if [[ "${RUN_XENIUM_EXPORT}" == "1" ]]; then
    if xenium_export_supported_for_input "${input_dir}"; then
      [[ -f "${xenium_file}" ]] || return 1
    elif [[ "${SKIP_UNSUPPORTED_XENIUM_EXPORT}" != "1" ]]; then
      return 1
    fi
  fi

  return 0
}

dataset_missing_primary_results() {
  local dataset="$1"
  local dataset_root="$2"
  local input_dir=""
  local line=""
  local job=""
  local fragment_mode=""
  input_dir="$(dataset_input_dir "${dataset}" 2>/dev/null || true)"
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    IFS='|' read -r \
      job _group _mode _use_3d _expansion _txk _txdist _layers _heads _cellsmin _minqv _alignment _align_weight _pred_scale fragment_mode \
      <<< "${line}"
    if [[ "${FORCE_RERUN_FRAGMENT_JOBS}" == "1" && "${fragment_mode}" == "true" ]]; then
      return 0
    fi
    if ! job_primary_output_exists "${dataset_root}" "${job}"; then
      return 0
    fi
    if [[ -n "${input_dir}" ]] && ! job_export_outputs_complete "${dataset_root}" "${input_dir}" "${job}"; then
      return 0
    fi
  done <<EOF
$(job_specs)
EOF
  return 1
}

ensure_alignment_reference_ready() {
  local dataset="$1"
  local scrna_ref="$2"
  local tissue_type="$3"

  [[ "${ENABLE_ALIGNMENT_JOBS}" == "1" ]] || return 0

  case "${ALIGNMENT_REFERENCE_MODE}" in
    path)
      if [[ -z "${scrna_ref}" ]]; then
        echo "ERROR: dataset=${dataset} requires a local scRNA reference in ALIGNMENT_REFERENCE_MODE=path, but none is configured." >&2
        return 1
      fi
      if [[ ! -f "${scrna_ref}" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then
          printf "[%s] PLAN dataset=%s local scRNA reference missing in path mode (dry-run): %s\n" \
            "$(timestamp)" "${dataset}" "${scrna_ref}" >&2
          return 0
        fi
        echo "ERROR: dataset=${dataset} local scRNA reference not found: ${scrna_ref}" >&2
        return 1
      fi
      ;;
    auto)
      if [[ -z "${scrna_ref}" ]]; then
        echo "ERROR: dataset=${dataset} requires a local scRNA reference in ALIGNMENT_REFERENCE_MODE=auto, but none is configured." >&2
        return 1
      fi
      if [[ ! -f "${scrna_ref}" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then
          printf "[%s] PLAN dataset=%s local scRNA reference missing in auto mode (dry-run): %s\n" \
            "$(timestamp)" "${dataset}" "${scrna_ref}" >&2
          return 0
        fi
        echo "ERROR: dataset=${dataset} local scRNA reference not found: ${scrna_ref}" >&2
        return 1
      fi
      ;;
  esac
}

refresh_scrna_reference_from_atlas() {
  local dataset="$1"
  local tissue_type="$2"
  local reason="$3"
  local organism
  local -a fetch_cmd

  organism="$(dataset_organism "${dataset}")"
  if [[ -z "${tissue_type}" || -z "${organism}" ]]; then
    printf "[%s] WARN dataset=%s atlas refresh skipped (missing tissue/organism mapping)\n" \
      "$(timestamp)" "${dataset}" >&2
    return 0
  fi

  if ! command -v "${SEGGER_BIN}" >/dev/null 2>&1; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf "[%s] PLAN dataset=%s atlas fetch skipped in dry-run (missing command: %s)\n" \
        "$(timestamp)" "${dataset}" "${SEGGER_BIN}"
      return 0
    fi
    printf "[%s] ERROR dataset=%s atlas fetch requested but SEGGER_BIN not found: %s\n" \
      "$(timestamp)" "${dataset}" "${SEGGER_BIN}" >&2
    return 1
  fi

  fetch_cmd=(
    "${SEGGER_BIN}" atlas fetch
    "${tissue_type}"
    --organism "${organism}"
    --cache-dir "${SCRNA_CACHE_DIR}"
  )
  if [[ "${FORCE_SCRNA_REFETCH}" == "1" ]]; then
    fetch_cmd+=(--force)
  fi

  printf "[%s] INFO dataset=%s atlas refresh (%s): %s\n" \
    "$(timestamp)" "${dataset}" "${reason}" "$(shell_join "${fetch_cmd[@]}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${fetch_cmd[@]}"
}

job_exports_requested() {
  [[ "${RUN_EXPORT_JOBS}" == "1" && ( "${RUN_ANNDATA_EXPORT}" == "1" || "${RUN_XENIUM_EXPORT}" == "1" ) ]]
}

xenium_export_supported_for_input() {
  local input_dir="$1"
  [[ -f "${input_dir}/cells.zarr.zip" ]]
}

dataset_segment_mem_gb() {
  case "${1:-}" in
    xenium_mouse_brain) printf '512' ;;
    xenium_nsclc|xenium_mouse_liver) printf '256' ;;
    xenium_breast) printf '128' ;;
    xenium_crc|xenium_v1_colon) printf '128' ;;
    *) printf '96' ;;
  esac
}

dataset_segment_gmem() {
  case "${1:-}" in
    xenium_mouse_brain) printf '39' ;;
    xenium_crc) printf '25' ;;
    *) printf '30' ;;
  esac
}

dataset_segment_wall_time() {
  case "${1:-}" in
    xenium_mouse_brain) printf '12:00' ;;
    xenium_nsclc|xenium_mouse_liver) printf '10:00' ;;
    *) printf '%s' "${SEGMENT_WALL_TIME_DEFAULT}" ;;
  esac
}

dataset_predict_mem_gb() {
  local dataset="$1"
  local fragment="$2"
  if [[ "${fragment}" == "true" ]]; then
    case "${dataset}" in
      xenium_mouse_brain|merscope_mouse_brain) printf '512' ;;
      cosmx_human_pancreas) printf '384' ;;
      *) printf '256' ;;
    esac
  else
    case "${dataset}" in
      xenium_nsclc|xenium_mouse_liver) printf '160' ;;
      xenium_crc|xenium_v1_colon) printf '128' ;;
      *) printf '96' ;;
    esac
  fi
}

dataset_predict_gmem() {
  local dataset="$1"
  local fragment="$2"
  if [[ "${dataset}" == "xenium_crc" ]]; then
    printf '25'
    return 0
  fi
  if [[ "${fragment}" == "true" ]]; then
    case "${dataset}" in
      cosmx_human_pancreas) printf '39' ;;
      *) printf '%s' "${PREDICT_FRAGMENT_GMEM_GB}" ;;
    esac
  else
    printf '30'
  fi
}

dataset_predict_wall_time() {
  local dataset="$1"
  local fragment="$2"
  if [[ "${fragment}" == "true" ]]; then
    case "${dataset}" in
      xenium_mouse_brain) printf '16:00' ;;
      xenium_nsclc|xenium_mouse_liver|xenium_crc|xenium_v1_colon) printf '12:00' ;;
      *) printf '10:00' ;;
    esac
    return 0
  fi
  case "${dataset}" in
    xenium_mouse_brain) printf '12:00' ;;
    xenium_nsclc|xenium_mouse_liver) printf '10:00' ;;
    *) printf '%s' "${SEGMENT_WALL_TIME_DEFAULT}" ;;
  esac
}

clamp_gmem_for_queue() {
  local queue_name="$1"
  local gmem="$2"
  local upper=""
  if [[ "${queue_name}" == "${GPU_QUEUE_DEFAULT}" ]]; then
    upper="${GPU_QUEUE_MAX_GMEM}"
  elif [[ "${queue_name}" == "${GPU_QUEUE_PRO}" ]]; then
    upper="${GPU_QUEUE_PRO_MAX_GMEM}"
  fi
  if [[ -n "${upper}" ]]; then
    gmem="$(number_min "${gmem}" "${upper}")"
  fi
  printf '%s' "${gmem}"
}

clamp_mem_for_queue() {
  local queue_name="$1"
  local mem_gb="$2"
  local upper=""
  if [[ "${queue_name}" == "${GPU_QUEUE_DEFAULT}" ]]; then
    upper="${GPU_QUEUE_MAX_MEM_GB}"
  elif [[ "${queue_name}" == "${GPU_QUEUE_PRO}" ]]; then
    upper="${GPU_QUEUE_PRO_MAX_MEM_GB}"
  fi
  if [[ -n "${upper}" ]]; then
    mem_gb="$(number_min "${mem_gb}" "${upper}")"
  fi
  printf '%s' "${mem_gb}"
}

dataset_tiling_margin_training() {
  case "${1:-}" in
    xenium_mouse_brain) printf '4' ;;
    xenium_breast|xenium_v1_breast) printf '6' ;;
    *) printf '8' ;;
  esac
}

dataset_tiling_margin_prediction() {
  case "${1:-}" in
    xenium_mouse_brain) printf '4' ;;
    xenium_breast|xenium_v1_breast) printf '6' ;;
    *) printf '8' ;;
  esac
}

dataset_segment_prediction_mode() {
  local mode_lc
  mode_lc="$(printf '%s' "${SEGMENT_PREDICTION_MODE:-nucleus}" | tr '[:upper:]' '[:lower:]')"
  case "${mode_lc}" in
    nucleus|cell)
      printf '%s' "${mode_lc}"
      ;;
    *)
      printf '%s' "nucleus"
      ;;
  esac
}

dataset_effective_min_qv() {
  local default_min_qv="$2"
  printf '%s' "${default_min_qv}"
}

resolve_primary_resources() {
  local dataset="$1"
  local mode="$2"
  local fragment="$3"
  local alignment="${4:-false}"
  local queue_name="${GPU_QUEUE_PRO}"
  local gmem=""
  local mem_gb=""
  local wall_time=""
  local wall_minutes=0
  local max_wall_minutes=0
  local route_mem_gb=""

  if [[ "${mode}" == "segment" ]]; then
    gmem="$(dataset_segment_gmem "${dataset}")"
    mem_gb="$(dataset_segment_mem_gb "${dataset}")"
    wall_time="$(dataset_segment_wall_time "${dataset}")"
  else
    gmem="$(dataset_predict_gmem "${dataset}" "${fragment}")"
    mem_gb="$(dataset_predict_mem_gb "${dataset}" "${fragment}")"
    wall_time="$(dataset_predict_wall_time "${dataset}" "${fragment}")"
  fi

  if [[ "${ROUTE_ELIGIBLE_TO_GPU}" == "1" ]]; then
    # Routing should use the effective RAM request that would be used on gpu.
    route_mem_gb="$(clamp_mem_for_queue "${GPU_QUEUE_DEFAULT}" "${mem_gb}")"
    wall_minutes="$(wall_to_minutes "${wall_time}")"
    max_wall_minutes=$((GPU_QUEUE_MAX_WALL_H * 60))
    if number_le "${gmem}" "${GPU_QUEUE_MAX_GMEM}" && \
       number_le "${route_mem_gb}" "${GPU_QUEUE_MAX_MEM_GB}" && \
       [[ "${wall_minutes}" -le "${max_wall_minutes}" ]]; then
      queue_name="${GPU_QUEUE_DEFAULT}"
      mem_gb="${route_mem_gb}"
    fi
  fi

  # Keep known unstable datasets on the pro queue without changing base gmem.
  if dataset_forces_gpu_pro "${dataset}"; then
    queue_name="${GPU_QUEUE_PRO}"
  fi

  if [[ -n "${ALIGNMENT_GPU_QUEUE}" && "${alignment}" == "true" ]]; then
    queue_name="${ALIGNMENT_GPU_QUEUE}"
  fi

  gmem="$(clamp_gmem_for_queue "${queue_name}" "${gmem}")"
  mem_gb="$(clamp_mem_for_queue "${queue_name}" "${mem_gb}")"

  printf "%s\t%s\t%s\t%s\n" "${queue_name}" "${gmem}" "${mem_gb}" "${wall_time}"
}

resolve_export_resources() {
  local queue_name="${EXPORT_QUEUE}"
  local mem_gb="128"
  mem_gb="$(clamp_mem_for_queue "${queue_name}" "${mem_gb}")"
  printf "%s\t%s\t%s\n" "${queue_name}" "${mem_gb}" "${EXPORT_WALL_TIME_DEFAULT}"
}

primary_success_status() {
  local mode="$1"
  if [[ "${mode}" == "predict" ]]; then
    printf 'predict_ok'
  else
    printf 'segment_ok'
  fi
}

classify_primary_exit_status() {
  local mode="$1"
  local exit_code="$2"
  local prefix="segment"
  if [[ "${mode}" == "predict" ]]; then
    prefix="predict"
  fi
  case "${exit_code}" in
    137|140) printf '%s_oom' "${prefix}" ;;
    138|143) printf '%s_runlimit' "${prefix}" ;;
    *) printf '%s_error' "${prefix}" ;;
  esac
}

job_complete() {
  local dataset_root="$1"
  local input_dir="$2"
  local job="$3"

  if ! job_primary_output_exists "${dataset_root}" "${job}"; then
    return 1
  fi
  if ! job_export_outputs_complete "${dataset_root}" "${input_dir}" "${job}"; then
    return 1
  fi
  return 0
}

render_primary_attempt_script() {
  local dataset_root="$1"
  local dataset="$2"
  local input_dir="$3"
  local scrna_ref="$4"
  local tissue_type="$5"
  local job="$6"
  local group="$7"
  local mode="$8"
  local use_3d="$9"
  local expansion="${10}"
  local txk="${11}"
  local txdist="${12}"
  local layers="${13}"
  local heads="${14}"
  local cellsmin="${15}"
  local minqv="${16}"
  local alignment="${17}"
  local align_weight="${18}"
  local pred_scale="${19}"
  local fragment="${20}"
  local attempt="${21}"
  local queue_name="${22}"
  local gmem="${23}"
  local mem_gb="${24}"
  local wall_time="${25}"
  local baseline_dependency_name="${26:-}"
  local paired_predict_dependency_name="${27:-}"

  local seg_dir="${dataset_root}/runs/${job}"
  local seg_file="${seg_dir}/segger_segmentation.parquet"
  local baseline_ckpt="${dataset_root}/runs/baseline/checkpoints/last.ckpt"
  local segment_status_file="${seg_dir}/.segment_status"
  local out_log="${dataset_root}/logs/${job}.attempt${attempt}.out"
  local err_log="${dataset_root}/logs/${job}.attempt${attempt}.err"
  local script_path="${dataset_root}/bsub/${job}.attempt${attempt}.sh"
  local lsf_name="segger_${dataset}_${job}_${RUN_LABEL}_a${attempt}"
  local success_status
  local failure_prefix="segment"
  local segger_cmd_line=""
  local align_ref_args=""
  local dependency_directive=""
  local activation_block=":"
  local tiling_margin_training=""
  local tiling_margin_prediction=""
  local segment_prediction_mode=""
  local effective_minqv=""

  success_status="$(primary_success_status "${mode}")"
  if [[ "${mode}" == "predict" ]]; then
    failure_prefix="predict"
  fi

  tiling_margin_training="$(dataset_tiling_margin_training "${dataset}")"
  tiling_margin_prediction="$(dataset_tiling_margin_prediction "${dataset}")"

  if [[ "${mode}" == "segment" ]]; then
    segment_prediction_mode="$(dataset_segment_prediction_mode "${dataset}")"
    effective_minqv="$(dataset_effective_min_qv "${dataset}" "${minqv}")"
    if [[ "${CLI_IO_STYLE}" == "named" ]]; then
      # Main branch: pass only flags detected in `segger segment --help`.
      segger_cmd_line="$(shell_join \
        "${SEGGER_BIN}" segment \
        -i "${input_dir}" \
        -o "${seg_dir}")"
      if [[ "${CLI_SEGMENT_SUPPORTS_N_EPOCHS}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --n-epochs "${N_EPOCHS}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_PREDICTION_MODE}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --prediction-mode "${segment_prediction_mode}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_EXPANSION_FLAG}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join "${CLI_EXPANSION_FLAG}" "${pred_scale}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_TRANSCRIPTS_MAX_K}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --transcripts-max-k "${txk}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_TRANSCRIPTS_MAX_DIST}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --transcripts-max-dist "${txdist}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_TILING_MARGIN_TRAINING}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --tiling-margin-training "${tiling_margin_training}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_TILING_MARGIN_PREDICTION}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --tiling-margin-prediction "${tiling_margin_prediction}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_N_MID_LAYERS}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --n-mid-layers "${layers}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_N_HEADS}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --n-heads "${heads}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_CELLS_MIN_COUNTS}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --cells-min-counts "${cellsmin}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_MIN_QV}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --min-qv "${effective_minqv}")"
      fi
      if [[ "${CLI_SEGMENT_SUPPORTS_USE_3D}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --use-3d "${use_3d}")"
      fi
    else
      # v2-incremental: segger segment <in> <out> --prediction-scale-factor ...
      segger_cmd_line="$(shell_join \
        "${SEGGER_BIN}" segment \
        "${input_dir}" \
        "${seg_dir}" \
        --n-epochs "${N_EPOCHS}" \
        --prediction-mode "${segment_prediction_mode}" \
        "${CLI_EXPANSION_FLAG}" "${pred_scale}" \
        --use-3d "${use_3d}" \
        --transcripts-max-k "${txk}" \
        --transcripts-max-dist "${txdist}" \
        --tiling-margin-training "${tiling_margin_training}" \
        --tiling-margin-prediction "${tiling_margin_prediction}" \
        --n-mid-layers "${layers}" \
        --n-heads "${heads}" \
        --cells-min-counts "${cellsmin}" \
        --min-qv "${effective_minqv}")"
    fi
    if [[ "${alignment}" == "true" ]]; then
      align_ref_args="$(alignment_reference_args "${scrna_ref}" "${ALIGNMENT_REFERENCE_MODE}")"
      if [[ "${CLI_IO_STYLE}" == "named" ]]; then
        if [[ "${CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS}" == "1" ]]; then
          segger_cmd_line+=" $(shell_join --alignment-loss)"
        else
          printf "[%s] WARN dataset=%s job=%s alignment requested but --alignment-loss is not supported by segment CLI; skipping alignment flags.\n" \
            "$(timestamp)" "${dataset}" "${job}" >&2
        fi
        if [[ "${CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS_WEIGHT_START}" == "1" ]]; then
          segger_cmd_line+=" $(shell_join --alignment-loss-weight-start 0.0)"
        fi
        if [[ "${CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS_WEIGHT_END}" == "1" ]]; then
          segger_cmd_line+=" $(shell_join --alignment-loss-weight-end "${align_weight}")"
        fi
      else
        segger_cmd_line+=" $(shell_join \
          --alignment-loss \
          --alignment-loss-weight-start 0.0 \
          --alignment-loss-weight-end "${align_weight}")"
      fi
      if [[ -n "${align_ref_args}" ]]; then
        segger_cmd_line+=" ${align_ref_args}"
      fi
    fi
  else
    local dependency_expr=""
    if [[ -n "${baseline_dependency_name}" ]]; then
      dependency_expr="done(${baseline_dependency_name})"
    fi
    if [[ -n "${paired_predict_dependency_name}" ]]; then
      if [[ -n "${dependency_expr}" ]]; then
        dependency_expr+=" && "
      fi
      dependency_expr+="done(${paired_predict_dependency_name})"
    fi
    if [[ -n "${dependency_expr}" ]]; then
      dependency_directive="#BSUB -w \"${dependency_expr}\""
    fi
    segger_cmd_line="$(shell_join \
      "${SEGGER_BIN}" predict \
      -c "${baseline_ckpt}" \
      -i "${input_dir}" \
      -o "${seg_dir}")"
    if [[ "${CLI_PREDICT_SUPPORTS_EXPANSION_FLAG}" == "1" ]]; then
      segger_cmd_line+=" $(shell_join "${CLI_EXPANSION_FLAG}" "${pred_scale}")"
    fi
    if [[ "${CLI_PREDICT_SUPPORTS_USE_3D}" == "1" ]]; then
      segger_cmd_line+=" $(shell_join --use-3d checkpoint)"
    fi
    if [[ "${fragment}" == "true" ]]; then
      if [[ "${CLI_PREDICT_SUPPORTS_FRAGMENT_MODE}" == "1" ]]; then
        segger_cmd_line+=" $(shell_join --fragment-mode)"
      fi
    fi
  fi

  if [[ -n "${MAMBA_ACTIVATE_CMD}" ]]; then
    activation_block=$'set +u\n'"${MAMBA_ACTIVATE_CMD}"$'\nset -u'
  fi

  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#BSUB -J ${lsf_name}
#BSUB -o ${out_log}
#BSUB -e ${err_log}
#BSUB -L ${LSF_EXEC_SHELL}
#BSUB -q ${queue_name}
#BSUB -W ${wall_time}
#BSUB -n 8
#BSUB -R "rusage[mem=${mem_gb}G]"
#BSUB -M ${mem_gb}G
#BSUB -gpu "num=1:j_exclusive=yes:gmem=${gmem}G"
${dependency_directive}

set -euo pipefail

GPU_USAGE_POLL_SEC=$(printf '%q' "${GPU_USAGE_POLL_SEC}")
VRAM_TRACK_FILE=$(printf '%q' "${seg_dir}/.vram_peak_attempt${attempt}.txt")
GPU_MONITOR_PID=""

read_peak_vram_mb() {
  local peak_mb="0"
  if [[ -f "\${VRAM_TRACK_FILE}" ]]; then
    peak_mb="\$(tr -cd '0-9' < "\${VRAM_TRACK_FILE}" | head -c 16)"
  fi
  [[ -n "\${peak_mb}" ]] || peak_mb="0"
  printf '%s' "\${peak_mb}"
}

read_peak_vram_gb() {
  local peak_mb
  peak_mb="\$(read_peak_vram_mb)"
  awk -v mb="\${peak_mb}" 'BEGIN { printf "%.2f", (mb + 0.0) / 1024.0 }'
}

update_peak_vram_once() {
  local used current
  used="\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -cd '0-9' || true)"
  [[ "\${used}" =~ ^[0-9]+$ ]] || return 0

  current="\$(read_peak_vram_mb)"
  [[ "\${current}" =~ ^[0-9]+$ ]] || current="0"
  if (( used > current )); then
    printf '%s\n' "\${used}" > "\${VRAM_TRACK_FILE}"
  fi
}

start_peak_vram_monitor() {
  (
    while true; do
      update_peak_vram_once || true
      sleep "\${GPU_USAGE_POLL_SEC}" || break
    done
  ) &
  GPU_MONITOR_PID="\$!"
}

stop_peak_vram_monitor() {
  if [[ -n "\${GPU_MONITOR_PID}" ]]; then
    kill "\${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "\${GPU_MONITOR_PID}" 2>/dev/null || true
    GPU_MONITOR_PID=""
  fi
}

write_segment_status() {
  local stage_status="\$1"
  local stage_rc="\${2:-0}"
  local peak_mb peak_gb
  peak_mb="\$(read_peak_vram_mb)"
  peak_gb="\$(read_peak_vram_gb)"
  printf 'segment_status=%s\nsegment_rc=%s\nmax_vram_mb=%s\nmax_vram_gb=%s\n' \
    "\${stage_status}" "\${stage_rc}" "\${peak_mb}" "\${peak_gb}" > $(printf '%q' "${segment_status_file}")
}

classify_segment_status() {
  local stage_rc="\$1"
  case "\${stage_rc}" in
    137|140) printf '%s_oom' $(printf '%q' "${failure_prefix}") ;;
    138|143) printf '%s_runlimit' $(printf '%q' "${failure_prefix}") ;;
    *) printf '%s_error' $(printf '%q' "${failure_prefix}") ;;
  esac
}

on_primary_exit() {
  local rc=\$?
  local current_status=""
  trap - EXIT INT TERM HUP
  stop_peak_vram_monitor
  update_peak_vram_once || true
  if [[ -f $(printf '%q' "${segment_status_file}") ]]; then
    current_status="\$(awk -F'=' '\$1 == \"segment_status\" { print \$2; exit }' $(printf '%q' "${segment_status_file}") 2>/dev/null || true)"
  fi
  if [[ "\${rc}" -ne 0 ]] && [[ "\${current_status}" != $(printf '%q' "${success_status}") ]]; then
    write_segment_status "\$(classify_segment_status "\${rc}")" "\${rc}"
  elif [[ -z "\${current_status}" ]]; then
    write_segment_status "pending" "\${rc}"
  fi
  exit "\${rc}"
}

trap on_primary_exit EXIT INT TERM HUP

if [[ -d $(printf '%q' "${CLUSTER_CODE_ROOT}") ]]; then
  cd $(printf '%q' "${CLUSTER_CODE_ROOT}")
else
  echo "[JOB] WARN missing code_root=$(printf '%q' "${CLUSTER_CODE_ROOT}"); continuing from \$(pwd)" >&2
fi
${activation_block}

echo "[JOB] dataset=${dataset} job=${job} attempt=${attempt}"
echo "[JOB] queue=${queue_name} gmem=${gmem}G mem=${mem_gb}G wall=${wall_time}"
hostname || true
date || true
echo "[JOB] code_root=$(printf '%q' "${CLUSTER_CODE_ROOT}")"
echo "[JOB] segger_bin=\$(command -v $(printf '%q' "${SEGGER_BIN}") || true)"
if command -v git >/dev/null 2>&1; then
  if git -C $(printf '%q' "${CLUSTER_CODE_ROOT}") rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[JOB] git_commit=\$(git -C $(printf '%q' "${CLUSTER_CODE_ROOT}") rev-parse --short=12 HEAD 2>/dev/null || true)"
    echo "[JOB] git_dirty_files=\$(git -C $(printf '%q' "${CLUSTER_CODE_ROOT}") status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  fi
fi
if $(printf '%q' "${SEGGER_BIN}") --version >/dev/null 2>&1; then
  echo "[JOB] segger_version=\$($(printf '%q' "${SEGGER_BIN}") --version 2>/dev/null | head -n1)"
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits || true

mkdir -p $(printf '%q' "${seg_dir}")
printf '0\n' > "\${VRAM_TRACK_FILE}"
update_peak_vram_once || true
start_peak_vram_monitor
write_segment_status "running" "0"
$(printf '%s' "${segger_cmd_line}")
segger_rc=\$?
stop_peak_vram_monitor
update_peak_vram_once || true
echo "[JOB] segger_rc=\${segger_rc}"
if [[ "\${segger_rc}" -ne 0 ]]; then
  write_segment_status "\$(classify_segment_status "\${segger_rc}")" "\${segger_rc}"
  exit "\${segger_rc}"
fi

if [[ -f $(printf '%q' "${seg_file}") ]]; then
  ln -sfn segger_segmentation.parquet $(printf '%q' "${seg_dir}/transcripts.parquet")
fi

write_segment_status $(printf '%q' "${success_status}") "0"
echo "[JOB] completed"
exit 0
EOF

  chmod +x "${script_path}"
  printf '%s' "${script_path}"
}

render_export_attempt_script() {
  local dataset_root="$1"
  local dataset="$2"
  local input_dir="$3"
  local job="$4"
  local attempt="$5"
  local queue_name="$6"
  local mem_gb="$7"
  local wall_time="$8"
  local primary_dependency_name="${9:-}"

  local seg_dir="${dataset_root}/runs/${job}"
  local seg_file="${seg_dir}/segger_segmentation.parquet"
  local anndata_dir="${dataset_root}/exports/${job}/anndata"
  local xenium_dir="${dataset_root}/exports/${job}/xenium_explorer"
  local export_status_file="${seg_dir}/.export_status"
  local out_log="${dataset_root}/logs/${job}.export.attempt${attempt}.out"
  local err_log="${dataset_root}/logs/${job}.export.attempt${attempt}.err"
  local script_path="${dataset_root}/bsub/${job}.export.attempt${attempt}.sh"
  local lsf_name="segger_${dataset}_${job}_${RUN_LABEL}_a${attempt}_export"
  local dependency_directive=""
  local activation_block=":"
  local xenium_supported="0"
  local export_anndata_cmd=""
  local export_xenium_cmd=""
  local -a export_anndata_parts=()

  if [[ -n "${primary_dependency_name}" ]]; then
    dependency_directive="#BSUB -w \"done(${primary_dependency_name})\""
  fi

  if xenium_export_supported_for_input "${input_dir}"; then
    xenium_supported="1"
  fi

  export_anndata_parts=(
    "${SEGGER_BIN}" export
    -s "${seg_file}"
    -i "${input_dir}"
    -o "${anndata_dir}"
    --format anndata
  )
  if awk -v v="${ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT}" 'BEGIN { exit !((v + 0.0) > 0) }'; then
    export_anndata_parts+=(--min-similarity-shift "${ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT}")
  fi
  export_anndata_cmd="$(shell_join "${export_anndata_parts[@]}")"
  export_xenium_cmd="$(shell_join \
    "${SEGGER_BIN}" export \
    -s "${seg_file}" \
    -i "${input_dir}" \
    -o "${xenium_dir}" \
    --format xenium_explorer \
    --boundary-method convex_hull \
    --boundary-voxel-size 5 \
    --num-workers 8)"

  if [[ -n "${MAMBA_ACTIVATE_CMD}" ]]; then
    activation_block=$'set +u\n'"${MAMBA_ACTIVATE_CMD}"$'\nset -u'
  fi

  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
#BSUB -J ${lsf_name}
#BSUB -o ${out_log}
#BSUB -e ${err_log}
#BSUB -L ${LSF_EXEC_SHELL}
#BSUB -q ${queue_name}
#BSUB -W ${wall_time}
#BSUB -n 4
#BSUB -R "rusage[mem=${mem_gb}G]"
#BSUB -M ${mem_gb}G
${dependency_directive}

set -euo pipefail

write_export_status() {
  local stage_status="\$1"
  local stage_rc="\${2:-0}"
  local anndata_rc="\${3:-0}"
  local xenium_rc="\${4:-0}"
  local export_note="\${5:-}"
  printf 'export_status=%s\nexport_rc=%s\nanndata_rc=%s\nxenium_rc=%s\nexport_note=%s\n' \
    "\${stage_status}" "\${stage_rc}" "\${anndata_rc}" "\${xenium_rc}" "\${export_note}" \
    > $(printf '%q' "${export_status_file}")
}

on_export_exit() {
  local rc=\$?
  local current_status=""
  trap - EXIT INT TERM HUP
  if [[ -f $(printf '%q' "${export_status_file}") ]]; then
    current_status="\$(awk -F'=' '\$1 == \"export_status\" { print \$2; exit }' $(printf '%q' "${export_status_file}") 2>/dev/null || true)"
  fi
  if [[ "\${rc}" -ne 0 ]] && [[ "\${current_status}" != "export_ok" ]] && [[ "\${current_status}" != "export_skipped_xenium_missing_source" ]]; then
    write_export_status "export_error" "\${rc}" "0" "0" ""
  elif [[ -z "\${current_status}" ]]; then
    write_export_status "pending" "\${rc}" "0" "0" ""
  fi
  exit "\${rc}"
}

trap on_export_exit EXIT INT TERM HUP

if [[ -d $(printf '%q' "${CLUSTER_CODE_ROOT}") ]]; then
  cd $(printf '%q' "${CLUSTER_CODE_ROOT}")
else
  echo "[JOB] WARN missing code_root=$(printf '%q' "${CLUSTER_CODE_ROOT}"); continuing from \$(pwd)" >&2
fi
${activation_block}

echo "[JOB] dataset=${dataset} job=${job} export attempt=${attempt}"
echo "[JOB] queue=${queue_name} mem=${mem_gb}G wall=${wall_time}"
hostname || true
date || true

if [[ $(printf '%q' "${RUN_ANNDATA_EXPORT}") != "1" && $(printf '%q' "${RUN_XENIUM_EXPORT}") != "1" ]]; then
  write_export_status "not_requested" "0" "0" "0" ""
  echo "[JOB] export disabled"
  exit 0
fi

if [[ ! -f $(printf '%q' "${seg_file}") ]]; then
  write_export_status "export_error" "1" "0" "0" "missing_segmentation"
  echo "[JOB] export missing segmentation input"
  exit 1
fi

mkdir -p $(printf '%q' "${anndata_dir}") $(printf '%q' "${xenium_dir}")
write_export_status "running" "0" "0" "0" ""

anndata_rc=0
xenium_rc=0
overall_rc=0
final_status="export_ok"
export_note=""

if [[ $(printf '%q' "${RUN_ANNDATA_EXPORT}") == "1" ]]; then
  ${export_anndata_cmd} || anndata_rc=\$?
fi

if [[ $(printf '%q' "${RUN_XENIUM_EXPORT}") == "1" ]]; then
  if [[ $(printf '%q' "${xenium_supported}") == "1" ]]; then
    ${export_xenium_cmd} || xenium_rc=\$?
  elif [[ $(printf '%q' "${SKIP_UNSUPPORTED_XENIUM_EXPORT}") == "1" ]]; then
    export_note="xenium_missing_source"
    final_status="export_skipped_xenium_missing_source"
  else
    xenium_rc=1
    export_note="xenium_missing_source_required"
  fi
fi

if [[ "\${anndata_rc}" -ne 0 ]]; then
  overall_rc="\${anndata_rc}"
  final_status="export_error"
elif [[ "\${xenium_rc}" -ne 0 ]]; then
  overall_rc="\${xenium_rc}"
  final_status="export_error"
fi

write_export_status "\${final_status}" "\${overall_rc}" "\${anndata_rc}" "\${xenium_rc}" "\${export_note}"
echo "[JOB] export_status=\${final_status} export_rc=\${overall_rc}"
if [[ "\${overall_rc}" -ne 0 ]]; then
  exit "\${overall_rc}"
fi
echo "[JOB] completed"
exit 0
EOF

  chmod +x "${script_path}"
  printf '%s' "${script_path}"
}

status_note() {
  local queue_name="$1"
  local gmem="$2"
  local mem_gb="$3"
  local wall_time="$4"
  local attempt="$5"
  local job_id="$6"
  local suffix="${7:-}"
  local note
  note="queue=${queue_name} gmem=${gmem}G mem=${mem_gb}G wall=${wall_time} attempt=${attempt}"
  if [[ -n "${job_id}" ]]; then
    note+=" lsf_job=${job_id}"
  fi
  if [[ -n "${suffix}" ]]; then
    note+=" ${suffix}"
  fi
  printf '%s' "${note}"
}

next_attempt_for_job() {
  local dataset_root="$1"
  local job="$2"
  local best=0
  local f num
  for f in \
    "${dataset_root}/logs/${job}.attempt"*.out \
    "${dataset_root}/logs/${job}.attempt"*.err \
    "${dataset_root}/bsub/${job}.attempt"*.sh \
    "${dataset_root}/bsub/${job}.export.attempt"*.sh; do
    [[ -f "${f}" ]] || continue
    num="$(printf '%s\n' "$(basename "${f}")" | sed -n 's/.*\.attempt\([0-9][0-9]*\)\..*/\1/p')"
    [[ -n "${num}" ]] || continue
    if [[ "${num}" -gt "${best}" ]]; then
      best="${num}"
    fi
  done
  printf '%s' "$((best + 1))"
}

contains_pattern() {
  local file_path="$1"
  local pattern="$2"
  [[ -f "${file_path}" ]] || return 1
  if command -v rg >/dev/null 2>&1; then
    rg -qi "${pattern}" "${file_path}"
  else
    grep -Eiq "${pattern}" "${file_path}"
  fi
}

detect_prior_gpu_failure() {
  local dataset_root="$1"
  local job="$2"
  local attempt="$3"
  local out_log="${dataset_root}/logs/${job}.attempt${attempt}.out"
  local err_log="${dataset_root}/logs/${job}.attempt${attempt}.err"
  local summary_file summary_hit

  if [[ "${attempt}" -le 0 ]]; then
    printf 'none'
    return 0
  fi

  if contains_pattern "${out_log}" 'cudaerrorillegaladdress|cuda_error_illegal_address|illegal memory access|cudadrivererror' || \
     contains_pattern "${err_log}" 'cudaerrorillegaladdress|cuda_error_illegal_address|illegal memory access|cudadrivererror'; then
    printf 'gpu_illegal_access'
    return 0
  fi

  if contains_pattern "${out_log}" 'term_memlimit|exited with exit code 137|exited with exit code 140' || \
     contains_pattern "${err_log}" 'term_memlimit|exited with exit code 137|exited with exit code 140|out of memory|cuda error: out of memory|cudaerrormemoryallocation|memoryerror: std::bad_alloc|killed[[:space:]]+segger|user defined signal 2'; then
    printf 'gpu_oom_or_memlimit'
    return 0
  fi

  # Fallback: when prior attempt logs were rotated/cleaned, recover the last
  # known failure class from summary snapshots.
  for summary_file in \
    "${dataset_root}/summaries/status_snapshot.tsv" \
    "${dataset_root}/summaries/all_jobs.tsv"; do
    [[ -f "${summary_file}" ]] || continue
    summary_hit="$(awk -F'\t' -v target_job="${job}" -v target_attempt="${attempt}" '
      function field(name) {
        if ((name in idx) && idx[name] > 0 && idx[name] <= NF) return $(idx[name])
        return ""
      }
      NR == 1 {
        for (i = 1; i <= NF; i++) idx[$i] = i
        next
      }
      {
        if (field("job") != target_job) next

        run_count = field("run_count")
        if (run_count ~ /^[0-9]+$/ && (run_count + 0) < (target_attempt + 0)) next

        status = tolower(field("status"))
        seg_status = tolower(field("segment_status"))
        note = tolower(field("note"))

        combined = status " " seg_status " " note
        if (combined ~ /cudaerrorillegaladdress|cuda_error_illegal_address|illegal memory access|cudadrivererror/) {
          print "gpu_illegal_access"
          exit
        }
        if (combined ~ /term_memlimit|out of memory|cuda error: out of memory|cudaerrormemoryallocation|memoryerror: std::bad_alloc|exited with exit code 137|exited with exit code 140|segment_oom|oom/) {
          print "gpu_oom_or_memlimit"
          exit
        }
      }
    ' "${summary_file}")"
    if [[ "${summary_hit}" == "gpu_illegal_access" || "${summary_hit}" == "gpu_oom_or_memlimit" ]]; then
      printf '%s' "${summary_hit}"
      return 0
    fi
  done

  printf 'none'
}

apply_retry_gpu_fallback() {
  local dataset="$1"
  local mode="$2"
  local fragment="$3"
  local queue_name="$4"
  local gmem="$5"
  local mem_gb="$6"
  local wall_time="$7"
  local reason="$8"

  local prior_queue="${queue_name}"
  local prior_gmem="${gmem}"
  local fallback_queue="${GPU_FALLBACK_QUEUE}"
  local fallback_gmem="${GPU_FALLBACK_MIN_GMEM}"
  local fallback_mem="${GPU_FALLBACK_MIN_MEM_GB}"
  local bumped_gmem
  local fallback_wall_minutes
  local wall_minutes

  if [[ "${fragment}" == "true" ]]; then
    fallback_gmem="${GPU_FALLBACK_FRAGMENT_GMEM}"
  fi
  if [[ "${dataset}" == "xenium_mouse_brain" && "${mode}" == "predict" ]]; then
    fallback_gmem="$(number_max "${fallback_gmem}" "${GPU_FALLBACK_FRAGMENT_GMEM}")"
    fallback_mem="$(number_max "${fallback_mem}" "512")"
  fi

  bumped_gmem="$(awk -v current="${gmem}" -v bump="${GPU_RETRY_GMEM_BUMP_GB}" 'BEGIN { printf "%d", (current + 0) + (bump + 0) }')"

  queue_name="${fallback_queue}"
  gmem="$(number_max "${bumped_gmem}" "${fallback_gmem}")"
  mem_gb="$(number_max "${mem_gb}" "${fallback_mem}")"
  gmem="$(clamp_gmem_for_queue "${queue_name}" "${gmem}")"
  mem_gb="$(clamp_mem_for_queue "${queue_name}" "${mem_gb}")"
  wall_minutes="$(wall_to_minutes "${wall_time}")"
  fallback_wall_minutes=$((GPU_FALLBACK_MIN_WALL_H * 60))
  if [[ "${wall_minutes}" -lt "${fallback_wall_minutes}" ]]; then
    wall_time="$(minutes_to_wall "${fallback_wall_minutes}")"
  fi

  printf "%s\t%s\t%s\t%s\tretry_fallback=%s queue_shift=%s->%s gmem_shift=%sG->%sG gmem_bump=+%sG\n" \
    "${queue_name}" "${gmem}" "${mem_gb}" "${wall_time}" "${reason}" \
    "${prior_queue}" "${queue_name}" "${prior_gmem}" "${gmem}" "${GPU_RETRY_GMEM_BUMP_GB}"
}

diagnostics_for_job() {
  local job_id="$1"
  local tmp_file="$2"
  : > "${tmp_file}"
  if is_remote_submit_enabled; then
    ssh -o BatchMode=yes "${LSF_SUBMIT_HOST}" bjobs -l "${job_id}" >> "${tmp_file}" 2>/dev/null || true
  elif command -v bjobs >/dev/null 2>&1; then
    bjobs -l "${job_id}" >> "${tmp_file}" 2>/dev/null || true
  fi
  if is_remote_submit_enabled; then
    ssh -o BatchMode=yes "${LSF_SUBMIT_HOST}" bacct -l "${job_id}" >> "${tmp_file}" 2>/dev/null || true
  elif command -v bacct >/dev/null 2>&1; then
    bacct -l "${job_id}" >> "${tmp_file}" 2>/dev/null || true
  fi
}

classify_failure() {
  local job="$1"
  local attempt="$2"
  local dataset_root="$3"
  local mode="$4"
  local job_id="$5"
  local diag_file
  local out_log="${dataset_root}/logs/${job}.attempt${attempt}.out"
  local err_log="${dataset_root}/logs/${job}.attempt${attempt}.err"

  diag_file="$(mktemp)"
  diagnostics_for_job "${job_id}" "${diag_file}"

  if contains_pattern "${diag_file}" 'TERM_MEMLIMIT' || \
     contains_pattern "${out_log}" 'TERM_MEMLIMIT|out of memory|cuda error: out of memory|cublas status alloc failed' || \
     contains_pattern "${err_log}" 'TERM_MEMLIMIT|out of memory|cuda error: out of memory|cublas status alloc failed'; then
    rm -f "${diag_file}"
    printf 'oom'
    return 0
  fi
  if contains_pattern "${diag_file}" 'TERM_RUNLIMIT' || \
     contains_pattern "${out_log}" 'TERM_RUNLIMIT' || \
     contains_pattern "${err_log}" 'TERM_RUNLIMIT'; then
    rm -f "${diag_file}"
    printf 'runlimit'
    return 0
  fi

  rm -f "${diag_file}"
  if [[ "${mode}" == "predict" ]]; then
    printf 'predict_error'
  else
    printf 'segment_error'
  fi
}

query_job_state() {
  local job_id="$1"
  local state=""
  if is_remote_submit_enabled; then
    state="$(ssh -o BatchMode=yes "${LSF_SUBMIT_HOST}" bjobs -a -noheader -o stat "${job_id}" 2>/dev/null | awk 'NF { print $1; exit }' || true)"
  elif command -v bjobs >/dev/null 2>&1; then
    state="$(bjobs -a -noheader -o stat "${job_id}" 2>/dev/null | awk 'NF { print $1; exit }' || true)"
  fi
  printf '%s' "${state}"
}

submit_attempt_script() {
  local script_path="$1"
  local qtmp=""
  printf -v qtmp '%q' "${WORK_TMP_ROOT}"
  if is_remote_submit_enabled; then
    ssh -o BatchMode=yes "${LSF_SUBMIT_HOST}" "mkdir -p ${qtmp} && TMPDIR=${qtmp} TMP=${qtmp} TEMP=${qtmp} bsub" < "${script_path}"
  else
    TMPDIR="${WORK_TMP_ROOT}" TMP="${WORK_TMP_ROOT}" TEMP="${WORK_TMP_ROOT}" bsub < "${script_path}"
  fi
}

require_submit_transport() {
  if [[ "${AUTO_SUBMIT}" != "1" || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if is_remote_submit_enabled; then
    if ! command -v ssh >/dev/null 2>&1; then
      echo "ERROR: ssh is required for remote LSF submission." >&2
      exit 1
    fi
    if ! ssh -o BatchMode=yes "${LSF_SUBMIT_HOST}" 'command -v bsub >/dev/null 2>&1'; then
      echo "ERROR: Could not reach remote bsub on ${LSF_SUBMIT_HOST}. Use SSH keys/agent or set LSF_SUBMIT_HOST=local if bsub is available locally." >&2
      exit 1
    fi
    return 0
  fi

  if ! command -v bsub >/dev/null 2>&1; then
    echo "ERROR: bsub not found locally. Set LSF_SUBMIT_HOST to a reachable submission host." >&2
    exit 1
  fi
}

require_segger_binary() {
  if [[ -z "${SEGGER_BIN}" ]]; then
    echo "ERROR: SEGGER_BIN is empty." >&2
    exit 1
  fi
}

detect_cli_flag_support() {
  local subcommand="$1"
  local flag="$2"
  local help_out=""

  help_out="$("${SEGGER_BIN}" "${subcommand}" --help 2>&1 || true)"
  if printf '%s\n' "${help_out}" | grep -Fq -- "${flag}"; then
    printf '1'
  else
    printf '0'
  fi
}

run_dashboard_for_root() {
  local dataset_root="$1"
  bash "${SCRIPT_DIR}/benchmark_status_dashboard.sh" --root "${dataset_root}" >/dev/null
}

run_validation_for_root() {
  local dataset_root="$1"
  local input_dir="$2"
  local scrna_ref="$3"
  local tissue_type="$4"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if [[ -z "${scrna_ref}" ]]; then
    echo "ERROR: dataset_root=${dataset_root} missing SCRNA_REFERENCE_PATH; refusing tissue fallback for validation." >&2
    return 1
  fi
  if [[ ! -f "${scrna_ref}" ]]; then
    echo "ERROR: dataset_root=${dataset_root} SCRNA_REFERENCE_PATH not found; refusing tissue fallback for validation: ${scrna_ref}" >&2
    return 1
  fi

  bash "${SCRIPT_DIR}/build_benchmark_validation_table.sh" \
    --root "${dataset_root}" \
    --input-dir "${input_dir}" \
    --scrna-reference-path "${scrna_ref}" \
    --include-default-10x true \
    >/dev/null
}

run_or_plan_job() {
  local dataset_root="$1"
  local dataset="$2"
  local input_dir="$3"
  local scrna_ref="$4"
  local tissue_type="$5"
  local job="$6"
  local group="$7"
  local mode="$8"
  local use_3d="$9"
  local expansion="${10}"
  local txk="${11}"
  local txdist="${12}"
  local layers="${13}"
  local heads="${14}"
  local cellsmin="${15}"
  local minqv="${16}"
  local alignment="${17}"
  local align_weight="${18}"
  local pred_scale="${19}"
  local fragment="${20}"

  local lane
  local summary_file
  local skipped_file="${dataset_root}/summaries/skipped_existing.tsv"
  local seg_dir="${dataset_root}/runs/${job}"
  local log_file
  local attempt=1
  local queue_name=""
  local gmem=""
  local mem_gb=""
  local wall_time=""
  local export_queue_name=""
  local export_mem_gb=""
  local export_wall_time=""
  local segment_job_id=""
  local export_job_id=""
  local state=""
  local attempt_status="pending"
  local segment_status="pending"
  local export_status="not_requested"
  local export_note=""
  local note=""
  local primary_attempt_script=""
  local export_attempt_script=""
  local bsub_output=""
  local export_submit_output=""
  local export_suffix=""
  local baseline_dep_name=""
  local fallback_note=""
  local previous_attempt=0
  local prior_gpu_failure="none"
  local segment_outputs_present="0"
  local export_outputs_ready="0"
  local force_rerun_fragment_job="0"
  local paired_predict_dep_name=""
  local paired_predict_job=""
  local predict_dependency_list=""
  local primary_dependency_name=""
  local baseline_ckpt_path="${dataset_root}/runs/baseline/checkpoints/last.ckpt"
  local allow_export_only="0"

  lane="$(job_lane "${group}")"
  summary_file="$(summary_file_for_group "${dataset_root}" "${group}")"
  log_file="$(canonical_log_for_job "${dataset_root}" "${job}" "${group}")"

  if [[ "${FORCE_RERUN_FRAGMENT_JOBS}" == "1" && "${fragment}" == "true" ]]; then
    force_rerun_fragment_job="1"
  fi

  if job_primary_output_exists "${dataset_root}" "${job}"; then
    segment_outputs_present="1"
  fi
  if job_export_outputs_complete "${dataset_root}" "${input_dir}" "${job}"; then
    export_outputs_ready="1"
  fi

  if [[ "${RESUME_IF_EXISTS}" == "1" && "${force_rerun_fragment_job}" != "1" ]] && \
     job_complete "${dataset_root}" "${input_dir}" "${job}"; then
    note="existing results found in runs/ or exports/; skipping"
    upsert_status_row \
      "${skipped_file}" "${job}" "${lane}" "skipped_existing" "0" "${note}" "${seg_dir}" "${log_file}" \
      "skipped_existing" "skipped_existing" "" "" "existing_results"
    append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} status=skipped_existing"
    rebuild_all_jobs_summary "${dataset_root}"
    return 0
  fi

  attempt="$(next_attempt_for_job "${dataset_root}" "${job}")"
  previous_attempt=$((attempt - 1))
  if [[ "${ONLY_RETRY_OOM_JOBS}" == "1" ]]; then
    if [[ "${segment_outputs_present}" == "1" ]]; then
      note="filter_only_retry_oom segment_output_present"
      upsert_status_row \
        "${skipped_file}" "${job}" "${lane}" "skipped_filter" "0" "${note}" "${seg_dir}" "${log_file}" \
        "skipped_filter" "skipped_filter" "" "" "only_retry_oom"
      append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} status=skipped_filter reason=only_retry_oom segment_output_present"
      rebuild_all_jobs_summary "${dataset_root}"
      return 0
    fi
    if [[ "${previous_attempt}" -lt 1 ]]; then
      note="filter_only_retry_oom no_prior_attempt"
      upsert_status_row \
        "${skipped_file}" "${job}" "${lane}" "skipped_filter" "0" "${note}" "${seg_dir}" "${log_file}" \
        "skipped_filter" "skipped_filter" "" "" "only_retry_oom"
      append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} status=skipped_filter reason=only_retry_oom no_prior_attempt"
      rebuild_all_jobs_summary "${dataset_root}"
      return 0
    fi

    prior_gpu_failure="$(detect_prior_gpu_failure "${dataset_root}" "${job}" "${previous_attempt}")"
    if [[ "${prior_gpu_failure}" != "gpu_oom_or_memlimit" ]]; then
      note="filter_only_retry_oom prior_failure=${prior_gpu_failure}"
      upsert_status_row \
        "${skipped_file}" "${job}" "${lane}" "skipped_filter" "0" "${note}" "${seg_dir}" "${log_file}" \
        "skipped_filter" "skipped_filter" "" "" "only_retry_oom"
      append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} status=skipped_filter reason=only_retry_oom prior_failure=${prior_gpu_failure}"
      rebuild_all_jobs_summary "${dataset_root}"
      return 0
    fi
  fi
  baseline_dep_name=""
  paired_predict_dep_name=""
  paired_predict_job=""
  predict_dependency_list=""
  if [[ "${mode}" == "predict" && ! -f "${baseline_ckpt_path}" ]]; then
    baseline_dep_name="segger_${dataset}_baseline_${RUN_LABEL}_a${attempt}"
  fi
  if [[ "${mode}" == "predict" && "${fragment}" == "true" ]]; then
    paired_predict_job="$(paired_non_fragment_job "${job}")"
    if [[ -n "${paired_predict_job}" ]] && ! job_primary_output_exists "${dataset_root}" "${paired_predict_job}"; then
      paired_predict_dep_name="segger_${dataset}_${paired_predict_job}_${RUN_LABEL}_a${attempt}"
    fi
  fi
  if [[ -n "${baseline_dep_name}" ]]; then
    predict_dependency_list="${baseline_dep_name}"
  fi
  if [[ -n "${paired_predict_dep_name}" ]]; then
    if [[ -n "${predict_dependency_list}" ]]; then
      predict_dependency_list+=","
    fi
    predict_dependency_list+="${paired_predict_dep_name}"
  fi

  IFS=$'\t' read -r queue_name gmem mem_gb wall_time <<< "$(resolve_primary_resources "${dataset}" "${mode}" "${fragment}" "${alignment}")"
  if [[ "${previous_attempt}" -ge 1 && "${prior_gpu_failure}" == "none" ]]; then
    prior_gpu_failure="$(detect_prior_gpu_failure "${dataset_root}" "${job}" "${previous_attempt}")"
  fi
  if [[ "${previous_attempt}" -ge 1 && "${prior_gpu_failure}" != "none" ]]; then
    if [[ "${GPU_FALLBACK_ON_RETRY}" == "1" || "${ONLY_RETRY_OOM_JOBS}" == "1" ]]; then
      IFS=$'\t' read -r queue_name gmem mem_gb wall_time fallback_note <<< \
        "$(apply_retry_gpu_fallback "${dataset}" "${mode}" "${fragment}" "${queue_name}" "${gmem}" "${mem_gb}" "${wall_time}" "${prior_gpu_failure}")"
    fi
  fi
  if job_exports_requested; then
    IFS=$'\t' read -r export_queue_name export_mem_gb export_wall_time <<< "$(resolve_export_resources)"
    export_status="planned"
    if [[ "${RUN_XENIUM_EXPORT}" == "1" ]] && ! xenium_export_supported_for_input "${input_dir}"; then
      if [[ "${SKIP_UNSUPPORTED_XENIUM_EXPORT}" == "1" ]]; then
        export_note="xenium_missing_source_will_skip"
      else
        export_note="xenium_missing_source_required"
      fi
    fi
  else
    export_status="not_requested"
  fi

  if [[ "${segment_outputs_present}" == "1" ]] && job_exports_requested && [[ "${export_outputs_ready}" != "1" ]]; then
    if [[ "${RESUME_IF_EXISTS}" == "1" && "${force_rerun_fragment_job}" != "1" ]]; then
      allow_export_only="1"
    fi
    if [[ "${RUN_PRIMARY_JOBS}" != "1" ]]; then
      allow_export_only="1"
    fi
  fi
  if [[ "${allow_export_only}" == "1" ]]; then
    segment_status="skipped_existing"
    attempt_status="pending"
    note="export_only existing_segment=true attempt=${attempt}"
    export_attempt_script="$(render_export_attempt_script \
      "${dataset_root}" "${dataset}" "${input_dir}" "${job}" "${attempt}" \
      "${export_queue_name}" "${export_mem_gb}" "${export_wall_time}" "")"

    if [[ "${AUTO_SUBMIT}" != "1" || "${DRY_RUN}" == "1" ]]; then
      upsert_status_row \
        "${summary_file}" "${job}" "${lane}" "pending" "0" "planned ${note}" "${seg_dir}" "${log_file}" \
        "${segment_status}" "${export_status}" "" "" "${export_note}"
      append_canonical_log "${dataset_root}" "${job}" "${group}" "START job=${job} planned_only export_only ${note}"
      append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} planned_only export_manifest_write_complete export_note=${export_note:-none}"
      rebuild_all_jobs_summary "${dataset_root}"
      announce_job_event "PLANNED" "${dataset}" "${job}" "${note}"
      return 0
    fi

    if [[ ! -x "${export_attempt_script}" ]]; then
      export_status="export_error"
      export_note="export_script_missing"
      upsert_status_row \
        "${summary_file}" "${job}" "${lane}" "export_error" "0" "${note}" "${seg_dir}" "${log_file}" \
        "${segment_status}" "${export_status}" "" "" "${export_note}"
      append_canonical_log "${dataset_root}" "${job}" "${group}" "FAIL job=${job} step=export_submit (attempt script missing)"
      rebuild_all_jobs_summary "${dataset_root}"
      announce_job_event "ERROR" "${dataset}" "${job}" "export-only submit failed (attempt script missing)"
      return 1
    fi

    export_submit_output="$(submit_attempt_script "${export_attempt_script}" 2>&1 || true)"
    export_job_id="$(printf '%s\n' "${export_submit_output}" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -n1 || true)"
    if [[ -z "${export_job_id}" ]]; then
      export_status="export_error"
      export_note="export_submit_failed"
      upsert_status_row \
        "${summary_file}" "${job}" "${lane}" "export_error" "0" "${note}" "${seg_dir}" "${log_file}" \
        "${segment_status}" "${export_status}" "" "" "${export_note}"
      append_canonical_log "${dataset_root}" "${job}" "${group}" "FAIL job=${job} step=export_submit (could not parse job id)"
      append_canonical_log "${dataset_root}" "${job}" "${group}" "SUBMIT_STDERR ${export_submit_output}"
      rebuild_all_jobs_summary "${dataset_root}"
      announce_job_event "ERROR" "${dataset}" "${job}" "export-only submit failed (${export_submit_output})"
      return 1
    fi

    note="${note} export_lsf_job=${export_job_id}"
    upsert_status_row \
      "${summary_file}" "${job}" "${lane}" "running" "0" "${note}" "${seg_dir}" "${log_file}" \
      "${segment_status}" "running" "" "${export_job_id}" "${export_note}"
    append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} export_submit enqueue_complete export_lsf_job=${export_job_id} export_only=true"
    rebuild_all_jobs_summary "${dataset_root}"
    announce_job_event "SUBMITTED" "${dataset}" "${job}" "${note} export_only=true"
    return 0
  fi

  if [[ "${RUN_PRIMARY_JOBS}" != "1" ]]; then
    note="primary_disabled_missing_segment attempt=${attempt}"
    upsert_status_row \
      "${skipped_file}" "${job}" "${lane}" "skipped_filter" "0" "${note}" "${seg_dir}" "${log_file}" \
      "skipped_filter" "${export_status}" "" "" "primary_disabled"
    append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} status=skipped_filter reason=primary_disabled_missing_segment"
    rebuild_all_jobs_summary "${dataset_root}"
    announce_job_event "SKIPPED" "${dataset}" "${job}" "${note}"
    return 0
  fi

  primary_attempt_script="$(render_primary_attempt_script \
    "${dataset_root}" "${dataset}" "${input_dir}" "${scrna_ref}" "${tissue_type}" \
    "${job}" "${group}" "${mode}" "${use_3d}" "${expansion}" "${txk}" "${txdist}" \
    "${layers}" "${heads}" "${cellsmin}" "${minqv}" "${alignment}" "${align_weight}" \
    "${pred_scale}" "${fragment}" "${attempt}" "${queue_name}" "${gmem}" "${mem_gb}" "${wall_time}" "${baseline_dep_name}" "${paired_predict_dep_name}")"
  if job_exports_requested; then
    primary_dependency_name="segger_${dataset}_${job}_${RUN_LABEL}_a${attempt}"
    export_attempt_script="$(render_export_attempt_script \
      "${dataset_root}" "${dataset}" "${input_dir}" "${job}" "${attempt}" \
      "${export_queue_name}" "${export_mem_gb}" "${export_wall_time}" "${primary_dependency_name}")"
  fi

  note="$(status_note "${queue_name}" "${gmem}" "${mem_gb}" "${wall_time}" "${attempt}" "" "")"
  if [[ "${mode}" == "predict" && -n "${predict_dependency_list}" ]]; then
    note="$(status_note "${queue_name}" "${gmem}" "${mem_gb}" "${wall_time}" "${attempt}" "" "depends_on=${predict_dependency_list}")"
  fi
  if [[ -n "${fallback_note}" ]]; then
    note="${note} ${fallback_note}"
  fi
  if [[ "${force_rerun_fragment_job}" == "1" ]]; then
    note="${note} force_fragment_rerun=true"
  fi

  if [[ "${AUTO_SUBMIT}" != "1" || "${DRY_RUN}" == "1" ]]; then
    upsert_status_row \
      "${summary_file}" "${job}" "${lane}" "pending" "0" "planned ${note}" "${seg_dir}" "${log_file}" \
      "pending" "${export_status}" "" "" "${export_note}"
    append_canonical_log "${dataset_root}" "${job}" "${group}" "START job=${job} planned_only manifest_write_begin ${note}"
    if [[ -n "${export_attempt_script}" ]]; then
      append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} planned_only export_manifest_write_complete export_note=${export_note:-none}"
    fi
    append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} planned_only manifest_write_complete"
    rebuild_all_jobs_summary "${dataset_root}"
    announce_job_event "PLANNED" "${dataset}" "${job}" "${note}"
    return 0
  fi

  if [[ ! -x "${primary_attempt_script}" ]]; then
    segment_status="$(classify_primary_exit_status "${mode}" "1")"
    upsert_status_row \
      "${summary_file}" "${job}" "${lane}" "${segment_status}" "0" "attempt script missing" "${seg_dir}" "${log_file}" \
      "${segment_status}" "${export_status}" "" "" "${export_note}"
    append_canonical_log "${dataset_root}" "${job}" "${group}" "START job=${job} submit_failed"
    append_canonical_log "${dataset_root}" "${job}" "${group}" "FAIL job=${job} step=submit (attempt script missing)"
    rebuild_all_jobs_summary "${dataset_root}"
    announce_job_event "ERROR" "${dataset}" "${job}" "submit failed (attempt script missing)"
    return 1
  fi

  bsub_output="$(submit_attempt_script "${primary_attempt_script}" 2>&1 || true)"
  segment_job_id="$(printf '%s\n' "${bsub_output}" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -n1 || true)"
  if [[ -z "${segment_job_id}" ]]; then
    segment_status="$(classify_primary_exit_status "${mode}" "1")"
    upsert_status_row \
      "${summary_file}" "${job}" "${lane}" "${segment_status}" "0" "submit failed" "${seg_dir}" "${log_file}" \
      "${segment_status}" "${export_status}" "" "" "${export_note}"
    append_canonical_log "${dataset_root}" "${job}" "${group}" "START job=${job} submit_failed"
    append_canonical_log "${dataset_root}" "${job}" "${group}" "FAIL job=${job} step=submit (could not parse job id)"
    append_canonical_log "${dataset_root}" "${job}" "${group}" "SUBMIT_STDERR ${bsub_output}"
    rebuild_all_jobs_summary "${dataset_root}"
    announce_job_event "ERROR" "${dataset}" "${job}" "submit failed (${bsub_output})"
    return 1
  fi

  state="$(query_job_state "${segment_job_id}")"
  attempt_status="pending"
  segment_status="pending"
  case "${state}" in
    RUN|PROV)
      attempt_status="running"
      segment_status="running"
      ;;
    PEND|"")
      attempt_status="pending"
      segment_status="pending"
      ;;
  esac

  if [[ -n "${export_attempt_script}" ]]; then
    if [[ ! -x "${export_attempt_script}" ]]; then
      export_status="export_error"
      export_note="export_script_missing"
    else
      export_submit_output="$(submit_attempt_script "${export_attempt_script}" 2>&1 || true)"
      export_job_id="$(printf '%s\n' "${export_submit_output}" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -n1 || true)"
      if [[ -z "${export_job_id}" ]]; then
        export_status="export_error"
        export_note="export_submit_failed"
      fi
    fi
  fi

  export_suffix=""
  if [[ "${mode}" == "predict" && -n "${predict_dependency_list}" ]]; then
    export_suffix="depends_on=${predict_dependency_list}"
  fi
  if [[ -n "${export_job_id}" ]]; then
    if [[ -n "${export_suffix}" ]]; then
      export_suffix+=" "
    fi
    export_suffix+="export_lsf_job=${export_job_id}"
  fi
  note="$(status_note "${queue_name}" "${gmem}" "${mem_gb}" "${wall_time}" "${attempt}" "${segment_job_id}" "${export_suffix}")"

  append_canonical_log "${dataset_root}" "${job}" "${group}" "START job=${job} submit_only enqueue_begin ${note}"
  append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} submit_only enqueue_complete lsf_state=${state:-unknown}"
  if [[ -n "${export_job_id}" ]]; then
    append_canonical_log "${dataset_root}" "${job}" "${group}" "DONE job=${job} export_submit enqueue_complete export_lsf_job=${export_job_id}"
  fi
  upsert_status_row \
    "${summary_file}" "${job}" "${lane}" "${attempt_status}" "0" "${note}" "${seg_dir}" "${log_file}" \
    "${segment_status}" "${export_status}" "${segment_job_id}" "${export_job_id}" "${export_note}"
  rebuild_all_jobs_summary "${dataset_root}"
  announce_job_event "SUBMITTED" "${dataset}" "${job}" "${note} state=${state:-unknown}"
  return 0
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CLUSTER_CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/omics/groups/OE0606/internal/elihei/projects/segger_lsf_benchmark_fixed}"
WORK_TMP_ROOT="${WORK_TMP_ROOT:-${OUTPUT_ROOT}/_workflow/tmp}"
DATASET_KEYS="${DATASET_KEYS:-root_missing}"
DRY_RUN="$(normalize_bool "${DRY_RUN:-1}")"
AUTO_SUBMIT="$(normalize_bool "${AUTO_SUBMIT:-0}")"
RUN_PRIMARY_JOBS="$(normalize_bool "${RUN_PRIMARY_JOBS:-1}")"
RUN_EXPORT_JOBS="$(normalize_bool "${RUN_EXPORT_JOBS:-1}")"
RUN_EXPORTS="$(normalize_bool "${RUN_EXPORTS:-1}")"
RUN_ANNDATA_EXPORT="$(normalize_bool "${RUN_ANNDATA_EXPORT:-${RUN_EXPORTS}}")"
RUN_XENIUM_EXPORT="$(normalize_bool "${RUN_XENIUM_EXPORT:-${RUN_EXPORTS}}")"
SKIP_UNSUPPORTED_XENIUM_EXPORT="$(normalize_bool "${SKIP_UNSUPPORTED_XENIUM_EXPORT:-1}")"
RUN_VALIDATION_TABLE="$(normalize_bool "${RUN_VALIDATION_TABLE:-1}")"
RUN_STATUS_SNAPSHOT="$(normalize_bool "${RUN_STATUS_SNAPSHOT:-1}")"
ENABLE_ALIGNMENT_JOBS="$(normalize_bool "${ENABLE_ALIGNMENT_JOBS:-1}")"
RESUME_IF_EXISTS="$(normalize_bool "${RESUME_IF_EXISTS:-1}")"
FORCE_RERUN_FRAGMENT_JOBS="$(normalize_bool "${FORCE_RERUN_FRAGMENT_JOBS:-0}")"
ONLY_RETRY_OOM_JOBS="$(normalize_bool "${ONLY_RETRY_OOM_JOBS:-0}")"
RESET_DATASET_ROOT="$(normalize_bool "${RESET_DATASET_ROOT:-0}")"
AUTO_FETCH_SCRNA_REFS="$(normalize_bool "${AUTO_FETCH_SCRNA_REFS:-1}")"
FORCE_SCRNA_REFETCH="$(normalize_bool "${FORCE_SCRNA_REFETCH:-0}")"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
PEND_FALLBACK_MIN="${PEND_FALLBACK_MIN:-15}"
MAX_ACTIVE_STANDARD="${MAX_ACTIVE_STANDARD:-6}"
MAX_ACTIVE_FRAGMENT="${MAX_ACTIVE_FRAGMENT:-2}"
ROUTE_ELIGIBLE_TO_GPU="$(normalize_bool "${ROUTE_ELIGIBLE_TO_GPU:-1}")"
GPU_QUEUE_DEFAULT="${GPU_QUEUE_DEFAULT:-gpu}"
GPU_QUEUE_PRO="${GPU_QUEUE_PRO:-gpu-pro}"
EXPORT_QUEUE="${EXPORT_QUEUE:-long}"
ALIGNMENT_GPU_QUEUE="${ALIGNMENT_GPU_QUEUE:-}"
GPU_QUEUE_MAX_GMEM="${GPU_QUEUE_MAX_GMEM:-31}"
GPU_QUEUE_MAX_MEM_GB="${GPU_QUEUE_MAX_MEM_GB:-84}"
GPU_QUEUE_MAX_WALL_H="${GPU_QUEUE_MAX_WALL_H:-24}"
GPU_QUEUE_PRO_MAX_GMEM="${GPU_QUEUE_PRO_MAX_GMEM:-39}"
GPU_QUEUE_PRO_MAX_MEM_GB="${GPU_QUEUE_PRO_MAX_MEM_GB:-84}"
FORCE_GPU_PRO_DATASETS="${FORCE_GPU_PRO_DATASETS:-xenium_crc,xenium_nsclc,xenium_mouse_liver,xenium_breast}"
REQUIRE_SUCCESS_STATUS_FOR_RESUME="$(normalize_bool "${REQUIRE_SUCCESS_STATUS_FOR_RESUME:-1}")"
ALLOW_UNVERIFIED_PRIMARY_OUTPUTS="$(normalize_bool "${ALLOW_UNVERIFIED_PRIMARY_OUTPUTS:-0}")"
GPU_FALLBACK_ON_RETRY="$(normalize_bool "${GPU_FALLBACK_ON_RETRY:-1}")"
GPU_FALLBACK_QUEUE="${GPU_FALLBACK_QUEUE:-${GPU_QUEUE_PRO}}"
GPU_RETRY_GMEM_BUMP_GB="${GPU_RETRY_GMEM_BUMP_GB:-20}"
GPU_FALLBACK_MIN_GMEM="${GPU_FALLBACK_MIN_GMEM:-36}"
GPU_FALLBACK_FRAGMENT_GMEM="${GPU_FALLBACK_FRAGMENT_GMEM:-${GPU_QUEUE_PRO_MAX_GMEM}}"
GPU_FALLBACK_MIN_MEM_GB="${GPU_FALLBACK_MIN_MEM_GB:-84}"
GPU_FALLBACK_MIN_WALL_H="${GPU_FALLBACK_MIN_WALL_H:-12}"
GPU_USAGE_POLL_SEC="${GPU_USAGE_POLL_SEC:-15}"
ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT="${ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT:-0.0}"
if ! [[ "${GPU_RETRY_GMEM_BUMP_GB}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: GPU_RETRY_GMEM_BUMP_GB must be a non-negative integer (got: ${GPU_RETRY_GMEM_BUMP_GB})." >&2
  exit 1
fi
if ! [[ "${GPU_QUEUE_PRO_MAX_GMEM}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: GPU_QUEUE_PRO_MAX_GMEM must be a non-negative integer (got: ${GPU_QUEUE_PRO_MAX_GMEM})." >&2
  exit 1
fi
if ! [[ "${ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
  echo "ERROR: ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT must be a decimal in [0,1] (got: ${ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT})." >&2
  exit 1
fi
if ! awk -v v="${ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT}" 'BEGIN { exit !((v + 0.0) >= 0 && (v + 0.0) <= 1) }'; then
  echo "ERROR: ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT must be in [0,1] (got: ${ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT})." >&2
  exit 1
fi
if ! [[ "${GPU_QUEUE_PRO_MAX_MEM_GB}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: GPU_QUEUE_PRO_MAX_MEM_GB must be a non-negative integer (got: ${GPU_QUEUE_PRO_MAX_MEM_GB})." >&2
  exit 1
fi
if ! [[ "${GPU_QUEUE_MAX_WALL_H}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: GPU_QUEUE_MAX_WALL_H must be a non-negative integer (got: ${GPU_QUEUE_MAX_WALL_H})." >&2
  exit 1
fi
N_EPOCHS="${N_EPOCHS:-20}"
SEGGER_BIN="${SEGGER_BIN:-segger}"
CLI_EXPANSION_FLAG="${CLI_EXPANSION_FLAG:---prediction-scale-factor}"
CLI_IO_STYLE="${CLI_IO_STYLE:-positional}"  # "positional" (v2) or "named" (main: -i/-o)
CLUSTER_CODE_ROOT="${CLUSTER_CODE_ROOT:-${DEFAULT_CLUSTER_CODE_ROOT}}"
SEGMENT_WALL_TIME_DEFAULT="${SEGMENT_WALL_TIME_DEFAULT:-6:00}"
PREDICT_FRAGMENT_GMEM_GB="${PREDICT_FRAGMENT_GMEM_GB:-36}"
EXPORT_WALL_TIME_DEFAULT="${EXPORT_WALL_TIME_DEFAULT:-6:00}"
SEGMENT_PREDICTION_MODE="${SEGMENT_PREDICTION_MODE:-nucleus}"
SEGMENT_PREDICTION_MODE="$(printf '%s' "${SEGMENT_PREDICTION_MODE}" | tr '[:upper:]' '[:lower:]')"
case "${SEGMENT_PREDICTION_MODE}" in
  nucleus|cell)
    ;;
  *)
    echo "ERROR: SEGMENT_PREDICTION_MODE must be nucleus|cell (got: ${SEGMENT_PREDICTION_MODE})." >&2
    exit 1
    ;;
esac
if ! [[ "${PREDICT_FRAGMENT_GMEM_GB}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: PREDICT_FRAGMENT_GMEM_GB must be a non-negative integer (got: ${PREDICT_FRAGMENT_GMEM_GB})." >&2
  exit 1
fi
RUN_LABEL="${RUN_LABEL:-$(date '+%Y%m%d_%H%M%S')}"
RUN_LABEL="$(printf '%s' "${RUN_LABEL}" | tr -cd '[:alnum:]_-')"
if [[ -z "${RUN_LABEL}" ]]; then
  RUN_LABEL="$(date '+%Y%m%d_%H%M%S')"
fi
ALIGNMENT_REFERENCE_MODE="$(normalize_alignment_reference_mode "${ALIGNMENT_REFERENCE_MODE:-path}")"
ALIGNMENT_SCRNA_CELLTYPE_COLUMN="${ALIGNMENT_SCRNA_CELLTYPE_COLUMN:-cell_type}"
JOB_SPECS_FILE="${JOB_SPECS_FILE:-}"
LSF_SUBMIT_HOST="${LSF_SUBMIT_HOST:-local}"
case "${LSF_SUBMIT_HOST}" in
  local|LOCAL|none|NONE|-)
    LSF_SUBMIT_HOST=""
    ;;
esac
LSF_EXEC_SHELL="${LSF_EXEC_SHELL:-/bin/bash}"
SCRNA_REF_ROOT="${SCRNA_REF_ROOT:-/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_raw/scrnaseq}"
SCRNA_CACHE_DIR="${SCRNA_CACHE_DIR:-${SCRNA_REF_ROOT}}"
if [[ "${SCRNA_CACHE_DIR}" != */.segger_references ]]; then
  SCRNA_CACHE_DIR="${SCRNA_CACHE_DIR}/.segger_references"
fi
MICROMAMBA_BIN="${MICROMAMBA_BIN:-${HOME}/.local/bin/micromamba}"
MICROMAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT_PREFIX:-/dkfz/cluster/gpu/data/OE0606/elihei/micromamba}"
SEGGER_ENV_PATH="${SEGGER_ENV_PATH:-/omics/groups/OE0606/internal/elihei/projects/conda_envs/seggerv2}"
PRESERVE_PYTHONPATH="$(normalize_bool "${PRESERVE_PYTHONPATH:-0}")"
DEFAULT_ACTIVATE_CMD="$(cat <<EOF
if [[ $(printf '%q' "${PRESERVE_PYTHONPATH}") != "1" ]]; then
    unset PYTHONPATH || true
fi
export MAMBA_NO_ENV_PROMPT=1
export CONDA_PROMPT_MODIFIER=""
export PYTHONNOUSERSITE=1
export MAMBA_EXE=$(printf '%q' "${MICROMAMBA_BIN}")
export MAMBA_ROOT_PREFIX=$(printf '%q' "${MICROMAMBA_ROOT_PREFIX}")
__mamba_setup="\$("\$MAMBA_EXE" shell hook --shell bash --root-prefix "\$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ \$? -eq 0 ]; then
    eval "\$__mamba_setup"
else
    alias micromamba="\$MAMBA_EXE"
fi
unset __mamba_setup
micromamba activate $(printf '%q' "${SEGGER_ENV_PATH}")
EOF
)"
MAMBA_ACTIVATE_CMD="${MAMBA_ACTIVATE_CMD:-${DEFAULT_ACTIVATE_CMD}}"

mkdir -p "${OUTPUT_ROOT}/datasets" "${OUTPUT_ROOT}/summaries" "${WORK_TMP_ROOT}"
export TMPDIR="${WORK_TMP_ROOT}"
export TMP="${WORK_TMP_ROOT}"
export TEMP="${WORK_TMP_ROOT}"

printf "[%s] OUTPUT_ROOT=%s\n" "$(timestamp)" "${OUTPUT_ROOT}"
printf "[%s] WORK_TMP_ROOT=%s\n" "$(timestamp)" "${WORK_TMP_ROOT}"
printf "[%s] DATASET_KEYS=%s\n" "$(timestamp)" "${DATASET_KEYS}"
printf "[%s] DRY_RUN=%s AUTO_SUBMIT=%s RUN_PRIMARY_JOBS=%s RUN_EXPORT_JOBS=%s RUN_ANNDATA_EXPORT=%s RUN_XENIUM_EXPORT=%s SKIP_UNSUPPORTED_XENIUM_EXPORT=%s RUN_VALIDATION_TABLE=%s RUN_STATUS_SNAPSHOT=%s\n" \
  "$(timestamp)" "${DRY_RUN}" "${AUTO_SUBMIT}" "${RUN_PRIMARY_JOBS}" "${RUN_EXPORT_JOBS}" "${RUN_ANNDATA_EXPORT}" "${RUN_XENIUM_EXPORT}" "${SKIP_UNSUPPORTED_XENIUM_EXPORT}" "${RUN_VALIDATION_TABLE}" "${RUN_STATUS_SNAPSHOT}"
printf "[%s] ENABLE_ALIGNMENT_JOBS=%s\n" "$(timestamp)" "${ENABLE_ALIGNMENT_JOBS}"
printf "[%s] RESUME_IF_EXISTS=%s RESET_DATASET_ROOT=%s RUN_LABEL=%s\n" \
  "$(timestamp)" "${RESUME_IF_EXISTS}" "${RESET_DATASET_ROOT}" "${RUN_LABEL}"
printf "[%s] FORCE_RERUN_FRAGMENT_JOBS=%s\n" \
  "$(timestamp)" "${FORCE_RERUN_FRAGMENT_JOBS}"
printf "[%s] ONLY_RETRY_OOM_JOBS=%s\n" \
  "$(timestamp)" "${ONLY_RETRY_OOM_JOBS}"
printf "[%s] ALIGNMENT_REFERENCE_MODE=%s\n" "$(timestamp)" "${ALIGNMENT_REFERENCE_MODE}"
printf "[%s] ALIGNMENT_SCRNA_CELLTYPE_COLUMN=%s\n" "$(timestamp)" "${ALIGNMENT_SCRNA_CELLTYPE_COLUMN}"
printf "[%s] JOB_SPECS_FILE=%s\n" "$(timestamp)" "${JOB_SPECS_FILE:-<default_matrix>}"
if is_remote_submit_enabled; then
  printf "[%s] LSF submit transport=ssh host=%s\n" "$(timestamp)" "${LSF_SUBMIT_HOST}"
else
  printf "[%s] LSF submit transport=local\n" "$(timestamp)"
fi
printf "[%s] LSF_EXEC_SHELL=%s\n" "$(timestamp)" "${LSF_EXEC_SHELL}"
printf "[%s] CLUSTER_CODE_ROOT=%s\n" "$(timestamp)" "${CLUSTER_CODE_ROOT}"
printf "[%s] SEGGER_BIN=%s\n" "$(timestamp)" "${SEGGER_BIN}"
printf "[%s] MICROMAMBA_BIN=%s\n" "$(timestamp)" "${MICROMAMBA_BIN}"
printf "[%s] MICROMAMBA_ROOT_PREFIX=%s\n" "$(timestamp)" "${MICROMAMBA_ROOT_PREFIX}"
printf "[%s] SEGGER_ENV_PATH=%s\n" "$(timestamp)" "${SEGGER_ENV_PATH}"
printf "[%s] SCRNA_REF_ROOT=%s\n" "$(timestamp)" "${SCRNA_REF_ROOT}"
printf "[%s] SCRNA_CACHE_DIR=%s\n" "$(timestamp)" "${SCRNA_CACHE_DIR}"
printf "[%s] AUTO_FETCH_SCRNA_REFS=%s FORCE_SCRNA_REFETCH=%s\n" \
  "$(timestamp)" "${AUTO_FETCH_SCRNA_REFS}" "${FORCE_SCRNA_REFETCH}"
printf "[%s] SEGMENT_WALL_TIME_DEFAULT=%s EXPORT_WALL_TIME_DEFAULT=%s PREDICT_FRAGMENT_GMEM_GB=%s PRESERVE_PYTHONPATH=%s\n" \
  "$(timestamp)" "${SEGMENT_WALL_TIME_DEFAULT}" "${EXPORT_WALL_TIME_DEFAULT}" "${PREDICT_FRAGMENT_GMEM_GB}" "${PRESERVE_PYTHONPATH}"
printf "[%s] SEGMENT_PREDICTION_MODE=%s\n" \
  "$(timestamp)" "${SEGMENT_PREDICTION_MODE}"
printf "[%s] ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT=%s\n" \
  "$(timestamp)" "${ANNDATA_EXPORT_MIN_SIMILARITY_SHIFT}"
printf "[%s] MAX_ACTIVE_STANDARD=%s MAX_ACTIVE_FRAGMENT=%s (currently informational)\n" \
  "$(timestamp)" "${MAX_ACTIVE_STANDARD}" "${MAX_ACTIVE_FRAGMENT}"
printf "[%s] QUEUE_ROUTING route_eligible=%s gpu_default=%s gpu_pro=%s export=%s align_override=%s max_gmem=%sG pro_max_gmem=%sG max_mem=%sG pro_max_mem=%sG max_wall=%sh\n" \
  "$(timestamp)" "${ROUTE_ELIGIBLE_TO_GPU}" "${GPU_QUEUE_DEFAULT}" "${GPU_QUEUE_PRO}" "${EXPORT_QUEUE}" "${ALIGNMENT_GPU_QUEUE:-<none>}" "${GPU_QUEUE_MAX_GMEM}" "${GPU_QUEUE_PRO_MAX_GMEM}" "${GPU_QUEUE_MAX_MEM_GB}" "${GPU_QUEUE_PRO_MAX_MEM_GB}" "${GPU_QUEUE_MAX_WALL_H}"
printf "[%s] FORCE_GPU_PRO_DATASETS=%s\n" \
  "$(timestamp)" "${FORCE_GPU_PRO_DATASETS}"
printf "[%s] RESUME_STATUS_CHECK require_success=%s allow_unverified=%s\n" \
  "$(timestamp)" "${REQUIRE_SUCCESS_STATUS_FOR_RESUME}" "${ALLOW_UNVERIFIED_PRIMARY_OUTPUTS}"
printf "[%s] RETRY_FALLBACK enabled=%s queue=%s gmem_bump=+%sG min_gmem=%sG fragment_min_gmem=%sG min_mem=%sG min_wall=%sh\n" \
  "$(timestamp)" "${GPU_FALLBACK_ON_RETRY}" "${GPU_FALLBACK_QUEUE}" "${GPU_RETRY_GMEM_BUMP_GB}" "${GPU_FALLBACK_MIN_GMEM}" "${GPU_FALLBACK_FRAGMENT_GMEM}" "${GPU_FALLBACK_MIN_MEM_GB}" "${GPU_FALLBACK_MIN_WALL_H}"
printf "[%s] GPU_USAGE_POLL_SEC=%s\n" \
  "$(timestamp)" "${GPU_USAGE_POLL_SEC}"

require_submit_transport
require_segger_binary

CLI_SEGMENT_SUPPORTS_N_EPOCHS="1"
CLI_SEGMENT_SUPPORTS_PREDICTION_MODE="1"
CLI_SEGMENT_SUPPORTS_EXPANSION_FLAG="1"
CLI_SEGMENT_SUPPORTS_TRANSCRIPTS_MAX_K="1"
CLI_SEGMENT_SUPPORTS_TRANSCRIPTS_MAX_DIST="1"
CLI_SEGMENT_SUPPORTS_TILING_MARGIN_TRAINING="1"
CLI_SEGMENT_SUPPORTS_TILING_MARGIN_PREDICTION="1"
CLI_SEGMENT_SUPPORTS_N_MID_LAYERS="1"
CLI_SEGMENT_SUPPORTS_N_HEADS="1"
CLI_SEGMENT_SUPPORTS_CELLS_MIN_COUNTS="1"
CLI_SEGMENT_SUPPORTS_MIN_QV="1"
CLI_SEGMENT_SUPPORTS_USE_3D="1"
CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS="1"
CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS_WEIGHT_START="1"
CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS_WEIGHT_END="1"
CLI_SEGMENT_SUPPORTS_SCRNA_REFERENCE_PATH="1"
CLI_SEGMENT_SUPPORTS_SCRNA_CELLTYPE_COLUMN="1"

CLI_PREDICT_SUPPORTS_EXPANSION_FLAG="1"
CLI_PREDICT_SUPPORTS_USE_3D="1"
CLI_PREDICT_SUPPORTS_FRAGMENT_MODE="1"
HAS_PREDICT_JOBS="1"

if [[ "${CLI_IO_STYLE}" == "named" ]]; then
  if ! job_specs_contains_mode "predict"; then
    HAS_PREDICT_JOBS="0"
  fi
  CLI_SEGMENT_SUPPORTS_N_EPOCHS="$(detect_cli_flag_support segment "--n-epochs")"
  CLI_SEGMENT_SUPPORTS_PREDICTION_MODE="$(detect_cli_flag_support segment "--prediction-mode")"
  CLI_SEGMENT_SUPPORTS_EXPANSION_FLAG="$(detect_cli_flag_support segment "${CLI_EXPANSION_FLAG}")"
  CLI_SEGMENT_SUPPORTS_TRANSCRIPTS_MAX_K="$(detect_cli_flag_support segment "--transcripts-max-k")"
  CLI_SEGMENT_SUPPORTS_TRANSCRIPTS_MAX_DIST="$(detect_cli_flag_support segment "--transcripts-max-dist")"
  CLI_SEGMENT_SUPPORTS_TILING_MARGIN_TRAINING="$(detect_cli_flag_support segment "--tiling-margin-training")"
  CLI_SEGMENT_SUPPORTS_TILING_MARGIN_PREDICTION="$(detect_cli_flag_support segment "--tiling-margin-prediction")"
  CLI_SEGMENT_SUPPORTS_N_MID_LAYERS="$(detect_cli_flag_support segment "--n-mid-layers")"
  CLI_SEGMENT_SUPPORTS_N_HEADS="$(detect_cli_flag_support segment "--n-heads")"
  CLI_SEGMENT_SUPPORTS_CELLS_MIN_COUNTS="$(detect_cli_flag_support segment "--cells-min-counts")"
  CLI_SEGMENT_SUPPORTS_MIN_QV="$(detect_cli_flag_support segment "--min-qv")"
  CLI_SEGMENT_SUPPORTS_USE_3D="$(detect_cli_flag_support segment "--use-3d")"
  CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS="$(detect_cli_flag_support segment "--alignment-loss")"
  CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS_WEIGHT_START="$(detect_cli_flag_support segment "--alignment-loss-weight-start")"
  CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS_WEIGHT_END="$(detect_cli_flag_support segment "--alignment-loss-weight-end")"
  CLI_SEGMENT_SUPPORTS_SCRNA_REFERENCE_PATH="$(detect_cli_flag_support segment "--scrna-reference-path")"
  CLI_SEGMENT_SUPPORTS_SCRNA_CELLTYPE_COLUMN="$(detect_cli_flag_support segment "--scrna-celltype-column")"

  if [[ "${HAS_PREDICT_JOBS}" == "1" ]]; then
    CLI_PREDICT_SUPPORTS_EXPANSION_FLAG="$(detect_cli_flag_support predict "${CLI_EXPANSION_FLAG}")"
    CLI_PREDICT_SUPPORTS_USE_3D="$(detect_cli_flag_support predict "--use-3d")"
    CLI_PREDICT_SUPPORTS_FRAGMENT_MODE="$(detect_cli_flag_support predict "--fragment-mode")"
  else
    CLI_PREDICT_SUPPORTS_EXPANSION_FLAG="0"
    CLI_PREDICT_SUPPORTS_USE_3D="0"
    CLI_PREDICT_SUPPORTS_FRAGMENT_MODE="0"
  fi
fi
printf "[%s] CLI_FLAG_SUPPORT segment: n_epochs=%s prediction_mode=%s expansion=%s tx_k=%s tx_dist=%s tile_train=%s tile_pred=%s mid_layers=%s heads=%s cells_min=%s min_qv=%s use_3d=%s align=%s align_w_start=%s align_w_end=%s\n" \
  "$(timestamp)" \
  "${CLI_SEGMENT_SUPPORTS_N_EPOCHS}" \
  "${CLI_SEGMENT_SUPPORTS_PREDICTION_MODE}" \
  "${CLI_SEGMENT_SUPPORTS_EXPANSION_FLAG}" \
  "${CLI_SEGMENT_SUPPORTS_TRANSCRIPTS_MAX_K}" \
  "${CLI_SEGMENT_SUPPORTS_TRANSCRIPTS_MAX_DIST}" \
  "${CLI_SEGMENT_SUPPORTS_TILING_MARGIN_TRAINING}" \
  "${CLI_SEGMENT_SUPPORTS_TILING_MARGIN_PREDICTION}" \
  "${CLI_SEGMENT_SUPPORTS_N_MID_LAYERS}" \
  "${CLI_SEGMENT_SUPPORTS_N_HEADS}" \
  "${CLI_SEGMENT_SUPPORTS_CELLS_MIN_COUNTS}" \
  "${CLI_SEGMENT_SUPPORTS_MIN_QV}" \
  "${CLI_SEGMENT_SUPPORTS_USE_3D}" \
  "${CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS}" \
  "${CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS_WEIGHT_START}" \
  "${CLI_SEGMENT_SUPPORTS_ALIGNMENT_LOSS_WEIGHT_END}"
printf "[%s] CLI_FLAG_SUPPORT segment refs: scrna_reference_path=%s scrna_celltype_column=%s\n" \
  "$(timestamp)" \
  "${CLI_SEGMENT_SUPPORTS_SCRNA_REFERENCE_PATH}" \
  "${CLI_SEGMENT_SUPPORTS_SCRNA_CELLTYPE_COLUMN}"
printf "[%s] CLI_FLAG_SUPPORT predict: expansion=%s use_3d=%s fragment_mode=%s\n" \
  "$(timestamp)" \
  "${CLI_PREDICT_SUPPORTS_EXPANSION_FLAG}" \
  "${CLI_PREDICT_SUPPORTS_USE_3D}" \
  "${CLI_PREDICT_SUPPORTS_FRAGMENT_MODE}"
if [[ "${HAS_PREDICT_JOBS}" != "1" ]]; then
  printf "[%s] CLI_FLAG_SUPPORT predict probe skipped (no predict jobs in JOB_SPECS_FILE)\n" "$(timestamp)"
fi

REQUESTED_DATASETS="$(resolve_requested_datasets)"
if [[ -z "${REQUESTED_DATASETS//[[:space:]]/}" ]]; then
  printf "[%s] No datasets selected for execution (DATASET_KEYS=%s OUTPUT_ROOT=%s).\n" \
    "$(timestamp)" "${DATASET_KEYS}" "${OUTPUT_ROOT}"
  exit 0
fi
printf "[%s] RESOLVED_DATASETS=%s\n" \
  "$(timestamp)" "$(printf '%s' "${REQUESTED_DATASETS}" | tr '\n' ',' | sed 's/,$//')"

while IFS= read -r dataset; do
  [[ -z "${dataset}" ]] && continue

  dataset_root="${OUTPUT_ROOT}/datasets/${dataset}"
  input_dir="$(dataset_input_dir "${dataset}")"
  tissue_type="$(dataset_tissue_type "${dataset}")"
  scrna_ref="$(resolve_scrna_reference "${dataset}")"

  if [[ "${ENABLE_ALIGNMENT_JOBS}" == "1" ]]; then
    if [[ "${FORCE_SCRNA_REFETCH}" == "1" ]]; then
      refresh_scrna_reference_from_atlas "${dataset}" "${tissue_type}" "forced_refresh"
      scrna_ref="$(resolve_scrna_reference "${dataset}")"
    elif [[ "${AUTO_FETCH_SCRNA_REFS}" == "1" ]] && [[ -n "${tissue_type}" ]] && [[ ( -z "${scrna_ref}" ) || ( ! -f "${scrna_ref}" ) ]]; then
      refresh_scrna_reference_from_atlas "${dataset}" "${tissue_type}" "missing_reference" || true
      scrna_ref="$(resolve_scrna_reference "${dataset}")"
    fi
  fi

  printf "[%s] Preparing dataset=%s input_dir=%s tissue_type=%s scrna_ref=%s\n" \
    "$(timestamp)" "${dataset}" "${input_dir}" "${tissue_type:-<none>}" "${scrna_ref:-<none>}"
  ensure_alignment_reference_ready "${dataset}" "${scrna_ref}" "${tissue_type}"

  if [[ "${RESET_DATASET_ROOT}" == "1" ]]; then
    printf "[%s] RESET_DATASET_ROOT=1 removing %s\n" "$(timestamp)" "${dataset_root}"
    rm -rf "${dataset_root}"
  fi

  init_dataset_root "${dataset_root}"
  write_dataset_context "${dataset_root}" "${dataset}" "${input_dir}" "${scrna_ref}" "${tissue_type}"
  write_dataset_plan "${dataset_root}" "${dataset}"

  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    IFS='|' read -r \
      job group mode use_3d expansion txk txdist layers heads cellsmin minqv alignment align_weight pred_scale fragment \
      <<< "${line}"
    run_or_plan_job \
      "${dataset_root}" "${dataset}" "${input_dir}" "${scrna_ref}" "${tissue_type}" \
      "${job}" "${group}" "${mode}" "${use_3d}" "${expansion}" "${txk}" "${txdist}" \
      "${layers}" "${heads}" "${cellsmin}" "${minqv}" "${alignment}" "${align_weight}" \
      "${pred_scale}" "${fragment}" || true
  done <<EOF
$(job_specs)
EOF

  rebuild_all_jobs_summary "${dataset_root}"

  if [[ "${RUN_VALIDATION_TABLE}" == "1" ]]; then
    if [[ "${AUTO_SUBMIT}" != "1" && "${DRY_RUN}" != "1" ]]; then
      run_validation_for_root "${dataset_root}" "${input_dir}" "${scrna_ref}" "${tissue_type}" || true
    fi
  fi

  if [[ "${RUN_STATUS_SNAPSHOT}" == "1" ]]; then
    run_dashboard_for_root "${dataset_root}" || true
  fi
done <<EOF
${REQUESTED_DATASETS}
EOF

if [[ -f "${SCRIPT_DIR}/benchmark_status_dashboard_lsf_multi.sh" && "${RUN_STATUS_SNAPSHOT}" == "1" ]]; then
  bash "${SCRIPT_DIR}/benchmark_status_dashboard_lsf_multi.sh" --root "${OUTPUT_ROOT}" >/dev/null || true
fi
if [[ -f "${SCRIPT_DIR}/build_benchmark_validation_table_lsf_multi.sh" && "${RUN_VALIDATION_TABLE}" == "1" && "${AUTO_SUBMIT}" != "1" && "${DRY_RUN}" != "1" ]]; then
  bash "${SCRIPT_DIR}/build_benchmark_validation_table_lsf_multi.sh" --root "${OUTPUT_ROOT}" >/dev/null || true
fi

printf "[%s] Completed LSF benchmark orchestration.\n" "$(timestamp)"
