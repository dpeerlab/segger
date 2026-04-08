#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Benchmark status snapshot + terminal dashboard.

Usage:
  bash scripts/benchmark_status_dashboard.sh [options]

Options:
  --root <dir>       Benchmark root directory
                     (default: ./results/mossi_main_big_benchmark_nightly)
  --out-tsv <file>   Snapshot TSV output path
                     (default: <root>/summaries/status_snapshot.tsv)
  --watch [sec]      Refresh dashboard every N seconds (default: 20)
  --minimal          Minimal terminal output with compact jobs overview
  --no-color         Disable ANSI colors
  -h, --help         Show this help
EOF
}

ROOT="./results/mossi_main_big_benchmark_nightly"
OUT_TSV=""
OUT_TSV_SET=0
WATCH_SEC=0
MINIMAL_MODE=0
NO_COLOR=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --root requires a value." >&2
        exit 1
      fi
      ROOT="$2"
      shift 2
      ;;
    --out-tsv)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --out-tsv requires a value." >&2
        exit 1
      fi
      OUT_TSV="$2"
      OUT_TSV_SET=1
      shift 2
      ;;
    --watch)
      if [[ $# -ge 2 ]] && [[ ! "${2-}" =~ ^- ]]; then
        WATCH_SEC="$2"
        shift 2
      else
        WATCH_SEC=20
        shift
      fi
      ;;
    --minimal)
      MINIMAL_MODE=1
      shift
      ;;
    --no-color)
      NO_COLOR=1
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

if [[ -z "${OUT_TSV}" ]]; then
  OUT_TSV="${ROOT}/summaries/status_snapshot.tsv"
fi

if ! [[ "${WATCH_SEC}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --watch must be a non-negative integer." >&2
  exit 1
fi

PLAN_FILE="${ROOT}/job_plan.tsv"
SUMMARY_DIR="${ROOT}/summaries"
LOGS_DIR="${ROOT}/logs"
RUNS_DIR="${ROOT}/runs"
EXPORTS_DIR="${ROOT}/exports"
VALIDATION_TSV="${SUMMARY_DIR}/validation_metrics_.tsv"
if [[ ! -f "${VALIDATION_TSV}" ]] && [[ -f "${SUMMARY_DIR}/validation_metrics.tsv" ]]; then
  VALIDATION_TSV="${SUMMARY_DIR}/validation_metrics.tsv"
fi
if [[ ! -f "${VALIDATION_TSV}" ]] && [[ -f "${ROOT}/validation_metrics.tsv" ]]; then
  VALIDATION_TSV="${ROOT}/validation_metrics.tsv"
fi
if [[ ! -f "${VALIDATION_TSV}" ]] && [[ -f "./results/validation_metrics.tsv" ]]; then
  VALIDATION_TSV="./results/validation_metrics.tsv"
fi
LSF_SUBMIT_HOST="${LSF_SUBMIT_HOST:-local}"
case "${LSF_SUBMIT_HOST}" in
  local|LOCAL|none|NONE|-)
    LSF_SUBMIT_HOST=""
    ;;
esac

if [[ ! -f "${PLAN_FILE}" ]]; then
  if [[ -d "${ROOT}/datasets" ]] && [[ -f "${SCRIPT_DIR}/benchmark_status_dashboard_lsf_multi.sh" ]]; then
    cmd=(bash "${SCRIPT_DIR}/benchmark_status_dashboard_lsf_multi.sh" --root "${ROOT}")
    if [[ "${OUT_TSV_SET}" == "1" ]]; then
      cmd+=(--out-tsv "${OUT_TSV}")
    fi
    if [[ "${WATCH_SEC}" -gt 0 ]]; then
      cmd+=(--watch "${WATCH_SEC}")
    fi
    if [[ "${MINIMAL_MODE}" == "1" ]]; then
      cmd+=(--minimal)
    fi
    if [[ "${NO_COLOR}" == "1" ]]; then
      cmd+=(--no-color)
    fi
    exec "${cmd[@]}"
  fi
  echo "ERROR: Missing plan file: ${PLAN_FILE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_TSV}")"

if [[ "${NO_COLOR}" == "0" ]] && [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_BOLD_OFF=$'\033[22m'
  C_GREEN=$'\033[32m'
  C_RED=$'\033[31m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
  C_CYAN=$'\033[36m'
else
  C_RESET=""
  C_BOLD=""
  C_BOLD_OFF=""
  C_GREEN=""
  C_RED=""
  C_YELLOW=""
  C_BLUE=""
  C_CYAN=""
fi

collect_status_map() {
  local out_file="$1"
  local running_jobs_file="${2:-}"
  local -a status_files=()
  local f
  local include_recovery=1
  local newest_gpu_mtime=0
  local recovery_mtime=0

  if [[ -n "${running_jobs_file}" ]] && [[ -s "${running_jobs_file}" ]]; then
    include_recovery=0
  fi

  for f in "${SUMMARY_DIR}"/gpu*.tsv; do
    [[ -f "${f}" ]] || continue
    status_files+=("${f}")
    local mtime=0
    mtime="$(date -r "${f}" +%s 2>/dev/null || stat -f %m "${f}" 2>/dev/null || stat -c %Y "${f}" 2>/dev/null || echo 0)"
    if [[ "${mtime}" -gt "${newest_gpu_mtime}" ]]; then
      newest_gpu_mtime="${mtime}"
    fi
  done
  if [[ -f "${SUMMARY_DIR}/skipped_existing.tsv" ]]; then
    status_files+=("${SUMMARY_DIR}/skipped_existing.tsv")
    local mtime=0
    mtime="$(date -r "${SUMMARY_DIR}/skipped_existing.tsv" +%s 2>/dev/null || stat -f %m "${SUMMARY_DIR}/skipped_existing.tsv" 2>/dev/null || stat -c %Y "${SUMMARY_DIR}/skipped_existing.tsv" 2>/dev/null || echo 0)"
    if [[ "${mtime}" -gt "${newest_gpu_mtime}" ]]; then
      newest_gpu_mtime="${mtime}"
    fi
  fi
  if [[ "${include_recovery}" == "1" ]] && [[ -f "${SUMMARY_DIR}/recovery.tsv" ]]; then
    recovery_mtime="$(date -r "${SUMMARY_DIR}/recovery.tsv" +%s 2>/dev/null || stat -f %m "${SUMMARY_DIR}/recovery.tsv" 2>/dev/null || stat -c %Y "${SUMMARY_DIR}/recovery.tsv" 2>/dev/null || echo 0)"
    # Ignore stale recovery.tsv while a fresh run is underway.
    if [[ "${recovery_mtime}" -ge "${newest_gpu_mtime}" ]]; then
      status_files+=("${SUMMARY_DIR}/recovery.tsv")
    fi
  fi

  if [[ "${#status_files[@]}" -eq 0 ]]; then
    : > "${out_file}"
    return 0
  fi

  awk -F'\t' '
    FNR == 1 {
      delete idx
      for (i = 1; i <= NF; i++) {
        idx[$i] = i
      }
      next
    }
    {
      job = (("job" in idx) && idx["job"] > 0) ? $(idx["job"]) : $1
      gpu = (("gpu" in idx) && idx["gpu"] > 0) ? $(idx["gpu"]) : ((NF >= 2) ? $2 : "")
      status = (("status" in idx) && idx["status"] > 0) ? $(idx["status"]) : ((NF >= 3) ? $3 : "")
      elapsed = (("elapsed_s" in idx) && idx["elapsed_s"] > 0) ? $(idx["elapsed_s"]) : ((NF >= 4) ? $4 : "")
      note = (("note" in idx) && idx["note"] > 0) ? $(idx["note"]) : ""
      seg = (("seg_dir" in idx) && idx["seg_dir"] > 0) ? $(idx["seg_dir"]) : ""
      log_path = (("log_file" in idx) && idx["log_file"] > 0) ? $(idx["log_file"]) : ""
      segment_status = (("segment_status" in idx) && idx["segment_status"] > 0) ? $(idx["segment_status"]) : ""
      export_status = (("export_status" in idx) && idx["export_status"] > 0) ? $(idx["export_status"]) : ""
      segment_job_id = (("segment_job_id" in idx) && idx["segment_job_id"] > 0) ? $(idx["segment_job_id"]) : ""
      export_job_id = (("export_job_id" in idx) && idx["export_job_id"] > 0) ? $(idx["export_job_id"]) : ""
      export_note = (("export_note" in idx) && idx["export_note"] > 0) ? $(idx["export_note"]) : ""

      if (note == "") {
        note = "-"
      }
      if (seg == "") {
        seg = "-"
      }
      if (log_path == "") {
        log_path = "-"
      }
      if (segment_status == "") {
        segment_status = "-"
      }
      if (export_status == "") {
        export_status = "-"
      }
      if (segment_job_id == "") {
        segment_job_id = "-"
      }
      if (export_job_id == "") {
        export_job_id = "-"
      }
      if (export_note == "") {
        export_note = "-"
      }
      gpu_map[job] = gpu
      status_map[job] = status
      elapsed_map[job] = elapsed
      note_map[job] = note
      seg_map[job] = seg
      log_map[job] = log_path
      segment_status_map[job] = segment_status
      export_status_map[job] = export_status
      segment_job_id_map[job] = segment_job_id
      export_job_id_map[job] = export_job_id
      export_note_map[job] = export_note
    }
    END {
      for (job in status_map) {
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
          job,
          gpu_map[job],
          status_map[job],
          elapsed_map[job],
          note_map[job],
          seg_map[job],
          log_map[job],
          segment_status_map[job],
          export_status_map[job],
          segment_job_id_map[job],
          export_job_id_map[job],
          export_note_map[job]
      }
    }
  ' "${status_files[@]}" > "${out_file}"
}

collect_running_jobs() {
  local out_file="$1"
  if ! command -v pgrep >/dev/null 2>&1; then
    : > "${out_file}"
    return 0
  fi

  pgrep -af 'segger segment|segger predict|segger export' 2>/dev/null \
    | awk '
      {
        for (i = 1; i <= NF; i++) {
          if ($i == "-o" && (i + 1) <= NF) {
            out = $(i + 1)
            gsub(/\/+$/, "", out)
            n = split(out, a, "/")
            if (n > 0) {
              job = a[n]
              if ((job == "anndata" || job == "xenium_explorer") && n > 1) {
                job = a[n - 1]
              }
              print job
            }
          }
        }
      }
    ' \
    | sed '/^$/d' \
    | sort -u > "${out_file}" || : > "${out_file}"

  # Fallback: infer active job from logs when latest START has no terminal marker.
  local log_file
  for log_file in "${LOGS_DIR}"/*.gpu*.log; do
    [[ -f "${log_file}" ]] || continue
    awk -F'job=' '
      /\] START job=/ {
        split($2, a, /[[:space:]]+/)
        job = a[1]
        started = 1
        done = 0
      }
      /\] DONE job=/ || /\] FAIL job=/ {
        done = 1
      }
      END {
        if (started == 1 && done == 0 && job != "") {
          print job
        }
      }
    ' "${log_file}" >> "${out_file}"
  done
  if [[ -s "${out_file}" ]]; then
    sort -u -o "${out_file}" "${out_file}"
  fi
}

is_remote_lsf_enabled() {
  [[ -n "${LSF_SUBMIT_HOST:-}" ]]
}

collect_lsf_state_map() {
  local status_map="$1"
  local out_file="$2"
  local job_ids_file states_file raw_file id_list

  : > "${out_file}"
  job_ids_file="$(mktemp)"
  states_file="$(mktemp)"
  raw_file="$(mktemp)"

  awk -F'\t' '
    {
      segment_job_id = (NF >= 10 && $10 != "-") ? $10 : ""
      export_job_id = (NF >= 11 && $11 != "-") ? $11 : ""
      export_status = (NF >= 9 && $9 != "-") ? tolower($9) : ""
      note = (NF >= 5) ? $5 : ""
      selected_job_id = segment_job_id

      if (export_job_id != "" && (export_status == "planned" || export_status == "pending" || export_status == "running")) {
        selected_job_id = export_job_id
      }
      if (selected_job_id == "" && match(note, /lsf_job=[0-9]+/)) {
        selected_job_id = substr(note, RSTART + 8, RLENGTH - 8)
      }
      if (selected_job_id != "") {
        print $1 "\t" selected_job_id
      }
    }
  ' "${status_map}" > "${job_ids_file}"

  if [[ ! -s "${job_ids_file}" ]]; then
    rm -f "${job_ids_file}" "${states_file}" "${raw_file}"
    return 0
  fi

  id_list="$(awk -F'\t' '!seen[$2]++ { printf("%s%s", sep, $2); sep=" " }' "${job_ids_file}")"
  if [[ -z "${id_list}" ]]; then
    rm -f "${job_ids_file}" "${states_file}" "${raw_file}"
    return 0
  fi

  if is_remote_lsf_enabled; then
    if command -v ssh >/dev/null 2>&1; then
      ssh -o BatchMode=yes "${LSF_SUBMIT_HOST}" "bjobs -a -noheader -o 'jobid stat' ${id_list}" > "${raw_file}" 2>/dev/null || true
    fi
  elif command -v bjobs >/dev/null 2>&1; then
    bjobs -a -noheader -o "jobid stat" ${id_list} > "${raw_file}" 2>/dev/null || true
  fi

  awk '
    NF >= 2 {
      print $1 "\t" $2
    }
  ' "${raw_file}" > "${states_file}"

  awk -F'\t' '
    NR == FNR {
      state_by_id[$1] = $2
      next
    }
    {
      state = ""
      if ($2 in state_by_id) {
        state = state_by_id[$2]
      }
      print $1 "\t" $2 "\t" state
    }
  ' "${states_file}" "${job_ids_file}" > "${out_file}"

  rm -f "${job_ids_file}" "${states_file}" "${raw_file}"
}

pick_log_file() {
  local job="$1"
  local f
  local found=""
  for f in "${LOGS_DIR}/${job}.gpu"*.log; do
    [[ -f "${f}" ]] || continue
    found="${f}"
  done
  printf '%s' "${found}"
}

pick_attempt_log_file() {
  local job="$1"
  local suffix="$2"
  local f
  local best=""
  local best_num=-1
  local name num
  for f in "${LOGS_DIR}/${job}.attempt"*".${suffix}"; do
    [[ -f "${f}" ]] || continue
    name="$(basename "${f}")"
    num="$(printf '%s\n' "${name}" | sed -n 's/.*\.attempt\([0-9][0-9]*\)\..*/\1/p')"
    [[ -n "${num}" ]] || continue
    if [[ "${num}" -gt "${best_num}" ]]; then
      best="${f}"
      best_num="${num}"
    fi
  done
  printf '%s' "${best}"
}

read_export_rc() {
  local seg_dir="$1"
  local status_file="${seg_dir}/.export_status"
  if [[ ! -f "${status_file}" ]]; then
    return 0
  fi
  awk -F'=' '
    $1 == "export_rc" {
      print $2
      exit
    }
  ' "${status_file}" 2>/dev/null
}

read_stage_field() {
  local file_path="$1"
  local key="$2"
  if [[ ! -f "${file_path}" ]]; then
    return 0
  fi
  awk -F'=' -v target="${key}" '
    $1 == target {
      print $2
      exit
    }
  ' "${file_path}" 2>/dev/null
}

read_segment_stage_status() {
  local seg_dir="$1"
  read_stage_field "${seg_dir}/.segment_status" "segment_status"
}

read_export_stage_status() {
  local seg_dir="$1"
  read_stage_field "${seg_dir}/.export_status" "export_status"
}

read_export_stage_note() {
  local seg_dir="$1"
  read_stage_field "${seg_dir}/.export_status" "export_note"
}

read_segment_stage_max_vram_mb() {
  local seg_dir="$1"
  read_stage_field "${seg_dir}/.segment_status" "max_vram_mb"
}

read_segment_stage_max_vram_gb() {
  local seg_dir="$1"
  read_stage_field "${seg_dir}/.segment_status" "max_vram_gb"
}

read_note_field() {
  local note_text="$1"
  local key="$2"
  local token
  for token in ${note_text}; do
    if [[ "${token}" == "${key}="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  return 0
}

get_validation_status_for_job() {
  local job="$1"
  if [[ ! -f "${VALIDATION_TSV}" ]]; then
    return 0
  fi
  awk -F'\t' -v j="${job}" '
    FNR == 1 {
      for (i = 1; i <= NF; i++) {
        idx[$i] = i
      }
      next
    }
    (("job" in idx) && idx["job"] > 0) && $(idx["job"]) == j {
      validate_status = (("validate_status" in idx) && idx["validate_status"] > 0) ? $(idx["validate_status"]) : ""
      if (validate_status == "ok" && ("run_input_dir" in idx) && idx["run_input_dir"] > 0) {
        run_input_dir = $(idx["run_input_dir"])
        lower = tolower(run_input_dir)
        if (run_input_dir == "" || run_input_dir == "-" || lower == "nan" || lower == "none") {
          validate_status = "missing_input_dir"
        }
      }
      print validate_status
      exit
    }
  ' "${VALIDATION_TSV}"
}

infer_failure_status_from_log() {
  local log_file="$1"
  local fallback_status="${2:-}"
  local lower_status
  lower_status="$(printf '%s' "${fallback_status}" | tr '[:upper:]' '[:lower:]')"

  case "${lower_status}" in
    ""|ok|skipped_existing|recovered_predict_ok|running|in_progress|pending|partial|reference)
      printf '%s' "${fallback_status}"
      return 0
      ;;
    segment_oom|segment_runlimit|segment_ancdata|segment_error|predict_error|predict_oom|predict_runlimit)
      printf '%s' "${fallback_status}"
      return 0
      ;;
  esac

  if [[ -f "${log_file}" ]]; then
    if grep -Eq 'FAIL job=.*step=segment(_retry)? \(ancdata\)' "${log_file}" 2>/dev/null; then
      printf 'segment_ancdata'
      return 0
    fi
    if grep -Eq 'FAIL job=.*step=segment(_retry)? \(OOT ' "${log_file}" 2>/dev/null; then
      printf 'segment_runlimit'
      return 0
    fi
    if grep -Eq 'FAIL job=.*step=segment(_retry)? \(oom\)' "${log_file}" 2>/dev/null; then
      printf 'segment_oom'
      return 0
    fi
    if grep -Eq 'FAIL job=.*step=segment(_retry)?' "${log_file}" 2>/dev/null; then
      printf 'segment_error'
      return 0
    fi
    if grep -Eq 'FAIL job=.*step=predict|predict_only_failed|recovered_predict_failed' "${log_file}" 2>/dev/null; then
      printf 'predict_error'
      return 0
    fi
  fi

  case "${lower_status}" in
    recovery_no_checkpoint|missing_segmentation)
      printf 'segment_error'
      ;;
    recovered_predict_failed)
      printf 'predict_error'
      ;;
    *)
      printf '%s' "${fallback_status}"
      ;;
  esac
}

build_snapshot() {
  local status_map="$1"
  local running_jobs="$2"
  local out_file="$3"
  local tmp_file lsf_state_map
  local plan_header plan_has_study_block
  tmp_file="$(mktemp)"
  lsf_state_map="$(mktemp)"

  plan_header="$(head -n 1 "${PLAN_FILE}")"
  plan_has_study_block=0
  if printf '%s\n' "${plan_header}" | tr '\t' '\n' | grep -Fxq "study_block"; then
    plan_has_study_block=1
  fi
  collect_lsf_state_map "${status_map}" "${lsf_state_map}"

  printf "job\tgroup\tgpu\tstatus\tstate\trunning\telapsed_s\trun_count\thad_rerun\thad_anc_retry\thad_predict_fallback\thad_recovery_pass\tseg_exists\tanndata_exists\txenium_exists\tseg_dir\tlog_file\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\tsegment_status\texport_status\tsegment_job_id\texport_job_id\texport_note\tnote\trequested_queue\trequested_gmem_gb\trequested_mem_gb\trequested_wall\tmax_vram_mb\tmax_vram_gb\n" > "${tmp_file}"

  tail -n +2 "${PLAN_FILE}" | while IFS=$'\t' read -r -a cols; do
    local job group use_3d expansion tx_max_k tx_max_dist n_mid_layers n_heads cells_min_counts min_qv alignment_loss
    local row lsf_row gpu status elapsed note seg_dir log_file attempt_err_log attempt_out_log
    local row_segment_status row_export_status row_segment_job_id row_export_job_id row_export_note
    local running seg_exists anndata_exists xenium_exists
    local run_count had_rerun had_anc_retry had_predict_fallback had_recovery_pass
    local state validate_status lsf_job_id lsf_state export_rc
    local stage_segment_status stage_export_status stage_export_note
    local requested_queue requested_gmem_gb requested_mem_gb requested_wall
    local stage_max_vram_mb stage_max_vram_gb

    if [[ "${plan_has_study_block}" == "1" ]]; then
      job="${cols[0]-}"
      group="${cols[2]-}"
      use_3d="${cols[3]-}"
      expansion="${cols[4]-}"
      tx_max_k="${cols[5]-}"
      tx_max_dist="${cols[6]-}"
      n_mid_layers="${cols[7]-}"
      n_heads="${cols[8]-}"
      cells_min_counts="${cols[9]-}"
      min_qv="${cols[10]-}"
      alignment_loss="${cols[11]-}"
    else
      job="${cols[0]-}"
      group="${cols[1]-}"
      use_3d="${cols[2]-}"
      expansion="${cols[3]-}"
      tx_max_k="${cols[4]-}"
      tx_max_dist="${cols[5]-}"
      n_mid_layers="${cols[6]-}"
      n_heads="${cols[7]-}"
      cells_min_counts="${cols[8]-}"
      min_qv="${cols[9]-}"
      alignment_loss="${cols[10]-}"
    fi

    row="$(awk -F'\t' -v j="${job}" '$1 == j { print; exit }' "${status_map}")"
    gpu=""
    status=""
    elapsed=""
    note=""
    seg_dir=""
    log_file=""
    row_segment_status=""
    row_export_status=""
    row_segment_job_id=""
    row_export_job_id=""
    row_export_note=""
    if [[ -n "${row}" ]]; then
      IFS=$'\t' read -r _ gpu status elapsed note seg_dir log_file row_segment_status row_export_status row_segment_job_id row_export_job_id row_export_note <<< "${row}"
      if [[ "${note}" == "-" ]]; then
        note=""
      fi
      if [[ "${seg_dir}" == "-" ]]; then
        seg_dir=""
      fi
      if [[ "${log_file}" == "-" ]]; then
        log_file=""
      fi
      if [[ "${row_segment_status}" == "-" ]]; then
        row_segment_status=""
      fi
      if [[ "${row_export_status}" == "-" ]]; then
        row_export_status=""
      fi
      if [[ "${row_segment_job_id}" == "-" ]]; then
        row_segment_job_id=""
      fi
      if [[ "${row_export_job_id}" == "-" ]]; then
        row_export_job_id=""
      fi
      if [[ "${row_export_note}" == "-" ]]; then
        row_export_note=""
      fi
    fi

    [[ -n "${seg_dir}" ]] || seg_dir="${RUNS_DIR}/${job}"
    [[ -n "${log_file}" ]] || log_file="$(pick_log_file "${job}")"
    [[ -n "${log_file}" ]] || log_file="${LOGS_DIR}/${job}.gpu?.log"
    attempt_err_log="$(pick_attempt_log_file "${job}" "err")"
    attempt_out_log="$(pick_attempt_log_file "${job}" "out")"

    run_count=0
    had_rerun=0
    had_anc_retry=0
    had_predict_fallback=0
    had_recovery_pass=0
    if [[ -f "${log_file}" ]]; then
      run_count="$(awk -v needle="START job=${job}" 'index($0, needle) { count++ } END { print count + 0 }' "${log_file}" 2>/dev/null)"
      if [[ "${run_count}" -gt 1 ]]; then
        had_rerun=1
      fi
      grep -q "segment failed with ancdata; retrying" "${log_file}" 2>/dev/null && had_anc_retry=1 || true
      grep -q "predict fallback succeeded after OOM" "${log_file}" 2>/dev/null && had_predict_fallback=1 || true
      grep -q "RECOVERY job=${job}" "${log_file}" 2>/dev/null && had_recovery_pass=1 || true
    fi

    validate_status="$(get_validation_status_for_job "${job}")"
    lsf_row="$(awk -F'\t' -v j="${job}" '$1 == j { print; exit }' "${lsf_state_map}")"
    lsf_job_id=""
    lsf_state=""
    export_rc="$(read_export_rc "${seg_dir}")"
    stage_segment_status="$(read_segment_stage_status "${seg_dir}")"
    stage_export_status="$(read_export_stage_status "${seg_dir}")"
    stage_export_note="$(read_export_stage_note "${seg_dir}")"
    stage_max_vram_mb="$(read_segment_stage_max_vram_mb "${seg_dir}")"
    stage_max_vram_gb="$(read_segment_stage_max_vram_gb "${seg_dir}")"
    [[ -n "${stage_segment_status}" ]] || stage_segment_status="${row_segment_status}"
    [[ -n "${stage_export_status}" ]] || stage_export_status="${row_export_status}"
    [[ -n "${stage_export_note}" ]] || stage_export_note="${row_export_note}"
    if [[ -z "${stage_max_vram_gb}" && -n "${stage_max_vram_mb}" ]]; then
      stage_max_vram_gb="$(awk -v mb="${stage_max_vram_mb}" 'BEGIN { printf "%.2f", (mb + 0.0) / 1024.0 }')"
    fi
    requested_queue="$(read_note_field "${note}" "queue")"
    requested_gmem_gb="$(read_note_field "${note}" "gmem")"
    requested_mem_gb="$(read_note_field "${note}" "mem")"
    requested_wall="$(read_note_field "${note}" "wall")"
    requested_gmem_gb="${requested_gmem_gb%G}"
    requested_mem_gb="${requested_mem_gb%G}"
    [[ -n "${requested_queue}" ]] || requested_queue="-"
    [[ -n "${requested_gmem_gb}" ]] || requested_gmem_gb="-"
    [[ -n "${requested_mem_gb}" ]] || requested_mem_gb="-"
    [[ -n "${requested_wall}" ]] || requested_wall="-"
    [[ -n "${stage_max_vram_mb}" ]] || stage_max_vram_mb="-"
    [[ -n "${stage_max_vram_gb}" ]] || stage_max_vram_gb="-"
    [[ -n "${lsf_job_id}" ]] || lsf_job_id="${row_segment_job_id}"
    if [[ -n "${lsf_row}" ]]; then
      IFS=$'\t' read -r _ lsf_job_id lsf_state <<< "${lsf_row}"
    fi

    running=0
    if [[ -s "${running_jobs}" ]] && grep -Fxq "${job}" "${running_jobs}"; then
      running=1
    fi
    case "${lsf_state}" in
      RUN|PROV)
        running=1
        ;;
    esac

    seg_exists=0
    anndata_exists=0
    xenium_exists=0
    [[ -f "${RUNS_DIR}/${job}/segger_segmentation.parquet" ]] && seg_exists=1
    [[ -f "${EXPORTS_DIR}/${job}/anndata/segger_segmentation.h5ad" ]] && anndata_exists=1
    [[ -f "${EXPORTS_DIR}/${job}/xenium_explorer/seg_experiment.xenium" ]] && xenium_exists=1

    # Keep historical completion visible even if large artifacts were cleaned up.
    # Do not mark as complete when validation explicitly reports failure.
    case "${status}" in
      ok|skipped_existing|recovered_predict_ok)
        if [[ -z "${validate_status}" ]] || [[ "${validate_status}" == "ok" ]]; then
          if [[ "${seg_exists}" == "0" ]]; then
            seg_exists=1
          fi
          if [[ "${anndata_exists}" == "0" ]]; then
            anndata_exists=1
          fi
          if [[ "${xenium_exists}" == "0" ]]; then
            xenium_exists=1
          fi
        fi
        ;;
    esac

    state="pending"
    if [[ -n "${status}" ]]; then
      case "${status}" in
        ok|skipped_existing|recovered_predict_ok)
          state="done"
          ;;
        running|in_progress)
          state="running"
          ;;
        pending)
          state="pending"
          ;;
        partial)
          state="partial"
          ;;
        *)
          state="failed"
          ;;
      esac
    else
      if [[ "${run_count}" -gt 0 ]] && [[ "${seg_exists}" == "0" ]] && [[ "${running}" == "0" ]]; then
        state="failed"
      elif [[ "${seg_exists}" == "1" && "${anndata_exists}" == "1" && "${xenium_exists}" == "1" ]]; then
        state="done"
      elif [[ "${seg_exists}" == "1" || "${anndata_exists}" == "1" || "${xenium_exists}" == "1" ]]; then
        state="partial"
      else
        state="pending"
      fi
    fi
    if [[ "${status}" == "pending" || "${status}" == "running" || "${status}" == "in_progress" || -z "${status}" ]]; then
      if [[ "${seg_exists}" == "1" && "${anndata_exists}" == "1" && "${xenium_exists}" == "1" ]]; then
        state="done"
        if [[ -z "${validate_status}" || "${validate_status}" == "ok" ]]; then
          status="ok"
        elif [[ -n "${validate_status}" ]]; then
          status="${validate_status}"
        fi
      elif [[ "${seg_exists}" == "1" || "${anndata_exists}" == "1" || "${xenium_exists}" == "1" ]]; then
        if [[ "${running}" == "1" ]]; then
          state="running"
          status="running"
        else
          state="partial"
          status="partial"
        fi
      fi
    fi
    if [[ "${seg_exists}" == "0" && -n "${export_rc}" && "${export_rc}" != "0" ]] && [[ "${anndata_exists}" == "0" || "${xenium_exists}" == "0" ]]; then
      state="failed"
      status="export_error"
    fi
    if [[ "${seg_exists}" == "0" && "${anndata_exists}" == "0" && "${xenium_exists}" == "0" ]]; then
      case "${lsf_state}" in
        RUN|PROV)
          state="running"
          status="running"
          ;;
        PEND|PSUSP|USUSP|SSUSP|WAIT)
          if [[ -z "${status}" || "${status}" == "pending" || "${status}" == "running" || "${status}" == "in_progress" ]]; then
            state="pending"
            status="pending"
          fi
          ;;
        EXIT|ZOMBI|UNKWN)
          if [[ -z "${status}" || "${status}" == "pending" || "${status}" == "running" || "${status}" == "in_progress" ]]; then
            state="failed"
            status="lsf_exit"
          fi
          ;;
        DONE)
          if [[ -z "${status}" || "${status}" == "pending" || "${status}" == "running" || "${status}" == "in_progress" ]]; then
            state="failed"
            status="missing_segmentation"
          fi
          ;;
      esac
    fi
    if [[ "${state}" == "failed" || "${state}" == "partial" ]]; then
      if [[ -n "${attempt_err_log}" ]]; then
        log_file="${attempt_err_log}"
      elif [[ -n "${attempt_out_log}" ]]; then
        log_file="${attempt_out_log}"
      fi
    fi
    if [[ "${running}" == "1" ]] && [[ "${state}" == "pending" ]]; then
      state="running"
    fi
    if [[ "${state}" == "running" ]] && [[ -z "${status}" ]]; then
      status="running"
    elif [[ "${state}" == "failed" ]] && [[ -z "${status}" ]]; then
      status="missing_segmentation"
    fi
    status="$(infer_failure_status_from_log "${log_file}" "${status}")"

    case "${stage_segment_status}" in
      segment_oom|segment_runlimit|segment_error|predict_oom|predict_runlimit|predict_error)
        state="failed"
        status="${stage_segment_status}"
        ;;
      segment_ok|predict_ok)
        state="done"
        if [[ -z "${validate_status}" || "${validate_status}" == "ok" ]]; then
          if [[ "${anndata_exists}" == "1" ]]; then
            status="ok"
          else
            status="segment_only"
          fi
        elif [[ -n "${validate_status}" ]]; then
          status="${validate_status}"
        fi
        ;;
      running)
        state="running"
        status="running"
        ;;
      pending)
        if [[ -z "${status}" || "${status}" == "pending" ]]; then
          state="pending"
          status="pending"
        fi
        ;;
    esac

    # Segmentation availability is the source of truth for completion.
    if [[ "${seg_exists}" == "1" ]]; then
      state="done"
      running=0
      if [[ -z "${validate_status}" || "${validate_status}" == "ok" ]]; then
        if [[ "${anndata_exists}" == "1" ]]; then
          status="ok"
        else
          status="segment_only"
        fi
      elif [[ -n "${validate_status}" ]]; then
        status="${validate_status}"
      fi
      if [[ "${log_file}" == "${attempt_err_log}" || "${log_file}" == "${attempt_out_log}" ]]; then
        log_file="$(pick_log_file "${job}")"
        [[ -n "${log_file}" ]] || log_file="${LOGS_DIR}/${job}.gpu?.log"
      fi
    fi

    if [[ "${state}" == "failed" || "${state}" == "partial" ]]; then
      if [[ -n "${attempt_err_log}" ]]; then
        log_file="${attempt_err_log}"
      elif [[ -n "${attempt_out_log}" ]]; then
        log_file="${attempt_out_log}"
      fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job}" "${group}" "${gpu}" "${status}" "${state}" "${running}" "${elapsed}" \
      "${run_count}" "${had_rerun}" "${had_anc_retry}" "${had_predict_fallback}" "${had_recovery_pass}" \
      "${seg_exists}" "${anndata_exists}" "${xenium_exists}" "${seg_dir}" "${log_file}" \
      "${use_3d}" "${expansion}" "${tx_max_k}" "${tx_max_dist}" "${n_mid_layers}" "${n_heads}" \
      "${cells_min_counts}" "${min_qv}" "${alignment_loss}" \
      "${stage_segment_status}" "${stage_export_status}" "${row_segment_job_id}" "${row_export_job_id}" "${stage_export_note}" "${note}" \
      "${requested_queue}" "${requested_gmem_gb}" "${requested_mem_gb}" "${requested_wall}" "${stage_max_vram_mb}" "${stage_max_vram_gb}" \
      >> "${tmp_file}"
  done

  mv "${tmp_file}" "${out_file}"
  rm -f "${lsf_state_map}"
}

draw_progress_bar() {
  local current="$1"
  local total="$2"
  local width=40
  local fill=0
  local pct=0
  if [[ "${total}" -gt 0 ]]; then
    fill=$((current * width / total))
    pct=$((current * 100 / total))
  fi
  local empty=$((width - fill))
  local left right
  left="$(printf '%*s' "${fill}" '' | tr ' ' '#')"
  right="$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  printf "[%s%s] %d/%d (%d%%)" "${left}" "${right}" "${current}" "${total}" "${pct}"
}

render_tsv_table() {
  awk -F'\t' '
    function repeat_char(ch, n, out, i) {
      out = ""
      for (i = 0; i < n; i++) out = out ch
      return out
    }
    function visible_len(s, t) {
      t = s
      gsub(/\033\[[0-9;]*m/, "", t)
      return length(t)
    }
    function print_cell(val, width, vis, i) {
      vis = visible_len(val)
      printf(" %s", val)
      for (i = vis; i < width; i++) {
        printf(" ")
      }
      printf(" |")
    }
    {
      rows = NR
      if ($1 == "__ROW_SEP__") {
        row_sep[NR] = 1
        next
      }
      if (NF > ncols) ncols = NF
      for (i = 1; i <= NF; i++) {
        val = $i
        sub(/\r$/, "", val)
        cells[NR, i] = val
        vis = visible_len(val)
        if (vis > widths[i]) widths[i] = vis
      }
    }
    END {
      if (rows == 0) exit

      sep = "+"
      for (i = 1; i <= ncols; i++) {
        sep = sep repeat_char("-", widths[i] + 2) "+"
      }

      print sep
      printf("|")
      for (i = 1; i <= ncols; i++) {
        print_cell(cells[1, i], widths[i])
      }
      printf("\n")
      print sep

      for (r = 2; r <= rows; r++) {
        if (row_sep[r]) {
          print sep
          continue
        }
        printf("|")
        for (i = 1; i <= ncols; i++) {
          print_cell(cells[r, i], widths[i])
        }
        printf("\n")
      }
      print sep
    }
  '
}

colorize_rows_by_state_column() {
  local state_col="$1"
  awk -F'\t' \
    -v state_col="${state_col}" \
    -v c_done="${C_GREEN}" \
    -v c_run="${C_BLUE}" \
    -v c_fail="${C_RED}" \
    -v c_pending="${C_YELLOW}" \
    -v c_partial="${C_CYAN}" \
    -v c_reset="${C_RESET}" \
    '
      function color_for_state(s, lower) {
        lower = tolower(s)
        if (lower == "done") return c_done
        if (lower == "running") return c_run
        if (lower == "failed") return c_fail
        if (lower == "pending") return c_pending
        if (lower == "reference") return c_partial
        if (lower == "partial") return c_partial
        return ""
      }
      NR == 1 { print; next }
      {
        state = (state_col > 0 && state_col <= NF) ? $state_col : ""
        color = color_for_state(state)
        if (color != "") {
          for (i = 1; i <= NF; i++) {
            $i = color $i c_reset
          }
        }
        print
      }
    ' OFS=$'\t'
}

colorize_rows_by_status_column() {
  local status_col="$1"
  awk -F'\t' \
    -v status_col="${status_col}" \
    -v c_done="${C_GREEN}" \
    -v c_run="${C_BLUE}" \
    -v c_fail="${C_RED}" \
    -v c_pending="${C_YELLOW}" \
    -v c_partial="${C_CYAN}" \
    -v c_reset="${C_RESET}" \
    '
      function state_from_status(status, lower) {
        lower = tolower(status)
        if (status == "" || status == "<none>") return "pending"
        if (lower ~ /running|in_progress/) return "running"
        if (lower ~ /oom|oot|ancdata|fail|error|missing|recovery_no_checkpoint/) return "failed"
        if (lower == "ok" || lower == "skipped_existing" || lower == "recovered_predict_ok") return "done"
        return "pending"
      }
      function color_for_state(s, lower) {
        lower = tolower(s)
        if (lower == "done") return c_done
        if (lower == "running") return c_run
        if (lower == "failed") return c_fail
        if (lower == "pending") return c_pending
        if (lower == "reference") return c_partial
        if (lower == "partial") return c_partial
        return ""
      }
      NR == 1 { print; next }
      {
        status = (status_col > 0 && status_col <= NF) ? $status_col : ""
        state = state_from_status(status)
        color = color_for_state(state)
        if (color != "") {
          for (i = 1; i <= NF; i++) {
            $i = color $i c_reset
          }
        }
        print
      }
    ' OFS=$'\t'
}

render_dashboard() {
  local snapshot="$1"
  local total done_count running_count pending_count partial_count failed_count
  local oom_count runlimit_count anc_count rerun_count recovered_count processed
  local now

  total="$(awk 'END { print NR-1 }' "${snapshot}")"
  done_count="$(awk -F'\t' 'NR>1 && $5=="done" {c++} END{print c+0}' "${snapshot}")"
  running_count="$(awk -F'\t' 'NR>1 && $5=="running" {c++} END{print c+0}' "${snapshot}")"
  pending_count="$(awk -F'\t' 'NR>1 && $5=="pending" {c++} END{print c+0}' "${snapshot}")"
  partial_count="$(awk -F'\t' 'NR>1 && $5=="partial" {c++} END{print c+0}' "${snapshot}")"
  failed_count="$(awk -F'\t' 'NR>1 && $5=="failed" {c++} END{print c+0}' "${snapshot}")"

  oom_count="$(awk -F'\t' 'NR>1 && $4 ~ /oom/ {c++} END{print c+0}' "${snapshot}")"
  runlimit_count="$(awk -F'\t' 'NR>1 && $4 ~ /runlimit/ {c++} END{print c+0}' "${snapshot}")"
  anc_count="$(awk -F'\t' 'NR>1 && ($4=="segment_ancdata" || $10=="1") {c++} END{print c+0}' "${snapshot}")"
  rerun_count="$(awk -F'\t' 'NR>1 && $9=="1" {c++} END{print c+0}' "${snapshot}")"
  recovered_count="$(awk -F'\t' 'NR>1 && ($4=="recovered_predict_ok" || $11=="1" || $12=="1") {c++} END{print c+0}' "${snapshot}")"
  processed=$((done_count + partial_count + failed_count))

  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "${C_CYAN}Benchmark Dashboard${C_RESET} | ${now}"
  echo "Root: ${ROOT}"
  echo "Snapshot: ${snapshot}"
  printf "Progress: "
  draw_progress_bar "${processed}" "${total}"
  echo
  echo
  printf "%b\n" "${C_BLUE}running=${running_count}${C_RESET}  ${C_YELLOW}pending=${pending_count}${C_RESET}  ${C_RED}failed=${failed_count}${C_RESET}  ${C_GREEN}done=${done_count}${C_RESET}  ${C_CYAN}partial=${partial_count}${C_RESET}"
  printf "oom=%s  runlimit=%s  ancdata=%s  rerun=%s  recovered=%s\n" "${oom_count}" "${runlimit_count}" "${anc_count}" "${rerun_count}" "${recovered_count}"

  echo
  echo "State Counts:"
  awk -F'\t' '
    NR > 1 {
      c[$5]++
    }
    END {
      print "state\tcount"
      order[1] = "running"
      order[2] = "pending"
      order[3] = "failed"
      order[4] = "done"
      order[5] = "partial"

      for (i = 1; i <= 5; i++) {
        s = order[i]
        print s "\t" (c[s] + 0)
        seen[s] = 1
      }
      for (k in c) {
        if (!(k in seen)) {
          print k "\t" c[k]
        }
      }
    }
  ' "${snapshot}" \
    | colorize_rows_by_state_column 1 \
    | render_tsv_table

  echo
  echo "Status Counts:"
  awk -F'\t' '
    function state_from_status(status, lower) {
      lower = tolower(status)
      if (status == "" || status == "<none>") return "pending"
      if (lower ~ /running|in_progress/) return "running"
      if (lower ~ /oom|oot|ancdata|fail|error|missing|recovery_no_checkpoint/) return "failed"
      if (lower == "ok" || lower == "skipped_existing" || lower == "recovered_predict_ok") return "done"
      return "pending"
    }
    function rank(state, lower) {
      lower = tolower(state)
      if (lower == "running") return 1
      if (lower == "pending") return 2
      if (lower == "failed") return 3
      if (lower == "done") return 4
      if (lower == "partial") return 5
      return 99
    }
    NR > 1 {
      key = $4
      if (key == "") key = "<none>"
      c[key]++
    }
    END {
      print "rank\tstatus\tcount"
      for (k in c) {
        st = state_from_status(k)
        print rank(st) "\t" k "\t" c[k]
      }
    }
  ' "${snapshot}" \
    | {
        IFS= read -r header || true
        if [[ -n "${header}" ]]; then
          printf "status\tcount\n"
        fi
        sort -t $'\t' -k1,1n -k3,3nr -k2,2 \
          | cut -f2-
      } \
    | colorize_rows_by_status_column 1 \
    | render_tsv_table

  echo
  echo "All Jobs Overview:"
  awk -F'\t' '
    function rank(st, lower) {
      lower = tolower(st)
      if (lower == "done") return 1
      if (lower == "running") return 2
      if (lower == "failed") return 3
      if (lower == "pending") return 4
      if (lower == "partial") return 5
      if (lower == "reference") return 6
      return 99
    }
    function fmt_minutes(v, lower, n) {
      lower = tolower(v)
      if (v == "" || lower == "nan" || lower == "none" || lower == "-") return "nan"
      n = (v + 0.0) / 60.0
      if (n < 0) n = 0
      return sprintf("%.2f", n)
    }
    function col(name, fallback) {
      if ((name in idx) && idx[name] > 0 && idx[name] <= NF) return $(idx[name])
      return fallback
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      print "rank\tjob\tgroup\tgpu\tstate\tstatus\truns\telapsed_min\treq_gmem_gb\tmax_vram_gb\trerun\tanc_retry\toom_pred_fallback\trecovery\tseg\tanndata\txenium"
      next
    }
    NR > 1 {
      st = col("state", "pending")
      if (st == "") st = "pending"
      status = col("status", "")
      if (status == "") status = "<none>"
      print rank(st) "\t" \
            col("job", "-") "\t" \
            col("group", "-") "\t" \
            col("gpu", "-") "\t" \
            st "\t" \
            status "\t" \
            col("run_count", "0") "\t" \
            fmt_minutes(col("elapsed_s", "0")) "\t" \
            col("requested_gmem_gb", "-") "\t" \
            col("max_vram_gb", "-") "\t" \
            col("had_rerun", "0") "\t" \
            col("had_anc_retry", "0") "\t" \
            col("had_predict_fallback", "0") "\t" \
            col("had_recovery_pass", "0") "\t" \
            col("seg_exists", "0") "\t" \
            col("anndata_exists", "0") "\t" \
            col("xenium_exists", "0")
    }
  ' "${snapshot}" \
    | {
        IFS= read -r header || true
        if [[ -n "${header}" ]]; then
          printf "job\tgroup\tgpu\tstate\tstatus\truns\telapsed_min\treq_gmem_gb\tmax_vram_gb\trerun\tanc_retry\toom_pred_fallback\trecovery\tseg\tanndata\txenium\n"
        fi
        sort -t $'\t' -k1,1n -k2,2 \
          | cut -f2-
      } \
    | colorize_rows_by_state_column 4 \
    | render_tsv_table

  echo
  echo "Model Parameterization:"
  awk -F'\t' '
    function block_rank(block, lower) {
      lower = tolower(block)
      if (lower == "stability") return 1
      if (lower == "interaction") return 2
      if (lower == "stress") return 3
      if (block == "-") return 4
      return 5
    }
    function model_label(job, lower) {
      lower = tolower(job)
      if (lower == "baseline") return "baseline"
      if (index(lower, "stbl_baseline_") == 1) return "baseline_repeat"
      if (index(lower, "stbl_anchor_") == 1) return "anchor_repeat"
      if (index(lower, "stbl_sens_") == 1) return "sensitivity_repeat"
      if (index(lower, "int_") == 1) return "interaction_ablation"
      if (index(lower, "stress_") == 1) return "stress_test"
      if (lower == "abl_full") return "ablation_anchor"
      if (index(lower, "abl_sg_") == 1) return "loss_decomposition"
      if (index(lower, "abl_sgloss_") == 1) return "segmentation_loss"
      if (index(lower, "abl_aw_") == 1) return "alignment_sweep"
      if (index(lower, "abl_scale_") == 1) return "scale_factor"
      if (index(lower, "abl_use3d_") == 1) return "use_3d"
      if (index(lower, "abl_txk_") == 1) return "graph_neighbors"
      if (index(lower, "abl_txdist_") == 1) return "graph_radius"
      if (index(lower, "abl_graph_") == 1) return "graph_density"
      if (index(lower, "abl_depth_") == 1) return "gnn_depth"
      if (index(lower, "abl_net_") == 1) return "learnable_params"
      if (index(lower, "abl_width_") == 1) return "gnn_width"
      if (index(lower, "abl_heads_") == 1) return "attention_heads"
      if (index(lower, "abl_no_pos") == 1) return "positional_encoding"
      if (index(lower, "abl_no_norm") == 1) return "embedding_norm"
      if (index(lower, "abl_morph") == 1) return "cell_features"
      if (index(lower, "abl_pred_") == 1) return "prediction_mode"
      if (index(lower, "abl_frag_") == 1) return "fragments"
      if (index(lower, "abl_lr_") == 1) return "learning_rate"
      if (index(lower, "use3d_") == 1) return "ablation_use3d"
      if (index(lower, "expansion_") == 1) return "ablation_expansion"
      if (index(lower, "txk_") == 1) return "ablation_txk"
      if (index(lower, "txdist_") == 1) return "ablation_txdist"
      if (index(lower, "layers_") == 1) return "ablation_layers"
      if (index(lower, "heads_") == 1) return "ablation_heads"
      if (index(lower, "cellsmin_") == 1) return "ablation_cellsmin"
      if (index(lower, "align_") == 1) return "ablation_alignment"
      return "custom"
    }
    function col(name, fallback) {
      if ((name in idx) && idx[name] > 0) return $(idx[name])
      return fallback
    }
    FNR == NR {
      if (FNR == 1) {
        for (i = 1; i <= NF; i++) {
          if ($i == "job") snap_job_col = i
          if ($i == "state") snap_state_col = i
          if ($i == "status") snap_status_col = i
        }
        next
      }
      if (snap_job_col > 0) {
        j = $snap_job_col
        state_by_job[j] = (snap_state_col > 0 ? $snap_state_col : "")
        status_by_job[j] = (snap_status_col > 0 ? $snap_status_col : "")
      }
      next
    }
    FNR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      has_block = (("study_block" in idx) && idx["study_block"] > 0) ? 1 : 0
      print "rank\tjob\tmodel\tstudy_block\tgroup\tstate\tstatus\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\tprediction_mode\tfragment_mode\tqueue\treq_gmem_gb\treq_mem_gb\treq_wall"
      next
    }
    {
      job = col("job", $1)
      block = has_block ? col("study_block", "-") : "-"
      group = col("group", "-")
      if (group == "-" && ("gpu_group" in idx) && idx["gpu_group"] > 0) {
        group = $(idx["gpu_group"])
      }
      state = state_by_job[job]
      if (state == "") state = "pending"
      status = status_by_job[job]
      if (status == "") status = "<none>"

      print block_rank(block) "\t" \
            job "\t" \
            model_label(job) "\t" \
            block "\t" \
            group "\t" \
            state "\t" \
            status "\t" \
            col("use_3d", "-") "\t" \
            col("expansion", "-") "\t" \
            col("tx_max_k", "-") "\t" \
            col("tx_max_dist", "-") "\t" \
            col("n_mid_layers", "-") "\t" \
            col("n_heads", "-") "\t" \
            col("cells_min_counts", "-") "\t" \
            col("min_qv", "-") "\t" \
            col("alignment_loss", "-") "\t" \
            col("prediction_mode", "-") "\t" \
            col("fragment_mode", "-") "\t" \
            col("queue", "-") "\t" \
            col("gmem_gb", "-") "\t" \
            col("mem_gb", "-") "\t" \
            col("wall_time", "-")
    }
  ' "${snapshot}" "${PLAN_FILE}" \
    | {
        IFS= read -r header || true
        if [[ -n "${header}" ]]; then
          printf "job\tmodel\tstudy_block\tgroup\tstate\tstatus\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\tprediction_mode\tfragment_mode\tqueue\treq_gmem_gb\treq_mem_gb\treq_wall\n"
        fi
        sort -t $'\t' -k1,1n -k2,2 \
          | cut -f2-
      } \
    | colorize_rows_by_state_column 5 \
    | render_tsv_table

  echo
  echo "Validation Metrics:"
  if [[ -f "${VALIDATION_TSV}" ]] && [[ "$(awk 'END { print NR-1 }' "${VALIDATION_TSV}")" -gt 0 ]]; then
    awk -F'\t' -v snapshot_file="${snapshot}" -v plan_file="${PLAN_FILE}" -v validation_file="${VALIDATION_TSV}" '
      function has_col(name) {
        return (name in idx) && idx[name] > 0
      }
      function get_col(name) {
        if (has_col(name)) return $(idx[name])
        return ""
      }
      function fmt_float(v, lower) {
        lower = tolower(v)
        if (v == "" || lower == "nan" || lower == "none" || lower == "-") return "nan"
        return sprintf("%.4f", v + 0.0)
      }
      function fmt_nonneg_int(v, lower, n) {
        lower = tolower(v)
        if (v == "" || lower == "nan" || lower == "none" || lower == "-") return "0"
        n = v + 0.0
        if (n < 0) n = 0
        return sprintf("%.0f", n)
      }
      function fmt_minutes(v, lower, n) {
        lower = tolower(v)
        if (v == "" || lower == "nan" || lower == "none" || lower == "-") return "nan"
        n = (v + 0.0) / 60.0
        if (n < 0) n = 0
        return sprintf("%.2f", n)
      }
      FILENAME == snapshot_file {
        if (FNR == 1) {
          for (i = 1; i <= NF; i++) {
            if ($i == "job") snap_job_col = i
            if ($i == "state") snap_state_col = i
            if ($i == "status") snap_status_col = i
            if ($i == "elapsed_s") snap_elapsed_col = i
          }
          next
        }
        if (snap_job_col > 0) {
          job_key = $snap_job_col
          if (snap_state_col > 0) state_by_job[job_key] = $snap_state_col
          if (snap_status_col > 0) status_by_job[job_key] = $snap_status_col
          if (snap_elapsed_col > 0) elapsed_by_job[job_key] = $snap_elapsed_col
        }
        next
      }
      FILENAME == plan_file {
        if (FNR > 1 && $1 != "") {
          planned[$1] = 1
        }
        next
      }
      FILENAME == validation_file && FNR == 1 {
        for (i = 1; i <= NF; i++) {
          idx[$i] = i
        }
        print "job\tkind\tst\tval\tgpu_m v\tcells\tfrag\tasg% ^\tmecr v\tbdr% v\trslv v\taes ^\tvsi% v"
        next
      }
      FILENAME != validation_file {
        next
      }
      {
        job = get_col("job")
        job_disp = job
        group = get_col("group")
        is_reference = get_col("is_reference")
        reference_kind = get_col("reference_kind")
        is_reference_row = 0
        if (reference_kind != "" && reference_kind != "-") {
          is_reference_row = 1
        } else if (is_reference == "1" || group == "R") {
          is_reference_row = 1
        }
        if (!(job in planned) && is_reference_row == 0) {
          next
        }
        snapshot_status = status_by_job[job]
        if (reference_kind != "" && reference_kind != "-") {
          kind = reference_kind
        } else if (is_reference == "1" || group == "R") {
          kind = "reference"
        } else {
          kind = "segger"
        }
        if (job == "baseline" && kind == "segger") {
          job_disp = "baseline*"
        }

        state = state_by_job[job]
        if (kind != "segger") {
          state = "reference"
        } else if (state == "") {
          state = "<none>"
        }

        validate_status = get_col("validate_status")
        if (validate_status == "") validate_status = "<none>"
        if (kind == "segger") {
          lower_validate = tolower(validate_status)
          lower_snapshot = tolower(snapshot_status)
          if ((lower_validate == "missing_segmentation" || lower_validate == "recovery_no_checkpoint") &&
              snapshot_status != "" && lower_snapshot != "missing_segmentation" && lower_snapshot != "recovery_no_checkpoint") {
            validate_status = snapshot_status
          }
        }

        gpu_time = get_col("gpu_time_s")
        if (gpu_time == "") gpu_time = get_col("elapsed_s")
        if (gpu_time == "") gpu_time = elapsed_by_job[job]
        gpu_time = fmt_minutes(gpu_time)

        cells = get_col("cells")
        if (cells == "") cells = get_col("cells_non_fragment_total")
        if (cells == "") cells = get_col("cells_assigned")
        if (cells == "") cells = get_col("cells_total")
        cells = fmt_nonneg_int(cells)

        fragments = get_col("fragments")
        if (fragments == "") fragments = get_col("fragments_total")
        if (fragments == "") fragments = get_col("fragments_assigned")
        fragments = fmt_nonneg_int(fragments)

        assigned = get_col("assigned_pct")
        if (assigned == "") assigned = get_col("coverage_pct")

        mecr = get_col("mecr")
        if (mecr == "") mecr = get_col("mecr_fast")

        contamination = get_col("contamination_pct")
        if (contamination == "") contamination = get_col("border_contaminated_cells_pct_fast")

        resolvi = get_col("contamination_pct")
        if (resolvi == "") resolvi = get_col("contamination_pct_fast")
        if (resolvi == "") resolvi = get_col("resolvi_contamination_pct")

        aes = get_col("aes")
        if (aes == "") aes = get_col("expression_angular_uniformity_fast")

        doublet = get_col("doublet_pct")
        if (doublet == "") {
          doublet = get_col("vsi_doublet_fraction_fast")
          if (doublet == "") doublet = get_col("signal_hotspot_doublet_fraction_fast")
          if (doublet == "") doublet = get_col("signal_doublet_like_fraction_fast")
          if (doublet != "" && tolower(doublet) != "nan" && tolower(doublet) != "none") {
            doublet = 100.0 * (doublet + 0.0)
          }
        }

        print job_disp "\t" kind "\t" state "\t" validate_status "\t" gpu_time "\t" cells "\t" fragments "\t" \
              fmt_float(assigned) "\t" fmt_float(mecr) "\t" fmt_float(contamination) "\t" \
              fmt_float(resolvi) "\t" fmt_float(aes) "\t" fmt_float(doublet)
      }
    ' "${snapshot}" "${PLAN_FILE}" "${VALIDATION_TSV}" \
      | {
          IFS= read -r header || true
          if [[ -n "${header}" ]]; then
            printf "%s\n" "${header}"
          fi
          awk -F'\t' -v b_on="${C_BOLD}" -v b_off="${C_BOLD_OFF}" '
            function is_num(v, lower) {
              lower = tolower(v)
              return !(v == "" || lower == "nan" || lower == "none" || lower == "-")
            }
            function to_num(v) {
              return v + 0.0
            }
            function update_top2_up(v,    x) {
              if (!is_num(v)) return
              x = to_num(v)
              if (!have1 || x > top1) {
                top2 = top1
                have2 = have1
                top1 = x
                have1 = 1
              } else if (!have2 || x > top2) {
                top2 = x
                have2 = 1
              }
            }
            function update_top2_down(v,    x) {
              if (!is_num(v)) return
              x = to_num(v)
              if (!have1 || x < top1) {
                top2 = top1
                have2 = have1
                top1 = x
                have1 = 1
              } else if (!have2 || x < top2) {
                top2 = x
                have2 = 1
              }
            }
            function is_top2_up(v, best1, best2, have_2,    x) {
              if (!is_num(v) || !is_num(best1)) return 0
              x = to_num(v)
              if (!have_2) return (x >= best1)
              return (x >= best2)
            }
            function is_top2_down(v, best1, best2, have_2,    x) {
              if (!is_num(v) || !is_num(best1)) return 0
              x = to_num(v)
              if (!have_2) return (x <= best1)
              return (x <= best2)
            }
            function sort_key_up(v) {
              if (!is_num(v)) return 1e12
              return -to_num(v)
            }
            function sort_key_down(v) {
              if (!is_num(v)) return 1e12
              return to_num(v)
            }
            {
              n++
              for (j = 1; j <= NF; j++) {
                cell[n, j] = $j
              }
              nf[n] = NF
              m_assigned[n] = $8
              m_mecr[n] = $9
              m_border[n] = $10
              m_resolvi[n] = $11
              m_aes[n] = $12
              m_doublet[n] = $13
              m_gpu[n] = $5
              st[n] = tolower($4)
              is_ref[n] = (tolower($3) == "reference")

              if (!is_ref[n] && st[n] == "ok") {
                have1 = have_a_best1; top1 = a_best1; have2 = have_a_best2; top2 = a_best2
                update_top2_up($8)
                have_a_best1 = have1; a_best1 = top1; have_a_best2 = have2; a_best2 = top2

                have1 = have_m_best1; top1 = m_best1; have2 = have_m_best2; top2 = m_best2
                update_top2_down($9)
                have_m_best1 = have1; m_best1 = top1; have_m_best2 = have2; m_best2 = top2

                have1 = have_b_best1; top1 = b_best1; have2 = have_b_best2; top2 = b_best2
                update_top2_down($10)
                have_b_best1 = have1; b_best1 = top1; have_b_best2 = have2; b_best2 = top2

                have1 = have_r_best1; top1 = r_best1; have2 = have_r_best2; top2 = r_best2
                update_top2_down($11)
                have_r_best1 = have1; r_best1 = top1; have_r_best2 = have2; r_best2 = top2

                have1 = have_t_best1; top1 = t_best1; have2 = have_t_best2; top2 = t_best2
                update_top2_up($12)
                have_t_best1 = have1; t_best1 = top1; have_t_best2 = have2; t_best2 = top2

                have1 = have_d_best1; top1 = d_best1; have2 = have_d_best2; top2 = d_best2
                update_top2_down($13)
                have_d_best1 = have1; d_best1 = top1; have_d_best2 = have2; d_best2 = top2

                have1 = have_g_best1; top1 = g_best1; have2 = have_g_best2; top2 = g_best2
                update_top2_down($5)
                have_g_best1 = have1; g_best1 = top1; have_g_best2 = have2; g_best2 = top2
              }
            }
            END {
              for (i = 1; i <= n; i++) {
                if (b_on != "" && st[i] == "ok") {
                  if (!is_ref[i] && is_top2_down(m_gpu[i], g_best1, g_best2, have_g_best2)) cell[i, 5] = b_on cell[i, 5] b_off
                  if (is_top2_up(m_assigned[i], a_best1, a_best2, have_a_best2)) cell[i, 8] = b_on cell[i, 8] b_off
                  if (is_top2_down(m_mecr[i], m_best1, m_best2, have_m_best2)) cell[i, 9] = b_on cell[i, 9] b_off
                  if (is_top2_down(m_border[i], b_best1, b_best2, have_b_best2)) cell[i, 10] = b_on cell[i, 10] b_off
                  if (is_top2_down(m_resolvi[i], r_best1, r_best2, have_r_best2)) cell[i, 11] = b_on cell[i, 11] b_off
                  if (is_top2_up(m_aes[i], t_best1, t_best2, have_t_best2)) cell[i, 12] = b_on cell[i, 12] b_off
                  if (is_top2_down(m_doublet[i], d_best1, d_best2, have_d_best2)) cell[i, 13] = b_on cell[i, 13] b_off
                }

                if (is_ref[i]) {
                  bucket = 0
                  k_assigned = 0
                  k_mecr = 0
                  k_border = 0
                  k_resolvi = 0
                  k_aes = 0
                  k_doublet = 0
                } else if (st[i] == "ok") {
                  bucket = 1
                  k_assigned = sort_key_up(m_assigned[i])
                  k_mecr = sort_key_down(m_mecr[i])
                  k_border = sort_key_down(m_border[i])
                  k_resolvi = sort_key_down(m_resolvi[i])
                  k_aes = sort_key_up(m_aes[i])
                  k_doublet = sort_key_down(m_doublet[i])
                } else {
                  bucket = 2
                  k_assigned = 1e12
                  k_mecr = 1e12
                  k_border = 1e12
                  k_resolvi = 1e12
                  k_aes = 1e12
                  k_doublet = 1e12
                }

                row = cell[i, 1]
                for (j = 2; j <= nf[i]; j++) {
                  row = row "\t" cell[i, j]
                }
                printf "%d\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%08d\t%s\n", \
                  bucket, k_assigned, k_mecr, k_border, k_resolvi, k_aes, k_doublet, i, row
              }
            }
          ' \
            | sort -t $'\t' -k1,1n -k2,2n -k3,3n -k4,4n -k5,5n -k6,6n -k7,7n -k8,8n \
            | cut -f9- \
            | awk -F'\t' '
                {
                  state = tolower($3)
                  if (!seen_data) {
                    seen_data = 1
                  }
                  if (state == "reference") {
                    seen_ref = 1
                    print
                    next
                  }
                  if (seen_ref && !inserted_sep) {
                    print "__ROW_SEP__"
                    inserted_sep = 1
                  }
                  print
                }
              '
        } \
      | colorize_rows_by_state_column 3 \
      | render_tsv_table
  else
    echo "No validation TSV found at ${VALIDATION_TSV}"
    echo "Run: bash scripts/build_benchmark_validation_table.sh --root ${ROOT}"
  fi

  if [[ "${running_count}" -gt 0 ]]; then
    echo
    echo "Running Jobs:"
    awk -F'\t' '
      BEGIN {
        print "job\tgpu\tstatus\tstate\truns\tlog_file"
      }
      NR > 1 && $5 == "running" {
        st = $4
        if (st == "") st = "<none>"
        print $1 "\t" $3 "\t" st "\t" $5 "\t" $8 "\t" $17
      }
    ' "${snapshot}" \
      | colorize_rows_by_state_column 4 \
      | render_tsv_table
  fi

  if [[ "${failed_count}" -gt 0 ]]; then
    echo
    echo "Failed Jobs:"
    awk -F'\t' '
      BEGIN {
        print "job\tgpu\tstatus\tstate\truns\trerun\tlog_file"
      }
      NR > 1 && $5 == "failed" {
        st = $4
        if (st == "") st = "<none>"
        print $1 "\t" $3 "\t" st "\t" $5 "\t" $8 "\t" $9 "\t" $17
      }
    ' "${snapshot}" \
      | colorize_rows_by_state_column 4 \
      | render_tsv_table
  fi

  if [[ "${partial_count}" -gt 0 ]]; then
    echo
    echo "Partial Jobs:"
    awk -F'\t' '
      BEGIN {
        print "job\tgpu\tstatus\tstate\truns\tseg\tanndata\txenium\tlog_file"
      }
      NR > 1 && $5 == "partial" {
        st = $4
        if (st == "") st = "<none>"
        print $1 "\t" $3 "\t" st "\t" $5 "\t" $8 "\t" $13 "\t" $14 "\t" $15 "\t" $17
      }
    ' "${snapshot}" \
      | colorize_rows_by_state_column 4 \
      | render_tsv_table
  fi

  if [[ "${rerun_count}" -gt 0 ]]; then
    echo
    echo "Rerun/Retry Jobs:"
    awk -F'\t' '
      BEGIN {
        print "job\tstate\tstatus\truns\tanc_retry\toom_predict_fallback\trecovery_pass"
      }
      NR > 1 && $9 == "1" {
        st = $4
        if (st == "") st = "<none>"
        print $1 "\t" $5 "\t" st "\t" $8 "\t" $10 "\t" $11 "\t" $12
      }
    ' "${snapshot}" \
      | {
          IFS= read -r header || true
          if [[ -n "${header}" ]]; then
            printf "%s\n" "${header}"
          fi
          awk -F'\t' '
            function rank(state, lower) {
              lower = tolower(state)
              if (lower == "running") return 1
              if (lower == "pending") return 2
              if (lower == "failed") return 3
              if (lower == "done") return 4
              if (lower == "partial") return 5
              return 99
            }
            {
              print rank($2) "\t" $0
            }
          ' \
            | sort -t $'\t' -k1,1n -k2,2 \
            | cut -f2-
        } \
      | colorize_rows_by_state_column 2 \
      | render_tsv_table
  fi
}

render_minimal_dashboard() {
  local snapshot="$1"
  local total done_count running_count pending_count partial_count failed_count processed now
  local jobs_overview_file done_jobs_file failed_jobs_file

  total="$(awk 'END { print NR-1 }' "${snapshot}")"
  done_count="$(awk -F'\t' 'NR>1 && $5=="done" {c++} END{print c+0}' "${snapshot}")"
  running_count="$(awk -F'\t' 'NR>1 && $5=="running" {c++} END{print c+0}' "${snapshot}")"
  pending_count="$(awk -F'\t' 'NR>1 && $5=="pending" {c++} END{print c+0}' "${snapshot}")"
  partial_count="$(awk -F'\t' 'NR>1 && $5=="partial" {c++} END{print c+0}' "${snapshot}")"
  failed_count="$(awk -F'\t' 'NR>1 && $5=="failed" {c++} END{print c+0}' "${snapshot}")"
  processed=$((done_count + partial_count + failed_count))

  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "${C_CYAN}Benchmark Dashboard (Minimal)${C_RESET} | ${now}"
  echo "Root: ${ROOT}"
  echo "Snapshot: ${snapshot}"
  printf "Progress: "
  draw_progress_bar "${processed}" "${total}"
  echo
  printf "%b\n" "${C_BLUE}running=${running_count}${C_RESET}  ${C_YELLOW}pending=${pending_count}${C_RESET}  ${C_RED}failed=${failed_count}${C_RESET}  ${C_GREEN}done=${done_count}${C_RESET}  ${C_CYAN}partial=${partial_count}${C_RESET}"

  jobs_overview_file="$(mktemp)"
  done_jobs_file="$(mktemp)"
  failed_jobs_file="$(mktemp)"

  awk -F'\t' '
    function rank(st, lower) {
      lower = tolower(st)
      if (lower == "failed") return 1
      if (lower == "running") return 2
      if (lower == "partial") return 3
      if (lower == "pending") return 4
      if (lower == "done") return 5
      if (lower == "reference") return 6
      return 99
    }
    function fmt_minutes(v, lower, n) {
      lower = tolower(v)
      if (v == "" || lower == "nan" || lower == "none" || lower == "-") return "0.00"
      n = (v + 0.0) / 60.0
      if (n < 0) n = 0
      return sprintf("%.2f", n)
    }
    function col(name, fallback) {
      if ((name in idx) && idx[name] > 0 && idx[name] <= NF) return $(idx[name])
      return fallback
    }
    function bool_label(v, lower) {
      lower = tolower(v)
      if (v == "1" || lower == "true" || lower == "yes") return "yes"
      return "no"
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      print "rank\tjob\tgroup\tstate\tstatus\truns\telapsed_min\treq_queue\treq_gmem_gb\tanndata\tmax_vram_gb"
      next
    }
    NR > 1 {
      st = col("state", "pending")
      if (st == "") st = "pending"
      status = col("status", "")
      if (status == "") status = "<none>"
      print rank(st) "\t" \
            col("job", "-") "\t" \
            col("group", "-") "\t" \
            st "\t" \
            status "\t" \
            col("run_count", "0") "\t" \
            fmt_minutes(col("elapsed_s", "0")) "\t" \
            col("requested_queue", "-") "\t" \
            col("requested_gmem_gb", "-") "\t" \
            bool_label(col("anndata_exists", "0")) "\t" \
            col("max_vram_gb", "-")
    }
  ' "${snapshot}" \
    | {
        IFS= read -r header || true
        if [[ -n "${header}" ]]; then
          printf "job\tgroup\tstate\tstatus\truns\telapsed_min\treq_queue\treq_gmem_gb\tanndata\tmax_vram_gb\n"
        fi
        sort -t $'\t' -k1,1n -k2,2 | cut -f2-
      } > "${jobs_overview_file}"

  awk -F'\t' 'NR == 1 || tolower($3) == "done" { print }' "${jobs_overview_file}" > "${done_jobs_file}"
  awk -F'\t' 'NR == 1 || tolower($3) == "failed" { print }' "${jobs_overview_file}" > "${failed_jobs_file}"

  echo
  echo "Done Jobs:"
  colorize_rows_by_state_column 3 < "${done_jobs_file}" | render_tsv_table

  echo
  echo "Failed Jobs:"
  colorize_rows_by_state_column 3 < "${failed_jobs_file}" | render_tsv_table

  rm -f "${jobs_overview_file}" "${done_jobs_file}" "${failed_jobs_file}"
}

snapshot_once() {
  local tmp_status_map tmp_running
  tmp_status_map="$(mktemp)"
  tmp_running="$(mktemp)"
  collect_running_jobs "${tmp_running}"
  collect_status_map "${tmp_status_map}" "${tmp_running}"
  build_snapshot "${tmp_status_map}" "${tmp_running}" "${OUT_TSV}"
  rm -f "${tmp_status_map}" "${tmp_running}"
}

if [[ "${WATCH_SEC}" -gt 0 ]]; then
  while true; do
    snapshot_once
    if [[ -t 1 ]]; then
      clear
    fi
    if [[ "${MINIMAL_MODE}" == "1" ]]; then
      render_minimal_dashboard "${OUT_TSV}"
    else
      render_dashboard "${OUT_TSV}"
    fi
    sleep "${WATCH_SEC}"
  done
else
  snapshot_once
  if [[ "${MINIMAL_MODE}" == "1" ]]; then
    render_minimal_dashboard "${OUT_TSV}"
  else
    render_dashboard "${OUT_TSV}"
  fi
fi
