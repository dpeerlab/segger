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
  --no-color         Disable ANSI colors
  -h, --help         Show this help
EOF
}

ROOT="./results/mossi_main_big_benchmark_nightly"
OUT_TSV=""
WATCH_SEC=0
NO_COLOR=0

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
VALIDATION_TSV="${SUMMARY_DIR}/validation_metrics.tsv"
if [[ ! -f "${VALIDATION_TSV}" ]] && [[ -f "${ROOT}/validation_metrics.tsv" ]]; then
  VALIDATION_TSV="${ROOT}/validation_metrics.tsv"
fi

if [[ ! -f "${PLAN_FILE}" ]]; then
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

  if [[ -n "${running_jobs_file}" ]] && [[ -s "${running_jobs_file}" ]]; then
    include_recovery=0
  fi

  for f in "${SUMMARY_DIR}"/gpu*.tsv; do
    [[ -f "${f}" ]] || continue
    status_files+=("${f}")
  done
  if [[ "${include_recovery}" == "1" ]] && [[ -f "${SUMMARY_DIR}/recovery.tsv" ]]; then
    status_files+=("${SUMMARY_DIR}/recovery.tsv")
  fi

  if [[ "${#status_files[@]}" -eq 0 ]]; then
    : > "${out_file}"
    return 0
  fi

  awk -F'\t' '
    FNR == 1 { next }
    {
      job = $1
      gpu = $2
      status = $3
      elapsed = $4
      note = ""
      seg = ""
      log_path = ""
      if (NF >= 7) {
        note = $5
        seg = $6
        log_path = $7
      } else if (NF >= 6) {
        seg = $5
        log_path = $6
      }

      if (note == "") {
        note = "-"
      }
      gpu_map[job] = gpu
      status_map[job] = status
      elapsed_map[job] = elapsed
      note_map[job] = note
      seg_map[job] = seg
      log_map[job] = log_path
    }
    END {
      for (job in status_map) {
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
          job,
          gpu_map[job],
          status_map[job],
          elapsed_map[job],
          note_map[job],
          seg_map[job],
          log_map[job]
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

  pgrep -af 'segger segment|segger predict' 2>/dev/null \
    | awk '
      {
        for (i = 1; i <= NF; i++) {
          if ($i == "-o" && (i + 1) <= NF) {
            out = $(i + 1)
            gsub(/\/+$/, "", out)
            n = split(out, a, "/")
            if (n > 0) {
              print a[n]
            }
          }
        }
      }
    ' \
    | sed '/^$/d' \
    | sort -u > "${out_file}" || : > "${out_file}"
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

build_snapshot() {
  local status_map="$1"
  local running_jobs="$2"
  local out_file="$3"
  local tmp_file
  local plan_header plan_has_study_block
  tmp_file="$(mktemp)"

  plan_header="$(head -n 1 "${PLAN_FILE}")"
  plan_has_study_block=0
  if printf '%s\n' "${plan_header}" | tr '\t' '\n' | grep -Fxq "study_block"; then
    plan_has_study_block=1
  fi

  printf "job\tgroup\tgpu\tstatus\tstate\trunning\telapsed_s\trun_count\thad_rerun\thad_anc_retry\thad_predict_fallback\thad_recovery_pass\tseg_exists\tanndata_exists\txenium_exists\tseg_dir\tlog_file\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\tnote\n" > "${tmp_file}"

  tail -n +2 "${PLAN_FILE}" | while IFS=$'\t' read -r -a cols; do
    local job group use_3d expansion tx_max_k tx_max_dist n_mid_layers n_heads cells_min_counts min_qv alignment_loss
    local row gpu status elapsed note seg_dir log_file
    local running seg_exists anndata_exists xenium_exists
    local run_count had_rerun had_anc_retry had_predict_fallback had_recovery_pass
    local state

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
    if [[ -n "${row}" ]]; then
      IFS=$'\t' read -r _ gpu status elapsed note seg_dir log_file <<< "${row}"
      if [[ "${note}" == "-" ]]; then
        note=""
      fi
    fi

    [[ -n "${seg_dir}" ]] || seg_dir="${RUNS_DIR}/${job}"
    [[ -n "${log_file}" ]] || log_file="$(pick_log_file "${job}")"
    [[ -n "${log_file}" ]] || log_file="${LOGS_DIR}/${job}.gpu?.log"

    running=0
    if [[ -s "${running_jobs}" ]] && grep -Fxq "${job}" "${running_jobs}"; then
      running=1
    fi

    seg_exists=0
    anndata_exists=0
    xenium_exists=0
    [[ -f "${RUNS_DIR}/${job}/segger_segmentation.parquet" ]] && seg_exists=1
    [[ -f "${EXPORTS_DIR}/${job}/anndata/segger_segmentation.h5ad" ]] && anndata_exists=1
    [[ -f "${EXPORTS_DIR}/${job}/xenium_explorer/seg_experiment.xenium" ]] && xenium_exists=1

    run_count=0
    had_rerun=0
    had_anc_retry=0
    had_predict_fallback=0
    had_recovery_pass=0
    if [[ -f "${log_file}" ]]; then
      run_count="$(grep -c "START job=${job}" "${log_file}" 2>/dev/null || printf '0')"
      if [[ "${run_count}" -gt 1 ]]; then
        had_rerun=1
      fi
      grep -q "segment failed with ancdata; retrying" "${log_file}" 2>/dev/null && had_anc_retry=1 || true
      grep -q "predict fallback succeeded after OOM" "${log_file}" 2>/dev/null && had_predict_fallback=1 || true
      grep -q "RECOVERY job=${job}" "${log_file}" 2>/dev/null && had_recovery_pass=1 || true
    fi

    state="pending"
    if [[ "${running}" == "1" ]]; then
      state="running"
    elif [[ "${seg_exists}" == "1" && "${anndata_exists}" == "1" && "${xenium_exists}" == "1" ]]; then
      state="done"
    elif [[ -n "${status}" ]]; then
      case "${status}" in
        ok|skipped_existing|recovered_predict_ok)
          state="partial"
          ;;
        *)
          state="failed"
          ;;
      esac
    else
      if [[ "${seg_exists}" == "1" || "${anndata_exists}" == "1" || "${xenium_exists}" == "1" ]]; then
        state="partial"
      else
        state="pending"
      fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job}" "${group}" "${gpu}" "${status}" "${state}" "${running}" "${elapsed}" \
      "${run_count}" "${had_rerun}" "${had_anc_retry}" "${had_predict_fallback}" "${had_recovery_pass}" \
      "${seg_exists}" "${anndata_exists}" "${xenium_exists}" "${seg_dir}" "${log_file}" \
      "${use_3d}" "${expansion}" "${tx_max_k}" "${tx_max_dist}" "${n_mid_layers}" "${n_heads}" \
      "${cells_min_counts}" "${min_qv}" "${alignment_loss}" "${note}" \
      >> "${tmp_file}"
  done

  mv "${tmp_file}" "${out_file}"
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
  local oom_count oot_count anc_count rerun_count recovered_count processed
  local now

  total="$(awk 'END { print NR-1 }' "${snapshot}")"
  done_count="$(awk -F'\t' 'NR>1 && $5=="done" {c++} END{print c+0}' "${snapshot}")"
  running_count="$(awk -F'\t' 'NR>1 && $5=="running" {c++} END{print c+0}' "${snapshot}")"
  pending_count="$(awk -F'\t' 'NR>1 && $5=="pending" {c++} END{print c+0}' "${snapshot}")"
  partial_count="$(awk -F'\t' 'NR>1 && $5=="partial" {c++} END{print c+0}' "${snapshot}")"
  failed_count="$(awk -F'\t' 'NR>1 && $5=="failed" {c++} END{print c+0}' "${snapshot}")"

  oom_count="$(awk -F'\t' 'NR>1 && $4 ~ /oom/ {c++} END{print c+0}' "${snapshot}")"
  oot_count="$(awk -F'\t' 'NR>1 && $4=="segment_oot" {c++} END{print c+0}' "${snapshot}")"
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
  printf "oom=%s  oot=%s  ancdata=%s  rerun=%s  recovered=%s\n" "${oom_count}" "${oot_count}" "${anc_count}" "${rerun_count}" "${recovered_count}"

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
    BEGIN {
      print "rank\tjob\tgroup\tgpu\tstate\tstatus\truns\telapsed_min\trerun\tanc_retry\toom_pred_fallback\trecovery\tseg\tanndata\txenium"
    }
    NR > 1 {
      st = $5
      if (st == "") st = "pending"
      status = $4
      if (status == "") status = "<none>"
      print rank(st) "\t" $1 "\t" $2 "\t" $3 "\t" st "\t" status "\t" $8 "\t" fmt_minutes($7) "\t" $9 "\t" $10 "\t" $11 "\t" $12 "\t" $13 "\t" $14 "\t" $15
    }
  ' "${snapshot}" \
    | {
        IFS= read -r header || true
        if [[ -n "${header}" ]]; then
          printf "job\tgroup\tgpu\tstate\tstatus\truns\telapsed_min\trerun\tanc_retry\toom_pred_fallback\trecovery\tseg\tanndata\txenium\n"
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
      print "rank\tjob\tmodel\tstudy_block\tgroup\tstate\tstatus\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss"
      next
    }
    {
      job = col("job", $1)
      block = has_block ? col("study_block", "-") : "-"
      group = col("group", "-")
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
            col("alignment_loss", "-")
    }
  ' "${snapshot}" "${PLAN_FILE}" \
    | {
        IFS= read -r header || true
        if [[ -n "${header}" ]]; then
          printf "job\tmodel\tstudy_block\tgroup\tstate\tstatus\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\n"
        fi
        sort -t $'\t' -k1,1n -k2,2 \
          | cut -f2-
      } \
    | colorize_rows_by_state_column 5 \
    | render_tsv_table

  echo
  echo "Validation Metrics:"
  if [[ -f "${VALIDATION_TSV}" ]] && [[ "$(awk 'END { print NR-1 }' "${VALIDATION_TSV}")" -gt 0 ]]; then
    awk -F'\t' '
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
      FNR == NR {
        if (FNR == 1) {
          for (i = 1; i <= NF; i++) {
            if ($i == "job") snap_job_col = i
            if ($i == "state") snap_state_col = i
            if ($i == "elapsed_s") snap_elapsed_col = i
          }
          next
        }
        if (snap_job_col > 0) {
          job_key = $snap_job_col
          if (snap_state_col > 0) state_by_job[job_key] = $snap_state_col
          if (snap_elapsed_col > 0) elapsed_by_job[job_key] = $snap_elapsed_col
        }
        next
      }
      FNR == 1 {
        for (i = 1; i <= NF; i++) {
          idx[$i] = i
        }
        print "job\tkind\tstate\tvalidate_status\tgpu_time_min v\tcells\tassigned_pct ^\tmecr v\tcontamination_pct v\tresolvi_contam_pct v\ttco ^\tdoublet_pct v"
        next
      }
      {
        job = get_col("job")
        job_disp = job
        group = get_col("group")
        is_reference = get_col("is_reference")
        reference_kind = get_col("reference_kind")
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

        gpu_time = get_col("gpu_time_s")
        if (gpu_time == "") gpu_time = get_col("elapsed_s")
        if (gpu_time == "") gpu_time = elapsed_by_job[job]
        gpu_time = fmt_minutes(gpu_time)

        cells = get_col("cells")
        if (cells == "") cells = get_col("cells_total")
        if (cells == "") cells = get_col("cells_assigned")
        cells = fmt_nonneg_int(cells)

        assigned = get_col("assigned_pct")
        if (assigned == "") assigned = get_col("transcripts_assigned_pct")

        mecr = get_col("mecr")
        if (mecr == "") mecr = get_col("mecr_fast")

        contamination = get_col("contamination_pct")
        if (contamination == "") contamination = get_col("border_contaminated_cells_pct_fast")

        resolvi = get_col("resolvi_contamination_pct")
        if (resolvi == "") resolvi = get_col("resolvi_contamination_pct_fast")

        tco = get_col("tco")
        if (tco == "") tco = get_col("transcript_centroid_offset_fast")

        doublet = get_col("doublet_pct")
        if (doublet == "") {
          doublet = get_col("signal_doublet_like_fraction_fast")
          if (doublet != "" && tolower(doublet) != "nan" && tolower(doublet) != "none") {
            doublet = 100.0 * (doublet + 0.0)
          }
        }

        print job_disp "\t" kind "\t" state "\t" validate_status "\t" gpu_time "\t" cells "\t" \
              fmt_float(assigned) "\t" fmt_float(mecr) "\t" fmt_float(contamination) "\t" \
              fmt_float(resolvi) "\t" fmt_float(tco) "\t" fmt_float(doublet)
      }
    ' "${snapshot}" "${VALIDATION_TSV}" \
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
              if (!have_2) return (x == best1)
              return (x >= best2)
            }
            function is_top2_down(v, best1, best2, have_2,    x) {
              if (!is_num(v) || !is_num(best1)) return 0
              x = to_num(v)
              if (!have_2) return (x == best1)
              return (x <= best2)
            }
            function norm_up(v, lo, hi) {
              if (!is_num(v)) return ""
              if (hi <= lo) return 1.0
              return (to_num(v) - lo) / (hi - lo)
            }
            function norm_down(v, lo, hi) {
              if (!is_num(v)) return ""
              if (hi <= lo) return 1.0
              return (hi - to_num(v)) / (hi - lo)
            }
            {
              n++
              for (j = 1; j <= NF; j++) {
                cell[n, j] = $j
              }
              nf[n] = NF
              m_assigned[n] = $7
              m_mecr[n] = $8
              m_contam[n] = $9
              m_resolvi[n] = $10
              m_tco[n] = $11
              m_doublet[n] = $12
              m_gpu[n] = $5
              st[n] = tolower($4)
              is_ref[n] = (tolower($3) == "reference")

              if (is_num($7)) {
                v = to_num($7)
                if (!has_a || v < min_a) min_a = v
                if (!has_a || v > max_a) max_a = v
                has_a = 1
              }
              if (is_num($8)) {
                v = to_num($8)
                if (!has_m || v < min_m) min_m = v
                if (!has_m || v > max_m) max_m = v
                has_m = 1
              }
              if (is_num($9)) {
                v = to_num($9)
                if (!has_c || v < min_c) min_c = v
                if (!has_c || v > max_c) max_c = v
                has_c = 1
              }
              if (is_num($10)) {
                v = to_num($10)
                if (!has_r || v < min_r) min_r = v
                if (!has_r || v > max_r) max_r = v
                has_r = 1
              }
              if (is_num($11)) {
                v = to_num($11)
                if (!has_t || v < min_t) min_t = v
                if (!has_t || v > max_t) max_t = v
                has_t = 1
              }
              if (is_num($12)) {
                v = to_num($12)
                if (!has_d || v < min_d) min_d = v
                if (!has_d || v > max_d) max_d = v
                has_d = 1
              }

              if (st[n] == "ok") {
                have1 = have_a_best1; top1 = a_best1; have2 = have_a_best2; top2 = a_best2
                update_top2_up($7)
                have_a_best1 = have1; a_best1 = top1; have_a_best2 = have2; a_best2 = top2

                have1 = have_m_best1; top1 = m_best1; have2 = have_m_best2; top2 = m_best2
                update_top2_down($8)
                have_m_best1 = have1; m_best1 = top1; have_m_best2 = have2; m_best2 = top2

                have1 = have_c_best1; top1 = c_best1; have2 = have_c_best2; top2 = c_best2
                update_top2_down($9)
                have_c_best1 = have1; c_best1 = top1; have_c_best2 = have2; c_best2 = top2

                have1 = have_r_best1; top1 = r_best1; have2 = have_r_best2; top2 = r_best2
                update_top2_down($10)
                have_r_best1 = have1; r_best1 = top1; have_r_best2 = have2; r_best2 = top2

                have1 = have_t_best1; top1 = t_best1; have2 = have_t_best2; top2 = t_best2
                update_top2_up($11)
                have_t_best1 = have1; t_best1 = top1; have_t_best2 = have2; t_best2 = top2

                have1 = have_d_best1; top1 = d_best1; have2 = have_d_best2; top2 = d_best2
                update_top2_down($12)
                have_d_best1 = have1; d_best1 = top1; have_d_best2 = have2; d_best2 = top2

                if (!is_ref[n]) {
                  have1 = have_g_best1; top1 = g_best1; have2 = have_g_best2; top2 = g_best2
                  update_top2_down($5)
                  have_g_best1 = have1; g_best1 = top1; have_g_best2 = have2; g_best2 = top2
                }
              }
            }
            END {
              for (i = 1; i <= n; i++) {
                # Rank rows by overall score across assigned/mecr/contam/tco/doublet.
                # Rows with no numeric metrics (or non-ok status) stay at the bottom.
                if (st[i] != "ok") {
                  score = -1e9
                } else {
                  score = 0.0
                  cnt = 0

                  s = norm_up(m_assigned[i], min_a, max_a)
                  if (s != "") { score += s; cnt++ }

                  s = norm_down(m_mecr[i], min_m, max_m)
                  if (s != "") { score += s; cnt++ }

                  s = norm_down(m_contam[i], min_c, max_c)
                  if (s != "") { score += s; cnt++ }

                  s = norm_up(m_tco[i], min_t, max_t)
                  if (s != "") { score += s; cnt++ }

                  s = norm_down(m_doublet[i], min_d, max_d)
                  if (s != "") { score += s; cnt++ }

                  if (cnt > 0) {
                    score /= cnt
                  } else {
                    score = -1e9
                  }
                }

                if (b_on != "" && st[i] == "ok") {
                  if (!is_ref[i] && is_top2_down(m_gpu[i], g_best1, g_best2, have_g_best2)) cell[i, 5] = b_on cell[i, 5] b_off
                  if (is_top2_up(m_assigned[i], a_best1, a_best2, have_a_best2)) cell[i, 7] = b_on cell[i, 7] b_off
                  if (is_top2_down(m_mecr[i], m_best1, m_best2, have_m_best2)) cell[i, 8] = b_on cell[i, 8] b_off
                  if (is_top2_down(m_contam[i], c_best1, c_best2, have_c_best2)) cell[i, 9] = b_on cell[i, 9] b_off
                  if (is_top2_down(m_resolvi[i], r_best1, r_best2, have_r_best2)) cell[i, 10] = b_on cell[i, 10] b_off
                  if (is_top2_up(m_tco[i], t_best1, t_best2, have_t_best2)) cell[i, 11] = b_on cell[i, 11] b_off
                  if (is_top2_down(m_doublet[i], d_best1, d_best2, have_d_best2)) cell[i, 12] = b_on cell[i, 12] b_off
                }

                row = cell[i, 1]
                for (j = 2; j <= nf[i]; j++) {
                  row = row "\t" cell[i, j]
                }
                gkey = 1
                if (tolower(cell[i, 3]) == "reference") gkey = 0
                printf "%d\t%.10f\t%s\n", gkey, score, row
              }
            }
          ' \
            | sort -t $'\t' -k1,1n -k2,2gr -k3,3 \
            | cut -f3- \
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
      NR > 1 && $6 == "1" {
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
    render_dashboard "${OUT_TSV}"
    sleep "${WATCH_SEC}"
  done
else
  snapshot_once
  render_dashboard "${OUT_TSV}"
fi
