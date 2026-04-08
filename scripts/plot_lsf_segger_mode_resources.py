#!/usr/bin/env python3
"""Plot Segger LSF runtime and memory by mode against transcript counts.

This script parses LSF `*.attempt*.out` logs produced by benchmark runs,
extracts per-job resource summaries, and renders a two-panel figure:

1) Runtime (minutes) vs number of transcripts
2) Max RAM (GB) vs number of transcripts

Defaults are tuned for the benchmark folders present in this workspace.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from plot_reference_aesthetics import apply_reference_rcparams


# Fallback transcript counts (used only if segbench overview values are unavailable).
FALLBACK_DATASET_TRANSCRIPTS = {
    "xenium_crc": 35_042_827.0,
    "xenium_v1_colon": 65_047_433.0,
    "xenium_breast": 92_810_244.0,
    "xenium_v1_breast": 995_676.0,
    "xenium_nsclc": 147_069_015.0,
    "xenium_mouse_brain": 186_756_695.0,
    "xenium_mouse_liver": 178_711_670.0,
    "merscope_mouse_brain": 70_514_304.0,
    "cosmx_human_pancreas": 64_960_677.0,
}

DATASET_LABELS = {
    "xenium_crc": "Xenium Human CRC",
    "xenium_v1_colon": "Xenium Human Colon",
    "xenium_breast": "Xenium Human Breast",
    "xenium_v1_breast": "Xenium 5K Human Breast",
    "xenium_nsclc": "Xenium Lung Cancer",
    "xenium_mouse_brain": "Xenium 5K Mouse Brain",
    "xenium_mouse_liver": "Xenium Mouse Liver",
    "merscope_mouse_brain": "MERSCOPE Mouse Brain",
    "cosmx_human_pancreas": "CosMX Human Pancreas",
}

MODE_ORDER = [
    "baseline",
    "use3d_true",
    "pred_frag_off",
    "pred_frag_on",
    "pred_sf1p2_fragoff",
    "pred_sf1p2_fragon",
    "pred_sf2p2_fragoff",
    "pred_sf2p2_fragon",
    "pred_sf3p2_fragoff",
    "pred_sf3p2_fragon",
    "align_combined",
    "align_0p01",
    "align_0p03",
    "align_0p10",
]

MODE_LABELS = {
    "baseline": "Baseline",
    "use3d_true": "Use3D",
    "pred_frag_off": "Frag. off (baseline)",
    "pred_frag_on": "Frag. on",
    "pred_sf1p2_fragoff": "Pred SF1.2 (frag off)",
    "pred_sf1p2_fragon": "Pred SF1.2 (frag on)",
    "pred_sf2p2_fragoff": "Pred SF2.2 (frag off)",
    "pred_sf2p2_fragon": "Pred SF2.2 (frag on)",
    "pred_sf3p2_fragoff": "Pred SF3.2 (frag off)",
    "pred_sf3p2_fragon": "Pred SF3.2 (frag on)",
    "align_combined": "Alignment (avg)",
    "align_0p01": "Align 0.01",
    "align_0p03": "Align 0.03",
    "align_0p10": "Align 0.10",
}

MODE_COLORS = {
    "baseline": "#0077BB",
    "use3d_true": "#33BBEE",
    "pred_frag_off": "#009988",
    "pred_frag_on": "#EE3377",
    "pred_sf1p2_fragoff": "#009988",
    "pred_sf2p2_fragoff": "#22AA99",
    "pred_sf3p2_fragoff": "#117733",
    "pred_sf1p2_fragon": "#EE7733",
    "pred_sf2p2_fragon": "#CC3311",
    "pred_sf3p2_fragon": "#EE3377",
    "align_combined": "#111111",
    "align_0p01": "#882255",
    "align_0p03": "#AA4499",
    "align_0p10": "#332288",
}

MODE_MARKERS = {
    "baseline": "o",
    "use3d_true": "o",
    "pred_frag_off": "o",
    "pred_frag_on": "o",
    "pred_sf1p2_fragoff": "o",
    "pred_sf2p2_fragoff": "o",
    "pred_sf3p2_fragoff": "o",
    "pred_sf1p2_fragon": "o",
    "pred_sf2p2_fragon": "o",
    "pred_sf3p2_fragon": "o",
    "align_combined": "o",
    "align_0p01": "o",
    "align_0p03": "o",
    "align_0p10": "o",
}

MODE_LINESTYLES = {
    "baseline": "-",
    "use3d_true": "-",
    "pred_frag_off": "-",
    "pred_frag_on": "-",
    "pred_sf1p2_fragoff": "-",
    "pred_sf2p2_fragoff": "-",
    "pred_sf3p2_fragoff": "-",
    "pred_sf1p2_fragon": "-",
    "pred_sf2p2_fragon": "-",
    "pred_sf3p2_fragon": "-",
    "align_combined": "-",
    "align_0p01": "-",
    "align_0p03": "-",
    "align_0p10": "-",
}

ALIGN_VARIANT_MODES = ("align_0p01", "align_0p03", "align_0p10")
PRED_FRAG_OFF_MODES = ("pred_sf1p2_fragoff", "pred_sf2p2_fragoff", "pred_sf3p2_fragoff")
PRED_FRAG_ON_MODES = ("pred_sf1p2_fragon", "pred_sf2p2_fragon", "pred_sf3p2_fragon")

SEG_BENCH_ID_TO_DATASET_KEY = {
    "xenium_v1_colon": "xenium_v1_colon",
    "xenium_nscls": "xenium_nsclc",
    "xenium_mouse_liver": "xenium_mouse_liver",
    "xenium_CRC": "xenium_crc",
    "xenium_breast": "xenium_breast",
    "xenium_mouse_brain": "xenium_mouse_brain",
    "MERSCOPE_brain": "merscope_mouse_brain",
    "CosMx_pancreas": "cosmx_human_pancreas",
}

DATASET_ICON_FILES = {
    "xenium_crc": "icon_colon_v2.png",
    "xenium_v1_colon": "icon_colon_v2.png",
    "xenium_breast": "icon_breast_small.png",
    "xenium_v1_breast": "icon_breast_small.png",
    "xenium_nsclc": "icon_lung_v2.png",
    "xenium_mouse_brain": "icon_brain_small.png",
    "xenium_mouse_liver": "icon_liver_small.png",
    "merscope_mouse_brain": "icon_brain_small.png",
    "cosmx_human_pancreas": "icon_pancreas_small.png",
}

DATASET_ICON_ZOOM_SCALE = {
    "cosmx_human_pancreas": 1.15,
}

DATASET_ICON_PLATFORM_COLORS = {
    # From segbench dataset_overview.html platformColors.
    "xenium_crc": "#7CB77E",            # xenium_segkit
    "xenium_v1_colon": "#6BA3D4",       # xenium_v1
    "xenium_nsclc": "#6BA3D4",          # xenium_v1
    "xenium_mouse_liver": "#6BA3D4",    # xenium_v1
    "xenium_breast": "#B8A3D4",         # xenium_prime5k
    "xenium_v1_breast": "#B8A3D4",      # xenium_prime5k
    "xenium_mouse_brain": "#B8A3D4",    # xenium_prime5k
    "merscope_mouse_brain": "#D47B8A",  # merscope
    "cosmx_human_pancreas": "#D4A574",  # cosmx
}

SUBJECT_RE = re.compile(r"^Subject: Job\s+(?P<job_id>\d+):\s+<[^>]+>\s+in cluster <[^>]+>\s+(?P<state>Done|Exited)\s*$")
STARTED_RE = re.compile(r"^Started at\s+(?P<value>.+)$")
TERMINATED_RE = re.compile(r"^Terminated at\s+(?P<value>.+)$")
RUN_TIME_RE = re.compile(r"^\s*Run time\s*:\s*(?P<value>[0-9.]+)\s*sec\.?\s*$")
MAX_MEMORY_RE = re.compile(r"^\s*Max Memory\s*:\s*(?P<value>[0-9.]+)\s*MB\s*$")
SEGGER_RC_RE = re.compile(r"^\[JOB\]\s+segger_rc=(?P<value>-?\d+)\s*$")
JOB_QUEUE_RE = re.compile(
    r"^\[JOB\]\s+queue=(?P<queue>\S+)\s+gmem=(?P<gmem>[0-9]+(?:\.[0-9]+)?)G\s+mem=(?P<mem>[0-9]+(?:\.[0-9]+)?)G\s+wall=(?P<wall>[0-9:]+)\s*$"
)
GPU_SNAPSHOT_RE = re.compile(
    r"^(?P<gpu_name>[^,]+),\s*(?P<total_mb>[0-9]+(?:\.[0-9]+)?),\s*(?P<used_mb>[0-9]+(?:\.[0-9]+)?)\s*$"
)
OOM_PROCESS_MEM_IN_USE_RE = re.compile(
    r"this process has\s+(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>GiB|MiB)\s+memory in use",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    mode: str
    source_root: str
    source_log: str
    lsf_job_id: str
    lsf_exit_state: str
    completed: bool
    started_at: datetime | None
    terminated_at: datetime | None
    run_time_s: float
    max_memory_mb: float
    max_vram_mb: float | None
    segger_rc: int | None

    def rank_key(self) -> tuple[int, datetime, datetime]:
        return (
            1 if self.completed else 0,
            self.terminated_at or datetime.min,
            self.started_at or datetime.min,
        )


@dataclass(frozen=True)
class AggregatedRecord:
    dataset: str
    mode: str
    runtime_s: float
    max_memory_mb: float
    max_vram_mb: float | None
    n_runs: int
    source_roots: str
    source_logs_count: int


@dataclass(frozen=True)
class VramEstimate:
    worked_max_mb: float | None
    failed_max_mb: float | None
    estimate_mb: float | None
    worked_n: int
    failed_n: int


def parse_lsf_datetime(value: str) -> datetime | None:
    cleaned = " ".join(value.strip().split())
    try:
        return datetime.strptime(cleaned, "%a %b %d %H:%M:%S %Y")
    except ValueError:
        return None


def _to_mb(value: float, unit: str) -> float:
    unit_norm = unit.strip().lower()
    if unit_norm == "gib":
        return value * 1024.0
    if unit_norm == "mib":
        return value
    return value


def parse_oom_vram_mb(err_text: str) -> float | None:
    """Best-effort VRAM peak from stderr OOM traces.

    Example line:
    "... this process has 39.43 GiB memory in use ..."
    """
    values_mb: list[float] = []
    for match in OOM_PROCESS_MEM_IN_USE_RE.finditer(err_text or ""):
        value = float(match.group("value"))
        unit = match.group("unit")
        values_mb.append(_to_mb(value, unit))
    if not values_mb:
        return None
    return max(values_mb)


def _safe_float(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"-", "nan", "None", "none", "NULL", "null"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def discover_status_snapshot_paths(
    roots: list[Path],
    explicit_paths: list[Path] | None,
) -> list[Path]:
    candidates: list[Path] = []
    if explicit_paths:
        candidates.extend(explicit_paths)
    else:
        for root in roots:
            candidates.append(root / "summaries" / "aggregate_status_snapshot.tsv")

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_file():
            continue
        if resolved in seen:
            continue
        out.append(resolved)
        seen.add(resolved)
    return out


def _infer_dataset_from_status_path(path: Path) -> str | None:
    # Supports .../datasets/<dataset>/summaries/status_snapshot.tsv inputs.
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "datasets" and (idx + 2) < len(parts):
            return parts[idx + 1]
    return None


def _status_row_vram_mb(
    row: dict[str, str],
    *,
    fallback_to_requested_gmem: bool,
) -> float | None:
    max_vram_mb = _safe_float(row.get("max_vram_mb"))
    if max_vram_mb is not None and max_vram_mb > 0:
        return float(max_vram_mb)

    max_vram_gb = _safe_float(row.get("max_vram_gb"))
    if max_vram_gb is not None and max_vram_gb > 0:
        return float(max_vram_gb * 1024.0)

    if not fallback_to_requested_gmem:
        return None

    requested_gmem_gb = _safe_float(row.get("requested_gmem_gb"))
    if requested_gmem_gb is not None and requested_gmem_gb > 0:
        return float(requested_gmem_gb * 1024.0)
    return None


def load_status_snapshot_vram_mb(
    paths: list[Path],
    *,
    fallback_to_requested_gmem: bool,
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for path in paths:
        dataset_hint = _infer_dataset_from_status_path(path)
        try:
            with path.open("r", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    mode = str(row.get("job", "")).strip() or str(row.get("mode", "")).strip()
                    if not mode:
                        continue

                    dataset = str(row.get("dataset", "")).strip()
                    if not dataset and dataset_hint is not None:
                        dataset = dataset_hint
                    if not dataset:
                        continue

                    max_vram_mb = _status_row_vram_mb(
                        row,
                        fallback_to_requested_gmem=fallback_to_requested_gmem,
                    )
                    if max_vram_mb is None or max_vram_mb <= 0:
                        continue

                    key = (dataset, mode)
                    prev = out.get(key)
                    if prev is None or max_vram_mb > prev:
                        out[key] = float(max_vram_mb)
        except OSError:
            continue
    return out


def is_success_record(record: RunRecord) -> bool:
    return record.completed and (record.segger_rc is None or record.segger_rc == 0)


def estimate_vram_from_worked_failed(
    records: list[RunRecord],
) -> dict[tuple[str, str], VramEstimate]:
    grouped: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        if record.max_vram_mb is None:
            continue
        grouped[(record.dataset, record.mode)].append(record)

    out: dict[tuple[str, str], VramEstimate] = {}
    for key, recs in grouped.items():
        worked_values = [float(rec.max_vram_mb) for rec in recs if is_success_record(rec) and rec.max_vram_mb is not None]
        failed_values = [float(rec.max_vram_mb) for rec in recs if (not is_success_record(rec)) and rec.max_vram_mb is not None]
        worked_max = max(worked_values) if worked_values else None
        failed_max = max(failed_values) if failed_values else None
        if worked_max is not None and failed_max is not None:
            estimate = (worked_max + failed_max) / 2.0
        elif worked_max is not None:
            estimate = worked_max
        elif failed_max is not None:
            estimate = failed_max
        else:
            estimate = None
        out[key] = VramEstimate(
            worked_max_mb=worked_max,
            failed_max_mb=failed_max,
            estimate_mb=estimate,
            worked_n=len(worked_values),
            failed_n=len(failed_values),
        )
    return out


def iter_lsf_blocks(lines: Iterable[str]) -> Iterable[list[str]]:
    block: list[str] = []
    for line in lines:
        if line.startswith("Sender: LSF System"):
            if block:
                yield block
            block = [line]
        elif block:
            block.append(line)
    if block:
        yield block


def parse_block(
    dataset: str,
    mode: str,
    source_root: Path,
    source_log: Path,
    block: list[str],
    err_text: str,
    vram_fallback: str,
) -> RunRecord | None:
    job_id: str | None = None
    exit_state = ""
    completed = False
    started_at: datetime | None = None
    terminated_at: datetime | None = None
    run_time_s: float | None = None
    max_memory_mb: float | None = None
    requested_gmem_gb: float | None = None
    gpu_used_mb: float | None = None
    segger_rc: int | None = None

    for line in block:
        subject_match = SUBJECT_RE.match(line)
        if subject_match:
            job_id = subject_match.group("job_id")
            exit_state = subject_match.group("state")
            continue

        started_match = STARTED_RE.match(line)
        if started_match:
            started_at = parse_lsf_datetime(started_match.group("value"))
            continue

        terminated_match = TERMINATED_RE.match(line)
        if terminated_match:
            terminated_at = parse_lsf_datetime(terminated_match.group("value"))
            continue

        run_time_match = RUN_TIME_RE.match(line)
        if run_time_match:
            run_time_s = float(run_time_match.group("value"))
            continue

        max_memory_match = MAX_MEMORY_RE.match(line)
        if max_memory_match:
            max_memory_mb = float(max_memory_match.group("value"))
            continue

        queue_match = JOB_QUEUE_RE.match(line.strip())
        if queue_match:
            requested_gmem_gb = float(queue_match.group("gmem"))
            continue

        gpu_snapshot_match = GPU_SNAPSHOT_RE.match(line.strip())
        if gpu_snapshot_match:
            gpu_used_mb = float(gpu_snapshot_match.group("used_mb"))
            continue

        segger_rc_match = SEGGER_RC_RE.match(line)
        if segger_rc_match:
            segger_rc = int(segger_rc_match.group("value"))
            continue

        if "Successfully completed." in line:
            completed = True
            continue

        if line.strip() == "[JOB] completed":
            completed = True

    if job_id is None or run_time_s is None or max_memory_mb is None:
        return None

    oom_vram_mb = parse_oom_vram_mb(err_text)
    max_vram_mb: float | None = None
    if oom_vram_mb is not None and oom_vram_mb > 0:
        max_vram_mb = oom_vram_mb
    elif gpu_used_mb is not None and gpu_used_mb > 0:
        max_vram_mb = gpu_used_mb
    elif vram_fallback == "gmem" and requested_gmem_gb is not None and requested_gmem_gb > 0:
        # Optional fallback only: gmem is a reservation/limit, not measured usage.
        max_vram_mb = requested_gmem_gb * 1024.0

    return RunRecord(
        dataset=dataset,
        mode=mode,
        source_root=str(source_root),
        source_log=str(source_log),
        lsf_job_id=job_id,
        lsf_exit_state=exit_state,
        completed=completed,
        started_at=started_at,
        terminated_at=terminated_at,
        run_time_s=run_time_s,
        max_memory_mb=max_memory_mb,
        max_vram_mb=max_vram_mb,
        segger_rc=segger_rc,
    )


def collect_records(
    roots: list[Path],
    datasets: set[str],
    *,
    vram_fallback: str,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for root in roots:
        datasets_root = root / "datasets"
        if not datasets_root.exists():
            continue
        for dataset_dir in sorted(datasets_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            if dataset not in datasets:
                continue

            logs_dir = dataset_dir / "logs"
            if not logs_dir.exists():
                continue

            for out_log in sorted(logs_dir.glob("*.attempt*.out")):
                mode = out_log.name.split(".attempt", maxsplit=1)[0]
                try:
                    lines = out_log.read_text(errors="ignore").splitlines()
                except OSError:
                    continue
                err_log = out_log.with_suffix(".err")
                try:
                    err_text = err_log.read_text(errors="ignore")
                except OSError:
                    err_text = ""

                for block in iter_lsf_blocks(lines):
                    parsed = parse_block(
                        dataset,
                        mode,
                        root,
                        out_log,
                        block,
                        err_text,
                        vram_fallback=vram_fallback,
                    )
                    if parsed is not None:
                        records.append(parsed)
    return records


def aggregate_records(
    records: list[RunRecord],
    *,
    statistic: str,
    runtime_statistic: str | None,
    include_incomplete: bool,
) -> dict[tuple[str, str], AggregatedRecord]:
    runtime_stat = runtime_statistic or statistic
    grouped: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        if not include_incomplete:
            if not record.completed:
                continue
            if record.segger_rc is not None and record.segger_rc != 0:
                continue
        grouped[(record.dataset, record.mode)].append(record)

    out: dict[tuple[str, str], AggregatedRecord] = {}
    for key, recs in grouped.items():
        latest: RunRecord | None = None
        if runtime_stat == "latest" or statistic == "latest":
            latest = max(recs, key=lambda rec: rec.rank_key())

        runtimes = [rec.run_time_s for rec in recs]
        memories = [rec.max_memory_mb for rec in recs]
        vrams = [rec.max_vram_mb for rec in recs if rec.max_vram_mb is not None]

        if runtime_stat == "latest":
            runtime_s = float(latest.run_time_s) if latest is not None else float(max(runtimes))
        elif runtime_stat == "min":
            runtime_s = float(min(runtimes))
        elif runtime_stat == "median":
            runtime_s = float(statistics.median(runtimes))
        else:
            runtime_s = float(sum(runtimes) / len(runtimes))

        if statistic == "latest":
            max_memory_mb = float(latest.max_memory_mb) if latest is not None else float(max(memories))
            max_vram_mb = float(latest.max_vram_mb) if latest is not None and latest.max_vram_mb is not None else None
        elif statistic == "median":
            max_memory_mb = float(statistics.median(memories))
            max_vram_mb = float(statistics.median(vrams)) if vrams else None
        else:
            max_memory_mb = float(sum(memories) / len(memories))
            max_vram_mb = float(sum(vrams) / len(vrams)) if vrams else None

        if runtime_stat == "latest" and statistic == "latest" and latest is not None:
            n_runs = 1
            source_roots = [latest.source_root]
            source_logs = {latest.source_log}
        else:
            n_runs = len(recs)
            source_roots = sorted({rec.source_root for rec in recs})
            source_logs = {rec.source_log for rec in recs}

        out[key] = AggregatedRecord(
            dataset=key[0],
            mode=key[1],
            runtime_s=runtime_s,
            max_memory_mb=max_memory_mb,
            max_vram_mb=max_vram_mb,
            n_runs=n_runs,
            source_roots=";".join(source_roots),
            source_logs_count=len(source_logs),
        )
    return out


def sort_modes(modes: Iterable[str]) -> list[str]:
    mode_rank = {mode: idx for idx, mode in enumerate(MODE_ORDER)}
    return sorted(modes, key=lambda mode: (mode_rank.get(mode, 999), mode))


def add_combined_status_snapshot_vram_mb(
    status_snapshot_vram_mb: dict[tuple[str, str], float],
    *,
    include_align: bool,
    include_pred: bool,
) -> dict[tuple[str, str], float]:
    """Add synthetic-mode VRAM rows from measured status-snapshot values."""
    out = dict(status_snapshot_vram_mb)
    datasets = sorted({dataset for dataset, _mode in status_snapshot_vram_mb.keys()})

    for dataset in datasets:
        if include_align:
            align_values = [
                status_snapshot_vram_mb[(dataset, mode)]
                for mode in ALIGN_VARIANT_MODES
                if (dataset, mode) in status_snapshot_vram_mb
            ]
            if align_values:
                out[(dataset, "align_combined")] = float(sum(align_values) / len(align_values))

        if include_pred:
            pred_off_values = [
                status_snapshot_vram_mb[(dataset, mode)]
                for mode in PRED_FRAG_OFF_MODES
                if (dataset, mode) in status_snapshot_vram_mb
            ]
            if pred_off_values:
                out[(dataset, "pred_frag_off")] = float(sum(pred_off_values) / len(pred_off_values))

            pred_on_values = [
                status_snapshot_vram_mb[(dataset, mode)]
                for mode in PRED_FRAG_ON_MODES
                if (dataset, mode) in status_snapshot_vram_mb
            ]
            if pred_on_values:
                out[(dataset, "pred_frag_on")] = float(sum(pred_on_values) / len(pred_on_values))

    return out


def add_combined_mode_records(
    records: list[RunRecord],
    *,
    include_align: bool,
    include_pred: bool,
) -> list[RunRecord]:
    """Duplicate raw records into combined synthetic modes for correct aggregation.

    This keeps aggregation semantics correct for all statistics:
    - mean: mean over all raw runs in the combined mode
    - median: true median over all raw runs in the combined mode
    - latest: latest successful run across all source variants in the group
    """
    out = list(records)
    for record in records:
        if include_align and record.mode in ALIGN_VARIANT_MODES:
            out.append(replace(record, mode="align_combined"))

        if include_pred:
            if record.mode in PRED_FRAG_OFF_MODES:
                out.append(replace(record, mode="pred_frag_off"))
            elif record.mode in PRED_FRAG_ON_MODES:
                out.append(replace(record, mode="pred_frag_on"))
    return out


def add_combined_alignment_mode(
    aggregated_records: dict[tuple[str, str], AggregatedRecord],
) -> dict[tuple[str, str], AggregatedRecord]:
    by_dataset: dict[str, list[AggregatedRecord]] = defaultdict(list)
    for (dataset, mode), record in aggregated_records.items():
        if mode in ALIGN_VARIANT_MODES:
            by_dataset[dataset].append(record)

    out = dict(aggregated_records)
    for dataset, records in by_dataset.items():
        total_runs = sum(rec.n_runs for rec in records)
        if total_runs <= 0:
            continue

        weighted_runtime_s = sum(rec.runtime_s * rec.n_runs for rec in records) / total_runs
        weighted_memory_mb = sum(rec.max_memory_mb * rec.n_runs for rec in records) / total_runs
        vram_records = [rec for rec in records if rec.max_vram_mb is not None]
        total_vram_runs = sum(rec.n_runs for rec in vram_records)
        weighted_vram_mb = (
            sum(float(rec.max_vram_mb) * rec.n_runs for rec in vram_records) / total_vram_runs
            if total_vram_runs > 0
            else None
        )
        source_roots = sorted(
            {
                root
                for rec in records
                for root in rec.source_roots.split(";")
                if root
            }
        )

        out[(dataset, "align_combined")] = AggregatedRecord(
            dataset=dataset,
            mode="align_combined",
            runtime_s=float(weighted_runtime_s),
            max_memory_mb=float(weighted_memory_mb),
            max_vram_mb=float(weighted_vram_mb) if weighted_vram_mb is not None else None,
            n_runs=total_runs,
            source_roots=";".join(source_roots),
            source_logs_count=sum(rec.source_logs_count for rec in records),
        )

    return out


def add_combined_pred_fragment_modes(
    aggregated_records: dict[tuple[str, str], AggregatedRecord],
) -> dict[tuple[str, str], AggregatedRecord]:
    out = dict(aggregated_records)
    datasets = sorted({dataset for dataset, _mode in aggregated_records.keys()})

    for dataset in datasets:
        for source_modes, target_mode in (
            (PRED_FRAG_OFF_MODES, "pred_frag_off"),
            (PRED_FRAG_ON_MODES, "pred_frag_on"),
        ):
            records = [
                aggregated_records[(dataset, mode)]
                for mode in source_modes
                if (dataset, mode) in aggregated_records
            ]
            total_runs = sum(rec.n_runs for rec in records)
            if total_runs <= 0:
                continue

            weighted_runtime_s = sum(rec.runtime_s * rec.n_runs for rec in records) / total_runs
            weighted_memory_mb = sum(rec.max_memory_mb * rec.n_runs for rec in records) / total_runs
            vram_records = [rec for rec in records if rec.max_vram_mb is not None]
            total_vram_runs = sum(rec.n_runs for rec in vram_records)
            weighted_vram_mb = (
                sum(float(rec.max_vram_mb) * rec.n_runs for rec in vram_records) / total_vram_runs
                if total_vram_runs > 0
                else None
            )
            source_roots = sorted(
                {
                    root
                    for rec in records
                    for root in rec.source_roots.split(";")
                    if root
                }
            )

            out[(dataset, target_mode)] = AggregatedRecord(
                dataset=dataset,
                mode=target_mode,
                runtime_s=float(weighted_runtime_s),
                max_memory_mb=float(weighted_memory_mb),
                max_vram_mb=float(weighted_vram_mb) if weighted_vram_mb is not None else None,
                n_runs=total_runs,
                source_roots=";".join(source_roots),
                source_logs_count=sum(rec.source_logs_count for rec in records),
            )

    return out


def write_summary_tsv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_label",
        "mode",
        "mode_label",
        "transcripts",
        "runtime_min",
        "max_ram_gb",
        "max_vram_gb",
        "vram_worked_max_gb",
        "vram_failed_max_gb",
        "vram_estimate_gb",
        "plot_memory_gb",
        "memory_metric",
        "n_runs",
        "source_logs_count",
        "aggregate_statistic",
        "runtime_statistic",
        "transcript_source",
        "source_roots",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _configure_plot_style() -> tuple[str, str]:
    apply_reference_rcparams()
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 20,
            "axes.labelweight": "light",
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "legend.fontsize": 10,
        }
    )
    bg_color = "#ffffff"
    grid_color = "#d9d9d9"
    return bg_color, grid_color


def _apply_non_title_font(ax: plt.Axes) -> None:
    prop = _non_title_font(size=ax.xaxis.label.get_size())
    ax.xaxis.label.set_fontproperties(prop)
    ax.yaxis.label.set_fontproperties(_non_title_font(size=ax.yaxis.label.get_size()))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(_non_title_font(size=label.get_size()))


def _non_title_font(*, size: float | None = None) -> FontProperties:
    # Matplotlib cannot directly resolve "Helvetica Neue Light" by family name on macOS TTCs.
    # Use the light face from HelveticaNeue.ttc (index 7) when possible.
    cache_dir = Path.home() / ".cache" / "segger_fonts"
    light_ttf = cache_dir / "HelveticaNeue-Light.ttf"
    if not light_ttf.exists():
        try:
            from fontTools.ttLib import TTCollection  # type: ignore

            ttc = TTCollection("/System/Library/Fonts/HelveticaNeue.ttc")
            if len(ttc.fonts) > 7:
                cache_dir.mkdir(parents=True, exist_ok=True)
                ttc.fonts[7].save(str(light_ttf))
        except Exception:
            pass

    if light_ttf.exists():
        if size is None:
            return FontProperties(fname=str(light_ttf))
        return FontProperties(fname=str(light_ttf), size=size)

    if size is None:
        return FontProperties(family="Helvetica Neue", weight="light")
    return FontProperties(family="Helvetica Neue", weight="light", size=size)


def _style_axes(ax: plt.Axes, grid_color: str, *, log_scale: bool = True) -> None:
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Keep horizontal guides only; vertical guides come from dataset lines.
        ax.grid(True, which="major", axis="y", color=grid_color, linewidth=0.9, alpha=0.9)
        ax.grid(True, which="minor", axis="y", color=grid_color, linewidth=0.5, alpha=0.5)
    else:
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        # Keep horizontal guides only; vertical guides come from dataset lines.
        ax.grid(True, which="major", axis="y", color=grid_color, linewidth=0.9, alpha=0.85)
        ax.grid(False, which="minor", axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.tick_params(axis="both", which="major", length=10, width=1.5)
    if log_scale:
        ax.tick_params(axis="both", which="minor", length=6, width=1.0)
    else:
        ax.tick_params(axis="both", which="minor", length=0, width=0)


def _format_value_compact(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _set_data_driven_linear_yticks(
    ax: plt.Axes,
    y_values: list[float],
    *,
    min_floor_zero: bool = True,
) -> None:
    clean = sorted(value for value in y_values if value > 0)
    if not clean:
        return

    y_min = 0.0 if min_floor_zero else clean[0] * 0.90
    if clean[0] < 4:
        y_min = min(y_min, clean[0] * 0.85)
    y_max = clean[-1] * 1.12
    if y_max <= y_min:
        return

    locator = mticker.MaxNLocator(nbins=6, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
    ticks = [
        tick
        for tick in locator.tick_values(y_min, y_max)
        if y_min - 1e-9 <= tick <= y_max + 1e-9
    ]
    if len(ticks) < 4:
        step = (y_max - y_min) / 4.0 if y_max > y_min else 1.0
        ticks = [y_min + (step * idx) for idx in range(5)]

    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: _format_value_compact(value)))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def _set_data_driven_log_yticks(
    ax: plt.Axes,
    y_values: list[float],
    *,
    max_ticks: int = 6,
) -> None:
    clean = sorted(value for value in y_values if value > 0)
    if not clean:
        return

    y_min = clean[0] * 0.88
    y_max = clean[-1] * 1.15
    if y_max <= y_min:
        return

    min_pow = int(math.floor(math.log10(y_min))) - 1
    max_pow = int(math.ceil(math.log10(y_max))) + 1
    candidates: list[float] = []
    for power in range(min_pow, max_pow + 1):
        base = 10 ** power
        for multiplier in (1, 2, 3, 4, 5, 6, 8):
            candidate = multiplier * base
            if y_min <= candidate <= y_max:
                candidates.append(candidate)
    if not candidates:
        return

    ticks: set[float] = set()
    for value in clean:
        nearest = min(candidates, key=lambda candidate: abs(math.log10(candidate) - math.log10(value)))
        ticks.add(nearest)

    ordered = sorted(ticks)
    if len(ordered) > max_ticks:
        keep = {ordered[0], ordered[-1]}
        slots = max_ticks - 2
        if slots > 0:
            for idx in range(1, slots + 1):
                src = int(round(idx * (len(ordered) - 1) / (slots + 1)))
                keep.add(ordered[src])
        ordered = sorted(keep)

    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(mticker.FixedLocator(ordered))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: _format_value_compact(value)))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def _set_data_driven_log_xticks(
    ax: plt.Axes,
    x_values: list[float],
    *,
    max_ticks: int = 5,
    min_sep_log10: float = 0.20,
) -> None:
    clean = sorted(value for value in x_values if value > 0)
    if not clean:
        return

    x_min = clean[0] * 0.88
    x_max = clean[-1] * 1.15
    if x_max <= x_min:
        return

    ax.set_xlim(x_min, x_max)
    locator = mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
    candidates = sorted(
        {
            tick
            for tick in locator.tick_values(x_min, x_max)
            if x_min <= tick <= x_max
        }
    )
    if not candidates:
        return

    # Keep readable spacing between labels, especially on the left side where
    # the range is densest.
    major_ticks = [candidates[0]]
    for tick in candidates[1:]:
        if math.log10(tick) - math.log10(major_ticks[-1]) >= min_sep_log10:
            major_ticks.append(tick)
    if major_ticks[-1] != candidates[-1]:
        major_ticks.append(candidates[-1])

    if len(major_ticks) > max_ticks:
        keep_indices = {0, len(major_ticks) - 1}
        slots = max_ticks - 2
        if slots > 0:
            for idx in range(1, slots + 1):
                src = int(round(idx * (len(major_ticks) - 1) / (slots + 1)))
                keep_indices.add(src)
        major_ticks = [major_ticks[idx] for idx in sorted(keep_indices)]

    ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{_format_value_compact(value / 1e6)}M")
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def _set_linear_xticks(ax: plt.Axes, x_min: float, x_max: float) -> None:
    locator = mticker.MaxNLocator(nbins=5, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
    ticks = [
        tick
        for tick in locator.tick_values(x_min, x_max)
        if x_min - 1e-9 <= tick <= x_max + 1e-9 and tick > 0
    ]
    if len(ticks) < 4:
        span = x_max - x_min
        if span <= 0:
            return
        step = span / 4.0
        ticks = [x_min + (step * idx) for idx in range(5)]

    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{_format_value_compact(value / 1e6)}M")
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def _add_dataset_labels(
    ax: plt.Axes,
    transcript_values: list[float],
    dataset_labels: dict[float, str],
    grid_color: str,
    *,
    draw_labels: bool,
) -> None:
    for idx, tx in enumerate(transcript_values):
        ax.axvline(tx, color=grid_color, linewidth=1.0, alpha=0.95, zorder=1)
        if draw_labels:
            ax.text(
                tx,
                1.02 + (0.03 if idx % 2 else 0.0),
                dataset_labels.get(tx, ""),
                rotation=-60,
                ha="right",
                va="bottom",
                fontsize=10,
                color="#222222",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )


def _discover_segbench_assets_dir(project_root: Path) -> Path | None:
    candidates = [
        project_root.parent / "segbench" / "scripts" / "js_schematics" / "assets",
        project_root / "segbench" / "scripts" / "js_schematics" / "assets",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _build_dataset_icon_paths(project_root: Path) -> dict[str, Path]:
    assets_dir = _discover_segbench_assets_dir(project_root)
    if assets_dir is None:
        return {}
    out: dict[str, Path] = {}
    for dataset, filename in DATASET_ICON_FILES.items():
        icon_path = assets_dir / filename
        if icon_path.is_file():
            out[dataset] = icon_path
    return out


def _jitter_positions_log_space(
    x_values: list[float],
    *,
    min_sep_log10: float = 0.055,
) -> dict[float, float]:
    if not x_values:
        return {}
    sorted_vals = sorted(x_values)
    log_vals = [math.log10(x) for x in sorted_vals]
    adjusted = list(log_vals)
    for idx in range(1, len(adjusted)):
        if adjusted[idx] - adjusted[idx - 1] < min_sep_log10:
            adjusted[idx] = adjusted[idx - 1] + min_sep_log10

    shift = (sum(adjusted) - sum(log_vals)) / len(adjusted)
    adjusted = [value - shift for value in adjusted]
    return {x: 10 ** value for x, value in zip(sorted_vals, adjusted)}


def _load_icon_image(icon_path: Path) -> np.ndarray | None:
    def _normalize_with_transparent_white(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 3:
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
            arr = np.concatenate([arr.astype(np.uint8, copy=False), alpha], axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr.astype(np.uint8, copy=False)
        else:
            return arr
        arr = np.array(arr, copy=True)

        # Remove only edge-connected white canvas so icon interiors are preserved.
        rgb = arr[..., :3]
        white_like = np.all(rgb >= 245, axis=-1)
        low_chroma = (rgb.max(axis=-1) - rgb.min(axis=-1)) <= 12
        bg_candidate = white_like & low_chroma

        h, w = bg_candidate.shape
        edge_bg = np.zeros((h, w), dtype=bool)
        queue: deque[tuple[int, int]] = deque()

        def _push(y: int, x: int) -> None:
            if bg_candidate[y, x] and not edge_bg[y, x]:
                edge_bg[y, x] = True
                queue.append((y, x))

        for x in range(w):
            _push(0, x)
            _push(h - 1, x)
        for y in range(h):
            _push(y, 0)
            _push(y, w - 1)

        while queue:
            y, x = queue.popleft()
            if y > 0:
                _push(y - 1, x)
            if y + 1 < h:
                _push(y + 1, x)
            if x > 0:
                _push(y, x - 1)
            if x + 1 < w:
                _push(y, x + 1)

        arr[edge_bg, 3] = 0
        return arr

    try:
        return _normalize_with_transparent_white(mpimg.imread(icon_path))
    except Exception:
        # Some assets are JPEG files with .png extensions; Pillow can still decode these.
        try:
            from PIL import Image  # type: ignore

            return _normalize_with_transparent_white(np.asarray(Image.open(icon_path).convert("RGBA")))
        except Exception:
            return None


def _hex_to_rgb01(value: str) -> np.ndarray:
    stripped = value.strip().lstrip("#")
    if len(stripped) != 6:
        return np.array([0.45, 0.45, 0.45], dtype=np.float32)
    return np.array(
        [
            int(stripped[0:2], 16),
            int(stripped[2:4], 16),
            int(stripped[4:6], 16),
        ],
        dtype=np.float32,
    ) / 255.0


def _tint_icon_image(image: np.ndarray, color_hex: str) -> np.ndarray:
    arr = np.array(image, copy=True).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        return image

    alpha = arr[..., 3:4] / 255.0
    if float(alpha.max()) <= 0.0:
        return image

    rgb = arr[..., :3] / 255.0
    luminance = (0.2126 * rgb[..., 0]) + (0.7152 * rgb[..., 1]) + (0.0722 * rgb[..., 2])
    darkness = np.clip(1.0 - luminance, 0.0, 1.0)
    shade = 0.58 + (0.42 * darkness)
    tint_rgb = _hex_to_rgb01(color_hex)
    recolored = np.clip(tint_rgb[None, None, :] * shade[..., None], 0.0, 1.0)

    arr[..., :3] = recolored * 255.0
    return arr.astype(np.uint8)


def _add_dataset_icons(
    ax: plt.Axes,
    transcript_values: list[float],
    transcript_to_dataset: dict[float, str],
    icon_paths: dict[str, Path],
    *,
    y: float = 1.06,
    icon_zoom: float = 0.023,
    y_jitter: float = 0.04,
) -> None:
    if not transcript_values:
        return

    x_lo, x_hi = ax.get_xlim()
    x_left = min(x_lo, x_hi)
    x_right = max(x_lo, x_hi)
    is_log_x = ax.get_xscale() == "log"
    image_cache: dict[tuple[Path, str], np.ndarray] = {}
    connector_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    placements: list[tuple[float, str, float]] = []
    if is_log_x:
        log_left = math.log10(max(x_left, 1e-12))
        log_right = math.log10(max(x_right, 1e-12))
        log_span = max(log_right - log_left, 1e-12)
    else:
        x_span = max(x_right - x_left, 1e-12)

    for tx in transcript_values:
        dataset = transcript_to_dataset.get(tx)
        if dataset is None:
            continue
        if is_log_x:
            x_frac = (math.log10(max(tx, 1e-12)) - log_left) / log_span
        else:
            x_frac = (tx - x_left) / x_span
        placements.append((tx, dataset, x_frac))

    placements.sort(key=lambda item: item[2])
    if not placements:
        return

    # Keep all icons at one height and spread only horizontally.
    # Use moderate adaptive spacing so icons stay close to their true transcript
    # lines while still being readable.
    n_icons = len(placements)
    if n_icons >= 10:
        desired_sep_frac = 0.054
    elif n_icons >= 8:
        desired_sep_frac = 0.060
    elif n_icons >= 6:
        desired_sep_frac = 0.068
    else:
        desired_sep_frac = 0.075
    left_bound = 0.01
    right_bound = 0.99
    if n_icons > 1:
        max_feasible_sep = (right_bound - left_bound) / float(n_icons - 1)
        min_sep_frac = min(desired_sep_frac, max_feasible_sep * 0.92)
    else:
        min_sep_frac = desired_sep_frac
    base_fracs = [item[2] for item in placements]
    adj_fracs = list(base_fracs)

    for idx in range(1, len(adj_fracs)):
        adj_fracs[idx] = max(adj_fracs[idx], adj_fracs[idx - 1] + min_sep_frac)
    if adj_fracs[-1] > right_bound:
        shift = adj_fracs[-1] - right_bound
        adj_fracs = [value - shift for value in adj_fracs]
    if adj_fracs[0] < left_bound:
        shift = left_bound - adj_fracs[0]
        adj_fracs = [value + shift for value in adj_fracs]
    for idx in range(len(adj_fracs) - 2, -1, -1):
        adj_fracs[idx] = min(adj_fracs[idx], adj_fracs[idx + 1] - min_sep_frac)
    if adj_fracs[0] < left_bound:
        shift = left_bound - adj_fracs[0]
        adj_fracs = [value + shift for value in adj_fracs]
    if adj_fracs[-1] > right_bound:
        shift = adj_fracs[-1] - right_bound
        adj_fracs = [value - shift for value in adj_fracs]

    for idx, (tx, dataset, x_frac) in enumerate(placements):
        icon_path = icon_paths.get(dataset)
        if icon_path is None:
            continue
        tint_hex = DATASET_ICON_PLATFORM_COLORS.get(dataset, "#666666")
        cache_key = (icon_path, tint_hex)
        if cache_key not in image_cache:
            image = _load_icon_image(icon_path)
            if image is None:
                continue
            image_cache[cache_key] = _tint_icon_image(image, tint_hex)
        image = image_cache[cache_key]

        adj_frac = adj_fracs[idx]
        if is_log_x:
            x_val = 10 ** (log_left + (adj_frac * log_span))
        else:
            x_val = x_left + (adj_frac * x_span)
        y_val = y

        # Always draw a connector so each icon is clearly linked to its
        # transcript guide line at the top edge.
        ax.plot(
            [tx, x_val],
            [1.0, y_val - 0.008],
            color="#111111",
            linewidth=1.05,
            linestyle=(0, (1.5, 2.2)),
            alpha=0.90,
            zorder=6,
            transform=connector_transform,
            clip_on=False,
        )

        zoom = icon_zoom * DATASET_ICON_ZOOM_SCALE.get(dataset, 1.0)
        icon = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(
            icon,
            (x_val, y_val),
            xycoords=("data", "axes fraction"),
            frameon=False,
            box_alignment=(0.5, 0.0),
            annotation_clip=False,
            pad=0.0,
            zorder=8,
        )
        ax.add_artist(ab)


def build_plot(
    rows: list[dict[str, str]],
    out_png: Path,
    out_pdf: Path,
    title: str,
    *,
    mode_view: str,
    memory_axis_label: str,
    memory_value_key: str,
) -> None:
    bg_color, grid_color = _configure_plot_style()
    rows_by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_mode[row["mode"]].append(row)

    transcript_values = sorted({float(row["transcripts"]) for row in rows})
    dataset_labels = {float(row["transcripts"]): row["dataset_label"] for row in rows}
    transcript_to_dataset = {float(row["transcripts"]): row["dataset"] for row in rows}
    project_root = Path(__file__).resolve().parents[1]
    dataset_icon_paths = _build_dataset_icon_paths(project_root)
    x_min = min(transcript_values) * 0.85 if transcript_values else 1.0
    x_max = max(transcript_values) * 1.20 if transcript_values else 10.0

    if mode_view == "core_pred_2x2":
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 12.4), sharex=True)
        fig.patch.set_facecolor(bg_color)
        group_to_modes = {
            0: ["baseline", "use3d_true", "align_combined"],
            1: ["pred_frag_off", "pred_frag_on"],
        }
        group_titles = {
            0: "End-to-End Segmentation",
            1: "Prediction Only",
        }

        mode_handle_map: dict[str, Line2D] = {}
        axis_y_values: dict[tuple[int, int], list[float]] = {
            (0, 0): [],
            (0, 1): [],
            (1, 0): [],
            (1, 1): [],
        }
        for row_idx in range(2):
            for col_idx in range(2):
                ax = axes[row_idx][col_idx]
                ax.set_facecolor(bg_color)
                _style_axes(ax, grid_color, log_scale=True)
                ax.set_box_aspect(1.0)
                _set_data_driven_log_xticks(ax, transcript_values)
                ax.tick_params(axis="x", which="both", labelbottom=True)
                _add_dataset_labels(
                    ax,
                    transcript_values,
                    dataset_labels,
                    grid_color,
                    draw_labels=False,
                )
                if row_idx == 0:
                    ax.text(
                        0.5,
                        1.18,
                        group_titles[col_idx],
                        ha="center",
                        va="bottom",
                        fontsize=16,
                        fontfamily="Helvetica Neue",
                        fontweight="bold",
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                    _add_dataset_icons(
                        ax,
                        transcript_values,
                        transcript_to_dataset,
                        dataset_icon_paths,
                        y=1.04,
                        icon_zoom=0.023,
                        y_jitter=0.048,
                    )

                for mode in group_to_modes[col_idx]:
                    if mode not in rows_by_mode:
                        continue
                    mode_rows = sorted(rows_by_mode[mode], key=lambda row: float(row["transcripts"]))
                    x_vals = [float(row["transcripts"]) for row in mode_rows]
                    y_vals = [
                        float(row["runtime_min"]) if row_idx == 0 else float(row[memory_value_key])
                        for row in mode_rows
                    ]
                    axis_y_values[(row_idx, col_idx)].extend(y_vals)
                    color = MODE_COLORS.get(mode, "#555555")
                    marker = MODE_MARKERS.get(mode, "o")
                    linestyle = MODE_LINESTYLES.get(mode, "-")
                    base_label = MODE_LABELS.get(mode, mode)
                    label = base_label

                    ax.plot(
                        x_vals,
                        y_vals,
                        color=color,
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=2.3,
                        markersize=7.5,
                        alpha=0.96,
                        zorder=3,
                    )

                    if mode not in mode_handle_map:
                        mode_handle_map[mode] = Line2D(
                            [0],
                            [0],
                            color=color,
                            marker=marker,
                            linestyle=linestyle,
                            linewidth=2.6,
                            markersize=8.5,
                            label=label,
                        )

        for row_idx in range(2):
            for col_idx in range(2):
                panel_max_ticks = 4 if (row_idx, col_idx) == (0, 0) else 6
                _set_data_driven_log_yticks(
                    axes[row_idx][col_idx],
                    axis_y_values[(row_idx, col_idx)],
                    max_ticks=panel_max_ticks,
                )

        axes[0][0].set_ylabel("Runtime (min.)")
        axes[1][0].set_ylabel(memory_axis_label)
        axes[1][0].set_xlabel("No. Transcripts")
        axes[1][1].set_xlabel("No. Transcripts")
        for ax in axes.flat:
            _apply_non_title_font(ax)

        fig.suptitle(title, fontsize=28, y=0.986, weight="bold", fontfamily="Helvetica Neue")
        left_handles = [
            mode_handle_map[mode] for mode in group_to_modes[0] if mode in mode_handle_map
        ]
        right_handles = [
            mode_handle_map[mode] for mode in group_to_modes[1] if mode in mode_handle_map
        ]
        axes[1][0].legend(
            handles=left_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=2 if len(left_handles) >= 3 else max(1, len(left_handles)),
            frameon=False,
            prop=_non_title_font(size=14),
            handlelength=2.8,
            columnspacing=1.1,
            handletextpad=0.6,
            markerscale=1.2,
        )
        axes[1][1].legend(
            handles=right_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=max(1, min(2, len(right_handles))),
            frameon=False,
            prop=_non_title_font(size=14),
            handlelength=2.8,
            columnspacing=1.1,
            handletextpad=0.6,
            markerscale=1.2,
        )
        fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.14, wspace=0.02, hspace=0.14)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), sharex=True)
        fig.patch.set_facecolor(bg_color)
        for ax in axes:
            ax.set_facecolor(bg_color)
            _style_axes(ax, grid_color)

        mode_handles: list[Line2D] = []
        for mode in sort_modes(rows_by_mode.keys()):
            mode_rows = sorted(rows_by_mode[mode], key=lambda row: float(row["transcripts"]))
            x_vals = [float(row["transcripts"]) for row in mode_rows]
            runtime_vals = [float(row["runtime_min"]) for row in mode_rows]
            memory_vals = [float(row[memory_value_key]) for row in mode_rows]

            color = MODE_COLORS.get(mode, "#555555")
            marker = MODE_MARKERS.get(mode, "o")
            linestyle = MODE_LINESTYLES.get(mode, "-")
            base_label = MODE_LABELS.get(mode, mode)
            label = base_label

            axes[0].plot(
                x_vals,
                runtime_vals,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.2,
                markersize=7.5,
                alpha=0.95,
                zorder=3,
            )
            axes[1].plot(
                x_vals,
                memory_vals,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.2,
                markersize=7.5,
                alpha=0.95,
                zorder=3,
            )

            mode_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.2,
                    markersize=7,
                    label=label,
                )
            )

        if transcript_values:
            for ax in axes:
                ax.set_xlim(x_min, x_max)
                _add_dataset_labels(
                    ax,
                    transcript_values,
                    dataset_labels,
                    grid_color,
                    draw_labels=True,
                )

        axes[0].set_ylabel("Runtime (min.)")
        axes[1].set_ylabel(memory_axis_label)
        axes[0].set_xlabel("No. Transcripts")
        axes[1].set_xlabel("No. Transcripts")
        fig.suptitle(title, fontsize=15, y=0.985, weight="bold")

        fig.legend(
            handles=mode_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            columnspacing=1.0,
            handlelength=2.4,
        )
        fig.subplots_adjust(left=0.085, right=0.985, top=0.68, bottom=0.24, wspace=0.22)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor=bg_color)
    fig.savefig(out_pdf, dpi=320, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)


def discover_default_roots(project_root: Path) -> list[Path]:
    roots: list[Path] = []
    local_archive = project_root / "segger_lsf_benchmark_fixed_archive_20260305_121246"
    if local_archive.is_dir():
        roots.append(local_archive)

    for candidate in sorted(project_root.parent.glob("benchmark_segger_lsf*")):
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def discover_segbench_overview_path(project_root: Path) -> Path | None:
    candidates = [
        project_root.parent / "segbench" / "scripts" / "js_schematics" / "dataset_overview.html",
        project_root / "segbench" / "scripts" / "js_schematics" / "dataset_overview.html",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_transcript_counts_from_segbench_overview(overview_path: Path) -> dict[str, float]:
    # Parse the JS `datasets = [...]` block by extracting `id` and `transcripts`.
    id_re = re.compile(r'id:\s*"([^"]+)"')
    tx_re = re.compile(r"transcripts:\s*([0-9]+(?:\.[0-9]+)?)")

    counts: dict[str, float] = {}
    current_segbench_id: str | None = None
    with overview_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            id_match = id_re.search(line)
            if id_match:
                current_segbench_id = id_match.group(1)
                continue

            if current_segbench_id is None:
                continue

            tx_match = tx_re.search(line)
            if tx_match:
                dataset_key = SEG_BENCH_ID_TO_DATASET_KEY.get(current_segbench_id)
                if dataset_key is not None:
                    counts[dataset_key] = float(tx_match.group(1)) * 1e6
                current_segbench_id = None

    return counts


def build_rows(
    aggregated_records: dict[tuple[str, str], AggregatedRecord],
    dataset_transcripts: dict[str, float],
    transcript_source_map: dict[str, str],
    aggregate_statistic: str,
    runtime_statistic: str,
    memory_metric: str,
    vram_estimates: dict[tuple[str, str], VramEstimate],
    status_snapshot_vram_mb: dict[tuple[str, str], float],
) -> list[dict[str, str]]:
    mode_rank = {mode: idx for idx, mode in enumerate(MODE_ORDER)}
    rows: list[dict[str, str]] = []
    for (dataset, mode), record in aggregated_records.items():
        transcripts = dataset_transcripts.get(dataset)
        if transcripts is None:
            continue

        max_ram_gb = record.max_memory_mb / 1024.0
        record_vram_gb = (record.max_vram_mb / 1024.0) if record.max_vram_mb is not None else None
        status_vram_mb = status_snapshot_vram_mb.get((dataset, mode))
        status_vram_gb = (status_vram_mb / 1024.0) if status_vram_mb is not None and status_vram_mb > 0 else None
        # Match dashboard semantics: prefer status-snapshot resource values first.
        max_vram_gb = status_vram_gb if status_vram_gb is not None else record_vram_gb
        vram_estimate = vram_estimates.get((dataset, mode))
        vram_worked_max_gb = (
            (vram_estimate.worked_max_mb / 1024.0)
            if vram_estimate is not None and vram_estimate.worked_max_mb is not None
            else None
        )
        vram_failed_max_gb = (
            (vram_estimate.failed_max_mb / 1024.0)
            if vram_estimate is not None and vram_estimate.failed_max_mb is not None
            else None
        )
        vram_estimate_gb = (
            (vram_estimate.estimate_mb / 1024.0)
            if vram_estimate is not None and vram_estimate.estimate_mb is not None
            else None
        )

        if memory_metric == "vram":
            # Use measured max_vram when available.
            if max_vram_gb is not None:
                plot_memory_gb = max_vram_gb
            # Otherwise use worked/failed estimator:
            # max_vram_estimate = mean(max_worked_vram, max_failed_vram)
            # with fallback to available side when one side is missing.
            elif vram_estimate_gb is not None:
                plot_memory_gb = vram_estimate_gb
            else:
                continue
        else:
            plot_memory_gb = max_ram_gb

        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "transcripts": f"{transcripts:.0f}",
                "runtime_min": f"{record.runtime_s / 60.0:.4f}",
                "max_ram_gb": f"{max_ram_gb:.4f}",
                "max_vram_gb": f"{max_vram_gb:.4f}" if max_vram_gb is not None else "",
                "vram_worked_max_gb": f"{vram_worked_max_gb:.4f}" if vram_worked_max_gb is not None else "",
                "vram_failed_max_gb": f"{vram_failed_max_gb:.4f}" if vram_failed_max_gb is not None else "",
                "vram_estimate_gb": f"{vram_estimate_gb:.4f}" if vram_estimate_gb is not None else "",
                "plot_memory_gb": f"{plot_memory_gb:.4f}",
                "memory_metric": memory_metric,
                "n_runs": str(record.n_runs),
                "source_logs_count": str(record.source_logs_count),
                "aggregate_statistic": aggregate_statistic,
                "runtime_statistic": runtime_statistic,
                "transcript_source": transcript_source_map.get(dataset, "unknown"),
                "source_roots": record.source_roots,
            }
        )

    rows.sort(
        key=lambda row: (
            float(row["transcripts"]),
            mode_rank.get(row["mode"], 999),
            row["mode"],
        )
    )
    return rows


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_out_dir = project_root / "segger_lsf_benchmark_fixed" / "summaries"

    parser = argparse.ArgumentParser(
        description="Plot Segger LSF runtime/memory by mode vs transcript count."
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=None,
        help="Benchmark roots that contain a datasets/ subfolder. Defaults to discovered LSF roots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out_dir,
        help="Directory for output TSV/PNG/PDF files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="segger_modes_runtime_memory_vs_transcripts",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include incomplete/failed blocks (default excludes failed runs).",
    )
    parser.add_argument(
        "--aggregate-statistic",
        choices=["mean", "median", "latest"],
        default="latest",
        help="Statistic for aggregating runtime/memory across successful runs.",
    )
    parser.add_argument(
        "--runtime-statistic",
        choices=["mean", "median", "min", "latest"],
        default=None,
        help=(
            "Optional runtime-only aggregation across successful runs. "
            "If omitted, uses --aggregate-statistic."
        ),
    )
    parser.add_argument(
        "--memory-metric",
        choices=["ram", "vram"],
        default="ram",
        help=(
            "Memory metric for plotting/summary: "
            "'ram' uses LSF Max Memory (host RAM), "
            "'vram' uses measured GPU VRAM when available "
            "(OOM in-use VRAM and runtime snapshots/status snapshots)."
        ),
    )
    parser.add_argument(
        "--vram-fallback",
        choices=["none", "gmem"],
        default="none",
        help=(
            "Fallback for missing VRAM measurements. "
            "'none' keeps only measured VRAM; "
            "'gmem' falls back to requested queue reservation."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Compute Resources",
        help="Figure title.",
    )
    parser.add_argument(
        "--min-points-per-mode",
        type=int,
        default=2,
        help="Minimum number of dataset points required for a mode to be plotted.",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=[],
        help="Dataset keys to exclude from aggregation/plotting (e.g. xenium_v1_breast).",
    )
    parser.add_argument(
        "--transcript-source",
        choices=["auto", "fallback"],
        default="auto",
        help=(
            "Transcript count source: 'auto' prefers segbench overview when available; "
            "'fallback' uses the local fixed map."
        ),
    )
    parser.add_argument(
        "--include-align-combined",
        action="store_true",
        help="Add a synthetic Alignment (avg) mode by combining align_0p01/0p03/0p10.",
    )
    parser.add_argument(
        "--mode-view",
        choices=["full", "core_pred_2x2"],
        default="full",
        help=(
            "Plot composition: 'full' keeps all selected modes; "
            "'core_pred_2x2' shows baseline/use3d/alignment and pred frag off/on in a 2x2 layout."
        ),
    )
    parser.add_argument(
        "--status-snapshot-tsv",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Optional status snapshot TSV(s) with measured max_vram fields. "
            "Defaults to auto-discovered <root>/summaries/aggregate_status_snapshot.tsv."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    roots = args.roots if args.roots is not None and len(args.roots) > 0 else discover_default_roots(project_root)
    roots = [root.resolve() for root in roots if root.exists()]
    if not roots:
        raise SystemExit("No benchmark roots found. Pass explicit paths via --roots.")

    status_snapshot_paths = discover_status_snapshot_paths(roots, args.status_snapshot_tsv)
    status_snapshot_vram_mb = load_status_snapshot_vram_mb(
        status_snapshot_paths,
        fallback_to_requested_gmem=(args.vram_fallback == "gmem"),
    )

    overview_path = None if args.transcript_source == "fallback" else discover_segbench_overview_path(project_root)
    segbench_counts: dict[str, float] = {}
    if overview_path is not None:
        segbench_counts = load_transcript_counts_from_segbench_overview(overview_path)

    dataset_transcripts: dict[str, float] = {}
    transcript_source_map: dict[str, str] = {}
    for dataset_key, fallback in FALLBACK_DATASET_TRANSCRIPTS.items():
        if dataset_key in segbench_counts:
            dataset_transcripts[dataset_key] = segbench_counts[dataset_key]
            transcript_source_map[dataset_key] = "segbench_dataset_overview"
        else:
            dataset_transcripts[dataset_key] = fallback
            transcript_source_map[dataset_key] = "fallback_map"

    excluded_datasets = set(args.exclude_datasets or [])
    if excluded_datasets:
        dataset_transcripts = {
            key: value
            for key, value in dataset_transcripts.items()
            if key not in excluded_datasets
        }
        transcript_source_map = {
            key: value
            for key, value in transcript_source_map.items()
            if key not in excluded_datasets
        }

    datasets = set(dataset_transcripts.keys())
    all_records = collect_records(
        roots,
        datasets=datasets,
        vram_fallback=args.vram_fallback,
    )
    if not all_records:
        raise SystemExit("No parseable LSF records found.")

    need_align_combined = args.include_align_combined or args.mode_view == "core_pred_2x2"
    records_for_aggregation = all_records

    aggregated_records = aggregate_records(
        records_for_aggregation,
        statistic=args.aggregate_statistic,
        runtime_statistic=args.runtime_statistic,
        include_incomplete=args.include_incomplete,
    )
    if not aggregated_records:
        raise SystemExit(
            "No successful records after filtering. "
            "Re-run with --include-incomplete if you want failed/incomplete blocks included."
        )

    if need_align_combined:
        aggregated_records = add_combined_alignment_mode(aggregated_records)
    if args.mode_view == "core_pred_2x2":
        aggregated_records = add_combined_pred_fragment_modes(aggregated_records)

    status_snapshot_vram_mb = add_combined_status_snapshot_vram_mb(
        status_snapshot_vram_mb,
        include_align=need_align_combined,
        include_pred=args.mode_view == "core_pred_2x2",
    )

    mode_whitelist = set(MODE_ORDER)
    aggregated_records = {
        key: value
        for key, value in aggregated_records.items()
        if key[1] in mode_whitelist
    }
    if not aggregated_records:
        raise SystemExit("No benchmark mode rows left after filtering export/non-mode logs.")

    if args.mode_view == "core_pred_2x2":
        core_pred_modes = {
            "baseline",
            "use3d_true",
            "align_combined",
            "pred_frag_off",
            "pred_frag_on",
        }
        aggregated_records = {
            key: value for key, value in aggregated_records.items() if key[1] in core_pred_modes
        }
        if not aggregated_records:
            raise SystemExit("No rows left after applying core_pred_2x2 mode filter.")

    vram_estimates = estimate_vram_from_worked_failed(records_for_aggregation)
    memory_axis_label = "Max. VRAM (GB)" if args.memory_metric == "vram" else "Max. RAM (GB)"
    memory_value_key = "plot_memory_gb"

    rows = build_rows(
        aggregated_records,
        dataset_transcripts=dataset_transcripts,
        transcript_source_map=transcript_source_map,
        aggregate_statistic=args.aggregate_statistic,
        runtime_statistic=(args.runtime_statistic or args.aggregate_statistic),
        memory_metric=args.memory_metric,
        vram_estimates=vram_estimates,
        status_snapshot_vram_mb=status_snapshot_vram_mb,
    )
    if not rows:
        raise SystemExit("No rows to plot after dataset/mode filtering.")

    points_per_mode: dict[str, int] = defaultdict(int)
    for row in rows:
        points_per_mode[row["mode"]] += 1

    rows_for_plot = [
        row for row in rows if points_per_mode[row["mode"]] >= args.min_points_per_mode
    ]
    if not rows_for_plot:
        raise SystemExit(
            "No rows left for plotting after applying --min-points-per-mode filter."
        )

    out_dir: Path = args.output_dir
    out_tsv = out_dir / f"{args.prefix}.tsv"
    out_png = out_dir / f"{args.prefix}.png"
    out_pdf = out_dir / f"{args.prefix}.pdf"

    write_summary_tsv(rows, out_tsv)
    build_plot(
        rows_for_plot,
        out_png=out_png,
        out_pdf=out_pdf,
        title=args.title,
        mode_view=args.mode_view,
        memory_axis_label=memory_axis_label,
        memory_value_key=memory_value_key,
    )

    datasets_in_rows = sorted({row["dataset"] for row in rows})
    modes_in_rows = sort_modes({row["mode"] for row in rows_for_plot})
    total_aggregated_runs = sum(int(row["n_runs"]) for row in rows)
    dropped_modes = sort_modes(
        {
            mode
            for mode, count in points_per_mode.items()
            if count < args.min_points_per_mode
        }
    )

    print(f"Parsed roots: {len(roots)}")
    for root in roots:
        print(f"  - {root}")
    print(f"Parsed LSF blocks: {len(all_records)}")
    print(f"Runs included after success filter: {total_aggregated_runs}")
    print(f"Rows written: {len(rows)}")
    print(f"Aggregate statistic: {args.aggregate_statistic}")
    print(f"Runtime statistic: {args.runtime_statistic or args.aggregate_statistic}")
    print(f"Memory metric: {args.memory_metric}")
    print(f"VRAM fallback mode: {args.vram_fallback}")
    print(f"Mode view: {args.mode_view}")
    print(f"Transcript source mode: {args.transcript_source}")
    print(f"Status snapshot TSVs used: {len(status_snapshot_paths)}")
    print(f"Status snapshot VRAM overrides: {len(status_snapshot_vram_mb)}")
    for status_path in status_snapshot_paths:
        print(f"  - status: {status_path}")
    if excluded_datasets:
        print(f"Excluded datasets: {', '.join(sorted(excluded_datasets))}")
    if overview_path is not None:
        print(f"Transcript source file: {overview_path}")
    else:
        print("Transcript source file: not found (using fallback map)")
    print(f"Datasets in plot: {', '.join(datasets_in_rows)}")
    print(f"Modes in plot: {', '.join(modes_in_rows)}")
    if dropped_modes:
        print(
            "Dropped from plot due to low coverage "
            f"(min {args.min_points_per_mode} points): {', '.join(dropped_modes)}"
        )
    print(f"Summary TSV: {out_tsv}")
    print(f"Figure PNG: {out_png}")
    print(f"Figure PDF: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
