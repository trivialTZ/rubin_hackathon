"""Watch CPU prep progress on SCC from submitted job logs.

Usage:
    python3 scripts/watch_cpu_prep.py
    python3 scripts/watch_cpu_prep.py --meta logs/cpu_prep_latest.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Union


_PROGRESS_RE = re.compile(r"^PROGRESS\s+(download|build)\s+(.+)$")


ProgressValue = Optional[Union[str, int]]


def _job_id_base(job_id: str | None) -> str:
    if not job_id or job_id == "skipped":
        return ""
    return str(job_id).split(".", 1)[0]


def _resolve_log_path(meta_path: Path, log_value: str | None, prefix: str, job_id: str) -> Path | None:
    if log_value:
        return Path(log_value)
    base = _job_id_base(job_id)
    if not base:
        return None
    return meta_path.parent / f"{prefix}.{base}.log"


def _parse_progress(log_path: Path, stage: str) -> Dict[str, ProgressValue]:
    info: Dict[str, ProgressValue] = {
        "phase": None,
        "current": 0,
        "total": 0,
        "ok": 0,
        "failed": 0,
        "rows": 0,
        "skipped": 0,
        "written": 0,
    }
    if not log_path.exists():
        return info

    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            m = _PROGRESS_RE.match(line.strip())
            if not m or m.group(1) != stage:
                continue
            payload = m.group(2).split()
            for token in payload:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                if key in {"phase"}:
                    info[key] = value
                else:
                    try:
                        info[key] = int(value)
                    except ValueError:
                        info[key] = value
    return info


def _job_states(job_ids):
    states = {job_id: "finished" for job_id in job_ids}
    user = os.environ.get("USER", "")
    result = subprocess.run(
        ["qstat", "-u", user],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if result.returncode != 0:
        return states

    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        job_id = parts[0]
        if job_id in states:
            states[job_id] = parts[4]
    return states


def _log_phase_hint(log_path: Path, stage: str) -> Optional[str]:
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None

    if "Traceback" in text or "ERROR:" in text:
        return "failed"

    if stage == "build":
        if "Processing " in text:
            return "epoch_table"
        if "Bronze → Silver..." in text:
            return "normalize"
        if "] wrote " in text or "[alerce]" in text or "[fink]" in text or "[lasair]" in text:
            return "backfill"
    else:
        if "=== Fetching " in text:
            return "fetch"
        if "=== Collecting " in text:
            return "collect"
    return None


def _has_failure(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return False
    return "Traceback" in text or "ERROR:" in text


def _bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "-" * width
    frac = max(0.0, min(1.0, current / total))
    done = int(round(frac * width))
    return "#" * done + "-" * (width - done)


def _render_line(name: str, state: str, info: Dict[str, ProgressValue]) -> str:
    current = int(info.get("current") or 0)
    total = int(info.get("total") or 0)
    phase = str(info.get("phase") or "waiting")
    bar = _bar(current, total)

    detail = f"{current}/{total}" if total else phase
    if name == "download":
        ok = int(info.get("ok") or 0)
        failed = int(info.get("failed") or 0)
        detail = f"{detail} ok={ok} failed={failed} phase={phase}"
    else:
        rows = int(info.get("rows") or 0)
        skipped = int(info.get("skipped") or 0)
        detail = f"{detail} rows={rows} skipped={skipped} phase={phase}"

    return f"{name:>8} [{bar}] {detail} state={state}"


def _is_complete(state: str, info: Dict[str, ProgressValue], log_path: Path) -> bool:
    if str(info.get("phase")) == "complete":
        return True
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return False
    if "Traceback" in text or "ERROR:" in text:
        return False
    return state == "finished" and ("Done:" in text or "Wrote " in text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch SCC CPU prep progress")
    parser.add_argument("--meta", default="logs/cpu_prep_latest.json")
    parser.add_argument("--poll", type=float, default=5.0)
    args = parser.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        print(f"Metadata file not found: {meta_path}")
        print("Run: bash jobs/submit_cpu_prep.sh --limit 2000")
        sys.exit(1)

    with open(meta_path) as fh:
        meta = json.load(fh)

    download_job = str(meta["download_job_id"])
    build_job = str(meta["build_job_id"])
    download_log = _resolve_log_path(meta_path, meta.get("download_log"), "download", download_job)
    build_log = _resolve_log_path(meta_path, meta.get("build_log"), "build_epochs", build_job)

    try:
        while True:
            active_jobs = [job_id for job_id in (download_job, build_job) if job_id.isdigit()]
            states = _job_states(active_jobs)
            states.setdefault(download_job, "skipped" if download_job == "skipped" else "finished")
            states.setdefault(build_job, "finished")

            dl_info = _parse_progress(download_log, "download") if download_log else {"phase": "skipped"}
            ep_info = _parse_progress(build_log, "build") if build_log else {"phase": "missing_log"}

            if download_log and not dl_info.get("phase"):
                dl_info["phase"] = _log_phase_hint(download_log, "download")
            if build_log and not ep_info.get("phase"):
                ep_info["phase"] = _log_phase_hint(build_log, "build")

            if download_log and _has_failure(download_log):
                states[download_job] = "failed"
            if build_log and _has_failure(build_log):
                states[build_job] = "failed"

            os.system("clear")
            print("DEBASS CPU Prep Watch")
            print(f"meta: {meta_path}")
            print(f"download log: {download_log or '<not recorded>'}")
            print(f"build log:    {build_log or '<not recorded>'}")
            print("")
            print(_render_line("download", states[download_job], dl_info))
            print(_render_line("build", states[build_job], ep_info))

            download_complete = download_job == "skipped" or (
                download_log is not None and _is_complete(states[download_job], dl_info, download_log)
            )
            build_complete = build_log is not None and _is_complete(states[build_job], ep_info, build_log)

            if download_complete and build_complete:
                print("\nCPU prep complete.")
                break

            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nStopped watcher.")


if __name__ == "__main__":
    main()
