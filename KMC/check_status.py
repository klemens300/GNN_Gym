#!/usr/bin/env python3
"""
Quick status check for KMC ensemble campaigns.

Scans run folders for per-realisation rolling checkpoints
(``checkpoint_T<T>K_real<k>.npz``) and reports, for each temperature and
realisation, how many BKL steps are done out of the configured target,
the simulated time reached, and how long ago the checkpoint was last
written (a stale timestamp means that realisation is not currently
advancing). Only numpy + the standard library are imported, so this is
safe to run on the GPU VM while a campaign is in progress -- it never
touches torch and only reads the two scalar fields it needs from each
.npz, so it is cheap even on large checkpoints.

ETA
---
The per-run throughput (ms/step) is the *instantaneous* rate, measured
from the wall-clock difference between the last two progress lines of the
active session (so a ``--resume`` does not dilute it -- see ``_read_log``).
A per-run ETA is the run's remaining steps times that rate. Runs that have
already finished (``diffusion_summary.csv`` present) contribute zero
remaining steps and never inflate the estimate. Because the batch wrapper
runs the configured runs *sequentially* (one GPU job at a time), the
campaign ETA is the sum of all remaining steps times the throughput of the
currently active run.

Usage
-----
    python check_status.py                 # scans ./runs or <repo>/runs
    python check_status.py /path/to/runs   # scans a specific runs dir
    python check_status.py /path/to/runs/MoNbTaW_quadslab_8x8x8_1000K_ens8
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

CKPT_RE = re.compile(r"checkpoint_T(?P<T>[-\d.]+)K_real(?P<k>\d+)\.npz$")
MS_RE = re.compile(r"([\d.]+)\s*ms/step")
# Progress line, e.g.:
#   [run] 46000/50000 ( 92.0%) | sim t = ... | elapsed 0:04:16 | ETA ... | ...
# Captures the absolute step and the session elapsed time (H:MM:SS).
PROGRESS_RE = re.compile(r"\]\s*(\d+)/\d+.*?elapsed\s+(\d+):(\d+):(\d+)")


def _fmt_age(seconds: float) -> str:
    """Human-readable 'time since last write'."""
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def _fmt_eta(seconds) -> str:
    """Human-readable remaining time."""
    if seconds is None or seconds < 0:
        return "n/a"
    s = int(round(seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, _ = divmod(s, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def _read_checkpoint_progress(path: Path) -> dict | None:
    """Read only the step counter, sim time and mtime from a checkpoint."""
    try:
        with np.load(path, allow_pickle=False) as data:
            step = int(data["n_steps"])
            sim_t = float(data["total_time_s"])
    except Exception:
        return None
    return {
        "step": step,
        "sim_t": sim_t,
        "age_s": time.time() - path.stat().st_mtime,
    }


def _load_config(run_dir: Path) -> dict:
    """Load config_used.json (preferred) or config.json from a run dir."""
    for name in ("config_used.json", "config.json"):
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


def _read_log(run_dir: Path):
    """Return (last_non_empty_line, ms_per_step) from run_until_done.log.

    ``ms_per_step`` is the *instantaneous* throughput, measured as the
    wall-clock difference between the last two progress lines of the
    current session (delta_elapsed / delta_step). This is robust to
    ``--resume``: a resumed session keeps climbing the absolute step
    counter but resets its elapsed clock to zero, so the per-line
    cumulative ``ms/step`` printed in the log (elapsed / absolute_step)
    is diluted by the steps already done in earlier sessions. Taking the
    difference of two same-session lines cancels that bias out. Falls
    back to the printed cumulative value only when fewer than two usable
    progress lines from one session are available.
    """
    p = run_dir / "run_until_done.log"
    if not p.exists():
        return None, None
    tail = None
    ms_cumulative = None
    points = []  # (abs_step, elapsed_s) per progress line, in file order
    for ln in p.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        tail = ln
        m = MS_RE.search(ln)
        if m:
            ms_cumulative = float(m.group(1))
        pm = PROGRESS_RE.search(ln)
        if pm:
            step = int(pm.group(1))
            h, mnt, sec = int(pm.group(2)), int(pm.group(3)), int(pm.group(4))
            points.append((step, h * 3600 + mnt * 60 + sec))

    # Walk backwards for the last consecutive pair from the SAME session:
    # within a session both step and elapsed increase; at a --resume
    # boundary elapsed drops back toward zero while step keeps climbing.
    ms_measured = None
    for i in range(len(points) - 1, 0, -1):
        (s1, e1), (s0, e0) = points[i], points[i - 1]
        if s1 > s0 and e1 > e0:
            ms_measured = (e1 - e0) / (s1 - s0) * 1000.0
            break

    return tail, (ms_measured if ms_measured is not None else ms_cumulative)


def _scan_run(run_dir: Path) -> dict | None:
    """Collect status for a single run folder."""
    ckpts = {}
    for f in run_dir.glob("checkpoint_T*K_real*.npz"):
        m = CKPT_RE.search(f.name)
        if not m:
            continue
        prog = _read_checkpoint_progress(f)
        if prog is not None:
            ckpts[int(m.group("k"))] = prog

    cfg = _load_config(run_dir)
    has_summary = (run_dir / "diffusion_summary.csv").exists()
    if not ckpts and not cfg and not has_summary:
        return None  # not a KMC run folder

    target = int(cfg.get("n_steps") or 0)
    n_real = int(cfg.get("n_realizations_per_T") or 0)
    if n_real == 0 and ckpts:
        n_real = max(ckpts) + 1

    log_tail, ms_per_step = _read_log(run_dir)
    steps_done = sum(p["step"] for p in ckpts.values())
    steps_total = target * n_real if (target and n_real) else 0
    # A finished run (diffusion_summary.csv written) has no remaining work,
    # regardless of what the checkpoints say -- never let a completed run
    # leak steps into the campaign ETA.
    if has_summary:
        remaining = 0
    else:
        remaining = max(0, steps_total - steps_done) if steps_total else 0
    min_age = min((p["age_s"] for p in ckpts.values()), default=float("inf"))

    return {
        "name": run_dir.name,
        "target": target,
        "n_real": n_real,
        "ckpts": ckpts,
        "has_summary": has_summary,
        "log_tail": log_tail,
        "ms_per_step": ms_per_step,
        "steps_done": steps_done,
        "steps_total": steps_total,
        "remaining": remaining,
        "min_age": min_age,
    }


def _print_run(run: dict) -> None:
    """Print one run's status block."""
    target = run["target"]
    n_real = run["n_real"]
    ckpts = run["ckpts"]

    print(f"\n{run['name']}")
    if run["has_summary"]:
        print("  [COMPLETE] diffusion_summary.csv present.")

    target_str = str(target) if target else "?"
    n_iter = n_real if n_real else (max(ckpts) + 1 if ckpts else 0)
    for k in range(n_iter):
        prog = ckpts.get(k)
        if prog is None:
            print(f"  real{k}:  --- no checkpoint yet")
            continue
        pct = (100.0 * prog["step"] / target) if target else float("nan")
        pct_str = f"{pct:5.1f}%" if target else "  ?  "
        print(
            f"  real{k}:  {prog['step']:>6}/{target_str:<6} {pct_str}"
            f"   sim t = {prog['sim_t']:.3e} s"
            f"   ({_fmt_age(prog['age_s'])})"
        )

    if run["steps_total"]:
        line = (
            f"  -- ensemble: {run['steps_done']}/{run['steps_total']} steps "
            f"({100.0 * run['steps_done'] / run['steps_total']:.1f}%)"
        )
        if run["remaining"] > 0 and run["ms_per_step"]:
            eta = run["remaining"] * run["ms_per_step"] / 1000.0
            line += (
                f" | {run['ms_per_step']:.0f} ms/step"
                f" | ETA {_fmt_eta(eta)}"
            )
        print(line)
    if run["log_tail"]:
        # Drop the trailing per-line "... | N.NN ms/step" from the wrapper
        # log: it is the loop's cumulative (resume-diluted) figure and only
        # confuses next to the correct measured rate shown above.
        tail = re.sub(r"\s*\|\s*[\d.]+\s*ms/step\s*$", "", run["log_tail"])
        print(f"  wrapper log: {tail}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv:
        root = Path(argv[0]).expanduser().resolve()
    else:
        cand = Path.cwd() / "runs"
        if not cand.is_dir():
            cand = Path(__file__).resolve().parents[2] / "runs"
        root = cand

    if not root.exists():
        print(f"ERROR: path not found: {root}", file=sys.stderr)
        return 2

    if list(root.glob("checkpoint_T*K_real*.npz")) or (root / "config.json").exists():
        run_dirs = [root]
    else:
        run_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    print("=" * 70)
    print(f"KMC campaign status  ({root})")
    print(f"scanned at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    runs = []
    for d in run_dirs:
        run = _scan_run(d)
        if run is None:
            continue
        runs.append(run)
        _print_run(run)

    if not runs:
        print("\nNo KMC run folders found here.")
        return 0

    grand_done = sum(r["steps_done"] for r in runs)
    grand_total = sum(r["steps_total"] for r in runs)
    grand_remaining = sum(r["remaining"] for r in runs)

    # Campaign throughput = ms/step of the run that is currently advancing
    # (the one with the freshest checkpoint). The batch is sequential, so
    # the total remaining wall time uses this single rate.
    active = min(
        (r for r in runs if r["ms_per_step"] and r["remaining"] > 0),
        key=lambda r: r["min_age"],
        default=None,
    )
    campaign_eta = (
        grand_remaining * active["ms_per_step"] / 1000.0
        if active else None
    )

    print("\n" + "=" * 70)
    if grand_total:
        print(
            f"CAMPAIGN TOTAL: {grand_done}/{grand_total} steps "
            f"({100.0 * grand_done / grand_total:.1f}%) across "
            f"{len(runs)} run folder(s)."
        )
        if campaign_eta is not None:
            print(
                f"  ETA: {_fmt_eta(campaign_eta)}  "
                f"(remaining {grand_remaining} steps, sequential)"
            )
        else:
            print(
                f"  remaining: {grand_remaining} steps  (ETA: n/a)"
            )
    else:
        print(f"Found {len(runs)} run folder(s) (targets unknown).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
