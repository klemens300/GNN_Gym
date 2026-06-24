#!/usr/bin/env bash
# run_until_done.sh — auto-recycle wrapper for long unattended KMC runs.
#
# What this does:
#   Runs `python -m KMC.main <config> --resume --max_steps_this_session N`
#   in a loop. After each Python invocation:
#     exit 0  -> full config target reached, exit the loop.
#     exit 99 -> session-cap reached (process recycled by design),
#                wait briefly and start the next session.
#     any other exit -> crash (CUDA OOB, segfault, OOM, ...). Wait
#                       longer to let the GPU/driver settle, then
#                       resume from the last good checkpoint.
#
# Why this exists:
#   On the RTX 5090 / Driver 591.86 / CUDA 13.1 stack we observe a
#   crash window between roughly 6000-12500 BKL steps per Python
#   process. By voluntarily ending each session well below that
#   window (default 5000 steps) and starting a fresh process, the
#   accumulated GPU/driver state is reset before it ever reaches
#   the failure regime. The bit-identical resume means the resulting
#   trajectory is the same as an uninterrupted single-process run.
#
# Usage:
#   ./run_until_done.sh <config.json> [session_steps] [max_restarts]
#
# Defaults:
#   session_steps  = 5000
#   max_restarts   = 500   # safety net; ~5000 steps * 500 = 2.5 M steps
#
# Exit codes from this script:
#   0  -> run completed (Python exited 0).
#   1  -> max_restarts hit before completion.
#   2  -> usage error.

set -u

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.json> [session_steps] [max_restarts]" >&2
    exit 2
fi

CONFIG="$1"
SESSION_STEPS="${2:-5000}"
MAX_RESTARTS="${3:-500}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG" >&2
    exit 2
fi

# Make path absolute so cd later does not break it.
CONFIG="$(readlink -f "$CONFIG")"

# Locate the run output directory and put the wrapper log there. We keep
# this lightweight (jq is not available everywhere) and parse output_dir
# from the JSON via grep+sed.
OUTPUT_DIR="$(grep -oP '"output_dir"\s*:\s*"\K[^"]+' "$CONFIG" || true)"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(dirname "$CONFIG")"
    echo "  (could not parse output_dir from config, using $OUTPUT_DIR)" >&2
fi
mkdir -p "$OUTPUT_DIR"

WRAPPER_LOG="$OUTPUT_DIR/run_until_done.log"

log() {
    local msg="$1"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $msg" | tee -a "$WRAPPER_LOG"
}

# ---------------------------------------------------------------------------
# Locate the KMC scipts/ directory so `python -m KMC.main` resolves.
# ---------------------------------------------------------------------------

# This script lives at scipts/KMC/run_until_done.sh; one level up is the
# scipts/ directory which has KMC/ as a package.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$SCIPTS_DIR/KMC" ]]; then
    echo "ERROR: cannot find KMC package at $SCIPTS_DIR/KMC" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Restart loop
# ---------------------------------------------------------------------------

log "=== run_until_done starting ==="
log "    config        : $CONFIG"
log "    output_dir    : $OUTPUT_DIR"
log "    session_steps : $SESSION_STEPS"
log "    max_restarts  : $MAX_RESTARTS"

count=0
crash_count=0
session_count=0
start_ts=$(date +%s)

while [[ $count -lt $MAX_RESTARTS ]]; do
    count=$((count + 1))
    log "--- attempt $count ---"

    # Run the KMC entry point. Python's stdout/stderr is teed into the
    # same wrapper log so all crash/output context is co-located.
    cd "$SCIPTS_DIR"
    python -m KMC.main \
        "$CONFIG" \
        --resume \
        --max_steps_this_session "$SESSION_STEPS" \
        2>&1 | tee -a "$WRAPPER_LOG"
    EXIT=${PIPESTATUS[0]}

    case "$EXIT" in
        0)
            elapsed=$(( $(date +%s) - start_ts ))
            log "complete: KMC exited 0 (target reached). " \
                "wallclock=${elapsed}s, sessions=$session_count, " \
                "crashes=$crash_count, attempts=$count"
            exit 0
            ;;
        99)
            session_count=$((session_count + 1))
            log "session cap: KMC exited 99 (designed recycle). " \
                "sessions so far: $session_count"
            sleep 3
            ;;
        *)
            crash_count=$((crash_count + 1))
            log "crash: KMC exited $EXIT. crashes so far: $crash_count. " \
                "sleeping 30s before next resume"
            sleep 30
            ;;
    esac
done

elapsed=$(( $(date +%s) - start_ts ))
log "ABORTED: max_restarts=$MAX_RESTARTS hit. " \
    "wallclock=${elapsed}s, sessions=$session_count, " \
    "crashes=$crash_count"
exit 1
