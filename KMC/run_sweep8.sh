#!/usr/bin/env bash
# Run the 8-cube MoNbTaW T-sweep (1000 / 1500 / 2000 K) sequentially.
# Each temperature uses run_until_done.sh, so it auto-resumes on crash
# and recycles the Python process every `session_steps` BKL steps.
#
# Usage (recommended via tmux):
#     tmux new -d -s kmc8 \
#         "/home/klechner/doctor/gnn_kmc/scipts/KMC/run_sweep8.sh"
#     tmux attach -t kmc8

set -u

# vGPU profile (GRID A100 / driver 525) does not support PyTorch's
# expandable_segments allocator. main.py setdefault()s it for Blackwell
# users, so we override it here with an empty value (which torch reads
# as "default allocator"). Without this the first CUDA op crashes with
#   RuntimeError: CUDA driver error: operation not supported
export PYTORCH_CUDA_ALLOC_CONF=

REPO="/home/klechner/doctor/gnn_kmc"
WRAPPER="${REPO}/scipts/KMC/run_until_done.sh"
SESSION_STEPS=5000
MAX_RESTARTS=500

TEMPS=(1000 1500 2000)

echo "[sweep] starting 8-cube MoNbTaW sweep at $(date -Iseconds)"
echo "[sweep] temperatures: ${TEMPS[*]} K"

for T in "${TEMPS[@]}"; do
    CONFIG="${REPO}/runs/sweep8_${T}K/config.json"
    if [[ ! -f "${CONFIG}" ]]; then
        echo "[sweep] !! missing config: ${CONFIG} -- skipping"
        continue
    fi

    echo "[sweep] === T = ${T} K === at $(date -Iseconds)"
    "${WRAPPER}" "${CONFIG}" "${SESSION_STEPS}" "${MAX_RESTARTS}"
    RC=$?
    echo "[sweep] T=${T}K finished with exit ${RC} at $(date -Iseconds)"

    if [[ ${RC} -ne 0 ]]; then
        echo "[sweep] !! non-zero exit at ${T} K; continuing to next T anyway"
    fi
done

echo "[sweep] all temperatures done at $(date -Iseconds)"
