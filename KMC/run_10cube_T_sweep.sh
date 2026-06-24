#!/usr/bin/env bash
# Run the 10-cube MoNbTaW T-sweep (1000 / 1500 / 2000 K) sequentially.
# Each temperature uses run_until_done.sh, so it auto-resumes on crash
# and recycles the Python process every `session_steps` BKL steps.
#
# Usage (recommended via tmux):
#     tmux new -d -s kmc10 \
#         "/path/to/GNN_Gym/KMC/run_10cube_T_sweep.sh"
#     tmux attach -t kmc10

set -u

REPO="/path/to/GNN_Gym"
WRAPPER="${REPO}/scipts/KMC/run_until_done.sh"
SESSION_STEPS=5000
MAX_RESTARTS=500

TEMPS=(1000 1500 2000)

echo "[sweep] starting 10-cube MoNbTaW sweep at $(date -Iseconds)"
echo "[sweep] temperatures: ${TEMPS[*]} K"

for T in "${TEMPS[@]}"; do
    CONFIG="${REPO}/runs/quad_slab_10cube_${T}K/config.json"
    if [[ ! -f "${CONFIG}" ]]; then
        echo "[sweep] !! missing config: ${CONFIG} -- skipping"
        continue
    fi

    echo "[sweep] === T = ${T} K === at $(date -Iseconds)"
    # run_until_done.sh handles its own retry / resume loop
    "${WRAPPER}" "${CONFIG}" "${SESSION_STEPS}" "${MAX_RESTARTS}"
    RC=$?
    echo "[sweep] T=${T}K finished with exit ${RC} at $(date -Iseconds)"

    if [[ ${RC} -ne 0 ]]; then
        echo "[sweep] !! non-zero exit at ${T} K; continuing to next T anyway"
    fi
done

echo "[sweep] all temperatures done at $(date -Iseconds)"
