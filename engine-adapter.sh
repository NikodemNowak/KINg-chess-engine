#!/usr/bin/env bash
# =============================================================================
# engine-adapter.sh
#
# This script IS the "engine executable" from cutechess-cli's point of view.
# It forwards UCI/XBoard stdin↔stdout through a Docker container.
#
# Usage in cutechess-cli:
#   -engine cmd=/path/to/engine-adapter.sh name=MyEngine proto=uci
#
# Environment variables (all optional):
#   ENGINE_IMAGE   Docker image to run         (default: chess-engine:latest)
#   ENGINE_CMD     Command inside the container (default: image ENTRYPOINT)
#   ENGINE_GPU     Set to "0" to disable GPU    (default: auto-detect)
#   ENGINE_MEMORY  Docker memory limit          (default: 2g)
#   ENGINE_CPUS    Docker CPU quota             (default: unlimited)
#   ENGINE_NET     Docker network mode          (default: none)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE="${ENGINE_IMAGE:-chess-engine:latest}"
MEMORY="${ENGINE_MEMORY:-2g}"
CPUS="${ENGINE_CPUS:-}"
NETWORK="${ENGINE_NET:-none}"

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
GPU_FLAGS=()
if [[ "${ENGINE_GPU:-auto}" != "0" ]]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        GPU_FLAGS=(--gpus all)
    fi
fi

# ---------------------------------------------------------------------------
# Optional extra engine command (overrides ENTRYPOINT)
# ---------------------------------------------------------------------------
ENGINE_CMD_ARGS=()
if [[ -n "${ENGINE_CMD:-}" ]]; then
    # Split safely on spaces (no arrays with IFS trick to keep it portable)
    read -r -a ENGINE_CMD_ARGS <<< "${ENGINE_CMD}"
fi

# ---------------------------------------------------------------------------
# Build docker run arguments
# ---------------------------------------------------------------------------
DOCKER_ARGS=(
    run
    --rm
    --interactive           # keep stdin open — UCI over pipe
    --init                  # proper PID 1 / signal handling
    --memory "${MEMORY}"
    --network "${NETWORK}"
    "${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"}"
)

if [[ -n "${CPUS:-}" ]]; then
    DOCKER_ARGS+=(--cpus "${CPUS}")
fi

# Add any extra mounts / env vars passed via ENGINE_EXTRA_ARGS
if [[ -n "${ENGINE_EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_ARRAY <<< "${ENGINE_EXTRA_ARGS}"
    DOCKER_ARGS+=("${EXTRA_ARRAY[@]}")
fi

DOCKER_ARGS+=("${IMAGE}")

# Append engine command override if set
if [[ "${#ENGINE_CMD_ARGS[@]}" -gt 0 ]]; then
    DOCKER_ARGS+=("${ENGINE_CMD_ARGS[@]}")
fi

# ---------------------------------------------------------------------------
# Hand off — exec replaces this shell with docker so signals propagate cleanly
# ---------------------------------------------------------------------------
exec docker "${DOCKER_ARGS[@]}"
