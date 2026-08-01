#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$WORKSPACE/.venv/bin/python"
EGODEX_SOURCE="$WORKSPACE/dataset/egodex_test"
EGODEX_FIRST="$WORKSPACE/dataset/.egodex_test_first"

tasks=()
for source in "$EGODEX_SOURCE"/*; do
    [[ -d "$source" ]] || continue
    task="${source##*/}"
    [[ -f "$source/0.hdf5" && -f "$source/0.mp4" ]] || {
        echo "Missing episode 0 pair: $source" >&2
        exit 1
    }
    mkdir -p "$EGODEX_FIRST/$task"
    ln -f "$source/0.hdf5" "$source/0.mp4" "$EGODEX_FIRST/$task/"
    tasks+=("$task")
done

((${#tasks[@]})) || {
    echo "No EgoDex tasks found under $EGODEX_SOURCE" >&2
    exit 1
}

"$PYTHON" -m repaint.egodex.egodex_vda_pipeline \
    --tasks "${tasks[@]}" \
    --dataset-root "$EGODEX_FIRST" \
    --output-root "$WORKSPACE/egodex" \
    --headless

"$PYTHON" -m repaint.nymeria.nymeria_vda_pipeline \
    --dataset-root "$WORKSPACE/dataset/nymeria" \
    --output-root "$WORKSPACE/nymeria/vda" \
    --headless
