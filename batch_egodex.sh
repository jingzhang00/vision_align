#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$WORKSPACE/.venv/bin/python"
EGODEX_SOURCE="$WORKSPACE/dataset/egodex_test"

tasks=()
for source in "$EGODEX_SOURCE"/*; do
    [[ -d "$source" ]] || continue
    task="${source##*/}"
    [[ -f "$source/0.hdf5" && -f "$source/0.mp4" ]] || {
        echo "Skipping missing episode 0 pair: $source" >&2
        continue
    }
    tasks+=("$task")
done

((${#tasks[@]})) || {
    echo "No EgoDex tasks found under $EGODEX_SOURCE" >&2
    exit 1
}

"$PYTHON" -m repaint.egodex.egodex_vda_pipeline \
    --tasks "${tasks[@]}" \
    --dataset-root "$EGODEX_SOURCE" \
    --output-root "$WORKSPACE/egodex" \
    --stem 0 \
    --batch \
    --headless
