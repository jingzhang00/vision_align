#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$WORKSPACE/.venv/bin/python"

"$PYTHON" -m repaint.nymeria.nymeria_vda_pipeline \
    --dataset-root "$WORKSPACE/dataset/nymeria" \
    --output-root "$WORKSPACE/nymeria/vda" \
    --headless
