#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR/.."

if [ ! -d ".venv" ]; then
    python3.9 -m venv .venv --version
fi

python3 --version

.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/python3 -m pip install -r requirements.txt

export CUDNN_PATH="$HOME/.local/lib/python3.9/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH:-}"

echo "Virtual environment setup complete"
