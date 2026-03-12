#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/base.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_step() {
  local title="$1"
  shift
  echo ""
  echo "========== ${title} =========="
  echo "Command: $*"
  "$@"
}

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import torch, yaml" >/dev/null 2>&1; then
  echo "Missing required Python packages. Please run: pip install -r requirement.txt" >&2
  exit 1
fi

echo "Project root: $ROOT_DIR"
echo "Using config: $CONFIG_PATH"
echo "Python bin : $PYTHON_BIN"

cd "$ROOT_DIR"

run_step "Train (semantic + mvcl)" \
  "$PYTHON_BIN" -m ufz train --stage all --config "$CONFIG_PATH"

run_step "Cluster" \
  "$PYTHON_BIN" -m ufz cluster --config "$CONFIG_PATH"

run_step "Export map" \
  "$PYTHON_BIN" -m ufz export --format map --config "$CONFIG_PATH"

run_step "Export GraphRAG" \
  "$PYTHON_BIN" -m ufz export --format graphrag --config "$CONFIG_PATH"

run_step "Visualize graph" \
  "$PYTHON_BIN" -m ufz visualize --type graph --config "$CONFIG_PATH"

run_step "Visualize embedding" \
  "$PYTHON_BIN" -m ufz visualize --type embedding --config "$CONFIG_PATH"

run_step "Visualize cluster" \
  "$PYTHON_BIN" -m ufz visualize --type cluster --config "$CONFIG_PATH"

echo ""
echo "Pipeline finished successfully."
