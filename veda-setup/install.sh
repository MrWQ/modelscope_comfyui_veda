#!/usr/bin/env bash
# Veda-on-ComfyUI one-click setup entry (LF line endings required)
# 仅负责找到 Python 解释器后调用 setup_veda.py，自身零业务逻辑。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "ERROR: Python 3.10-3.12 not found. Please install Python first." >&2
  exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/setup_veda.py" "$@"
