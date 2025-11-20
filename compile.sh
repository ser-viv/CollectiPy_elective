#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN=""

# Pick an available Python 3 interpreter (prefers 3.10, then 3.x).
for candidate in \
    python3.10 \
    python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3 interpreter not found. Please install Python 3.x." >&2
    exit 1
fi

# Use the cross-platform setup helper to create the venv and install deps.
"$PYTHON_BIN" "$ROOT_DIR/tools/setup_env.py"
