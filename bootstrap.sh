#!/usr/bin/env bash
set -eo pipefail

if [ -z "$PYTHON_BIN_PATH" ]; then
    PYTHON_BIN_PATH=$(which python3 || true)
fi

# Set all env variables
TOP_DIR=$(dirname "$0")
"$PYTHON_BIN_PATH" "${TOP_DIR}/tools/bootstrap.py" "$@"

echo "Configuration finished"
