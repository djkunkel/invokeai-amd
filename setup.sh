#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "=== Creating venv ==="
uv venv --relocatable --prompt invoke --python 3.12 --python-preference only-managed "${VENV_DIR}" --clear
source "${VENV_DIR}/bin/activate"

echo "=== Installing invokeai with CPU torch ==="
uv pip install invokeai==6.13.0 --python 3.12 --python-preference only-managed --torch-backend=cpu --force-reinstall

echo "=== Installing ROCm torch with device packages ==="
./retorch.sh

echo "=== Setting up bitsandbytes ROCm shim ==="
./bnb.sh

deactivate

echo "=== Setup complete ==="
echo "To activate: source .venv/bin/activate"
echo "To run: ./run.sh"