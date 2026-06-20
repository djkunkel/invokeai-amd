#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_URL="https://rocm.nightlies.amd.com/whl-multi-arch/"

echo "=== Installing ROCm torch with device packages ==="

# Install torch, torchvision, and the required device packages for gfx1201 (RX 9070)
# All versions are pinned via torch-overrides.txt
uv pip install \
    "rocm[libraries, device-gfx1201]" \
    "torch[device-gfx1201]" \
    "torchvision[device-gfx1201]" \
    --index-url "${INDEX_URL}" \
    --overrides torch-overrides.txt \
    --force-reinstall

echo "=== ROCm torch installation complete ==="
.venv/bin/python -c "import torch; print(f'torch: {torch.__version__}'); print(f'device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"