#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_URL="https://rocm.nightlies.amd.com/whl-multi-arch/"

echo "=== Installing ROCm torch with device packages ==="

# Install torch, torchvision, and the required device packages for gfx1201
# (AMD Radeon AI PRO R9700 / RX 9070 - both report gfx1201).
# All versions are pinned via torch-overrides.txt
#
# IMPORTANT: We install the amd-*-device-gfx1201 packages EXPLICITLY.
# These carry the per-arch kernel packs:
#   torch/.kpack/torch_gfx1201.kpack
#   torchvision/.kpack/torchvision_gfx1201.kpack
# torch loads its compute kernels from these at runtime (see the
# .rocm_kpack_ref section in libtorch_hip.so). They are normally pulled in
# transitively by the torch[device-gfx1201] extra, but combining that extra
# with --overrides can drop them, leaving torch with NO device kernels and
# producing "HIP error: device kernel image is invalid"
# (kpack_load_code_object failed with error: 13) on every GPU op.
uv pip install \
    "rocm[libraries, device-gfx1201]" \
    "torch[device-gfx1201]" \
    "torchvision[device-gfx1201]" \
    "amd-torch-device-gfx1201" \
    "amd-torchvision-device-gfx1201" \
    --index-url "${INDEX_URL}" \
    --overrides torch-overrides.txt \
    --force-reinstall

# Sanity check: the torch kernel pack MUST exist, otherwise all GPU ops fail.
TORCH_KPACK="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),".kpack","torch_gfx1201.kpack"))')"
if [[ ! -f "${TORCH_KPACK}" ]]; then
    echo "ERROR: missing torch kernel pack: ${TORCH_KPACK}" >&2
    echo "       GPU ops will fail with hipErrorInvalidImage. Aborting." >&2
    exit 1
fi
echo "Found torch kernel pack: ${TORCH_KPACK}"

echo "=== ROCm torch installation complete ==="
.venv/bin/python -c "import torch; print(f'torch: {torch.__version__}'); print(f'device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"