#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_URL="https://rocm.nightlies.amd.com/whl-multi-arch/"

echo "=== Installing ROCm torch with device packages ==="

# Install torch, torchvision, and the required device packages for gfx1201
# (AMD Radeon AI PRO R9700 / RX 9070 - both report gfx1201).
# All versions are pinned via torch-overrides.txt
#
# IMPORTANT: two SEPARATE device packages are needed, both installed
# EXPLICITLY because combining the [device-gfx1201] extra with --overrides
# drops the transitive deps:
#
#   amd-torch-device-gfx1201   -> torch/.kpack/torch_gfx1201.kpack
#       The COMPUTE kernels. torch loads these at runtime (see the
#       .rocm_kpack_ref section in libtorch_hip.so). Missing -> every GPU op
#       fails: "HIP error: device kernel image is invalid"
#       (kpack_load_code_object failed with error: 13).
#
#   amd-torch-device-gfx12-0   -> torch/lib/aotriton.images/amd-gfx120x/*.aks2
#       The FLASH/MEM-EFFICIENT ATTENTION kernels (aotriton). This is the
#       gfx12 FAMILY package, NOT the gfx1201-specific one. Missing -> flash
#       SDPA decompresses to a null kernel and crashes with "HIP error:
#       invalid argument", forcing the ~8x slower MATH fallback.
#       (Only present in torch >= 2.12.0.)
#
uv pip install \
    "rocm[libraries, device-gfx1201]" \
    "torch[device-gfx1201]" \
    "torchvision[device-gfx1201]" \
    "amd-torch-device-gfx1201" \
    "amd-torch-device-gfx12-0" \
    "amd-torchvision-device-gfx1201" \
    --index-url "${INDEX_URL}" \
    --overrides torch-overrides.txt \
    --force-reinstall

# Sanity check 1: the torch compute kernel pack MUST exist, otherwise every
# GPU op fails with "HIP error: device kernel image is invalid".
TORCH_KPACK="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),".kpack","torch_gfx1201.kpack"))')"
if [[ ! -f "${TORCH_KPACK}" ]]; then
    echo "ERROR: missing torch kernel pack: ${TORCH_KPACK}" >&2
    echo "       GPU ops will fail with hipErrorInvalidImage. Aborting." >&2
    exit 1
fi
echo "Found torch kernel pack: ${TORCH_KPACK}"

# Sanity check 2: the aotriton flash-attention kernel images MUST exist,
# otherwise flash/mem-efficient SDPA decompresses to a null kernel and crashes
# with "HIP error: invalid argument", forcing the slow MATH fallback.
# These are only present in torch >= 2.12.0 multi-arch wheels.
AOT_DIR="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib","aotriton.images","amd-gfx120x"))')"
if [[ ! -d "${AOT_DIR}" ]]; then
    echo "ERROR: missing aotriton gfx120x flash kernel images: ${AOT_DIR}" >&2
    echo "       Flash attention will crash. Pin torch >= 2.12.0 in torch-overrides.txt." >&2
    exit 1
fi
echo "Found aotriton gfx120x flash kernels: ${AOT_DIR}"

echo "=== ROCm torch installation complete ==="
.venv/bin/python -c "import torch; print(f'torch: {torch.__version__}'); print(f'device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"