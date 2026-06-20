# ---------------------------------------------------------------------------
# bitsandbytes ROCm compatibility shim
#
# BNB ships .so files up to rocm72 (HIP 7.2). Newer ROCm builds use higher
# HIP versions (e.g. HIP 7.13 → key 83, HIP 7.14 → key 84). The key is
# computed as major*10 + minor. We detect the actual HIP version at startup
# and symlink rocm72 → the required key. ABI-compatible.
# Remove these lines once BNB ships .so files for the HIP version in use.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BNB_DIR="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/bitsandbytes"
BNB_SOURCE="${BNB_DIR}/libbitsandbytes_rocm72.so"
if [[ -e "${BNB_SOURCE}" ]]; then
    # Detect HIP version from torch and compute the BNB key
    HIP_VER="$("${SCRIPT_DIR}/.venv/bin/python3" -c "import torch; print(torch.version.hip)" 2>/dev/null || true)"
    if [[ -n "${HIP_VER}" ]]; then
        HIP_MAJOR="${HIP_VER%%.*}"
        HIP_REST="${HIP_VER#*.}"
        HIP_MINOR="${HIP_REST%%.*}"
        BNB_KEY=$(( HIP_MAJOR * 10 + HIP_MINOR ))
        BNB_TARGET="${BNB_DIR}/libbitsandbytes_rocm${BNB_KEY}.so"
        if [[ ! -e "${BNB_TARGET}" ]]; then
            ln -sf "${BNB_SOURCE}" "${BNB_TARGET}"
            echo "BNB shim: symlinked rocm72 → rocm${BNB_KEY} (HIP ${HIP_VER})"
        fi
    fi
fi
