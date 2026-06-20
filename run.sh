#!/bin/bash
source .venv/bin/activate

# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 enables aotriton flash/mem-efficient
# attention on gfx1201 (R9700). Without it, torch refuses the fast kernels and
# SDPA falls back to slow MATH (~8x slower for the image transformer).
# Requires torch >= 2.12.0, which bundles the gfx120x aotriton kernel images.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Provide the unversioned libnuma.so that libtorch_rocshmem.so dlopen()s
# (see numa.sh). Silences the harmless "rocSHMEM Could not open libnuma" notice.
export LD_LIBRARY_PATH="$(dirname "$(readlink -f "$0")")/.venv/numa-shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

INVOKEAI_PATCHMATCH=false invokeai-web --root /mnt/extra/invokeai
