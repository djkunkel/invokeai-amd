#!/bin/bash
source .venv/bin/activate

# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 enables aotriton flash/mem-efficient
# attention on gfx1201 (R9700). Without it, torch refuses the fast kernels and
# SDPA falls back to slow MATH (~8x slower for the image transformer).
# Requires torch >= 2.12.0, which bundles the gfx120x aotriton kernel images.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

HIP_VISIBLE_DEVICES=0 INVOKEAI_PATCHMATCH=false invokeai-web --root /mnt/extra/invokeai
