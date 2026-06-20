#!/bin/bash
# Test script for PyTorch ROCm functionality
# Usage: ./test_torch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate venv and set env vars
source "$SCRIPT_DIR/.venv/bin/activate"

# Set ROCm library path for when ROCm is bundled in venv
export LD_LIBRARY_PATH="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$SCRIPT_DIR/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib:$SCRIPT_DIR/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Hide iGPU if present (use GPU 0 only)
export HIP_VISIBLE_DEVICES=0

echo "============================================"
echo "PyTorch ROCm Test Script"
echo "============================================"
echo ""

# Print environment info
echo "=== Environment ==="
python -c "
import torch, sys
print(f'Python: {sys.version.split()[0]}')
print(f'Torch: {torch.__version__}')
print(f'ROCm version: {torch.version.hip}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} (arch {props.major}.{props.minor}, {props.total_memory/1024**3:.1f}GB)')
" 2>/dev/null

echo ""
echo "=== Basic Tensor Operations ==="
python -c "
import torch

tests = [
    ('creation', lambda: torch.tensor([1.0], device='cuda')),
    ('addition', lambda: torch.tensor([1.0], device='cuda') + 1),
    ('subtraction', lambda: torch.tensor([1.0], device='cuda') - 1),
    ('multiplication', lambda: torch.tensor([2.0], device='cuda') * 3),
    ('division', lambda: torch.tensor([6.0], device='cuda') / 2),
    ('matmul', lambda: torch.randn(100, 100, device='cuda') @ torch.randn(100, 100, device='cuda')),
    ('sum', lambda: torch.randn(1000, device='cuda').sum()),
    ('mean', lambda: torch.randn(1000, device='cuda').mean()),
]

passed = 0
failed = 0
for name, test_fn in tests:
    try:
        result = test_fn()
        if hasattr(result, 'cpu'):
            result = result.cpu()
        print(f'  {name}: PASS')
        passed += 1
    except Exception as e:
        print(f'  {name}: FAIL - {type(e).__name__}')
        failed += 1

print(f'\nResults: {passed} passed, {failed} failed')
"

echo ""
echo "=== Neural Network Operations ==="
python -c "
import torch
import torch.nn as nn

tests = [
    ('Linear forward', lambda: nn.Linear(512, 128).to('cuda')(torch.randn(1, 512, device='cuda'))),
    ('Conv2d forward', lambda: nn.Conv2d(3, 64, 3).to('cuda')(torch.randn(1, 3, 32, 32, device='cuda'))),
    ('ReLU', lambda: nn.ReLU().to('cuda')(torch.randn(1, 100, device='cuda'))),
    ('Softmax', lambda: nn.Softmax(dim=1).to('cuda')(torch.randn(1, 100, device='cuda'))),
    ('Dropout forward', lambda: nn.Dropout(0.5).to('cuda')(torch.randn(1, 100, device='cuda'))),
]

passed = 0
failed = 0
for name, test_fn in tests:
    try:
        result = test_fn()
        if hasattr(result, 'cpu'):
            result = result.cpu()
        print(f'  {name}: PASS')
        passed += 1
    except Exception as e:
        print(f'  {name}: FAIL - {type(e).__name__}')
        failed += 1

print(f'\nResults: {passed} passed, {failed} failed')
"

echo ""
echo "=== TorchVision (if available) ==="
python -c "
import torch
try:
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Test basic transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Test tensor creation on GPU
    tensor = torch.randn(3, 224, 224, device='cuda')
    normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    print(f'  Basic transforms: PASS')

    # Test a simple model
    import torchvision.models as models
    model = models.resnet18(weights=None).to('cuda')
    x = torch.randn(1, 3, 224, 224, device='cuda')
    y = model(x)
    print(f'  ResNet18 forward: PASS')

    print('TorchVision: All tests PASS')
except ImportError as e:
    print(f'  TorchVision not installed: {e}')
except Exception as e:
    print(f'  TorchVision test FAIL: {type(e).__name__}: {e}')
"

echo ""
echo "=== Memory Test ==="
python -c "
import torch

# Check GPU memory
if torch.cuda.is_available():
    print(f'  Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
    print(f'  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB')

    # Test large tensor allocation
    try:
        large = torch.randn(1024, 1024, 512, device='cuda')
        print(f'  Large tensor (4GB): PASS')
    except Exception as e:
        print(f'  Large tensor FAIL: {e}')
else:
    print('  CUDA not available')
"

echo ""
echo "============================================"
echo "Test complete"
echo "============================================"