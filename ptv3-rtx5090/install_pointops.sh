#!/usr/bin/env bash
set -euo pipefail

pointops_dir="${1:-/workspace/libs/pointops}"
setup_file="${pointops_dir}/setup.py"

if [[ ! -f "${setup_file}" ]]; then
    echo "No se encontró ${setup_file}" >&2
    exit 2
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"

python -m pip install --verbose --no-build-isolation -e "${pointops_dir}"
python - <<'PY'
import torch
import pointops

print("pointops import OK", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
