#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "verify" ]]; then
    shift
    exec python /opt/ptv3/verify_rtx5090.py "$@"
fi

exec "$@"
