#!/usr/bin/env bash
set -eu
set -o pipefail 2>/dev/null || true
export NK_TEMP=0.1
export NK_TOPK=1
exec /home/rossduberry/uci.sh "$@"
