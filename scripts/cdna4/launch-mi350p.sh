#!/usr/bin/env bash
# Serve AMD's Quark MXFP4 checkpoint of Qwen3.8-Flash-Next on one AMD Instinct MI350P (CDNA4, gfx950) with stock
# vLLM 0.32 + AITER. This is the public default (unoptimized) path; it contains no Paiton engine.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/cdna4/launch-mi350p.sh [extra vllm serve arguments]

Starts an OpenAI-compatible server for amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8 on one MI350P.
The first start downloads the model (about 125 GB) into the Hugging Face cache; later starts reuse it.

Settings (environment variables):
  PAITON_IMAGE       container image   (default: ghcr.io/eliovp/paiton-vllm-plugin:cdna4-mxfp4-default-20261028)
  PAITON_MODEL       Hugging Face id or a local folder (default: amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8)
  PAITON_HF_CACHE    Hugging Face cache on the host    (default: ~/.cache/huggingface)
  PAITON_PORT        port                              (default: 8000)
  PAITON_BIND        listen address; 127.0.0.1 keeps the API on this machine (default: 0.0.0.0)
  PAITON_MAX_LEN     max context per request           (default: 262144)
  PAITON_MAX_SEQS    max concurrent requests           (default: 64)
  PAITON_GPU         GPU index                         (default: 0)
  PAITON_CONTAINER   container name                    (default: paiton-mi350p-default)
  PAITON_API_KEY     if set, /v1 requests must send this key

Power: our published numbers use a 450 W GPU power limit (the MI350P default is 600 W). This script does not change
it. To match them, as root: amd-smi set -g 0 --power-cap 450   (check with: amd-smi static -g 0 --limit)

Example:
  PAITON_BIND=127.0.0.1 scripts/cdna4/launch-mi350p.sh
  curl http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model": "qwen3.8-flash-next", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
EOF
}
[[ "${1:-}" == -h || "${1:-}" == --help ]] && { usage; exit 0; }

image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:cdna4-mxfp4-default-20261028}"
model="${PAITON_MODEL:-amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8}"
hf_cache="${PAITON_HF_CACHE:-$HOME/.cache/huggingface}"
port="${PAITON_PORT:-8000}"
bind="${PAITON_BIND:-0.0.0.0}"
name="${PAITON_CONTAINER:-paiton-mi350p-default}"
mkdir -p "$hf_cache"

extra=()
if [[ -d "$model" ]]; then  # local folder: mount it read-only
  extra+=(-v "$(realpath "$model"):/model:ro"); model=/model
fi
if [[ -n "${PAITON_API_KEY:-}" ]]; then
  export VLLM_API_KEY="$PAITON_API_KEY"; extra+=(-e VLLM_API_KEY)
fi

# Same settings as our MI350P benchmark row "default: AMD MXFP4 checkpoint on stock vLLM + AITER":
# AITER on, the n-gram (PLE) table in fp8, 92 % of GPU memory for weights + KV cache.
exec docker run --rm --name "$name" --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --shm-size 16g -e HIP_VISIBLE_DEVICES="${PAITON_GPU:-0}" -e VLLM_ROCM_USE_AITER=1 \
  -p "$bind:$port:8000" -v "$hf_cache:/root/.cache/huggingface" "${extra[@]}" "$image" \
  vllm serve "$model" --served-model-name qwen3.8-flash-next --host 0.0.0.0 --port 8000 \
    --max-model-len "${PAITON_MAX_LEN:-262144}" --max-num-seqs "${PAITON_MAX_SEQS:-64}" --gpu-memory-utilization 0.92 \
    --hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}' "$@"
