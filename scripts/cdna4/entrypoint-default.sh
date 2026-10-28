#!/usr/bin/env bash
# Entrypoint of the public CDNA4 default image: print what this image is, then run the given command.
cat >&2 <<'BANNER'
================================================================================================
 Paiton public default (unoptimized): stock vLLM + AITER with AMD's MXFP4 model.
 The optimized Paiton engine for this GPU is proprietary; see https://eliovp.com/blog/paiton-qwen38-flash-next-amd-instinct-mi350p for more info.
================================================================================================
BANNER
if [[ $# -eq 0 || "$1" == -* ]]; then  # no command, or only vllm serve options: serve
  set -- vllm serve "$@"
fi
exec "$@"
