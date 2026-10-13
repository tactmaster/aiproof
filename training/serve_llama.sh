#!/usr/bin/env bash
# Serve one GGUF with llama.cpp's server for aiproof's "llama.cpp server"
# provider preset (OpenAI-compatible, 127.0.0.1:8080/v1/chat/completions).
#
# Why: Ollama adds ~0.5 s of scheduler/load overhead to every request and
# reloads evicted models from disk (10-35 s); a dedicated llama-server keeps
# one model resident with per-request overhead in the milliseconds and the
# prompt prefix cached.
#
#   training/serve_llama.sh ~/.local/share/aiproof/models/aiproof-edit-1.5b-q8_0.gguf
#   training/serve_llama.sh "$(ollama show qwen2.5:3b --modelfile | awk '/^FROM/{print $2; exit}')"
#
# Env: LLAMA_SERVER (binary), PORT (8080), CTX (4096), NGL (99 = all layers on GPU).
set -euo pipefail
MODEL=${1:?usage: serve_llama.sh MODEL.gguf [ALIAS]}
ALIAS=${2:-$(basename "$MODEL" .gguf)}
LLAMA_SERVER=${LLAMA_SERVER:-$HOME/3rdparty/llama.cpp/build/bin/llama-server}
PORT=${PORT:-8080}
CTX=${CTX:-4096}
NGL=${NGL:-99}
[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER (build llama.cpp with -DGGML_CUDA=ON)"; exit 1; }
exec "$LLAMA_SERVER" \
  --model "$MODEL" --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  --n-gpu-layers "$NGL" --ctx-size "$CTX" --parallel 1 \
  --flash-attn on --cache-reuse 256 --jinja
