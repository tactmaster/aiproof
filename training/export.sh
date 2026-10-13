#!/usr/bin/env bash
# Merge a trained LoRA run, convert to GGUF, write a Modelfile and import it
# into Ollama.
#
#   training/export.sh training/runs/qwen1.5b-edit aiproof-edit-1.5b [q8_0|f16]
#
# Env: TRAIN_PY (python with torch/peft; default: conda env aiproof-train),
#      LLAMA_CPP (checkout with convert_hf_to_gguf.py), MODELS_DIR.
set -euo pipefail
RUN=${1:?usage: export.sh RUN_DIR NAME [QUANT]}
NAME=${2:?usage: export.sh RUN_DIR NAME [QUANT]}
QUANT=${3:-q8_0}
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN_PY=${TRAIN_PY:-conda run -n aiproof-train python}
LLAMA_CPP=${LLAMA_CPP:-$HOME/3rdparty/llama.cpp}
MODELS_DIR=${MODELS_DIR:-$HOME/.local/share/aiproof/models}
GGUF="$MODELS_DIR/$NAME-$QUANT.gguf"
mkdir -p "$MODELS_DIR"

if [ ! -d "$RUN/merged" ]; then
  echo "== merging LoRA into the base model"
  $TRAIN_PY "$HERE/merge_lora.py" "$RUN" "$RUN/merged"
fi
echo "== converting to GGUF ($QUANT)"
$TRAIN_PY "$LLAMA_CPP/convert_hf_to_gguf.py" "$RUN/merged" --outtype "$QUANT" --outfile "$GGUF"
echo "== writing Modelfile"
PYTHONPATH="$HERE/../src" python3 "$HERE/modelfile.py" "$GGUF" > "$MODELS_DIR/$NAME.Modelfile"
echo "== ollama create $NAME"
ollama create "$NAME" -f "$MODELS_DIR/$NAME.Modelfile"
ollama show "$NAME" --modelfile | sed -n '1,6p'
echo
echo "GGUF:      $GGUF"
echo "Modelfile: $MODELS_DIR/$NAME.Modelfile"
echo "Try:       make eval EVAL_ARGS=\"--model $NAME --output-mode edits --prompt-style minimal --warmup\""
