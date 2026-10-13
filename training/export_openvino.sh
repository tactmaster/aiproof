#!/usr/bin/env bash
# Optional: export the merged model for OpenVINO GenAI (Intel NPU / iGPU / CPU).
#
#   training/export_openvino.sh training/runs/qwen1.5b-edit/merged ~/.local/share/aiproof/models/aiproof-edit-1.5b-ov npu
#
# NPU wants channel-wise symmetric INT4; GPU/CPU do better with group-wise.
# Needs: pip install openvino-genai "optimum-intel[openvino]" nncf   (training env)
# and, for the NPU, Intel's Linux NPU user-mode driver (intel-level-zero-npu,
# intel-driver-compiler-npu, intel-fw-npu; libze1 >= 1.32) — check with
#   python -c "import openvino as ov; print(ov.Core().available_devices)"
set -euo pipefail
MERGED=${1:?usage: export_openvino.sh MERGED_DIR OUT_DIR [npu|gpu|cpu]}
OUT=${2:?usage: export_openvino.sh MERGED_DIR OUT_DIR [npu|gpu|cpu]}
TARGET=${3:-npu}
TRAIN_PY=${TRAIN_PY:-conda run -n aiproof-train python}
case "$TARGET" in
  npu) ARGS="--weight-format int4 --sym --ratio 1.0 --group-size -1" ;;
  *)   ARGS="--weight-format int4 --group-size 128" ;;
esac
$TRAIN_PY -m optimum.commands.optimum_cli export openvino -m "$MERGED" \
  --task text-generation-with-past $ARGS "$OUT"
echo "exported to $OUT; serve with: training/ov_server.py --model-dir $OUT --device ${TARGET^^}"
