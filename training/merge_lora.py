#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model (bf16, CPU) for GGUF conversion.

  python training/merge_lora.py training/runs/qwen1.5b-edit        # -> <run>/merged
"""

import json
import sys
from pathlib import Path


def main() -> int:
    run = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else run / "merged"
    adapter = run / "adapter"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"base {base} + {adapter} -> {out}")
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"})
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(out)
    print("merged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
