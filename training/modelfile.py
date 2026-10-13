#!/usr/bin/env python3
"""Render the Ollama Modelfile for an exported GGUF.

  PYTHONPATH=src python training/modelfile.py ~/.local/share/aiproof/models/x.gguf > x.Modelfile
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiproof.llm.prompts import EDIT_SYSTEM_PROMPT  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "Modelfile.template"


def render(gguf_path: str, system: str = EDIT_SYSTEM_PROMPT) -> str:
    return (TEMPLATE.read_text()
            .replace("@@GGUF@@", str(Path(gguf_path).expanduser().resolve()))
            .replace("@@SYSTEM@@", system))


if __name__ == "__main__":
    sys.stdout.write(render(sys.argv[1]))
