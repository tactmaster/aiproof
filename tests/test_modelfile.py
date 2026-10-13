"""The Modelfile SYSTEM, the daemon's system turn and the training data must
be the same bytes; this pins the first two."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from aiproof.llm.prompts import EDIT_SYSTEM_PROMPT  # noqa: E402
from modelfile import render  # noqa: E402


def test_modelfile_system_is_the_shared_constant():
    text = render("/tmp/x.gguf")
    assert f'SYSTEM """{EDIT_SYSTEM_PROMPT}"""' in text
    assert text.count("@@") == 0
    assert "FROM /tmp/x.gguf" in text
    assert "<|im_start|>assistant" in text
    assert "PARAMETER stop <|im_end|>" in text
