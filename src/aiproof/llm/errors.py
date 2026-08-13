"""LLM error types, split out so postprocess.py can distinguish LLMError
without importing client.py (which imports postprocess — would be circular).
"""


class LLMError(Exception):
    """User-presentable LLM failure."""


class TextTooLongError(LLMError):
    def __init__(self, length: int, limit: int):
        super().__init__(f"Selection too long ({length:,} > {limit:,} characters)")
        self.length = length
        self.limit = limit
