"""Provider registry, trimmed from the extension's shared/providers.js to the
four API formats that cover ~95% of its ~25 providers. Presets with
format="openai" all speak the OpenAI chat-completions dialect and differ only
in endpoint/model defaults.
"""

API_FORMATS = ("ollama", "openai", "anthropic", "google")

PROVIDERS = {
    "ollama": {
        "display_name": "Ollama (local)",
        "format": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "model": "phi3.5:latest",
        "needs_key": False,
    },
    "openai": {
        "display_name": "OpenAI",
        "format": "openai",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "needs_key": True,
    },
    "anthropic": {
        "display_name": "Anthropic (Claude)",
        "format": "anthropic",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-haiku-latest",
        "needs_key": True,
    },
    "google": {
        "display_name": "Google (Gemini)",
        "format": "google",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-1.5-flash",
        "needs_key": True,
    },
    "groq": {
        "display_name": "Groq",
        "format": "openai",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.1-8b-instant",
        "needs_key": True,
    },
    "mistral": {
        "display_name": "Mistral",
        "format": "openai",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "needs_key": True,
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "format": "openai",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "needs_key": True,
    },
    "lmstudio": {
        "display_name": "LM Studio (local)",
        "format": "openai",
        "endpoint": "http://127.0.0.1:1234/v1/chat/completions",
        "model": "local-model",
        "needs_key": False,
    },
    "llamacpp": {
        "display_name": "llama.cpp server (local)",
        "format": "openai",
        "endpoint": "http://127.0.0.1:8080/v1/chat/completions",
        "model": "local-model",
        "needs_key": False,
    },
    "custom": {
        "display_name": "Custom (OpenAI-compatible)",
        "format": "openai",
        "endpoint": "",
        "model": "",
        "needs_key": False,
    },
}


def get_provider(provider_id: str) -> dict:
    if provider_id not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_id!r}")
    return PROVIDERS[provider_id]
