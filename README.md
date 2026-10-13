# aiproof — OS-wide AI text proofreader for GNOME/Wayland

Select text in **any** application, press **Ctrl+Super+P**, and the selection
is proofread by an LLM and replaced in place. A native port of the
[AI Text Proofreader browser extension](https://github.com/tactmaster/ai-text-proofreader-extension) to
the whole desktop.

- **Local-first**: defaults to Ollama on `127.0.0.1:11434` — no text leaves
  your machine. OpenAI-compatible APIs (Groq, Mistral, LM Studio, llama.cpp,
  OpenRouter, …), Anthropic, and Google Gemini also supported.
- **Formatting-safe**: line breaks, bullets, and numbered lists are validated
  after correction; a repair pass runs on mismatch, and if the model still
  mangles the structure, your original text is kept untouched.
- **Seamless GNOME integration**: tray icon with status, desktop
  notifications, global hotkeys, GTK4/libadwaita settings, autostart, API
  keys in the GNOME Keyring.

## Hotkeys

| Keys | Action |
|---|---|
| `Ctrl+Super+P` | Proofread selection and replace it in place |
| `Ctrl+Shift+Super+P` | Proofread selection to clipboard (never types — safe for terminals) |

The in-place flow: your selection is copied (synthetic Ctrl+C via
ydotool/uinput), proofread, pasted back (synthetic Ctrl+V) — and your
previous clipboard contents (text *or image*) are restored afterwards.

> **Terminals:** use the clipboard hotkey there — a synthetic Ctrl+C in a
> terminal is SIGINT, not copy.

## Install

```sh
make deb
sudo apt install ../aiproof_*_all.deb
aiproof setup        # verifies ydotool, permissions, hotkeys, Ollama
```

Log out and back in once so the `input` group membership (for
`/dev/uinput`) and autostart take effect.

### Dev mode (no root)

```sh
make install-user    # wrapper in ~/.local/bin, runs from this source tree
aiproof daemon
```

## Right-click integration

GNOME on Wayland has **no OS-wide right-click hook for text** — context menus
inside apps belong to each app's toolkit and cannot be extended from outside.
What aiproof offers instead:

- **The hotkeys** (above) — work on selected text in *any* app; this *is* the
  Wayland-native equivalent of a global context-menu action.
- **Tray icon menu** — "Proofread selection now" acts on the current
  selection without a hotkey.
- **Files (Nautilus)**: right-click a text file → **Scripts → Proofread with
  AI** — proofreads the whole file in place, keeping a backup as
  `<name>.orig`. Installed by `aiproof setup`.

## How it works on Wayland

GNOME/Mutter allows no global key grabbing or synthetic input from regular
apps, so aiproof combines:

- **GNOME custom keybindings** (gsettings) → global hotkeys that `gdbus call`
  the daemon,
- **wl-clipboard** → reading the selection and setting the corrected text,
- **ydotool** (uinput kernel device) → synthetic Ctrl+C/Ctrl+V into the
  focused window,
- **AppIndicator** → tray icon (`gnome-shell-extension-appindicator`,
  enabled by default on Ubuntu).

[docs/flow.md](docs/flow.md) charts every route through the system — a
simple overview and a detailed flowchart of all branches.

## CLI

```
aiproof daemon              # the background service (autostarted at login)
aiproof trigger             # same as the hotkey (needs the daemon)
aiproof trigger-clipboard   # clipboard-only flow
aiproof settings            # GTK4 settings window
aiproof setup               # first-run system checks with fixes
aiproof history --tail 20   # saved proofreads (opt-in; --clear, --path)
echo "sum text" | aiproof proofread   # stdin→stdout, for scripts/debugging
```

## Configuration

`~/.config/aiproof/config.json` — edited by the settings app, live-reloaded
by the daemon. API keys live in the Secret Service (GNOME Keyring), never in
the config file unless no keyring is available.

**Proofread history** (opt-in, settings → Behavior): every completed
proofread — original, corrected, provider, pipeline path, timing — is
appended to `~/.local/share/aiproof/history.jsonl` (file mode 0600, local
only). View with `aiproof history`; it's JSONL, so `jq` works too.

**Claude CLI backend**: if Claude Code is installed on the machine, select
the `claude-cli` provider (default model `haiku`) to proofread with your
existing Claude subscription — no API key. It scores 27/28 on the eval suite
versus 23/28 for local qwen2.5:3b, at a median 3.1s versus 0.7s. Note the
text is sent to Anthropic, unlike the local Ollama default; it also works
well as the *fallback* provider when a local model is unreachable.

**Per-application profiles** (on by default): the focused window decides the
register. Slack and WhatsApp keep your lowercase starts, slang and missing
full stops; Zendesk and email get professional capitalization and
punctuation; code editors and terminals keep identifiers, flags and paths
untouched. Unknown apps use the default style. `aiproof app` shows the
current window and which profile it matched; set `"app_profiles": false` in
the config to turn it off. Needs the `focused-window-dbus` GNOME extension —
without it, everything falls back to the default profile.

**Context-aware proofreading** (opt-in, needs history): aiproof learns an
aggregate profile of what you write — domain vocabulary that must never be
"corrected" (a word qualifies only after surviving two or more proofreads
unchanged, passing dictionary/typo filters) and a one-sentence domain
summary written by the LLM in the background every ~25 proofreads. The
profile is a small capped prompt block (~100 tokens): proofreads are not
slowed down, and no per-run examples from your past are ever included.
Inspect or reset it any time with `aiproof context` / `--refresh` /
`--clear`.

**Correction output** (settings → Behavior): `"output_mode": "rewrite"`
(default) has the model return the whole corrected text; `"edits"` has it
return only `wrong => right` lines that aiproof applies and verifies itself —
much faster on small local models, and the layout cannot break.
`"prompt_style": "minimal"` drops the long instruction prompt in favour of a
one-line system prompt, for a model fine-tuned on the edit-list task (see
[training/README.md](training/README.md)). `aiproof proofread
--output-mode edits --prompt-style minimal` tries either per run.

## Tests

```sh
make test        # unit tests (offline, fast)
make test-flow   # route-by-route flow tests; writes build/flow-report.md
make eval        # live prompt/pipeline evaluation against the configured model
make eval EVAL_ARGS="--model gemma4:latest --only clean-informal --show-output"
```

The flow tests drive every route in [docs/flow.md](docs/flow.md) with the
LLM, clipboard, and keystrokes mocked, assert that each branch logs its
decision, and write the captured logs per route to `build/flow-report.md`.

`evals/cases.json` holds 28 realistic proofreading cases (spelling, homophones,
must-stay-unchanged texts, formatting, verbatim technical content, real
emails, edit-list edge cases); the runner checks structure automatically (line
counts, bullets, no added punctuation, quote wrapping) plus per-case content
expectations, and prints the pipeline path and median / p90 latency bucketed
by input length. Use it whenever you tweak the prompt or consider a different
model:

```sh
make eval EVAL_ARGS="--model qwen2.5:3b --output-mode edits --warmup --json evals/results/eval.json"
make eval-private              # your own hand-reviewed gold set (see training/README.md)
make bench BENCH_ARGS="qwen2.5:3b qwen2.5:1.5b"   # raw tokens/s per Ollama model
```

## Output modes and a custom model

By default the model rewrites the whole text and aiproof verifies the layout
afterwards. With `"output_mode": "edits"` the model returns only
`wrong => right` lines that aiproof applies itself — output tokens then
scale with the number of mistakes instead of the text length, and line
breaks and bullets cannot break. Stock chat models follow that format
unevenly, so [training/README.md](training/README.md) describes how to
fine-tune a small local model on it from your own proofread history (data
and models stay on your machine) and run it with `"prompt_style": "minimal"`.

**Serving tip (measured on an RTX 500 laptop):** Ollama adds roughly half a
second of scheduling overhead to every request and reloads an evicted model
from disk in 10-35 s. The same `qwen2.5:3b` weights served by llama.cpp's
`llama-server` answered the eval suite with a 0.4 s median instead of
1.1-1.5 s, and a fine-tuned 0.5B/1.5B edit-list model in 0.06-0.13 s (see
[training/README.md](training/README.md#results-so-far-rtx-500-ada-4-gb-laptop-28-public-eval-cases)). `training/serve_llama.sh <model.gguf>` starts one (an Ollama blob
works as the GGUF), `training/aiproof-llama-server.service` is a user-unit
template, and aiproof talks to it through the *llama.cpp server* provider
preset.

## License

[GPL-3.0-or-later](LICENSE). Share it, fork it, ship it — but if you
distribute a modified version, you must publish your changes under the
same license.
