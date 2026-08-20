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

`evals/cases.json` holds 17 realistic proofreading cases (spelling, homophones,
must-stay-unchanged texts, formatting, verbatim technical content, real
emails); the runner checks structure automatically (line counts, bullets, no
added punctuation, quote wrapping) plus per-case content expectations. Use it
whenever you tweak the prompt or consider a different model.

## License

[GPL-3.0-or-later](LICENSE). Share it, fork it, ship it — but if you
distribute a modified version, you must publish your changes under the
same license.
