# Training your own edit-list model

aiproof's **edit-list output mode** asks the model for corrections only:

```
inventroy => inventory
their going => they're going
```

aiproof applies those lines itself (`src/aiproof/llm/edits.py`), so output
tokens scale with the number of mistakes rather than the length of the text,
and line breaks, bullets and indentation can never be mangled. Stock chat
models follow the format unevenly (a 3B model under-corrects; smaller ones
ignore it), so this directory fine-tunes a small model on exactly that task.

Everything here runs locally. The data is mined from your own opt-in
proofread history, the model stays on your machine, and `training/data/`,
`training/runs/` and all model files are git-ignored (`make training-check`).

## Pipeline

| Step | Command | Produces |
|---|---|---|
| 1. Mine history | `make train-data` (runs `mine_history.py`, then `build_dataset.py`) | `data/error_profile.json` (your real error types), `data/clean_pool.jsonl` (your clean sentences), `data/protected_vocab.txt`, `data/train.jsonl` + `data/val.jsonl` |
| 2. Review gold (once, manual) | `PYTHONPATH=src python3 training/mine_history.py --review-candidates`, edit `data/review.jsonl`, then `--finalize-gold` | `~/.local/share/aiproof/evals_private.json` (private eval set) + `data/real_pairs.jsonl` |
| 3. Train | `make train TRAIN_ARGS="--base Qwen/Qwen2.5-1.5B-Instruct --name qwen1.5b-edit"` | `runs/<name>/adapter`, `val_edit_metrics.json` |
| 4. Export | `make export RUN=training/runs/qwen1.5b-edit NAME=aiproof-edit-1.5b` | GGUF in `~/.local/share/aiproof/models/`, Ollama model `aiproof-edit-1.5b` |
| 5. Evaluate | `make eval EVAL_ARGS="--model aiproof-edit-1.5b --output-mode edits --prompt-style minimal --warmup"` and `make eval-private EVAL_ARGS="..."` | pass counts, F0.5 on gold, latency by text length |

Then point aiproof at it: Settings → Provider → model `aiproof-edit-1.5b`,
Behavior → Correction output → *Edit list* and *Minimal* prompt style (or
`output_mode`/`prompt_style` in `~/.config/aiproof/config.json`).

### How the data is made

* `mine_history.py` word-diffs every saved proofread and classifies each
  change (missing letter, transposition, adjacent key, homophone, dropped
  function word, lowercase sentence start, ...). The counts become the
  weights `corrupt.py` samples from, so the synthetic mistakes look like
  yours. Texts the pipeline could not fix safely (`path=fallback`) and
  obvious model rewrites are excluded.
* `corrupt.py` injects those mistakes into clean text while leaving URLs,
  paths, flags, code, numbers, identifiers, bullets and the protected
  vocabulary untouched, and records the exact `wrong => right` list. Each
  example is verified to round-trip through aiproof's own applier.
* `build_dataset.py` mixes your corrupted texts with public English
  ([grammarly/coedit](https://huggingface.co/datasets/grammarly/coedit),
  Apache-2.0, and wikitext-103, CC BY-SA), reviewed real pairs, multi-line
  layouts (bullets, numbered lists, short e-mails) and "bait" texts that must
  not be changed (informal register, sign-offs, existing semicolons and
  dashes, quotes, code). About one row in six is clean and expects
  `NO CHANGES`. Rows are split into train/validation by source text.
* The system turn of every row is `aiproof.llm.prompts.EDIT_SYSTEM_PROMPT`.
  The same constant is baked into the Modelfile and sent by the daemon in
  *minimal* prompt style, so training and inference see identical bytes
  (`tests/test_modelfile.py` pins this).

## Environment

The daemon's system Python never gets any of this. Training uses its own
conda environment:

```bash
conda create -n aiproof-train python=3.12 -y
conda run -n aiproof-train pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
conda run -n aiproof-train pip install -r training/requirements.txt
```

GGUF conversion uses `convert_hf_to_gguf.py` from a llama.cpp checkout
(`LLAMA_CPP=~/3rdparty/llama.cpp` by default); Q8_0 needs no compiled
binaries.

## Fitting a 4 GB GPU

`train.py` defaults to 4-bit NF4 (QLoRA) for models above ~1B parameters and
plain bf16 LoRA below that, LoRA r=16/alpha=32 on all attention and MLP
projections, gradient checkpointing, sequence length 768, batch 1 x 32
accumulation. Stop Ollama's resident model first (`ollama ps`,
`ollama stop <model>`); the script refuses to start with less than ~3 GB
free. Expect roughly 1.5 h for a 0.5B model and 3-5 h for 1.5B on an RTX 500.

Smoke-test the whole loop before an overnight run:

```bash
make train TRAIN_ARGS="--base Qwen/Qwen2.5-0.5B-Instruct --name smoke --limit 40 --max-steps 5 --eval-samples 8"
```

## Measuring

`evals/run_evals.py` reports pass counts, the pipeline path per case, and
median / p90 latency bucketed by input length; `--json` writes everything
for side-by-side comparison. `training/bench_ollama.py` prints raw
prompt-eval and generation tokens/s per model, with Ollama's own `load`
overhead shown separately.

## Results so far (RTX 500 Ada 4 GB laptop, 28 public eval cases)

Median latency is per proofread on texts under 200 characters, measured
through `client.proofread` (prompt + model + applying/validating the result).

| model | served by | mode | pass | median | p90 | long cases |
|---|---|---|---|---|---|---|
| qwen2.5:3b (stock) | Ollama 0.20.6 | rewrite | 23/28 | 0.9-1.5 s | 1.4 s | 6.7 s / 25 s (salvage) |
| qwen2.5:3b (stock) | llama-server | rewrite | 21/28 | 0.40 s | 0.65 s | 4.1 s / 16 s |
| qwen2.5:3b (stock) | llama-server | edits | 18/28 | 0.38 s | 0.71 s | 0.65 s / 16 s |
| **aiproof-edit-0.5b** (fine-tuned, Q8_0) | llama-server | edits/minimal | 20/28 | **0.06 s** | 0.13 s | 0.05 s / 0.20 s |
| aiproof-edit-0.5b | Ollama | edits/minimal | 19/28 | 0.38 s | 0.51 s | 0.3 s / 0.5 s (24.9 s cold load) |
| **aiproof-edit-1.5b** (fine-tuned, Q8_0) | llama-server | edits/minimal | 19/28 | **0.13 s** | 0.27 s | 0.56 s / 0.42 s |
| aiproof-edit-1.5b (Q4_K_M) | llama-server | edits/minimal | 19/28 | 0.11 s | 0.29 s | 0.54 s / 0.36 s |
| aiproof-edit-1.5b (Q8_0) | Ollama | edits/minimal | 21/28 | 0.35 s | 0.77 s | 0.75 s / 0.66 s |
| **aiproof-edit-1.5b-v2** (v2 data, Q8_0) | llama-server | edits/minimal | 20/28 | 0.12 s | 0.28 s | 0.38 s / 0.48 s |
| aiproof-edit-1.5b-v2 (Q8_0) | Ollama | edits/minimal | 21/28 | 0.33 s | 0.53 s | 0.63 s / 0.81 s |

* The fine-tuned 0.5B never falls back to the rewrite path and never breaks
  layout; it generates at ~150 tokens/s. Its misses are content limits
  (homophones such as our/owe, their/they're, Deer/Dear; tense).
* Ollama adds 0.35-0.7 s of per-request scheduling overhead regardless of
  model size, and cold loads from disk take 10-35 s; the same GGUF behind
  `llama-server` has millisecond overhead and a warm prefix cache.
* Training the 0.5B (10k rows, 2 epochs, bf16 LoRA, batch 4) took ~2 h 15 min
  at 1.7 GiB peak; the 1.5B (QLoRA NF4, batch 2 x 16) took 2 h 30 min at
  2.8 GiB peak. Held-out exact edit-list match: 0.5B 47 %, 1.5B 57 %
  (span-mismatch 2.2 % / 1.6 %); generation 151 / 68 (Q8) / 82 (Q4) tok/s.
* The fine-tuned sizes tie on the 28 public cases. The v2 dataset
  (`--dense-frac 0.15`, `--min-homophone 0.08`, comma-only natural edits
  dropped) lowered the span-mismatch rate to 1.0 % and fixed the long
  support-email case, but the same seven cases still fail for every size:
  a 9-word sentence with five typos, our→owe, their/they're + loose/lose,
  buy→bought, Deer→Dear, a doubled-letter "bugg", and one-of-two "their".
  Next levers, in order: a third epoch or a larger dense/homophone share,
  reviewed real pairs from `--finalize-gold`, and the private gold set as
  the yardstick instead of these 28 cases. Differences of 1-2 cases between
  backends for the same weights are sampler/template noise at this size.
* Harness note: a background job here is limited to 2 h — run long trainings
  detached (`setsid nohup ... &`) and use `--resume` to continue from the last
  checkpoint.

## Optional: GPU-free backends

`export_openvino.sh` / `ov_server.py` (OpenVINO GenAI on the Intel NPU or
iGPU behind an OpenAI-compatible endpoint) and a plain `llama-server` on the
CPU both plug into aiproof through the existing *llama.cpp server* provider
preset with no daemon changes. See the measurements in the main README
before expecting them to beat the discrete GPU.
