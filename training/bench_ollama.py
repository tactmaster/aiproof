#!/usr/bin/env python3
"""Raw Ollama speed probe: prompt-eval and generation tokens/s per model.

Sends the same ~330-char proofreading request to each model (edit-list prompt
by default, the full rewrite prompt with --mode rewrite) and reads Ollama's
own timing counters, so the numbers exclude aiproof's post-processing.

  training/bench_ollama.py qwen2.5:3b qwen2.5:1.5b qwen2.5:0.5b
  training/bench_ollama.py aiproof-edit-1.5b --style minimal --runs 3
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiproof.llm.prompts import (  # noqa: E402
    EDIT_SYSTEM_PROMPT, build_edit_prompt, build_proofread_prompt,
)

SAMPLE = ("Hi team, I have finsihed the inventroy report for Q3 and attched it to "
          "the ticket. There are a couple of items were the counts dont match what "
          "the dashboard shows, so I think we shoud double check the devices in the "
          "Oslo lab before we send it to the customer. Let me know if you want me to "
          "walk you threw it tomorow morning.")


def call(endpoint, model, prompt, system=None, num_predict=256, temperature=0.0):
    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": num_predict}}
    if system:
        body["system"] = system
    req = urllib.request.Request(f"{endpoint.rstrip('/')}/api/generate",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    data["_wall"] = time.monotonic() - t0
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--endpoint", default="http://127.0.0.1:11434")
    ap.add_argument("--mode", choices=["edits", "rewrite"], default="edits")
    ap.add_argument("--style", choices=["full", "minimal"], default="full")
    ap.add_argument("--runs", type=int, default=2, help="timed runs after 1 warmup")
    ap.add_argument("--text", default=SAMPLE)
    ap.add_argument("--show", action="store_true", help="print the last response")
    args = ap.parse_args()

    if args.mode == "edits":
        prompt = build_edit_prompt(args.text, style=args.style)
        system = EDIT_SYSTEM_PROMPT if args.style == "minimal" else None
        num_predict = 256
    else:
        prompt, system, num_predict = build_proofread_prompt(args.text), None, 400
    print(f"text={len(args.text)} chars  prompt={len(prompt)} chars  "
          f"mode={args.mode}/{args.style}\n")
    print(f"{'model':<24}{'load':>6}{'prompt tok':>11}{'tok/s':>8}"
          f"{'gen tok':>9}{'tok/s':>8}{'wall s':>8}")
    for model in args.models:
        try:
            call(args.endpoint, model, prompt, system, num_predict)  # warmup/load
            runs = [call(args.endpoint, model, prompt, system, num_predict)
                    for _ in range(max(1, args.runs))]
        except Exception as e:  # noqa: BLE001
            print(f"{model:<24} ERROR {e}")
            continue
        med = lambda key: statistics.median(r.get(key, 0) for r in runs)  # noqa: E731
        pe, pc = med("prompt_eval_duration") / 1e9, med("prompt_eval_count")
        ge, gc = med("eval_duration") / 1e9, med("eval_count")
        print(f"{model:<24}{med('load_duration') / 1e9:>6.2f}{pc:>11.0f}"
              f"{(pc / pe if pe else 0):>8.0f}{gc:>9.0f}{(gc / ge if ge else 0):>8.0f}"
              f"{med('_wall'):>8.2f}")
        if args.show:
            print("  " + runs[-1]["response"].strip().replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
