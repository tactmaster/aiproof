#!/usr/bin/env python3
"""Prompt/pipeline evaluation: run realistic proofreading cases through the
REAL aiproof pipeline (prompt -> model -> cleanup -> formatting enforcement)
and score them.

Every case gets structural checks automatically (line count, bullet count,
no added ; : em/en-dash, quote-wrap preservation) plus its own content
checks from cases.json:

  unchanged          output must equal input exactly
  must_contain       required substrings (case failed if missing)
  must_contain_ci    required substrings, case-insensitive
  must_not_contain   forbidden substrings
  bonus_contain      nice-to-have substrings (reported, never fail the case)

Usage:
  run_evals.py                     # config's model+endpoint (live config)
  run_evals.py --model llama3:latest
  run_evals.py --endpoint http://127.0.0.1:11434 --model phi3.5:latest
  run_evals.py --only homophone-owe,bullets-email
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiproof import config  # noqa: E402
from aiproof.llm.client import LLMClient, LLMError  # noqa: E402

BULLET_RE = re.compile(r"^[ \t]*[•\-*][ \t]+", re.MULTILINE)
NUMBERED_RE = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.MULTILINE)
DASH_RE = re.compile("[—–]")


def structural_issues(inp: str, out: str) -> list[str]:
    issues = []
    if out.count("\n") != inp.count("\n"):
        issues.append(
            f"line structure: {inp.count(chr(10)) + 1} -> {out.count(chr(10)) + 1} lines"
        )
    if len(BULLET_RE.findall(out)) != len(BULLET_RE.findall(inp)):
        issues.append("bullet count changed")
    if len(NUMBERED_RE.findall(out)) != len(NUMBERED_RE.findall(inp)):
        issues.append("numbered-list count changed")
    for name, char_re in (("semicolon", re.compile(";")),
                          ("colon", re.compile(":")),
                          ("em/en dash", DASH_RE)):
        if len(char_re.findall(out)) > len(char_re.findall(inp)):
            issues.append(f"{name} added")
    if inp.startswith('"') and inp.endswith('"') and not (
        out.startswith('"') and out.endswith('"')
    ):
        issues.append("wrapping quotes lost")
    if not (inp.startswith('"') and inp.endswith('"')) and (
        out.startswith('"') and out.endswith('"')
    ):
        issues.append("wrapping quotes added")
    return issues


def run_case(client: LLMClient, case: dict) -> dict:
    inp = case["input"]
    start = time.monotonic()
    try:
        out, _ = client.proofread(inp)
    except LLMError as e:
        return {"id": case["id"], "ok": False, "path": "error",
                "elapsed": time.monotonic() - start,
                "failures": [f"pipeline error: {e}"], "bonus_hits": [],
                "bonus_total": 0, "output": ""}

    failures = structural_issues(inp, out)

    if case.get("unchanged"):
        if out != inp:
            failures.append(f"text was modified: {out!r}")
    for needle in case.get("must_contain", []):
        if needle not in out:
            failures.append(f"missing: {needle!r}")
    for needle in case.get("must_contain_ci", []):
        if needle.lower() not in out.lower():
            failures.append(f"missing (any case): {needle!r}")
    for needle in case.get("must_not_contain", []):
        if needle in out:
            failures.append(f"still present: {needle!r}")

    bonus = case.get("bonus_contain", [])
    bonus_hits = [b for b in bonus if b in out]

    return {
        "id": case["id"], "ok": not failures,
        "path": getattr(client, "last_path", "?"),
        "elapsed": time.monotonic() - start,
        "failures": failures, "bonus_hits": bonus_hits,
        "bonus_total": len(bonus), "output": out,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--only", default=None, help="comma-separated case ids")
    parser.add_argument("--context", action="store_true",
                        help="enable context-aware prompting (refreshes the "
                             "profile first; default is pinned OFF for "
                             "determinism)")
    parser.add_argument("--show-output", action="store_true",
                        help="print the corrected text for every case")
    args = parser.parse_args()

    cases = json.loads((Path(__file__).parent / "cases.json").read_text())
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    cfg = config.load()
    # Pin OFF by default: the runner uses the live user config, and a learned
    # profile would make eval results depend on the developer's history.
    cfg["context_aware"] = False
    if args.context:
        from aiproof import context as aiproof_context
        cfg["context_aware"] = True
        aiproof_context.refresh(cfg)
    if args.model:
        cfg["model"] = args.model
    if args.endpoint:
        cfg["endpoint"] = args.endpoint
    cfg["request_timeout_s"] = 300
    client = LLMClient(cfg, api_key=cfg.get("api_key_plaintext") or "")

    print(f"model={cfg['model']}  endpoint={cfg['endpoint']}  cases={len(cases)}\n")

    passed, bonus_hits, bonus_total = 0, 0, 0
    for case in cases:
        result = run_case(client, case)
        mark = "PASS" if result["ok"] else "FAIL"
        print(f"[{mark}] {result['id']:<26} ({result['elapsed']:5.1f}s, "
              f"path={result['path']})")
        for failure in result["failures"]:
            print(f"        - {failure}")
        for hit in result["bonus_hits"]:
            print(f"        + bonus: {hit!r}")
        if args.show_output:
            print("        out: " + result["output"].replace("\n", "\\n"))
        passed += result["ok"]
        bonus_hits += len(result["bonus_hits"])
        bonus_total += result["bonus_total"]

    print(f"\n{passed}/{len(cases)} cases passed"
          + (f"  (bonus: {bonus_hits}/{bonus_total})" if bonus_total else ""))
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
