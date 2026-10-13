#!/usr/bin/env python3
"""Prompt/pipeline evaluation: run realistic proofreading cases through the
REAL aiproof pipeline (prompt -> model -> cleanup -> formatting enforcement)
and score them.

Every case gets structural checks automatically (line count, bullet count,
no added ; : em/en-dash, quote-wrap preservation) plus its own content
checks from the cases file:

  unchanged          output must equal input exactly
  must_contain       required substrings (case failed if missing)
  must_contain_ci    required substrings, case-insensitive
  must_not_contain   forbidden substrings
  bonus_contain      nice-to-have substrings (reported, never fail the case)
  expected           gold corrected text (private sets): reported as exact
                     match + word-level F0.5, never fails the case by itself

Usage:
  run_evals.py                     # config's model+endpoint (live config)
  run_evals.py --model llama3:latest --output-mode edits
  run_evals.py --cases ~/.local/share/aiproof/evals_private.json --warmup
  run_evals.py --only homophone-owe,bullets-email --show-output
"""

import argparse
import difflib
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiproof import config  # noqa: E402
from aiproof.llm.client import LLMClient, LLMError  # noqa: E402

BULLET_RE = re.compile(r"^[ \t]*[•\-*][ \t]+", re.MULTILINE)
NUMBERED_RE = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.MULTILINE)
DASH_RE = re.compile("[—–]")
LENGTH_BUCKETS = (("<200", 0, 200), ("200-500", 200, 500), (">500", 500, 10 ** 9))


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


def word_edits(src: str, dst: str) -> set:
    """Word-level edit operations as (start, end, replacement) tuples."""
    a, b = src.split(), dst.split()
    ops = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            ops.add((i1, i2, " ".join(b[j1:j2])))
    return ops


def f_half(inp: str, out: str, expected: str) -> float:
    """GEC-style F0.5 of the system's word edits against the gold edits."""
    gold, got = word_edits(inp, expected), word_edits(inp, out)
    if not gold and not got:
        return 1.0
    tp = len(gold & got)
    precision = tp / len(got) if got else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def run_case(client: LLMClient, case: dict, repeat: int = 1) -> dict:
    inp = case["input"]
    timings, out, err = [], "", None
    for _ in range(max(1, repeat)):
        start = time.monotonic()
        try:
            out, _ = client.proofread(inp)
        except LLMError as e:
            err = e
        timings.append(time.monotonic() - start)
    elapsed = statistics.median(timings)
    base = {"id": case["id"], "chars": len(inp), "elapsed": elapsed,
            "timings": timings, "bonus_hits": [], "bonus_total": 0,
            "edits_fallback": getattr(client, "last_edits_fallback", False)}
    if err is not None:
        return dict(base, ok=False, path="error", output="",
                    failures=[f"pipeline error: {err}"])

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
    result = dict(
        base, ok=not failures, path=getattr(client, "last_path", "?"),
        failures=failures, bonus_hits=[b for b in bonus if b in out],
        bonus_total=len(bonus), output=out,
    )
    if "expected" in case:
        result["exact"] = out == case["expected"]
        result["f05"] = f_half(inp, out, case["expected"])
    return result


def latency_summary(results: list[dict]) -> dict:
    def stats(vals):
        if not vals:
            return None
        vals = sorted(vals)
        return {"n": len(vals), "median": statistics.median(vals),
                "p90": vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))],
                "max": vals[-1]}
    summary = {"all": stats([r["elapsed"] for r in results])}
    for name, lo, hi in LENGTH_BUCKETS:
        summary[name] = stats([r["elapsed"] for r in results if lo <= r["chars"] < hi])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--provider", default=None,
                        help="provider preset id (ollama, llamacpp, openai, ...); "
                             "default: the configured one")
    parser.add_argument("--cases", default=None,
                        help="cases file (default: evals/cases.json); a private "
                             "set may add an 'expected' gold text per case")
    parser.add_argument("--only", default=None, help="comma-separated case ids")
    parser.add_argument("--output-mode", choices=["rewrite", "edits"], default=None,
                        help="override the configured output mode")
    parser.add_argument("--prompt-style", choices=["full", "minimal"], default=None,
                        help="override the configured prompt style")
    parser.add_argument("--context", action="store_true",
                        help="enable context-aware prompting (refreshes the "
                             "profile first; default is pinned OFF for "
                             "determinism)")
    parser.add_argument("--warmup", action="store_true",
                        help="one discarded call first so model load time is "
                             "excluded from the timings")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="run every case N times; report the median time")
    parser.add_argument("--json", default=None, metavar="OUT",
                        help="write per-case results and the summary as JSON")
    parser.add_argument("--show-output", action="store_true",
                        help="print the corrected text for every case")
    args = parser.parse_args()

    cases_path = (Path(args.cases).expanduser() if args.cases
                  else Path(__file__).parent / "cases.json")
    cases = json.loads(cases_path.read_text())
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
    if args.provider:
        cfg["provider"] = args.provider
        # The preset's own endpoint/model unless overridden below.
        cfg["endpoint"], cfg["model"] = "", ""
    if args.model:
        cfg["model"] = args.model
    if args.endpoint:
        cfg["endpoint"] = args.endpoint
    if args.output_mode:
        cfg["output_mode"] = args.output_mode
    if args.prompt_style:
        cfg["prompt_style"] = args.prompt_style
    cfg["request_timeout_s"] = 300
    client = LLMClient(cfg, api_key=cfg.get("api_key_plaintext") or "")

    print(f"model={client.model}  endpoint={client.endpoint}  "
          f"mode={client.output_mode}/{client.prompt_style}  "
          f"cases={len(cases)} ({cases_path.name})\n")

    if args.warmup and cases:
        try:
            client.proofread(cases[0]["input"])
        except LLMError as e:
            print(f"warmup failed: {e}")

    results = []
    for case in cases:
        result = run_case(client, case, repeat=args.repeat)
        results.append(result)
        mark = "PASS" if result["ok"] else "FAIL"
        note = " (edits→rewrite)" if result["edits_fallback"] else ""
        extra = ""
        if "f05" in result:
            extra = f", exact={'y' if result['exact'] else 'n'} F0.5={result['f05']:.2f}"
        print(f"[{mark}] {result['id']:<26} ({result['elapsed']:5.1f}s, "
              f"path={result['path']}{note}{extra})")
        for failure in result["failures"]:
            print(f"        - {failure}")
        for hit in result["bonus_hits"]:
            print(f"        + bonus: {hit!r}")
        if args.show_output:
            print("        out: " + result["output"].replace("\n", "\\n"))

    passed = sum(r["ok"] for r in results)
    bonus_hits = sum(len(r["bonus_hits"]) for r in results)
    bonus_total = sum(r["bonus_total"] for r in results)
    print(f"\n{passed}/{len(results)} cases passed"
          + (f"  (bonus: {bonus_hits}/{bonus_total})" if bonus_total else ""))
    graded = [r for r in results if "f05" in r]
    if graded:
        print(f"gold: exact {sum(r['exact'] for r in graded)}/{len(graded)}, "
              f"mean F0.5 {statistics.mean(r['f05'] for r in graded):.3f}")
    paths = {}
    for r in results:
        paths[r["path"]] = paths.get(r["path"], 0) + 1
    print("paths: " + ", ".join(f"{k}={v}" for k, v in sorted(paths.items())))

    summary = latency_summary(results)
    print("latency (s):")
    for bucket, st in summary.items():
        if st:
            print(f"  {bucket:<8} n={st['n']:<3} median={st['median']:.2f}  "
                  f"p90={st['p90']:.2f}  max={st['max']:.2f}")

    if args.json:
        out_path = Path(args.json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": client.model, "endpoint": client.endpoint,
            "output_mode": client.output_mode, "prompt_style": client.prompt_style,
            "cases_file": str(cases_path), "passed": passed, "total": len(results),
            "latency": summary, "results": results,
        }, indent=1, ensure_ascii=False))
        print(f"written: {out_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
