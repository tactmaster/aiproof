#!/usr/bin/env python3
"""Side-by-side table from run_evals.py --json outputs.

  evals/compare.py build/eval-*.json
"""

import json
import sys


def main() -> int:
    rows = []
    for path in sys.argv[1:]:
        d = json.load(open(path))
        lat = d.get("latency", {})

        def med(bucket):
            st = lat.get(bucket)
            return f"{st['median']:.2f}/{st['p90']:.2f}" if st else "-"

        paths = {}
        for r in d["results"]:
            paths[r["path"]] = paths.get(r["path"], 0) + 1
        fallbacks = sum(1 for r in d["results"] if r.get("edits_fallback"))
        graded = [r for r in d["results"] if "f05" in r]
        f05 = f"{sum(r['f05'] for r in graded) / len(graded):.2f}" if graded else "-"
        ep = d.get("endpoint") or ""
        backend = "ollama" if (":11434" in ep or not ep) else ep.split("//")[-1].split("/")[0]
        rows.append((d["model"], backend, f"{d['output_mode']}/{d['prompt_style']}",
                     f"{d['passed']}/{d['total']}", f05, med("<200"), med("200-500"), med(">500"),
                     ", ".join(f"{k}={v}" for k, v in sorted(paths.items())) +
                     (f" (edits→rewrite {fallbacks})" if fallbacks else "")))
    head = ("model", "backend", "mode", "pass", "F0.5", "<200 med/p90", "200-500", ">500", "paths")
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    for row in (head, *rows):
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
