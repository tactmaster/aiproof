#!/usr/bin/env python3
"""Assemble the edit-list fine-tuning set (training/data/train.jsonl + val.jsonl).

Sources, mixed and seeded:
  ed        the writer's own clean texts (clean_pool.jsonl from mine_history.py),
            corrupted with corrupt.py using the mined error profile
  general   public English for coverage: grammarly/coedit GEC targets (Apache-2.0)
            corrupted the same way, plus CoEdIT's natural src->tgt pairs
            converted to edit lists, plus wikitext-103 paragraphs (CC BY-SA)
  real      reviewed real pairs (real_pairs.jsonl from --finalize-gold)
  bait      texts that must NOT be changed (informal register, sign-offs,
            existing ; and dashes, quoted spans, code, URLs, protected
            vocabulary): clean -> NO CHANGES, or errors only elsewhere
  layout    stacked units as bullets / numbered lists / short emails so the
            model sees multi-line inputs

Every row is TRL "prompt-completion" chat format; the system turn is
aiproof.llm.prompts.EDIT_SYSTEM_PROMPT, byte-identical to what the daemon
sends and what the Modelfile bakes in. Texts whose history id is in
gold_ids.json (private eval) are excluded. Nothing is committed.

  PYTHONPATH=src python training/build_dataset.py            # full mix
  PYTHONPATH=src python training/build_dataset.py --no-public --n-ed 300  # smoke
"""

import argparse
import collections
import difflib
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiproof.llm.edits import Edit, apply_edits  # noqa: E402
from aiproof.llm.edits import NO_CHANGES as NO_CHANGES_TOKEN  # noqa: E402
from aiproof.llm.prompts import EDIT_SYSTEM_PROMPT  # noqa: E402
from corrupt import AGREEMENT_SWAPS, Corruptor, load_profile, load_protected  # noqa: E402
from mine_history import FUNCTION_WORDS, HOMOPHONES, derive_edits  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

BAIT_SENTENCES = [
    "gonna grab lunch, back in 20",
    "kinda tired today but we're good to go",
    "wanna sync on the release plan asap?",
    "pushed the fix to prod, repo is green again",
    "lol that bug was fun, anyway fixed now",
    "ok cool, ty for the info",
    "Deployed to staging; rollback is one click away.",
    "The build failed — again — so I reverted it.",
    "Two things: the config and the docs.",
    "Is this expected? Or should I open a ticket?",
    "Done! Let me know if anything else breaks.",
    "Best regards,\nEd",
    "Thanks,\nEd",
    "Cheers,\nEd",
    "Let me know if you need anything else.",
    "Hope this helps.",
    "He said “ship it on Friday” and left.",
    "The error was 'connection refused' on port 8080.",
    "Run `make deb` and then `sudo dpkg -i ../aiproof_*.deb`.",
    "See https://docs.mender.io/overview for the full list.",
    "The file lives in /etc/mender/mender.conf on the device.",
    "Pass --force-bootstrap to the client to re-register.",
    "Version v2.3.1 shipped yesterday; v2.3.2 is in QA.",
    "Ping me on Slack or e-mail ed@example.com.",
    "FYI the API returns 404 for deleted devices.",
    "The ETA is EOD Thursday, CET.",
    "TODO: add the retry loop.",
    "LGTM, merging.",
    "idk, maybe tomorrow?",
    "brb, call in 5",
]
EMAIL_OPENERS = ["Hi {name},", "Hello {name},", "Hey {name},", "Dear {name},", "Hi team,", "Hi all,"]
EMAIL_CLOSERS = ["Best regards,\nEd", "Thanks,\nEd", "Cheers,\nEd", "Kind regards,\nEd",
                 "Thanks a lot,\nEd", "Let me know if you have questions.\n\nBest,\nEd"]
NAMES = ["Anna", "Tom", "Marcus", "Lena", "Jonas", "Priya", "Sam", "Kristian"]
INSTRUCTION_RE = re.compile(r"^[^:]{0,80}:\s*")
PUNCT = ".,;:!?\"'()[]{}<>\u201c\u201d\u2018\u2019\u2014\u2013-"
VERB_FORMS = set(AGREEMENT_SWAPS) | set(AGREEMENT_SWAPS.values()) | {
    "be", "been", "being", "am", "had", "having", "gone", "going", "come", "came",
    "see", "saw", "seen", "take", "taken", "give", "gave", "given", "write", "wrote",
    "written", "know", "knew", "known", "think", "thought", "say", "said", "run", "ran"}


def plausible_fix(wrong: str, right: str) -> bool:
    """Is this edit a spelling/grammar correction rather than a word-choice
    or style change? Public GEC data mixes both ("stare" -> "gaze"); only
    the former matches what aiproof is for."""
    if wrong.replace(",", "") == right.replace(",", "") and wrong != right:
        return False  # comma-only: style, and the 0.5B learned to sprinkle them
    w, r = wrong.strip(PUNCT).lower(), right.strip(PUNCT).lower()
    if w == r:
        return True  # punctuation / case only
    wt, rt = w.split(), r.split()
    if len(wt) == len(rt):
        for a, b in zip(wt, rt):
            if a == b or frozenset((a, b)) in HOMOPHONES:
                continue
            if a.replace("'", "") == b.replace("'", ""):
                continue
            if a in VERB_FORMS and b in VERB_FORMS:
                continue
            if difflib.SequenceMatcher(None, a, b).ratio() < 0.6:
                return False
        return True
    longer, shorter = (rt, wt) if len(rt) > len(wt) else (wt, rt)
    extra = list(longer)
    for tok in shorter:
        if tok in extra:
            extra.remove(tok)
    if len(extra) > 2 or len(extra) != len(longer) - len(shorter):
        return False
    return all(tok in FUNCTION_WORDS or tok in VERB_FORMS for tok in extra)


def edits_text(edits) -> str:
    return "\n".join(f"{e.wrong} => {e.right}" for e in edits) if edits else NO_CHANGES_TOKEN


def row(text: str, edits, source: str, unit_id: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": EDIT_SYSTEM_PROMPT},
                   {"role": "user", "content": text}],
        "completion": [{"role": "assistant", "content": edits_text(edits)}],
        "source": source, "unit": unit_id, "n_edits": len(edits),
    }


def split_key(unit_id: str) -> float:
    """Stable 0..1 hash so every example of one source unit lands in the same split."""
    return int(hashlib.sha1(unit_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def load_public(n_targets: int, n_natural: int, n_wiki: int, rng: random.Random):
    """CoEdIT GEC targets + natural pairs, and wikitext paragraphs. Needs the
    `datasets` package; returns empty lists when it is missing."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed: skipping public sources (pip install datasets)")
        return [], [], []
    targets, natural = [], []
    try:
        ds = load_dataset("grammarly/coedit", split="train")
        rows = [r for r in ds if r.get("task") == "gec"]
        rng.shuffle(rows)
        for r in rows:
            src, tgt = INSTRUCTION_RE.sub("", r["src"], count=1).strip(), r["tgt"].strip()
            if not (20 <= len(tgt) <= 400) or "\n" in tgt:
                continue
            if len(targets) < n_targets:
                targets.append((f"coedit:{r['_id']}", tgt))
            if len(natural) < n_natural and src != tgt:
                a, b = src.split(), tgt.split()
                sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
                ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
                if sm.ratio() >= 0.85 and 0 < len(ops) <= 4 and \
                        all(max(i2 - i1, j2 - j1) <= 3 for _, i1, i2, j1, j2 in ops):
                    for widen in (0, 1, 2):
                        edits = derive_edits(src, tgt, widen)
                        if edits and apply_edits(src, edits).text == tgt:
                            if all(plausible_fix(e.wrong, e.right) for e in edits):
                                natural.append((f"coedit:{r['_id']}", src, edits))
                            break
            if len(targets) >= n_targets and len(natural) >= n_natural:
                break
    except Exception as e:  # noqa: BLE001
        print(f"coedit unavailable: {e}")
    wiki = []
    if n_wiki:
        try:
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            for i, r in enumerate(ds):
                t = r["text"].strip()
                if 150 <= len(t) <= 500 and not t.startswith("=") and "@" not in t:
                    wiki.append((f"wiki:{i}", t))
                    if len(wiki) >= n_wiki:
                        break
                if i > 200000:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"wikitext unavailable: {e}")
    return targets, natural, wiki


def layout_examples(units: list[dict], n: int, rng: random.Random):
    """Multi-line compositions from Ed's sentence/line units."""
    sentences = [u for u in units if u["unit"] in ("sentence", "line") and "\n" not in u["text"]]
    out = []
    for k in range(n):
        picks = rng.sample(sentences, min(len(sentences), rng.randint(2, 4)))
        kind = rng.choice(["bullets", "numbered", "email", "paragraphs"])
        if kind == "bullets":
            marker = rng.choice(["- ", "* ", "• "])
            text = "\n".join(marker + p["text"] for p in picks)
        elif kind == "numbered":
            text = "\n".join(f"{i}. {p['text']}" for i, p in enumerate(picks, 1))
        elif kind == "email":
            opener = rng.choice(EMAIL_OPENERS).format(name=rng.choice(NAMES))
            body = "\n\n".join(p["text"] for p in picks)
            text = f"{opener}\n\n{body}\n\n{rng.choice(EMAIL_CLOSERS)}"
        else:
            text = "\n\n".join(p["text"] for p in picks)
        out.append((f"layout:{k}:" + "+".join(p["source_id"] for p in picks), text))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-ed", type=int, default=3500, help="corrupted examples from Ed's clean pool")
    ap.add_argument("--n-layout", type=int, default=800, help="multi-line compositions")
    ap.add_argument("--n-general", type=int, default=4000, help="corrupted CoEdIT targets")
    ap.add_argument("--n-natural", type=int, default=1500, help="natural CoEdIT pairs")
    ap.add_argument("--n-wiki", type=int, default=500)
    ap.add_argument("--n-bait", type=int, default=600)
    ap.add_argument("--clean-frac", type=float, default=0.17, help="share of NO CHANGES rows per source")
    ap.add_argument("--min-homophone", type=float, default=0.08,
                    help="minimum sampling weight for homophone / wrong-word swaps")
    ap.add_argument("--dense-frac", type=float, default=0.15,
                    help="share of corrupted rows with one mistake per ~3 words")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--no-public", action="store_true", help="skip CoEdIT/wikitext")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    profile, protected = load_profile(), load_protected()
    corruptor = Corruptor(profile, protected)
    # Homophones are rare in the mined profile (~2 %) but the costliest misses
    # (their/they're, our/owe, Deer/Dear): give them a floor.
    for kind, floor in (("homophone", args.min_homophone), ("wrong_word", args.min_homophone)):
        corruptor.weights[kind] = max(corruptor.weights.get(kind, 0), floor)
    gold_ids = set(json.loads((args.out / "gold_ids.json").read_text())) \
        if (args.out / "gold_ids.json").exists() else set()
    units = [u for u in read_jsonl(args.out / "clean_pool.jsonl") if u["source_id"] not in gold_ids]
    real_pairs = read_jsonl(args.out / "real_pairs.jsonl")
    print(f"clean pool units: {len(units)} (gold excluded: {len(gold_ids)}), real pairs: {len(real_pairs)}")

    rows = []

    def add_corrupted(source: str, unit_id: str, text: str, clean_frac: float, seeds: int = 1):
        if len(text) > args.max_chars:
            return
        for s in range(seeds):
            if rng.random() < clean_frac:
                rows.append(row(text, [], source, unit_id))
                continue
            n_errors = None
            if rng.random() < args.dense_frac:
                # Typo-dense sentences ("Ths sentnce has severall speling
                # erors"): one mistake in every ~3 words, up to 8.
                n_errors = max(2, min(8, round(len(text.split()) * 0.35)))
            res = corruptor.corrupt(text, random.Random(f"{unit_id}:{s}:{args.seed}"), n_errors)
            if res:
                rows.append(row(res[0], res[1], source, unit_id))

    # Ed's register: several seeds per unit until the budget is met.
    per_unit = max(1, round(args.n_ed / max(1, len(units))))
    for u in units:
        add_corrupted("ed", u["source_id"], u["text"], args.clean_frac, seeds=per_unit)
    for unit_id, text in layout_examples(units, args.n_layout, rng):
        add_corrupted("layout", unit_id.split(":")[2], text, args.clean_frac)

    if not args.no_public:
        targets, natural, wiki = load_public(args.n_general, args.n_natural, args.n_wiki, rng)
        for unit_id, text in targets:
            add_corrupted("general", unit_id, text, args.clean_frac)
        for unit_id, src, edits in natural:
            rows.append(row(src, edits, "natural", unit_id))
        for unit_id, text in wiki:
            add_corrupted("wiki", unit_id, text, args.clean_frac)
        print(f"public: {len(targets)} coedit targets, {len(natural)} natural pairs, {len(wiki)} wiki")

    for r in real_pairs:
        edits = [Edit(w, x) for w, x in r["edits"]]
        if apply_edits(r["original"], edits).text == r["corrected"]:
            rows.append(row(r["original"], edits, "real", r["id"]))

    # Bait: the sentence itself is sacred; errors (if any) land in a companion.
    for k in range(args.n_bait):
        bait = rng.choice(BAIT_SENTENCES)
        if protected and rng.random() < 0.35:
            word = rng.choice(protected)
            bait = rng.choice([f"The {word} update failed again on the test rig.",
                               f"Can you check {word} before the release?",
                               f"We saw it with {word} but not on the other setup."])
        companion = rng.choice(units)["text"] if units and "\n" not in bait else ""
        text = f"{companion}\n{bait}" if companion and rng.random() < 0.7 else bait
        if rng.random() < 0.5 or not companion:
            rows.append(row(text, [], "bait", f"bait:{k}"))
        else:
            res = corruptor.corrupt(companion, random.Random(f"bait:{k}:{args.seed}"))
            if res:
                corrupted, edits = res
                rows.append(row(f"{corrupted}\n{bait}", edits, "bait", f"bait:{k}"))

    # Split by source unit, shuffle, write.
    train, val = [], []
    for r in rows:
        (val if split_key(r["unit"]) < args.val_frac else train).append(r)
    rng.shuffle(train)
    rng.shuffle(val)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val)):
        with (args.out / f"{name}.jsonl").open("w") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_source = collections.Counter(r["source"] for r in rows)
    stats = {
        "train": len(train), "val": len(val), "by_source": dict(by_source),
        "no_changes_share": round(sum(1 for r in rows if r["n_edits"] == 0) / max(1, len(rows)), 3),
        "mean_edits": round(sum(r["n_edits"] for r in rows) / max(1, len(rows)), 2),
        "mean_chars": round(sum(len(r["prompt"][1]["content"]) for r in rows) / max(1, len(rows))),
        "multiline_share": round(sum(1 for r in rows if "\n" in r["prompt"][1]["content"]) / max(1, len(rows)), 3),
        "system_prompt_sha1": hashlib.sha1(EDIT_SYSTEM_PROMPT.encode()).hexdigest()[:12],
    }
    (args.out / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
