#!/usr/bin/env python3
"""Mine aiproof's proofread history for training signal.

Reads ~/.local/share/aiproof/history.jsonl (opt-in history: original text +
what the pipeline produced) and writes, under training/data/ (git-ignored):

  error_profile.json   what kinds of mistakes the writer actually makes, as
                       weights for corrupt.py (missing letter, homophone, ...)
  clean_pool.jsonl     corrected texts that survived the plausibility filter,
                       as whole texts, single lines and sentence windows:
                       the writer's own register for synthetic corruption
  protected_vocab.txt  product names / identifiers that must never be
                       "corrected" (context.json words + shape-signal tokens)

Gold review (the one manual step):

  --review-candidates  write data/review.jsonl: real pairs with a small,
                       plausible diff and the derived edit list, for Ed to
                       accept/fix in an editor ("accept": false to drop)
  --finalize-gold      split the accepted rows into the private eval set
                       (~/.local/share/aiproof/evals_private.json) and
                       data/real_pairs.jsonl for training; texts used as gold
                       are listed in data/gold_ids.json so build_dataset.py
                       keeps them out of the training mix.

Nothing here leaves the machine; see training/README.md.
"""

import argparse
import collections
import difflib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiproof.llm.edits import Edit, apply_edits  # noqa: E402

try:
    from aiproof.context import _has_shape_signal as has_shape_signal  # noqa: E402
    from aiproof.context import _load_wordlist as load_wordlist  # noqa: E402
except ImportError:  # pragma: no cover - older checkout
    def has_shape_signal(word: str) -> bool:
        return bool(re.search(r"[A-Z].*[a-z].*[A-Z]|\d|[._-]", word)) or (
            word.isupper() and len(word) > 1)

    def load_wordlist():
        return None

DATA_DIR = Path(__file__).resolve().parent / "data"
HISTORY = Path.home() / ".local/share/aiproof/history.jsonl"
CONTEXT = Path.home() / ".local/share/aiproof/context.json"
PRIVATE_EVAL = Path.home() / ".local/share/aiproof/evals_private.json"

PUNCT = ".,;:!?\"'()[]{}<>“”‘’—–-"
HOMOPHONE_GROUPS = [
    ("their", "there", "they're"), ("your", "you're"), ("its", "it's"),
    ("to", "too", "two"), ("then", "than"), ("were", "where", "we're"),
    ("lose", "loose"), ("our", "owe", "are"), ("affect", "effect"),
    ("accept", "except"), ("whether", "weather"), ("which", "witch"),
    ("here", "hear"), ("know", "no"), ("new", "knew"), ("by", "buy", "bye"),
    ("of", "off"), ("is", "if"), ("that", "than"), ("form", "from"),
    ("quite", "quiet"), ("advice", "advise"), ("passed", "past"),
    ("principal", "principle"), ("complement", "compliment"), ("led", "lead"),
    ("dear", "deer"), ("threw", "through"), ("week", "weak"), ("sent", "cent"),
]
HOMOPHONES = {frozenset(pair) for group in HOMOPHONE_GROUPS
              for pair in __import__("itertools").combinations(group, 2)}
FUNCTION_WORDS = {"the", "a", "an", "to", "is", "are", "of", "in", "on", "it",
                  "that", "and", "for", "be", "was", "were", "have", "has", "do",
                  "at", "with", "as", "we", "i", "you", "will", "can", "not"}
QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
URLISH_RE = re.compile(r"https?://|www\.|@\S+\.\S+|/[\w.-]+/|`")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(])")
TOKEN_RE = re.compile(r"\S+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9._\-']*")


# -- diff classification -------------------------------------------------------


def core(tok: str) -> str:
    return tok.strip(PUNCT)


def _adjacent_keys(a: str, b: str) -> bool:
    for row in QWERTY:
        if a in row and b in row:
            return abs(row.index(a) - row.index(b)) == 1
    for r1, r2 in zip(QWERTY, QWERTY[1:]):
        if (a in r1 and b in r2) or (a in r2 and b in r1):
            i = (r1.index(a) if a in r1 else r1.index(b))
            j = (r2.index(b) if b in r2 else r2.index(a))
            return abs(i - j) <= 1
    return False


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def _missing_letter(short: str, long: str):
    """The letter in ``long`` that ``short`` lacks, and whether it doubles a
    neighbour (comit -> commit)."""
    for i in range(len(long)):
        if long[:i] + long[i + 1:] == short:
            ch = long[i]
            doubled = (i > 0 and long[i - 1] == ch) or (i + 1 < len(long) and long[i + 1] == ch)
            return ch, doubled
    return None, False


def classify_word(a: str, b: str, wordlist) -> str:
    """Error type of the writer's token ``a`` against the corrected ``b``."""
    ca, cb = core(a), core(b)
    if not ca or not cb:
        return "punct_only"
    if ca == cb:
        return "punct_only"
    if ca.lower() == cb.lower():
        if ca == "i":
            return "lowercase_i"
        if ca[0].islower() and cb[0].isupper() and ca[1:] == cb[1:]:
            return "lowercase_start"
        return "case_other"
    la, lb = ca.lower(), cb.lower()
    strip_apos = str.maketrans("", "", "'’")
    if la.translate(strip_apos) == lb.translate(strip_apos):
        return "apostrophe"
    if frozenset((la, lb)) in HOMOPHONES:
        return "homophone"
    if len(la) == len(lb):
        diff = [i for i in range(len(la)) if la[i] != lb[i]]
        if (len(diff) == 2 and diff[1] == diff[0] + 1
                and la[diff[0]] == lb[diff[1]] and la[diff[1]] == lb[diff[0]]):
            return "transposition"
        if len(diff) == 1:
            return "adjacent_key" if _adjacent_keys(la[diff[0]], lb[diff[0]]) else "substitution"
    if len(lb) == len(la) + 1 and _is_subsequence(la, lb):
        _, doubled = _missing_letter(la, lb)
        return "missing_double" if doubled else "missing_letter"
    if len(la) == len(lb) + 1 and _is_subsequence(lb, la):
        _, doubled = _missing_letter(lb, la)
        return "doubled_letter" if doubled else "extra_letter"
    if wordlist and la in wordlist and lb in wordlist:
        return "wrong_word"
    return "misspelling_other"


def classify_opcode(tag, a_toks, b_toks, wordlist) -> list[tuple[str, str, str]]:
    """(type, wrong, right) triples for one non-equal opcode."""
    if tag == "insert":
        return [("missing_function_word" if core(w).lower() in FUNCTION_WORDS
                 else "missing_word", "", w) for w in b_toks]
    if tag == "delete":
        return [("extra_word", w, "") for w in a_toks]
    if len(a_toks) == 1 and len(b_toks) == 2 and \
            core(a_toks[0]).lower() == (core(b_toks[0]) + core(b_toks[1])).lower():
        return [("run_together", a_toks[0], " ".join(b_toks))]
    if len(a_toks) == 2 and len(b_toks) == 1 and \
            (core(a_toks[0]) + core(a_toks[1])).lower() == core(b_toks[0]).lower():
        return [("split_word", " ".join(a_toks), b_toks[0])]
    if len(a_toks) == len(b_toks):
        return [(classify_word(a, b, wordlist), a, b) for a, b in zip(a_toks, b_toks)]
    return [("rephrase", " ".join(a_toks), " ".join(b_toks))]


def diff_ops(original: str, corrected: str):
    a, b = original.split(), corrected.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return a, b, sm.ratio(), [op for op in sm.get_opcodes() if op[0] != "equal"]


# -- record filters ------------------------------------------------------------


def load_history(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open() as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_id"] = f"h{n}"
            records.append(rec)
    return records


def restyled(original: str, corrected: str) -> bool:
    return any(corrected.count(ch) > original.count(ch) for ch in ";—–")


def plausible(rec: dict, min_ratio: float, max_span: int, max_ops: int) -> bool:
    """A real pair whose diff looks like proofreading, not a rewrite."""
    orig, corr = rec.get("original", ""), rec.get("corrected", "")
    if rec.get("pipeline_path") == "fallback" or not orig or not corr:
        return False
    if orig.count("\n") != corr.count("\n") or restyled(orig, corr):
        return False
    _, _, ratio, ops = diff_ops(orig, corr)
    if ratio < min_ratio or len(ops) > max_ops:
        return False
    return all(max(i2 - i1, j2 - j1) <= max_span for _, i1, i2, j1, j2 in ops)


# -- outputs -------------------------------------------------------------------


def build_profile(records: list[dict], wordlist) -> dict:
    types = collections.Counter()
    examples = collections.defaultdict(list)
    pairs = collections.defaultdict(collections.Counter)
    missing_letters = collections.Counter()
    words_total = ops_total = 0
    for rec in records:
        if not rec.get("changed") or not plausible(rec, 0.75, 4, 20):
            continue
        a, b, _, ops = diff_ops(rec["original"], rec["corrected"])
        words_total += len(a)
        for tag, i1, i2, j1, j2 in ops:
            for kind, wrong, right in classify_opcode(tag, a[i1:i2], b[j1:j2], wordlist):
                types[kind] += 1
                ops_total += 1
                if len(examples[kind]) < 6:
                    examples[kind].append([wrong, right])
                if kind in ("homophone", "wrong_word", "apostrophe"):
                    pairs[kind][(core(wrong).lower(), core(right).lower())] += 1
                if kind in ("missing_letter", "missing_double"):
                    ch, _ = _missing_letter(core(wrong).lower(), core(right).lower())
                    if ch:
                        missing_letters[ch] += 1
    corruptable = {k: v for k, v in types.items()
                   if k not in ("punct_only", "rephrase", "case_other")}
    total = sum(corruptable.values()) or 1
    return {
        "records_used": sum(1 for r in records if r.get("changed") and plausible(r, 0.75, 4, 20)),
        "words_total": words_total, "ops_total": ops_total,
        "errors_per_word": round(ops_total / words_total, 4) if words_total else 0,
        "types": dict(types.most_common()),
        "weights": {k: round(v / total, 4) for k, v in corruptable.items()},
        "pairs": {k: [[w, r, c] for (w, r), c in v.most_common(40)] for k, v in pairs.items()},
        "missing_letters": dict(missing_letters.most_common()),
        "examples": dict(examples),
    }


def _units(text: str):
    """(unit_kind, text) pieces of one clean text."""
    text = text.strip()
    if len(text.split()) >= 3:
        yield "whole", text
    lines = [ln.strip() for ln in text.split("\n")]
    if len(lines) > 1:
        for ln in lines:
            if len(ln.split()) >= 4 and not URLISH_RE.search(ln):
                yield "line", ln
    flat = " ".join(ln for ln in lines if ln)
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(flat) if len(s.split()) >= 3]
    if len(sentences) > 1:
        for i, s in enumerate(sentences):
            yield "sentence", s
            if i + 1 < len(sentences):
                yield "sentence", s + " " + sentences[i + 1]


def build_clean_pool(records: list[dict]) -> list[dict]:
    seen, pool = set(), []
    for rec in records:
        if rec.get("pipeline_path") == "fallback":
            continue
        text = rec.get("corrected") or rec.get("original") or ""
        if rec.get("changed") and not plausible(rec, 0.80, 3, 20):
            continue
        for kind, unit in _units(text):
            key = re.sub(r"\s+", " ", unit.lower())
            if key in seen or URLISH_RE.search(unit) and kind != "whole":
                continue
            seen.add(key)
            pool.append({"id": f"{rec['_id']}:{kind}:{len(pool)}", "source_id": rec["_id"],
                         "unit": kind, "text": unit, "lines": unit.count("\n") + 1})
    return pool


def _vocab_candidate(word: str) -> str:
    """The token without sentence punctuation, or "" when it is just an
    ordinary word: a trailing full stop is not a shape signal."""
    word = word.strip(".,;:!?'")
    if len(word) < 2 or word.isdigit() or word.lower() in FUNCTION_WORDS:
        return ""
    plain = word.replace(".", "").replace("-", "").replace("_", "")
    if plain.isalpha() and (plain.islower() or plain.istitle()) \
            and not re.search(r"[._-]\w", word):
        return ""
    return word if has_shape_signal(word) else ""


def build_protected_vocab(records: list[dict], context_path: Path) -> list[str]:
    seen_in = collections.Counter()
    for rec in records:
        orig, corr = rec.get("original", ""), rec.get("corrected", "")
        for raw in set(WORD_RE.findall(orig)):
            word = _vocab_candidate(raw)
            if word and word in corr:
                seen_in[word] += 1
    vocab = {w for w, n in seen_in.items() if n >= 2}
    if context_path.exists():
        try:
            vocab.update(json.loads(context_path.read_text()).get("words", []))
        except (json.JSONDecodeError, AttributeError):
            pass
    return sorted(vocab, key=str.lower)


# -- gold review ---------------------------------------------------------------


def _spans(text: str):
    return [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def derive_edits(original: str, corrected: str, widen: int = 0):
    """Edit list for a real pair, with spans cut from the ORIGINAL string so
    whitespace inside a span is exact. ``widen`` adds neighbouring tokens
    (needed when a span occurs more than once)."""
    a_sp, b_sp = _spans(original), _spans(corrected)
    a, b = original.split(), corrected.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            if i1 > 0:
                i1, j1 = i1 - 1, j1 - 1
            else:
                i2, j2 = i2 + 1, j2 + 1
        elif tag == "delete":
            if i1 > 0:
                i1, j1 = i1 - 1, j1 - 1
            else:
                i2, j2 = i2 + 1, j2 + 1
        i1, j1 = max(0, i1 - widen), max(0, j1 - widen)
        i2, j2 = min(len(a), i2 + widen), min(len(b), j2 + widen)
        if i1 >= i2 or j1 >= j2:
            return None
        wrong = original[a_sp[i1][0]:a_sp[i2 - 1][1]]
        right = corrected[b_sp[j1][0]:b_sp[j2 - 1][1]]
        edits.append(Edit(wrong, right))
    out, seen = [], set()
    for e in edits:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def derive_edits_verified(orig: str, corr: str):
    """Edit list that provably turns ``orig`` into ``corr`` via the real
    applier, or None. Multi-line pairs (same line count by construction)
    are diffed line by line so no span crosses a newline; a span that is
    ambiguous in the whole text is widened with its neighbours."""
    o_lines, c_lines = orig.split("\n"), corr.split("\n")
    if len(o_lines) != len(c_lines):
        return None
    for widen in (0, 1, 2):
        edits = []
        for o, c in zip(o_lines, c_lines):
            if o == c:
                continue
            part = derive_edits(o, c, widen)
            if part is None:
                edits = None
                break
            edits.extend(part)
        if not edits:
            continue
        # dedupe, keep order
        seen, uniq = set(), []
        for e in edits:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        if apply_edits(orig, uniq).text == corr:
            return uniq
    return None


def review_candidates(records: list[dict], out_path: Path) -> int:
    rows = []
    for rec in records:
        if not rec.get("changed") or not plausible(rec, 0.85, 3, 12):
            continue
        orig, corr = rec["original"], rec["corrected"]
        edits = derive_edits_verified(orig, corr)
        verified = edits is not None
        if edits is None:
            edits = derive_edits(orig, corr, 0) or []
        rows.append({
            "id": rec["_id"], "timestamp": rec.get("timestamp"),
            "model": rec.get("model"), "accept": True, "verified": verified,
            "edits": [[e.wrong, e.right] for e in (edits or [])],
            "original": orig, "corrected": corr,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def finalize_gold(review_path: Path, gold_size: int, seed: int, vocab: list[str],
                  private_eval: Path, pairs_path: Path, gold_ids_path: Path) -> tuple[int, int]:
    rows = [json.loads(ln) for ln in review_path.read_text().splitlines() if ln.strip()]
    accepted = [r for r in rows if r.get("accept", True) and r.get("edits")]
    # Re-verify: Ed may have edited the lists by hand.
    good = []
    for r in accepted:
        edits = [Edit(w, x) for w, x in r["edits"]]
        result = apply_edits(r["original"], edits)
        if result.text == r["corrected"] or r.get("force"):
            r["corrected"] = result.text
            good.append(r)
    random.Random(seed).shuffle(good)
    gold, train = good[:gold_size], good[gold_size:]
    cases = []
    for n, r in enumerate(gold, 1):
        present = [w for w in vocab if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", r["corrected"])]
        cases.append({"id": f"gold-{n:03d}", "category": "gold", "input": r["original"],
                      "expected": r["corrected"], "must_contain": present,
                      "must_not_contain": []})
    private_eval.parent.mkdir(parents=True, exist_ok=True)
    private_eval.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    private_eval.chmod(0o600)
    with pairs_path.open("w") as fh:
        for r in train:
            fh.write(json.dumps({"id": r["id"], "original": r["original"],
                                 "corrected": r["corrected"], "edits": r["edits"]},
                                ensure_ascii=False) + "\n")
    gold_ids_path.write_text(json.dumps([r["id"] for r in gold]))
    return len(gold), len(train)


# -- main ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--history", type=Path, default=HISTORY)
    ap.add_argument("--context", type=Path, default=CONTEXT)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--review-candidates", action="store_true",
                    help="write data/review.jsonl for manual review")
    ap.add_argument("--finalize-gold", action="store_true",
                    help="split the reviewed rows into the private eval set + real_pairs.jsonl")
    ap.add_argument("--gold-size", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--private-eval", type=Path, default=PRIVATE_EVAL)
    args = ap.parse_args()

    records = load_history(args.history)
    if not records:
        print(f"no history at {args.history} (enable 'Save proofread history' first)")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    wordlist = load_wordlist()

    profile = build_profile(records, wordlist)
    (args.out / "error_profile.json").write_text(json.dumps(profile, indent=1, ensure_ascii=False))
    pool = build_clean_pool(records)
    with (args.out / "clean_pool.jsonl").open("w") as fh:
        for unit in pool:
            fh.write(json.dumps(unit, ensure_ascii=False) + "\n")
    vocab = build_protected_vocab(records, args.context)
    (args.out / "protected_vocab.txt").write_text("\n".join(vocab) + "\n")

    changed = sum(1 for r in records if r.get("changed"))
    print(f"history: {len(records)} records, {changed} changed, "
          f"{profile['records_used']} usable for the error profile")
    print(f"errors per word: {profile['errors_per_word']}  (ops={profile['ops_total']}, "
          f"words={profile['words_total']})")
    print("error types: " + ", ".join(f"{k}={v}" for k, v in list(profile["types"].items())[:12]))
    kinds = collections.Counter(u["unit"] for u in pool)
    print(f"clean pool: {len(pool)} units ({dict(kinds)}), "
          f"{sum(len(u['text']) for u in pool)} chars")
    print(f"protected vocabulary: {len(vocab)} words -> {args.out / 'protected_vocab.txt'}")

    if args.review_candidates:
        n = review_candidates(records, args.out / "review.jsonl")
        print(f"review candidates: {n} -> {args.out / 'review.jsonl'}  "
              "(edit 'edits' or set \"accept\": false, then --finalize-gold)")
    if args.finalize_gold:
        review = args.out / "review.jsonl"
        if not review.exists():
            print("run --review-candidates first")
            return 1
        gold, train = finalize_gold(review, args.gold_size, args.seed, vocab,
                                    args.private_eval, args.out / "real_pairs.jsonl",
                                    args.out / "gold_ids.json")
        print(f"gold eval cases: {gold} -> {args.private_eval}; "
              f"training pairs: {train} -> {args.out / 'real_pairs.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
