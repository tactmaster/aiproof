#!/usr/bin/env python3
"""Seeded synthetic corruption of clean text, with the exact gold edit list.

Given a clean text, inject realistic mistakes drawn from the writer's own
error profile (training/data/error_profile.json from mine_history.py) and
return the corrupted text plus the ``wrong => right`` edits that undo it.
Everything that must never be "corrected" is left alone and used as an
anchor: URLs, paths, flags, code, numbers, CamelCase/ALLCAPS identifiers,
bullets, indentation, emoji, quoted spans and the protected vocabulary.

Every example round-trips through the real applier
(``apply_edits(corrupted, edits).text == clean``) or it is discarded, so the
training target is exactly what aiproof will execute at inference time.

  training/corrupt.py --demo 8          # corrupt a few clean-pool units
  training/corrupt.py --selftest 500    # round-trip + failure-rate check
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiproof.llm.edits import Edit, _occurrences, apply_edits  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

PUNCT_LEAD = "\"'(“‘[<*_`"
PUNCT_TRAIL = ".,;:!?\"')”’]>*_`"
QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
URLISH_RE = re.compile(r"^(https?://|www\.|[\w.+-]+@[\w-]+\.|/|~/|\.{1,2}/)|`")
BULLET_RE = re.compile(r"^(?:[•\-*+]|\d+[.)])$")
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF☀-➿]")
CAMEL_RE = re.compile(r"[a-z][A-Z]")
FUNCTION_WORDS = ["the", "a", "an", "to", "is", "are", "of", "in", "on", "it",
                  "that", "and", "for", "be", "was", "were", "have", "has",
                  "at", "with", "as", "we", "will", "can", "not"]
HOMOPHONE_GROUPS = [
    ("their", "there", "they're"), ("your", "you're"), ("its", "it's"),
    ("to", "too"), ("then", "than"), ("were", "where", "we're"),
    ("lose", "loose"), ("affect", "effect"), ("accept", "except"),
    ("whether", "weather"), ("here", "hear"), ("know", "no"), ("new", "knew"),
    ("by", "buy"), ("of", "off"), ("is", "if"), ("form", "from"),
    ("quite", "quiet"), ("advice", "advise"), ("passed", "past"),
    ("led", "lead"), ("dear", "deer"), ("through", "threw"), ("week", "weak"),
    ("our", "are"), ("owe", "our"), ("whose", "who's"), ("principal", "principle"),
]
AGREEMENT_SWAPS = {"is": "are", "are": "is", "was": "were", "were": "was",
                   "has": "have", "have": "has", "does": "do", "do": "does",
                   "goes": "go", "went": "go", "bought": "buy", "took": "take",
                   "sent": "send", "made": "make", "got": "get", "did": "do"}
APOSTROPHE_WORDS = {"don't", "doesn't", "didn't", "can't", "won't", "isn't",
                    "aren't", "wasn't", "it's", "that's", "there's", "i'm",
                    "i've", "i'll", "i'd", "we're", "we've", "you're", "you've",
                    "they're", "let's", "what's", "he's", "she's", "haven't",
                    "hasn't", "wouldn't", "couldn't", "shouldn't"}

DEFAULT_WEIGHTS = {
    "missing_letter": 0.22, "misspelling_other": 0.10, "extra_letter": 0.05,
    "doubled_letter": 0.03, "missing_double": 0.03, "transposition": 0.07,
    "adjacent_key": 0.08, "lowercase_start": 0.09, "lowercase_i": 0.04,
    "apostrophe": 0.05, "homophone": 0.08, "wrong_word": 0.03,
    "run_together": 0.04, "split_word": 0.02, "missing_function_word": 0.05,
    "extra_word": 0.02, "agreement": 0.03,
}
DEFAULT_ERRORS_PER_WORD = 0.08


def _split_token(tok: str):
    """(leading punctuation, core, trailing punctuation)."""
    i, j = 0, len(tok)
    while i < j and tok[i] in PUNCT_LEAD:
        i += 1
    while j > i and tok[j - 1] in PUNCT_TRAIL:
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


def _neighbour_key(ch: str, rng: random.Random) -> str:
    low = ch.lower()
    for r, row in enumerate(QWERTY):
        if low in row:
            i = row.index(low)
            options = [row[k] for k in (i - 1, i + 1) if 0 <= k < len(row)]
            for rr in (r - 1, r + 1):
                if 0 <= rr < len(QWERTY):
                    options += [QWERTY[rr][k] for k in (i - 1, i, i + 1)
                                if 0 <= k < len(QWERTY[rr])]
            pick = rng.choice(options)
            return pick.upper() if ch.isupper() else pick
    return ch


class Corruptor:
    def __init__(self, profile: dict | None = None, protected=(), weights=None):
        profile = profile or {}
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        elif profile.get("weights"):
            for k, v in profile["weights"].items():
                if k in self.weights and v > 0:
                    self.weights[k] = v
        self.errors_per_word = profile.get("errors_per_word") or DEFAULT_ERRORS_PER_WORD
        self.missing_letters = profile.get("missing_letters") or {}
        # real wrong-word confusions, corrected -> what the writer typed
        self.wrong_words = {}
        for kind in ("wrong_word", "homophone"):
            for wrong, right, _count in profile.get("pairs", {}).get(kind, []):
                self.wrong_words.setdefault(right, []).append(wrong)
        self.homophones = {}
        for group in HOMOPHONE_GROUPS:
            for w in group:
                self.homophones.setdefault(w, []).extend(x for x in group if x != w)
        self.protected = {w.lower() for w in protected}

    # -- token eligibility -------------------------------------------------

    def _editable(self, tok: str, prev: str | None) -> bool:
        lead, core, trail = _split_token(tok)
        if not core or URLISH_RE.search(tok) or BULLET_RE.match(tok):
            return False
        if "`" in tok or "_" in core or "/" in core or "\\" in core or "=" in core:
            return False
        if any(ch.isdigit() for ch in core) or EMOJI_RE.search(tok):
            return False
        if core.startswith("-") or core.startswith("#") or core.startswith("@"):
            return False
        if CAMEL_RE.search(core) or (core.isupper() and len(core) > 1):
            return False
        if "." in core:  # docs.mender.io, e.g., v1.2
            return False
        if core.lower().strip("'’") in self.protected:
            return False
        if not core.replace("'", "").replace("’", "").isalpha():
            return False
        return True

    # -- single-token operations -----------------------------------------
    # each returns the corrupted core or None when not applicable

    def _op_missing_letter(self, core, rng):
        if len(core) < 4:
            return None
        pos = [i for i in range(1, len(core) - 1)]
        preferred = [i for i in pos if core[i].lower() in self.missing_letters]
        i = rng.choice(preferred if preferred and rng.random() < 0.7 else pos)
        return core[:i] + core[i + 1:]

    def _op_missing_double(self, core, rng):
        doubles = [i for i in range(1, len(core)) if core[i] == core[i - 1] and core[i].isalpha()]
        if not doubles:
            return None
        i = rng.choice(doubles)
        return core[:i] + core[i + 1:]

    def _op_doubled_letter(self, core, rng):
        if len(core) < 3:
            return None
        i = rng.randrange(1, len(core))
        return core[:i] + core[i] + core[i:]

    def _op_extra_letter(self, core, rng):
        if len(core) < 3:
            return None
        i = rng.randrange(1, len(core))
        return core[:i] + _neighbour_key(core[i], rng).lower() + core[i:]

    def _op_transposition(self, core, rng):
        if len(core) < 4:
            return None
        i = rng.randrange(1, len(core) - 2)
        if core[i] == core[i + 1]:
            return None
        return core[:i] + core[i + 1] + core[i] + core[i + 2:]

    def _op_adjacent_key(self, core, rng):
        if len(core) < 4:
            return None
        i = rng.randrange(1, len(core))
        return core[:i] + _neighbour_key(core[i], rng) + core[i + 1:]

    def _op_misspelling_other(self, core, rng):
        """Two independent slips in one word (the 'other' bucket is mostly
        phonetic respellings; two edits approximate it well enough)."""
        if len(core) < 6:
            return None
        first = self._op_missing_letter(core, rng)
        if not first:
            return None
        second = rng.choice([self._op_transposition, self._op_adjacent_key,
                             self._op_extra_letter])(first, rng)
        return second or first

    def _op_lowercase_i(self, core, rng):
        if core == "I" or core.startswith("I'") or core.startswith("I’"):
            return "i" + core[1:]
        return None

    def _op_lowercase_start(self, core, rng, sentence_start=False):
        if sentence_start and core[0].isupper() and core[1:].islower() and len(core) > 1:
            return core[0].lower() + core[1:]
        return None

    def _op_apostrophe(self, core, rng):
        if core.lower().replace("’", "'") in APOSTROPHE_WORDS:
            return core.replace("'", "").replace("’", "")
        return None

    def _op_homophone(self, core, rng):
        options = self.homophones.get(core.lower())
        if not options:
            return None
        pick = rng.choice(options)
        return pick.capitalize() if core[0].isupper() else pick

    def _op_wrong_word(self, core, rng):
        options = self.wrong_words.get(core.lower())
        if not options:
            return self._op_homophone(core, rng)
        pick = rng.choice(options)
        return pick.capitalize() if core[0].isupper() else pick

    def _op_agreement(self, core, rng):
        swap = AGREEMENT_SWAPS.get(core.lower())
        if not swap:
            return None
        return swap.capitalize() if core[0].isupper() else swap

    def _op_split_word(self, core, rng):
        if len(core) < 7:
            return None
        i = rng.randrange(3, len(core) - 2)
        return core[:i] + " " + core[i:]

    # -- main --------------------------------------------------------------

    def corrupt(self, text: str, rng: random.Random, n_errors: int | None = None):
        """Return (corrupted_text, [Edit...]) or None when the text offers no
        safe place to inject an error or the result does not round-trip."""
        toks = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if len(toks) < 2:
            return None
        seps = [text[toks[i][2]:toks[i + 1][1]] for i in range(len(toks) - 1)]
        words = [t[0] for t in toks]
        editable = [i for i, (tok, _, _) in enumerate(toks)
                    if self._editable(tok, words[i - 1] if i else None)]
        if not editable:
            return None
        if n_errors is None:
            n_errors = sum(1 for _ in editable if rng.random() < self.errors_per_word)
            n_errors = max(1, min(10, n_errors))
        ops_order = list(self.weights)
        weights = [self.weights[k] for k in ops_order]
        used, plan = set(), []   # plan: (index, kind, new_core_or_None, extra)
        attempts = 0
        while len(plan) < n_errors and attempts < n_errors * 12:
            attempts += 1
            i = rng.choice(editable)
            if i in used or (i - 1) in used or (i + 1) in used:
                continue
            kind = rng.choices(ops_order, weights)[0]
            lead, core, trail = _split_token(words[i])
            sentence_start = i == 0 or words[i - 1][-1:] in ".!?" or seps[i - 1].count("\n") > 0
            new = None
            if kind == "run_together":
                if i + 1 < len(toks) and (i + 1) in editable and seps[i] == " " and not trail:
                    plan.append((i, kind, None, i + 1))
                    used.update({i, i + 1})
                continue
            if kind == "missing_function_word":
                # Anchor on BOTH neighbours ("importance power" => "importance
                # to power"): with only the previous word the applier's
                # appended-content guard would reject the insertion.
                if core.lower() in FUNCTION_WORDS and not trail and not lead \
                        and 0 < i < len(toks) - 1 and seps[i - 1] == " " and seps[i] == " " \
                        and (i - 1) not in used and (i + 1) not in used:
                    plan.append((i, kind, None, None))
                    used.update({i - 1, i, i + 1})
                continue
            if kind == "extra_word":
                if core.lower() in FUNCTION_WORDS and len(core) <= 4 and not trail and not lead:
                    plan.append((i, kind, None, None))
                    used.add(i)
                continue
            if kind == "lowercase_start":
                new = self._op_lowercase_start(core, rng, sentence_start)
            else:
                fn = getattr(self, f"_op_{kind}", None)
                new = fn(core, rng) if fn else None
            if new is None or new == core:
                continue
            plan.append((i, kind, new, None))
            used.add(i)
        if not plan:
            return None

        # Build the corrupted token list and the edits in text order.
        out, skip = list(words), set()
        edits = []  # (anchor_index, wrong_tokens(list of corrupted idx), right_text)
        for i, kind, new, extra in sorted(plan, key=lambda p: p[0]):
            lead, core, trail = _split_token(words[i])
            if kind == "run_together":
                j = extra
                out[i] = words[i] + words[j]
                skip.add(j)
                edits.append(([i], words[i] + " " + words[j]))
            elif kind == "missing_function_word":
                skip.add(i)
                edits.append(([i - 1, i + 1], words[i - 1] + " " + words[i] + " " + words[i + 1]))
            elif kind == "extra_word":
                out[i] = words[i] + " " + words[i]
                edits.append(([i], words[i]))
            else:
                out[i] = lead + new + trail
                edits.append(([i], words[i]))

        # Assemble text, tracking where each surviving token landed.
        pieces, spans = [], {}
        pos = 0
        for i, tok in enumerate(out):
            if i in skip:
                continue
            if pieces:
                sep = seps[i - 1]
                # A dropped token takes the separator before it with it.
                pieces.append(sep)
                pos += len(sep)
            spans[i] = (pos, pos + len(tok))
            pieces.append(tok)
            pos += len(tok)
        corrupted = "".join(pieces)

        final = []
        for idxs, right in edits:
            i = idxs[0]
            start, end = spans[idxs[0]][0], spans[idxs[-1]][1]
            wrong = corrupted[start:end]
            if "\n" in wrong or "\n" in right:
                return None
            # Widen with unmodified neighbours until the span is unique.
            lo, hi = idxs[0], idxs[-1]
            for _ in range(3):
                if len(_occurrences(corrupted, wrong)) <= 1:
                    break
                grew = False
                if lo - 1 >= 0 and (lo - 1) not in used and (lo - 1) not in skip \
                        and seps[lo - 1] == " ":
                    lo -= 1
                    wrong = corrupted[spans[lo][0]:end]
                    right = words[lo] + " " + right
                    start, grew = spans[lo][0], True
                elif hi + 1 < len(words) and (hi + 1) not in used and (hi + 1) not in skip \
                        and seps[hi] == " ":
                    hi += 1
                    wrong = corrupted[start:spans[hi][1]]
                    right = right + " " + words[hi]
                    end, grew = spans[hi][1], True
                if not grew:
                    break
            if wrong == right:
                continue
            final.append(Edit(wrong, right))
        if not final:
            return None
        result = apply_edits(corrupted, final)
        if result.text != text or result.rejected:
            return None
        return corrupted, final


def load_profile(path: Path = DATA_DIR / "error_profile.json") -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_protected(path: Path = DATA_DIR / "protected_vocab.txt") -> list[str]:
    return [w for w in path.read_text().split() if w] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=0, help="corrupt N clean-pool units and print them")
    ap.add_argument("--selftest", type=int, default=0, help="round-trip N units, report failure rate")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--text", default=None, help="corrupt this text instead of pool units")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    c = Corruptor(load_profile(), load_protected())
    if args.text:
        res = c.corrupt(args.text, rng)
        print(res)
        return 0
    pool_path = DATA_DIR / "clean_pool.jsonl"
    units = [json.loads(ln) for ln in pool_path.read_text().splitlines() if ln.strip()]
    if args.demo:
        for unit in rng.sample(units, min(args.demo, len(units))):
            res = c.corrupt(unit["text"], rng)
            print("-" * 70)
            if not res:
                print("(no corruption possible)", unit["text"][:100])
                continue
            corrupted, edits = res
            print(corrupted)
            print("   " + "\n   ".join(f"{e.wrong} => {e.right}" for e in edits))
    if args.selftest:
        ok = fail = 0
        kinds = {}
        for unit in rng.sample(units, min(args.selftest, len(units))):
            res = c.corrupt(unit["text"], rng)
            if res is None:
                fail += 1
                continue
            ok += 1
            for e in res[1]:
                kinds[len(e.wrong.split())] = kinds.get(len(e.wrong.split()), 0) + 1
        print(f"selftest: {ok} ok, {fail} unusable ({fail / max(1, ok + fail):.1%}); "
              f"edit span lengths (words): {dict(sorted(kinds.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
