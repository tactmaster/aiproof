"""Edit-list output mode: parse ``wrong => right`` lines from the model and
apply them to the original text deterministically.

Why: a small model regenerating the whole text costs ~25 output tokens per
100 characters; an edit list costs ~5 tokens per correction, so latency
scales with the number of mistakes instead of the length of the text. And
because every untouched character is copied from the original, line breaks,
bullets and indentation cannot break — the repair/salvage machinery that
produced the 20-40 s outliers never runs.

The applier is the safety net: an edit is dropped (and logged) unless its
``wrong`` side is found in the original as a whole-word span outside URLs and
code, is short enough to be a correction rather than a rewrite, and adds no
restyling punctuation. Rejections degrade to "fewer fixes", never to
corrupted text.
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass
from typing import NamedTuple

from .postprocess import (
    desemicolon,
    strip_code_fences,
    strip_think_blocks,
    validate_added_punctuation,
    validate_formatting,
    validate_no_appended_content,
)

log = logging.getLogger(__name__)

# "=>" is what the prompt teaches; the others are what small models drift to.
DELIMITERS = ("=>", "->", "→")
NO_CHANGES = "NO CHANGES"
NO_CHANGES_RE = re.compile(
    r"^\W*no\s+(?:changes?|corrections?|errors?|edits?|fixes)\b", re.IGNORECASE
)
# A correction longer than this is a rewrite in disguise. Absolute, not
# relative: a whole-text edit on a 45-char input is fine, on a 500-char
# email it forces granularity.
MAX_EDIT_SPAN_CHARS = 100
# |words(right) - words(wrong)| beyond max(this, words(wrong)//2) is a
# rephrase, not a correction.
MAX_WORD_DELTA = 2

_LIST_PREFIX_RE = re.compile(r"^(?:[•\-*]|\d+[.)])[ \t]+")
_QUOTE_PAIRS = {'"': '"', "'": "'", "`": "`", "“": "”", "‘": "’"}
_QUOTE_CHARS = "\"'`“”‘’"
_FOLD_TABLE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
# Spans an edit may never land inside: URLs, e-mail addresses, inline code.
_PROTECTED_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+|`[^`\n]+`")
_JSON_KEYS = (
    ("wrong", "right"), ("from", "to"), ("original", "corrected"),
    ("old", "new"), ("incorrect", "correct"), ("error", "correction"),
)


@dataclass(frozen=True)
class Edit:
    wrong: str
    right: str


class ParseResult(NamedTuple):
    edits: list
    no_changes: bool   # the NO CHANGES sentinel was seen
    noise_lines: int   # non-blank lines that were neither an edit nor the sentinel


class ApplyResult(NamedTuple):
    text: str
    applied: list      # Edits actually spliced in (right side after normalisation)
    rejected: list     # (Edit, reason) pairs


# -- parsing -----------------------------------------------------------------


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] in _QUOTE_PAIRS and s[-1] == _QUOTE_PAIRS[s[0]]:
        return s[1:-1].strip()
    return s


def _parse_line(line: str):
    for delim in DELIMITERS:
        if delim in line:
            left, right = line.split(delim, 1)
            break
    else:
        return None
    left, right = left.strip(), right.strip()
    # "- teh => the" is list formatting the model added around the line; a
    # bullet that is part of the edit itself shows up on BOTH sides.
    m = _LIST_PREFIX_RE.match(left)
    if m:
        rm = _LIST_PREFIX_RE.match(right)
        if not (rm and rm.group(0).strip() == m.group(0).strip()):
            left = left[m.end():]
    left, right = _strip_quotes(left), _strip_quotes(right)
    # An empty right side is indistinguishable from a line cut off by the
    # token limit; deletions are expressed with a neighbouring word instead.
    if not left or not right:
        return None
    return Edit(left, right)


def _parse_json_edits(body: str) -> list:
    """Models that ignore the line format sometimes answer in JSON anyway."""
    try:
        data = json.loads(body)
    except ValueError:
        return []
    if isinstance(data, dict):
        for key in ("edits", "corrections", "changes"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    pairs = []
    if isinstance(data, dict):
        pairs = [(k, v) for k, v in data.items()
                 if isinstance(k, str) and isinstance(v, str)]
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for wk, rk in _JSON_KEYS:
                    if isinstance(item.get(wk), str) and isinstance(item.get(rk), str):
                        pairs.append((item[wk], item[rk]))
                        break
            elif (isinstance(item, list) and len(item) == 2
                  and all(isinstance(x, str) for x in item)):
                pairs.append((item[0], item[1]))
    edits, seen = [], set()
    for wrong, right in pairs:
        wrong, right = wrong.strip(), right.strip()
        if wrong and right and (wrong, right) not in seen:
            seen.add((wrong, right))
            edits.append(Edit(wrong, right))
    return edits


def parse_edits(response: str) -> ParseResult:
    """Extract the edit list from a model response. Tolerates think-blocks,
    code fences, list markers, quoted sides, alternative arrows, chatter
    lines, and JSON-shaped answers."""
    if not isinstance(response, str) or not response.strip():
        return ParseResult([], False, 0)
    body = strip_code_fences(strip_think_blocks(response.replace("\r\n", "\n")).strip())
    edits, seen, no_changes, noise = [], set(), False, 0
    for raw in body.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if NO_CHANGES_RE.match(line):
            no_changes = True
            continue
        edit = _parse_line(line)
        if edit is None:
            noise += 1
            log.debug("edit list: ignoring line %r", line[:80])
            continue
        if (edit.wrong, edit.right) in seen:
            continue
        seen.add((edit.wrong, edit.right))
        edits.append(edit)
    if not edits and not no_changes and body[:1] in "[{":
        edits = _parse_json_edits(body)
    return ParseResult(edits, no_changes, noise)


# -- applying ----------------------------------------------------------------


def _is_word_char(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch == "_")


def _occurrences(text: str, needle: str) -> list:
    """Start offsets of whole-word occurrences: wherever ``needle`` begins or
    ends with a word character, the neighbouring character in ``text`` must
    not be one ("is" never matches inside "this")."""
    starts = []
    for m in re.finditer(re.escape(needle), text):
        i, j = m.start(), m.end()
        if _is_word_char(needle[0]) and i > 0 and _is_word_char(text[i - 1]):
            continue
        if _is_word_char(needle[-1]) and j < len(text) and _is_word_char(text[j]):
            continue
        starts.append(i)
    return starts


def _protected_spans(text: str) -> list:
    return [(m.start(), m.end()) for m in _PROTECTED_RE.finditer(text)]


def _overlaps(start: int, end: int, spans) -> bool:
    return any(start < e and s < end for s, e, *_ in spans)


def _check_edit(original: str, folded: str, has_curly: bool, edit: Edit,
                accepted: list, protected: list):
    """Returns (reason, spans, right): reason is None when the edit is
    acceptable; spans is empty for a no-op."""
    wrong, right = edit.wrong, edit.right
    if "\n" in wrong or "\n" in right:
        return "spans a line break", [], right
    if wrong == right:
        return None, [], right
    if len(wrong) > MAX_EDIT_SPAN_CHARS:
        return f"span too long ({len(wrong)} chars)", [], right
    wc_wrong, wc_right = len(wrong.split()), len(right.split())
    if abs(wc_right - wc_wrong) > max(MAX_WORD_DELTA, wc_wrong // 2):
        return f"rephrase ({wc_wrong} -> {wc_right} words)", [], right
    right = desemicolon(right, wrong)
    for validator in (validate_added_punctuation, validate_formatting,
                      validate_no_appended_content):
        ok, issues = validator(wrong, right)
        if not ok:
            return issues[0], [], right

    wrong_f = wrong.translate(_FOLD_TABLE)
    if not has_curly:
        right = right.translate(_FOLD_TABLE)
    if wrong_f == right.translate(_FOLD_TABLE):
        return None, [], right  # only quote style differed
    starts = _occurrences(folded, wrong_f)
    if not starts and (wrong_f[0] in _QUOTE_CHARS or wrong_f[-1] in _QUOTE_CHARS):
        # The model copied the prompt's wrapping quote into the first/last
        # span ("teh => "the). Retry without it, on both sides.
        alt = wrong_f
        if alt[0] in _QUOTE_CHARS:
            if right[:1] == alt[0]:
                right = right[1:]
            alt = alt[1:]
        if alt and alt[-1] in _QUOTE_CHARS:
            if right[-1:] == alt[-1]:
                right = right[:-1]
            alt = alt[:-1]
        alt = alt.strip()
        if alt:
            starts = _occurrences(folded, alt)
            if starts:
                wrong_f = alt
    if not starts and wrong_f.lower() != wrong_f or not starts and wrong_f.islower():
        # Small models re-case the span they quote ("deploymnt" for
        # "Deploymnt"). Accept a case-insensitive whole-word match when it is
        # unambiguous, and carry the original's leading capital over.
        ci = _occurrences(folded.lower(), wrong_f.lower())
        if len(ci) == 1:
            found = folded[ci[0]:ci[0] + len(wrong_f)]
            if found[0].isupper() and right[:1].islower():
                right = right[0].upper() + right[1:]
            elif found[0].islower() and right[:1].isupper() and wrong_f[:1].isupper():
                right = right[0].lower() + right[1:]
            wrong_f, starts = found, ci
    if not starts:
        return "not in text", [], right
    spans = [(s, s + len(wrong_f)) for s in starts
             if not _overlaps(s, s + len(wrong_f), protected)]
    if not spans:
        return "only inside a URL/code span", [], right
    for s, e in spans:
        if _overlaps(s, e, accepted):
            return "overlaps an earlier edit", [], right
    if has_curly and "'" in right:
        # Keep the writer's apostrophe style inside the replaced span.
        span_text = original[spans[0][0]:spans[0][1]]
        if "’" in span_text:
            right = right.replace("'", "’")
    return None, spans, right


def apply_edits(original: str, edits: list) -> ApplyResult:
    """Splice accepted edits into ``original``. Every rejection carries a
    reason; the text is returned unchanged when nothing survives, and the
    final result must pass the same layout check as a full rewrite."""
    if not edits:
        return ApplyResult(original, [], [])
    if not original:
        return ApplyResult(original, [], [(e, "empty text") for e in edits])
    folded = original.translate(_FOLD_TABLE)
    has_curly = folded != original
    protected = _protected_spans(original)
    accepted, applied, rejected = [], [], []
    for edit in edits:
        reason, spans, right = _check_edit(original, folded, has_curly, edit,
                                           accepted, protected)
        if reason:
            rejected.append((edit, reason))
            continue
        if not spans:
            continue
        accepted.extend((s, e, right) for s, e in spans)
        applied.append(Edit(edit.wrong, right))
    if not accepted:
        return ApplyResult(original, [], rejected)
    result = original
    for s, e, right in sorted(accepted, key=lambda t: t[0], reverse=True):
        result = result[:s] + right + result[e:]
    ok, issues = validate_formatting(original, result)
    if not ok:
        reason = "final layout check: " + "; ".join(issues)
        return ApplyResult(original, [], rejected + [(e, reason) for e in applied])
    return ApplyResult(result, applied, rejected)


def looks_like_rewrite(original: str, candidate: str) -> bool:
    """Did the model answer with the full corrected text instead of a list?
    Then it can go through the rewrite pipeline without a second call."""
    if not original or not candidate:
        return False
    if not (0.5 * len(original) < len(candidate) < 1.5 * len(original)):
        return False
    return difflib.SequenceMatcher(None, original, candidate).ratio() >= 0.6
