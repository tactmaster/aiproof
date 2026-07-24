"""Response cleanup and formatting-preservation guards, ported from
ai-text-proofreader-extension/background/background.js (cleanResponse :2014,
enforceFormattingPreservation :2083, validateFormattingPreservation :2130).

Models routinely ignore "output only the corrected text", so this cleanup is
load-bearing: strip wrapper phrases, unwrap quotes, drop code fences, and if
the result mangles the original's line/bullet structure, retry once with a
repair prompt — falling back to the untouched original rather than ever losing
the user's text.
"""

import logging
import re
from typing import Callable

from .prompts import build_constrained_prompt, build_repair_prompt

log = logging.getLogger(__name__)

_WRAPPER_PHRASES = [
    # Generic: "Here is/Here's the corrected output/version/sentence/…"
    r"^[ \t]*(?:sure[,!]?\s+|certainly[,!]?\s+|of\s+course[,!]?\s+)?"
    r"here(?:'s|\s+is)\s+(?:the|your|an?)\s+"
    r"(?:corrected?|proofread|fixed|improved|revised|updated)\s+"
    r"(?:text|output|version|sentence|paragraph|email|message|result):?\s*",
    r"^[ \t]*here\s+is\s+the\s+corrected?\s+text:?\s*",
    r"^[ \t]*here\s+is\s+the\s+improved\s+text:?\s*",
    r"^[ \t]*here\s+is\s+an?\s+improved\s+version:?\s*",
    r"^[ \t]*corrected\s+text:\s*",
    r"^[ \t]*improved\s+text:\s*",
    r"^[ \t]*fixed\s+text:\s*",
    r"^[ \t]*revised\s+text:\s*",
    r"^[ \t]*the\s+corrected?\s+text\s+is:?\s*",
    r"^[ \t]*the\s+improved\s+text\s+is:?\s*",
    r"^[ \t]*the\s+fixed\s+text\s+is:?\s*",
    r"^[ \t]*here\s+you\s+go:?\s*",
    r"^[ \t]*here\s+it\s+is:?\s*",
    r"^[ \t]*sure[,!]?\s+here\s+is\s+the\s+corrected?\s+text:?\s*",
    r"^[ \t]*certainly[,!]?\s+here\s+is\s+the\s+corrected?\s+text:?\s*",
    r"^[ \t]*sure[,!]?\s+i'?d\s+be\s+happy\s+to\s+help.*?here'?s.*?:?\s*",
    r"^[ \t]*i'?d\s+be\s+happy\s+to\s+help.*?here'?s.*?:?\s*",
    r"^[ \t]*of\s+course[,!]?\s+here'?s.*?:?\s*",
    r"^[ \t]*absolutely[,!]?\s+here'?s.*?:?\s*",
    r"^[ \t]*here'?s\s+the\s+corrected?\s+version:?\s*",
    r"^[ \t]*here'?s\s+an?\s+improved\s+version:?\s*",
    r"^[ \t]*here'?s\s+the\s+proofread\s+text:?\s*",
    r"^[ \t]*let\s+me\s+help.*?:?\s*",
    r"^[ \t]*i\s+can\s+help.*?:?\s*",
    r"^[ \t]*sure[,!]?\s+here\s+is\s+the\s+corrected?\s+email:?\s*",
    r"^[ \t]*here\s+is\s+the\s+corrected?\s+email:?\s*",
    r"^[ \t]*corrected?\s+email:\s*",
    r"^[ \t]*here\s+is\s+your\s+corrected?\s+(?:text|email|message):?\s*",
    r"^[ \t]*your\s+corrected?\s+(?:text|email|message):?\s*",
    r"^[ \t]*corrected?\s+version:\s*",
    r"^[ \t]*improved?\s+version:\s*",
    r"^[ \t]*output:\s*",
    r"^[ \t]*result:\s*",
]
_WRAPPER_RES = [re.compile(p, re.IGNORECASE) for p in _WRAPPER_PHRASES]

_FENCE_OPEN = re.compile(r"^```[\w]*\s*\n?")
_FENCE_CLOSE = re.compile(r"\n?```\s*$")
_BULLET_RE = re.compile(r"[•\-*]\s+")
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)

# Chatty sign-offs some models append AFTER the corrected text ("Is there
# anything else I can help you with?"). Each pattern matches one full
# sign-off phrase at the very end of the response.
# The [^\n.!?]{0,N} tails absorb phrase variants ("…helps you out") but must
# never cross a sentence boundary, or a matching phrase mid-text would drag
# the user's following sentences away with it.
_SIGNOFF_PHRASES = [
    r"is there anything else (?:i can|you(?:'d| would) like me to)\s+"
    r"(?:help(?: you)? with|do(?: for you)?)[^\n.!?]{0,40}[.!?]*",
    r"anything else i can (?:help(?: you)? with|do(?: for you)?)[^\n.!?]{0,40}[.!?]*",
    r"is there anything else[^\n.!?]{0,60}[.!?]*",
    r"(?:i )?hope (?:this|that|it) helps[^\n.!?]{0,60}[.!?]*",
    r"let me know if (?:you|there)[^\n.!?]{0,100}[.!?]*",
    r"(?:please )?(?:don'?t hesitate|feel free) to"
    r" (?:ask|reach out|let me know|contact me)[^\n.!?]{0,80}[.!?]*",
    r"if you (?:have|need) any (?:other|more|further|additional)"
    r" (?:questions|help|assistance|changes|corrections)[^\n.!?]{0,80}[.!?]*",
    r"would you like me to [^\n.!?]{0,100}[.!?]*",
    r"happy to help[^\n.!?]{0,60}[.!?]*",
]
_SIGNOFF_RES = [
    re.compile(r"(?:(?<=\s)|^)(?:" + p + r")\s*$", re.IGNORECASE)
    for p in _SIGNOFF_PHRASES
]


def clean_response(response: str) -> str:
    if not isinstance(response, str):
        return response

    cleaned = response.replace("\r\n", "\n")

    # Reasoning models (qwen3, deepseek-r1, …) emit <think>…</think> blocks
    # before the answer; drop them (and any unclosed opener) entirely.
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*\Z", "", cleaned, flags=re.DOTALL)

    for phrase in _WRAPPER_RES:
        cleaned = phrase.sub("", cleaned, count=1)

    # Unwrap quotes only when the whole response is one quoted line.
    if "\n" not in cleaned and (
        (cleaned.startswith('"') and cleaned.endswith('"'))
        or (cleaned.startswith("'") and cleaned.endswith("'"))
    ):
        cleaned = cleaned[1:-1]

    cleaned = _FENCE_OPEN.sub("", cleaned, count=1)
    cleaned = _FENCE_CLOSE.sub("", cleaned, count=1)

    if not cleaned or not cleaned.strip():
        return ""

    return cleaned


_PARA_BREAK_RE = re.compile(r"(\n[ \t]*\n)")


def strip_trailing_explanation(text: str, original: str) -> str:
    """Some models answer correctly, then keep going: a parenthetical note
    about what they changed, a prose explanation, or even a second, relabeled
    attempt ('OUTPUT: "..."'). These always show up as extra paragraph(s) —
    separated by a blank line — that the original text didn't have. When the
    response has more blank-line-separated blocks than the original, keep
    only as many blocks as the original had; the rest is commentary, not the
    answer."""
    orig_blocks = (len(_PARA_BREAK_RE.split(original)) + 1) // 2
    parts = _PARA_BREAK_RE.split(text)
    text_blocks = (len(parts) + 1) // 2
    if text_blocks <= orig_blocks:
        return text
    return "".join(parts[: 2 * orig_blocks - 1])


def normalize_edges(text: str, original: str) -> str:
    """Strip leading/trailing whitespace the original didn't have (models
    sometimes emit a leading space or trailing newline)."""
    if not original.startswith((" ", "\t", "\n")):
        text = text.lstrip()
    if not original.endswith((" ", "\t", "\n")):
        text = text.rstrip()
    return text


def looks_like_wrapper_only(text: str, original: str) -> bool:
    """True when the model's answer is just boilerplate like "Here is the
    corrected output:" with the actual correction missing (proofreading never
    introduces a trailing colon that the original didn't have)."""
    stripped = text.rstrip()
    return bool(stripped) and stripped.endswith(":") and not original.rstrip().endswith(":")


_TRAILING_PAREN_RE = re.compile(r"\s*\([^()\n]{1,120}\)\s*$")


def strip_trailing_parenthetical(text: str, original: str) -> str:
    """Drop a trailing "(No changes needed.)"-style note the model appended
    on the same line. Only parentheticals ABSENT from the original are
    stripped — the user's own "(see attached)" endings survive."""
    orig_norm = re.sub(r"\s+", " ", original or "")
    result = text
    while True:
        m = _TRAILING_PAREN_RE.search(result)
        if not m:
            return result
        fragment = re.sub(r"\s+", " ", m.group(0)).strip()
        if fragment and fragment in orig_norm:
            return result
        result = result[: m.start()].rstrip()


_BRACKETED_URL_RE = re.compile(r"<(https?://[^>\s]+)>")


def unwrap_url_brackets(text: str, original: str) -> str:
    """Models love turning bare URLs into markdown autolinks (<https://…>).
    Unwrap them mechanically — unless the user's own text had that exact
    bracketed form."""
    def repl(m):
        if m.group(0) in (original or ""):
            return m.group(0)
        return m.group(1)

    return _BRACKETED_URL_RE.sub(repl, text)


def full_clean(response: str, original: str) -> str:
    """The complete response-cleanup chain, in the one order that works:
    wrapper phrases / fences / think-blocks, extra explanation paragraphs,
    edge whitespace (must precede quote-unwrapping — a trailing newline after
    the closing quote defeats it), appended parenthetical notes (also before
    unwrapping, for the same reason), quote unwrapping, sign-off chatter, and
    a final edge trim."""
    cleaned = clean_response(response)
    cleaned = strip_trailing_explanation(cleaned, original)
    cleaned = normalize_edges(cleaned, original)
    cleaned = strip_trailing_parenthetical(cleaned, original)
    cleaned = strip_trailing_signoff(unwrap_quotes(cleaned, original), original)
    cleaned = unwrap_url_brackets(cleaned, original)
    return normalize_edges(cleaned, original)


def unwrap_quotes(text: str, original: str) -> str:
    """Drop wrapping quotes the model added because the prompt shows the
    input inside quotes. Multiline-safe, unlike the single-line heuristic in
    clean_response: we know the original here, so only unwrap when the
    original itself wasn't quote-wrapped."""
    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in ('"', "'")
        and not (original.startswith(text[0]) and original.endswith(text[0]))
    ):
        return text[1:-1]
    return text


def strip_trailing_signoff(text: str, original: str) -> str:
    """Remove model sign-off chatter from the END of a response — but only
    phrases that are NOT part of the user's own text (emails legitimately end
    with "Let me know if…", so anything present in the original is kept)."""
    if not text:
        return text
    orig_norm = re.sub(r"\s+", " ", (original or "")).strip().lower()
    result = text
    changed = True
    while changed:
        changed = False
        for pattern in _SIGNOFF_RES:
            m = pattern.search(result)
            if not m:
                continue
            fragment = re.sub(r"\s+", " ", m.group(0)).strip().lower()
            if fragment and fragment in orig_norm:
                continue  # genuinely the user's words
            result = result[: m.start()].rstrip()
            changed = True
    return result


def validate_formatting(original: str, corrected: str) -> tuple[bool, list[str]]:
    norm_original = (original or "").replace("\r\n", "\n")
    norm_corrected = (corrected or "").replace("\r\n", "\n")

    original_lines = len(norm_original.split("\n"))
    corrected_lines = len(norm_corrected.split("\n"))
    original_newlines = norm_original.count("\n")
    corrected_newlines = norm_corrected.count("\n")

    original_bullets = len(_BULLET_RE.findall(original or ""))
    corrected_bullets = len(_BULLET_RE.findall(corrected or ""))
    original_numbered = len(_NUMBERED_RE.findall(original or ""))
    corrected_numbered = len(_NUMBERED_RE.findall(corrected or ""))

    issues = []
    if original_lines != corrected_lines:
        issues.append(f"Line count changed: {original_lines} -> {corrected_lines}")
    if original_newlines != corrected_newlines:
        issues.append(f"Newline count changed: {original_newlines} -> {corrected_newlines}")
    if original_bullets > 0 and original_bullets != corrected_bullets:
        issues.append(f"Bullet points changed: {original_bullets} -> {corrected_bullets}")
    if original_numbered > 0 and original_numbered != corrected_numbered:
        issues.append(f"Numbered list items changed: {original_numbered} -> {corrected_numbered}")

    return (len(issues) == 0, issues)


_DASH_RE = re.compile("[—–]")  # em dash U+2014, en dash U+2013


def validate_added_punctuation(original: str, corrected: str) -> tuple[bool, list[str]]:
    """Flag punctuation the model ADDED that wasn't in the original — never
    punctuation it removed. Semicolons and em/en dashes are the model
    "upgrading" wording (joining clauses, setting off an aside) rather than
    fixing an actual error — a rephrase that validate_formatting's line/bullet
    counts can't see, since it never touches line breaks."""
    orig = original or ""
    corr = corrected or ""

    orig_semicolons = orig.count(";")
    corr_semicolons = corr.count(";")

    issues = []
    if corr_semicolons > orig_semicolons:
        issues.append(f"Semicolons added: {orig_semicolons} -> {corr_semicolons}")
    if len(_DASH_RE.findall(corr)) > len(_DASH_RE.findall(orig)):
        issues.append("Em/en dashes added")
    if corr.count(":") > orig.count(":"):
        issues.append(
            f"Colons added: {orig.count(':')} -> {corr.count(':')}"
        )
    # Models like wrapping URLs in <angle brackets> (markdown autolinks).
    if corr.count("<") > orig.count("<") or corr.count(">") > orig.count(">"):
        issues.append("Angle brackets added")
    # Turning statements into questions/exclamations is restyling, not
    # proofreading ("rollback is it fails" -> "rollback? Is it failing?").
    if corr.count("?") > orig.count("?"):
        issues.append("Question marks added")
    if corr.count("!") > orig.count("!"):
        issues.append("Exclamation marks added")

    return (len(issues) == 0, issues)


_LAST_WORD_RE = re.compile(r"([A-Za-z0-9]+)\W*\Z")


def validate_no_appended_content(original: str, corrected: str) -> tuple[bool, list[str]]:
    """Catch the model finishing the user's thought ("back in 20" ->
    "back in 20 minutes.") or collapsing spaced em dashes. Both are silent
    modifications the count-based checks can't see."""
    orig = original or ""
    corr = corrected or ""
    issues = []

    orig_last = _LAST_WORD_RE.search(orig)
    corr_last = _LAST_WORD_RE.search(corr)
    if (
        orig_last
        and corr_last
        and corr_last.group(1) != orig_last.group(1)
        and len(corr.split()) > len(orig.split())
        and re.search(rf"\b{re.escape(orig_last.group(1))}\b", corr)
    ):
        issues.append(
            "Content appended after the input's final word "
            f"({orig_last.group(1)!r})"
        )

    if (
        orig.count(" — ") > corr.count(" — ")
        and len(_DASH_RE.findall(corr)) >= len(_DASH_RE.findall(orig))
    ):
        issues.append("Spacing around em dashes changed")

    return (len(issues) == 0, issues)


_TRAILING_NEWLINES_RE = re.compile(r"\n+\Z")


def enforce_formatting_preservation(
    original: str, candidate: str, query_fn: Callable[[str], str]
) -> str:
    """String-only wrapper around enforce_formatting_preservation_ex()."""
    return enforce_formatting_preservation_ex(original, candidate, query_fn)[0]


def enforce_formatting_preservation_ex(
    original: str, candidate: str, query_fn: Callable[[str], str]
) -> tuple[str, str]:
    """Return (text, path): candidate if formatting and wording survived
    (path "ok"), else retry once with the repair prompt (path "repaired"),
    else the original unchanged (path "fallback" — never lose text). The
    caller can use path "fallback" to trigger a smarter salvage.

    Models reliably drop a trailing newline at the very end of a response —
    there's no token that means "and stop right after this blank line" — so
    that alone shouldn't force a real fix back to the original. Compare
    without the original's trailing newlines, then splice them back onto
    whichever candidate wins."""
    trailing_match = _TRAILING_NEWLINES_RE.search(original)
    trailing = trailing_match.group(0) if trailing_match else ""
    core_original = original[: len(original) - len(trailing)]

    def finalize(text: str) -> str:
        return text.rstrip("\n") + trailing if trailing else text

    def check(candidate_text: str) -> tuple[bool, list[str]]:
        fmt_valid, fmt_issues = validate_formatting(core_original, candidate_text)
        punct_valid, punct_issues = validate_added_punctuation(core_original, candidate_text)
        app_valid, app_issues = validate_no_appended_content(core_original, candidate_text)
        return (
            fmt_valid and punct_valid and app_valid,
            fmt_issues + punct_issues + app_issues,
        )

    valid, issues = check(candidate.rstrip("\n"))
    if valid:
        return finalize(candidate), "ok"

    fmt_valid, _ = validate_formatting(core_original, candidate.rstrip("\n"))
    try:
        if fmt_valid:
            # Layout survived; only punctuation style broke. Re-proofread the
            # ORIGINAL with the violation named — the draft-based repair
            # prompt would entrench the bad punctuation, since the draft is
            # what contains it.
            log.warning(
                "Punctuation style changed by model (%s); "
                "retrying with named constraint", issues,
            )
            retried = full_clean(
                query_fn(build_constrained_prompt(original, issues)), original
            )
            if check(retried.rstrip("\n"))[0]:
                return finalize(retried), "repaired"
            log.warning(
                "Constrained retry still invalid (%s)",
                check(retried.rstrip("\n"))[1],
            )
            return original, "fallback"

        log.warning(
            "Formatting changed by model (%s); retrying with repair prompt",
            issues,
        )
        repaired = full_clean(
            query_fn(build_repair_prompt(original, candidate)), original
        )
        if check(repaired.rstrip("\n"))[0]:
            return finalize(repaired), "repaired"
        log.warning("Repair still invalid (%s)", check(repaired.rstrip("\n"))[1])
        return original, "fallback"
    except Exception:
        log.warning("Formatting repair failed", exc_info=True)
        return original, "fallback"


_SEGMENT_SPLIT_RE = re.compile(r"([.!?,;:][ \t]+)")


def proofread_segments(original: str, correct_fn) -> str:
    """Salvage for SINGLE-LINE text when whole-text correction keeps getting
    rejected: split on sentence/clause boundaries, correct each segment
    independently (same guards as line-by-line), and rejoin with the exact
    original separators. A rejected segment keeps its original text, so a
    restyling-happy model can still deliver the fixes it gets right."""
    parts = _SEGMENT_SPLIT_RE.split(original)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1 or not part.strip():
            out.append(part)  # separator (or whitespace) — verbatim
            continue
        corrected = correct_fn(part)
        if (
            not corrected
            or "\n" in corrected
            or not corrected.strip()
            or not validate_added_punctuation(part, corrected)[0]
            or not validate_no_appended_content(part, corrected)[0]
        ):
            out.append(part)
            continue
        # Mid-sentence fragments: don't let the model capitalize the start
        # or bolt a period onto a clause that had none.
        if part[0].islower() and corrected[:1].isupper():
            corrected = corrected[:1].lower() + corrected[1:]
        if corrected[-1:] in ".,;" and part[-1:] not in ".,;":
            corrected = corrected.rstrip(".,;")
        out.append(corrected)
    return "".join(out)


_LINE_PREFIX_RE = re.compile(r"^[ \t]*(?:(?:[•\-*]|\d+[.)])[ \t]+)?")


def proofread_line_by_line(original: str, correct_line_fn) -> str:
    """Deterministic formatting preservation: proofread each line separately
    and reassemble on the original line grid. Blank lines survive verbatim;
    each line's indentation and bullet/number marker is re-applied from the
    original; a per-line result that is empty, multiline, or adds punctuation
    style falls back to that original line. Newlines never enter the model,
    so the structure cannot be mangled — this is the salvage path for when
    whole-text correction keeps breaking the layout.

    correct_line_fn(line) -> corrected line, or None on failure."""
    out = []
    for line in original.split("\n"):
        if not line.strip():
            out.append(line)
            continue
        corrected = correct_line_fn(line)
        if (
            not corrected
            or "\n" in corrected
            or not corrected.strip()
            or not validate_added_punctuation(line, corrected)[0]
            or not validate_no_appended_content(line, corrected)[0]
        ):
            out.append(line)
            continue
        prefix_match = _LINE_PREFIX_RE.match(line)
        prefix = prefix_match.group(0) if prefix_match else ""
        if prefix and not corrected.startswith(prefix):
            corrected = prefix + _LINE_PREFIX_RE.sub("", corrected, count=1)
        # Restore a trailing colon the model dropped ("Steps:" -> "Steps").
        if line.rstrip().endswith(":") and not corrected.rstrip().endswith(":"):
            corrected = corrected.rstrip().rstrip(".") + ":"
        out.append(corrected)
    return "\n".join(out)
