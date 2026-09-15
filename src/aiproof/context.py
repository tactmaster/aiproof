"""Context profile mined from the opt-in proofread history.

Two things make a proofreader feel like it knows you:

1. VOCABULARY — the domain words you legitimately use (product names like
   Mender, tools like ydotool, hostnames, ALLCAPS acronyms) that a model will
   otherwise "correct" into something else. We only trust a word once the
   model itself has left it alone in at least two separate proofreads, one of
   which it *did* change something in — surviving an edit is the endorsement.
2. DOMAIN SUMMARY — one LLM-written sentence about the kind of text you write,
   regenerated rarely (every 25 new entries).

Both live in a small cache next to the history (~/.local/share/aiproof/
context.json) and are injected into the proofreading prompt by the caller.
Everything here is best-effort: the history may be absent, empty, or
hand-edited, and none of that may break a proofread.
"""

import json
import logging
import os
import re
import tempfile
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import history
from .llm.postprocess import clean_response

log = logging.getLogger(__name__)

CACHE_VERSION = 1

# A word is at most 24 chars (longer is a URL fragment or a mangled paste) and
# starts with a letter, so numbers and punctuation runs never qualify.
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,}")
_MAX_TOKEN_LEN = 24
_TRAILING_PUNCT = ".-_"

_HISTORY_WINDOW = 200   # entries mined per refresh
_MAX_WORDS = 12
_MAX_WORDS_CHARS = 120  # joined with ", " — this all goes into every prompt

_SUMMARY_MIN_ENTRIES = 5    # don't guess a domain from one or two samples
_SUMMARY_EVERY = 25         # new entries before the summary is regenerated
_SUMMARY_SAMPLES = 20
_SUMMARY_SAMPLE_CHARS = 200
_SUMMARY_MAX_CHARS = 160

_SIMPLE_CAPITALIZED_RE = re.compile(r"^[A-Z][a-z]+$")
_WORDLIST_PATH = "/usr/share/dict/words"

_SUMMARY_INSTRUCTION = (
    "In one sentence (maximum 20 words), describe the kind of text this user "
    "typically writes (domain, register, formats). Output ONLY the sentence."
)


def cache_path() -> Path:
    # Resolved at call time, never at import: tests (and a changed
    # XDG_DATA_HOME) must be able to move the data directory underneath us.
    return history.history_dir() / "context.json"


# -- tokenizing ---------------------------------------------------------------


def _tokens(text: str) -> list[tuple[str, int]]:
    """(token, start offset) pairs. Trailing .-_ is punctuation, not part of
    the word ("Mender." -> "Mender"), but an internal one is ("ai-proof")."""
    out = []
    for match in TOKEN_RE.finditer(text or ""):
        token = match.group(0).rstrip(_TRAILING_PUNCT)
        if token and len(token) <= _MAX_TOKEN_LEN:
            out.append((token, match.start()))
    return out


def _is_midsentence(text: str, start: int) -> bool:
    """True when the character before the token (skipping whitespace) exists
    and does not end a sentence — i.e. a capital here is the writer's choice,
    not English grammar."""
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return i >= 0 and text[i] not in ".!?:"


# -- dictionary ---------------------------------------------------------------


_wordlist_memo: "list | None" = None  # [frozenset | None] once loaded


def _load_wordlist() -> "frozenset[str] | None":
    """/usr/share/dict/words as a set, or None when the system has no word
    list installed. Possessives ("Ed's") carry no signal, so they're skipped.
    Memoized for the process lifetime (the daemon refreshes after every
    proofread — reparsing 104k lines each time would be pure waste);
    monkeypatched wholesale in the tests."""
    global _wordlist_memo
    if _wordlist_memo is not None:
        return _wordlist_memo[0]
    try:
        with open(_WORDLIST_PATH, encoding="utf-8", errors="ignore") as f:
            result = frozenset(
                word for word in (line.strip() for line in f)
                if word and "'" not in word
            )
    except OSError:
        log.debug("no system word list at %s", _WORDLIST_PATH)
        result = None
    _wordlist_memo = [result]
    return result


def _near_variants(word: str):
    """Edit-distance-1 DELETIONS and adjacent TRANSPOSITIONS only. Insertions
    and substitutions would match far too much ("Mender" -> "mender") — these
    two catch the shape of a persistent typo ("recieve" -> "receive")."""
    for i in range(len(word)):
        yield word[:i] + word[i + 1:]
    for i in range(len(word) - 1):
        yield word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _has_shape_signal(word: str) -> bool:
    """CamelCase / digits / internal ._- / ALLCAPS — shapes no English
    dictionary word has, so they are domain terms whatever the word list
    says."""
    if any(char.isdigit() for char in word):
        return True
    if any(char in "._-" for char in word):
        return True
    if len(word) >= 2 and word.isupper():
        return True
    # CamelCase: an interior capital with lowercase elsewhere.
    return any(char.islower() for char in word) and any(
        char.isupper() for char in word[1:]
    )


# URL plumbing that tokenizes into "words" but tells the model nothing.
_STOP_TOKENS = frozenset({"http", "https", "ftp", "www"})


def _keep_word(word: str, wordlist, midsentence: int) -> bool:
    """Decide whether a consensus token is domain vocabulary worth pinning."""
    if word.lower() in _STOP_TOKENS:
        return False
    intentional_proper_noun = (
        bool(_SIMPLE_CAPITALIZED_RE.match(word)) and midsentence >= 2
    )

    if wordlist is None:
        # No dictionary: trust only shape, plus capitals the writer uses where
        # English wouldn't. Plain lowercase words are indistinguishable from
        # ordinary English here, so they all go.
        return _has_shape_signal(word) or intentional_proper_noun

    if word in wordlist:
        return False  # plain English, or a known proper noun like "Debian"
    if _has_shape_signal(word):
        return True
    if word.islower():
        # A lowercase word the dictionary doesn't know is either jargon
        # ("ydotool") or a typo the model keeps failing to fix ("recieve").
        return not any(variant in wordlist for variant in _near_variants(word))
    if _SIMPLE_CAPITALIZED_RE.match(word) and word.lower() in wordlist:
        # "Mender" vs "Thanks": only the one capitalized mid-sentence is a name.
        return intentional_proper_noun
    return True


# -- vocabulary ---------------------------------------------------------------


def _usable(entry) -> bool:
    """Fallback entries had their correction rejected wholesale — "corrected"
    is the untouched original, so every token "survives" and the entry carries
    zero signal."""
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("original"), str)
        and isinstance(entry.get("corrected"), str)
        and entry.get("pipeline_path") != "fallback"
    )


def _build_words(entries: list[dict]) -> list[str]:
    """Consensus vocabulary: tokens the model left alone in >= 2 entries, at
    least one of which it otherwise changed, minus anything it was corrected
    away MORE often than it survived (a single model mangling of a good term
    — "ydotool" -> "Yocto tool" — must not veto its endorsements, but a typo
    that is usually fixed must never qualify), minus anything the dictionary
    explains."""
    entry_count: Counter = Counter()      # casefolded token -> distinct entries
    endorsed: set[str] = set()            # seen surviving a changed entry
    corrected_away: Counter = Counter()   # casefolded -> mangled-away entries
    spelling_fixed: set[str] = set()      # replaced by an edit-1 sibling
    surface: dict[str, Counter] = {}      # casefolded -> forms in corrected text
    last_entry: dict[str, int] = {}       # casefolded token -> entry index
    last_form: dict[str, int] = {}        # surface form -> entry index
    midsentence: Counter = Counter()      # surface form -> mid-sentence uses

    for index, entry in enumerate(entries):
        original = entry["original"]
        corrected = entry["corrected"]
        changed = bool(entry.get("changed"))

        original_forms = _tokens(original)
        corrected_forms = _tokens(corrected)
        original_keys = {token.casefold() for token, _ in original_forms}
        corrected_keys = {token.casefold() for token, _ in corrected_forms}

        for key in original_keys & corrected_keys:
            entry_count[key] += 1
            last_entry[key] = index
            if changed:
                endorsed.add(key)
        if changed:
            # Casefold comparison on purpose: "mender" -> "Mender" is a case
            # fix, which endorses the word rather than condemning it.
            for key in original_keys - corrected_keys:
                # What replaced the token tells us why it vanished: an
                # edit-distance-1 sibling in the corrected text means the
                # model FIXED a spelling ("ydotoool" -> "ydotool") — hard
                # veto. Unrelated words mean the model mangled a good term
                # ("ydotool" -> "Yocto tool") — count it, don't veto.
                variants = set(_near_variants(key))
                if any(
                    corr in variants or key in set(_near_variants(corr))
                    for corr in corrected_keys
                ):
                    spelling_fixed.add(key)
                else:
                    corrected_away[key] += 1

        for token, start in corrected_forms:
            key = token.casefold()
            surface.setdefault(key, Counter())[token] += 1
            last_form[token] = index
            if _is_midsentence(corrected, start):
                midsentence[token] += 1

    wordlist = _load_wordlist()
    ranked = []
    for key, count in entry_count.items():
        if count < 2 or key not in endorsed:
            continue
        if key in spelling_fixed or corrected_away[key] > count:
            continue
        forms = surface.get(key)
        if not forms:
            continue
        # Most common form in the corrected texts; ties go to the most recent.
        word = max(forms, key=lambda form: (forms[form], last_form.get(form, -1)))
        # A Capitalized word that ALSO shows up lowercase in corrected texts
        # is a common word wearing a capital (chat glue, HTTP headers) — not
        # a name. Real names ("Mender") never survive in lowercase.
        lowercase_sibling = word.lower() != word and forms.get(word.lower(), 0) > 0
        if lowercase_sibling or not _keep_word(word, wordlist, midsentence[word]):
            continue
        ranked.append((count, last_entry.get(key, -1), word))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    words = [word for _, _, word in ranked[:_MAX_WORDS]]
    # The list is pasted into every prompt, so keep it short enough to stay
    # cheap: drop the weakest evidence until it fits.
    while words and len(", ".join(words)) > _MAX_WORDS_CHARS:
        words.pop()
    return words


# -- LLM domain summary -------------------------------------------------------


def _summary_due(summary, entry_count: int, summary_entry_count: int) -> bool:
    if not summary:
        return entry_count >= _SUMMARY_MIN_ENTRIES
    return entry_count - summary_entry_count >= _SUMMARY_EVERY


def _generate_summary(cfg: dict, entries: list[dict]) -> "str | None":
    """One sentence describing what the user writes, or None if the model
    gave us nothing usable (the caller then keeps the previous summary and
    retries on the next refresh).

    Imported lazily on purpose: llm.client imports this module, so a top-level
    import would be circular."""
    from .llm.client import client_from_config

    lines = []
    for entry in entries[-_SUMMARY_SAMPLES:]:
        # Collapse whitespace first — a multi-line sample would otherwise
        # break the one-per-line list the prompt promises.
        sample = " ".join(entry["original"].split())[:_SUMMARY_SAMPLE_CHARS]
        if sample:
            lines.append("- " + sample)
    if not lines:
        return None

    prompt = "\n".join(lines) + "\n\n" + _SUMMARY_INSTRUCTION
    response = clean_response(client_from_config(cfg).query(prompt) or "")
    for line in (response or "").splitlines():
        line = " ".join(line.strip().strip("\"'“”").split())
        if line:
            return line[:_SUMMARY_MAX_CHARS]
    return None


# -- cache --------------------------------------------------------------------


def _history_mtime() -> float:
    try:
        return os.path.getmtime(history.history_path())
    except OSError:
        return 0.0


def _read_cache() -> dict:
    try:
        with open(cache_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(data: dict) -> None:
    """Atomic 0600 write, same pattern as config.save()."""
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".context-", suffix=".json")
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# -- public API ---------------------------------------------------------------


def refresh(cfg: dict, allow_summary: bool = False) -> None:
    """Re-mine the history and rewrite the cache. The vocabulary is always
    rebuilt; the summary costs an LLM call, so it only happens when
    allow_summary is set AND enough new entries have accumulated."""
    entries = [e for e in history.read_recent(_HISTORY_WINDOW) if _usable(e)]
    cached = _read_cache()
    summary = cached.get("summary") or None
    summary_entry_count = cached.get("summary_entry_count")
    if not isinstance(summary_entry_count, int):
        summary_entry_count = 0

    words = _build_words(entries)

    if allow_summary and _summary_due(summary, len(entries), summary_entry_count):
        try:
            generated = _generate_summary(cfg, entries)
            if generated:
                summary = generated
                summary_entry_count = len(entries)
            # An empty or failed answer keeps the old counts, so the next
            # refresh tries again rather than waiting another 25 entries.
        except Exception:
            log.warning("could not generate context summary", exc_info=True)

    _write_cache({
        "version": CACHE_VERSION,
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "history_mtime": _history_mtime(),
        "entry_count": len(entries),
        "words": words,
        "summary": summary,
        "summary_entry_count": summary_entry_count,
    })


def refresh_if_stale(cfg: dict, allow_summary: bool = False,
                     min_new_entries: int = 1) -> None:
    """refresh() unless the history hasn't changed since the cache was built.
    Never raises — a context profile is a nicety, never a reason to fail a
    proofread.

    The mtime check governs everything, including the summary: if a summary
    is due but no new proofread has been recorded, it waits for the next real
    refresh. That keeps the fast path to a single stat() call and costs
    nothing in practice — a due summary means entries have been arriving.

    min_new_entries > 1 additionally skips the refresh until the history has
    grown by that many entries since the cache was built — the CLI uses 5 so
    its synchronous post-output refresh doesn't reload the dictionary on
    every single proofread (an aggregate profile loses nothing to the lag)."""
    try:
        if not history.history_path().exists():
            return  # nothing recorded yet
        cached = _read_cache()
        if cached and cached.get("history_mtime") == _history_mtime():
            return
        if cached and min_new_entries > 1:
            grown = _history_entry_count() - int(cached.get("entry_count", 0))
            if 0 <= grown < min_new_entries:
                return
        refresh(cfg, allow_summary=allow_summary)
    except Exception:
        log.warning("could not refresh context profile", exc_info=True)


def _history_entry_count() -> int:
    """Cheap line count — no JSON parsing, no dictionary load."""
    try:
        with open(history.history_path(), "rb") as f:
            return sum(chunk.count(b"\n") for chunk in iter(
                lambda: f.read(1 << 20), b""))
    except OSError:
        return 0


def schedule_refresh(cfg: dict) -> None:
    """Kick off a background refresh (summary allowed). Gated on both flags:
    without save_history there is nothing to mine, and without context_aware
    nothing would read the result."""
    if not (cfg.get("context_aware") and cfg.get("save_history")):
        return
    threading.Thread(
        target=lambda: refresh_if_stale(cfg, allow_summary=True),
        daemon=True,
    ).start()


def load_words_and_summary(cfg: dict) -> "tuple[list[str], str | None]":
    """Cached vocabulary and domain summary for prompt injection. Touches the
    filesystem only when the feature is on."""
    if not cfg.get("context_aware"):
        return [], None
    data = _read_cache()
    raw_words = data.get("words")
    words = []
    if isinstance(raw_words, list):
        for word in raw_words:
            # The cache is a plain file a user may well have edited; re-check
            # every word rather than trust it into a prompt.
            if (
                isinstance(word, str)
                and len(word) <= _MAX_TOKEN_LEN
                and TOKEN_RE.fullmatch(word)
            ):
                words.append(word)
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = None
    return words[:_MAX_WORDS], summary


def clear() -> bool:
    """Delete the context cache. Returns True if there was one."""
    try:
        cache_path().unlink()
        return True
    except FileNotFoundError:
        return False
