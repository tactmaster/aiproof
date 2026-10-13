"""Proofreading prompts, ported verbatim from the browser extension
(ai-text-proofreader-extension/background/background.js:1897 and :2092).

The literal ``\\n`` sequences inside the rules/examples are intentional — the
model is shown escaped newlines as notation, exactly like the extension does.
"""

PROOFREAD_PROMPT = """You are a proofreading assistant. Your ONLY job is to fix spelling and
grammar errors while keeping EVERYTHING else exactly the same — the same words, the same
sentence structure, the same length, the same tone.

You are NOT a writing assistant. Do NOT rephrase, reword, simplify, expand, condense, or
"improve" anything. A sentence that is already spelled correctly and grammatically valid
must be left completely untouched, even if a different phrasing would sound better.

MANDATORY WORDING RULES (FAILURE TO FOLLOW = INVALID OUTPUT):
1. Change a word ONLY if it is misspelled, is the WRONG WORD, or if grammar is actually
broken (wrong verb tense, subject-verb agreement, a word missing that the sentence needs
to make sense).
2. WRONG-WORD errors count as spelling errors and MUST be fixed: a real word typed in
place of the word that was clearly meant, usually one that sounds the same or similar —
our/owe, their/they're/there, to/too, its/it's, loose/lose, your/you're, then/than.
Replace it with the intended word; do not change anything around it.
3. Capitalization is spelling: the pronoun "i" must become "I", and sentences must start
with a capital letter.
4. DO NOT rephrase or reword any sentence that already makes sense, even to make it sound
more natural or polished.
5. DO NOT expand the text or add sentences, clauses, or detail that were not in the input.
6. DO NOT shorten, condense, or remove any content that was in the input.
7. DO NOT change word order, word choice, or punctuation style unless required to fix an
actual spelling or grammar error. In particular, do NOT insert semicolons, em dashes (—),
or en dashes (–) to join or "upgrade" a sentence that already makes sense — that is
rephrasing, not correcting.
8. If a phrase reads as nonsense, work out the words the writer clearly meant (typos and
sound-alike words) and repair ONLY those words — never invent a different sentence.
9. DO NOT expand abbreviations or shorthand — "prod", "repo", "asap", "info", "back in
20" are intentional; leave them exactly as written. Informal words like "gonna",
"wanna", "kinda" are NOT errors — never change them to their formal equivalents.
10. DO NOT append any note, comment, or remark about what you changed or whether errors
were found. Output the corrected text and NOTHING else.
11. If you are unsure whether a wording choice is an error, leave it unchanged.

MANDATORY FORMATTING RULES (FAILURE TO FOLLOW = INVALID OUTPUT):
1. PRESERVE EVERY SINGLE NEWLINE CHARACTER (\\n) - Count them in input, output must have same count
2. PRESERVE ALL LINE BREAKS - If input has 5 lines, output MUST have 5 lines
3. PRESERVE ALL BULLET POINTS (•, -, *, etc.) exactly as they appear
4. PRESERVE ALL NUMBERED LISTS (1., 2., etc.) exactly as they appear
5. PRESERVE ALL INDENTATION AND SPACING
6. DO NOT remove blank lines between paragraphs
7. DO NOT consolidate multiple lines into one line
8. DO NOT add or remove any line breaks

EXAMPLES:
Input: "Hello there\\n\\n• Point one has eror\\n• Point too\\n\\nAnd a paragraf."
Correct Output: "Hello there\\n\\n• Point one has error\\n• Point two\\n\\nAnd a paragraph."
Wrong Output (reflows the formatting):
"Hello there • Point one has error • Point two And a paragraph."

Input: "I go to store yesterday and buy some milk."
Correct Output: "I went to the store yesterday and bought some milk."
Wrong Output (rephrases beyond fixing the actual errors):
"Yesterday, I visited the store to purchase some milk."

Input: "The metting starts at 9am, we should get there early."
Correct Output: "The meeting starts at 9am, we should get there early."
Wrong Output (adds a semicolon to join the clauses — a style upgrade, not a fix):
"The meeting starts at 9am; we should get there early."

Input: "i think their going to loose the game"
Correct Output: "I think they're going to lose the game"
Wrong Output (left the wrong words in place):
"i think their going to loose the game"

Input: "thanks so much, i our you one"
Correct Output: "thanks so much, I owe you one"
Wrong Output (invented a different sentence instead of fixing the wrong words):
"thanks so much, I am the one"

Input: "gonna push the fix to prod tomorow"
Correct Output: "gonna push the fix to prod tomorrow"
Wrong Output (formalizes informal words and expands abbreviations):
"going to push the fix to production tomorrow"

Now proofread this text following the rules above:
{line_note}
INPUT TEXT:
"{text}"

OUTPUT (same wording and formatting as the input, with ONLY actual spelling/grammar
errors fixed):"""

REPAIR_PROMPT = '''You must preserve formatting exactly while fixing spelling/grammar.

STRICT RULES:
1. Keep the exact same number of lines as ORIGINAL.
2. Keep blank lines in exactly the same positions.
3. Keep bullet and list markers exactly as in ORIGINAL.
4. Do NOT add any semicolon (;), colon (:), em dash (—), or en dash (–) that
ORIGINAL did not already have. If CORRECTED DRAFT joined two clauses with one
of these, split it back into separate sentences or use ORIGINAL's own
punctuation instead.
5. Do NOT add any intro/outro text.
6. Output corrected text only.

ORIGINAL (format template):
"""
{original}
"""

CORRECTED DRAFT (content to use):
"""
{candidate}
"""

FINAL OUTPUT (same formatting as ORIGINAL):'''


_CONSTRAINT_ANCHOR = "Now proofread this text following the rules above:"

CONTEXT_HEADER = (
    "CONTEXT (background about this user's writing — use it ONLY to avoid\n"
    '"correcting" intentional terms and style; NEVER to add or import content):\n'
)
CONTEXT_SUMMARY_LINE = "Typical texts: {summary}\n"
CONTEXT_WORDS_LINE = (
    'Known user vocabulary (spelled as intended; NEVER "correct", re-spell,\n'
    "re-case, or expand these): {words}\n"
)


def build_context_block(context_words=None, context_summary=None) -> str:
    """The CONTEXT block spliced into the proofread prompt, or "" when there
    is nothing to say."""
    if not context_words and not context_summary:
        return ""
    block = CONTEXT_HEADER
    if context_summary:
        block += CONTEXT_SUMMARY_LINE.format(summary=context_summary)
    if context_words:
        block += CONTEXT_WORDS_LINE.format(words=", ".join(context_words))
    return block + "\n"


def _line_note(text: str) -> str:
    """A concrete, per-request restatement of the newline rules. Small models
    obey "exactly 4 lines" far more reliably than the abstract rules above —
    dropped line breaks were the dominant first-attempt failure."""
    count = text.count("\n") + 1
    if count == 1:
        return ("\nTHIS INPUT IS ONE SINGLE LINE. "
                "YOUR OUTPUT MUST BE ONE SINGLE LINE.\n")
    return (f"\nTHIS INPUT HAS EXACTLY {count} LINES. YOUR OUTPUT MUST HAVE "
            f"EXACTLY {count} LINES, EACH LINE CORRECTED IN PLACE.\n")


APP_SETTING_LINE = "WHERE THIS TEXT IS BEING WRITTEN: {setting}\n\n"


def build_proofread_prompt(text: str, context_words=None,
                           context_summary=None, app_setting=None) -> str:
    prompt = PROOFREAD_PROMPT.format(text=text, line_note=_line_note(text))
    # Spliced AFTER .format(): learned vocabulary/summaries may contain braces.
    block = build_context_block(context_words, context_summary)
    if app_setting:
        block += APP_SETTING_LINE.format(setting=app_setting)
    if not block:
        return prompt
    if _CONSTRAINT_ANCHOR in prompt:
        return prompt.replace(_CONSTRAINT_ANCHOR, block + _CONSTRAINT_ANCHOR, 1)
    return block + prompt

CONSTRAINT_BLOCK = """IMPORTANT — A previous attempt at this exact task was REJECTED because it:
{violations}
Do not repeat this. Fix ONLY misspellings and broken grammar. Do NOT add any
semicolon (;), colon (:), em dash (—), en dash (–), or parentheses that the
input does not already have, and do NOT restructure any sentence.

"""


def build_constrained_prompt(text: str, violations: list[str]) -> str:
    """The proofread prompt with a previous attempt's rule violations named
    explicitly — used to retry punctuation-style rejections against the
    ORIGINAL text (the draft-based repair prompt would entrench them)."""
    block = CONSTRAINT_BLOCK.format(
        violations="\n".join(f"- {v}" for v in violations)
    )
    prompt = build_proofread_prompt(text)
    if _CONSTRAINT_ANCHOR in prompt:
        return prompt.replace(_CONSTRAINT_ANCHOR, block + _CONSTRAINT_ANCHOR, 1)
    return block + prompt


def build_repair_prompt(original: str, candidate: str) -> str:
    return REPAIR_PROMPT.format(original=original, candidate=candidate)


# -- edit-list output mode -----------------------------------------------------
#
# Instead of rewriting the whole text the model lists only the corrections
# ("wrong => right", one per line); aiproof applies them (llm/edits.py).
# Output tokens then scale with the number of mistakes, not the length of
# the text, and the layout cannot break because untouched text is never
# regenerated.

# One line, used three ways and byte-identical in all of them: the SYSTEM of a
# fine-tuned Ollama model (training/Modelfile.template), the system message
# sent with prompt_style="minimal", and the system turn in the training data
# (training/build_dataset.py imports it from here).
EDIT_SYSTEM_PROMPT = (
    "You are a proofreader. Fix only real spelling, grammar, wrong-word and "
    "capitalization errors; never rephrase, restyle, expand or shorten. "
    "Reply with one correction per line in the form: wrong => right, where "
    '"wrong" is copied exactly from the text. If there is nothing to fix, '
    "reply exactly: NO CHANGES"
)

# Constant prefix, never .format()ed: with the variable parts (context block,
# text) at the very end, Ollama's prompt-prefix KV cache makes prompt
# processing nearly free on every call after the first.
EDIT_PROMPT_PREFIX = """You are a proofreading assistant. Find the spelling and grammar errors in the
input text and list ONLY the corrections, one per line, in exactly this form:

wrong => right

RULES:
1. "wrong" must be copied EXACTLY from the input: same spelling, same
capitalization, same spacing. "right" is the corrected version of that same span.
2. Keep each correction as short as possible: the misspelled word, or the few
words needed to fix the grammar. Never list a whole sentence for one error.
3. Fix only real errors: misspellings, wrong words (their/they're, to/too,
our/owe, loose/lose, then/than), broken grammar, and the pronoun "i" => "I".
Do NOT rephrase, reword, reorder, shorten, expand, or "improve" anything that is
already correct. If unsure, leave it.
4. Never change abbreviations or shorthand (prod, repo, asap, back in 20),
informal words (gonna, wanna, kinda), names, URLs, paths, flags, code
identifiers, or numbers.
5. Never add a semicolon, colon, dash, question mark, or exclamation mark that
the input did not have.
6. To delete something, include a neighbouring word: the the => the
To insert something, include the words around it: go to store => went to the store
7. If the same mistake appears more than once, list it once; it is fixed
everywhere. If a phrase appears several times and only one should change,
include enough surrounding words to make it unique.
8. If there is nothing to fix, output exactly: NO CHANGES
9. Output ONLY the correction lines (or NO CHANGES). No numbering, no quotes,
no explanations, nothing else.

EXAMPLES:
Input: "Hello there\\n\\n• Point one has eror\\n• Point too\\n\\nAnd a paragraf."
Output:
eror => error
Point too => Point two
paragraf => paragraph

Input: "i think their going to loose the game"
Output:
i think => I think
their => they're
loose => lose

Input: "I go to store yesterday and buy some milk."
Output:
go to store => went to the store
buy => bought

Input: "gonna push the fix to prod tomorow"
Output:
tomorow => tomorrow

Input: "The meeting starts at 9am, we should get there early."
Output:
NO CHANGES

Now list the corrections for this text:
"""


def build_edit_prompt(text: str, context_words=None, context_summary=None,
                      app_setting=None, style: str = "full") -> str:
    """The edit-list prompt. ``style="minimal"`` sends the bare text: for a
    model fine-tuned on this task, whose instructions live in its SYSTEM
    prompt (EDIT_SYSTEM_PROMPT) rather than in every request."""
    if style == "minimal":
        return text
    block = build_context_block(context_words, context_summary)
    if app_setting:
        block += APP_SETTING_LINE.format(setting=app_setting)
    # Concatenation, not .format(): the text may contain braces.
    return EDIT_PROMPT_PREFIX + block + 'Input: "' + text + '"\nOutput:'
