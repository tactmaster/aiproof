import pytest

from aiproof.llm.postprocess import (
    full_clean,
    strip_trailing_parenthetical,
    clean_response,
    enforce_formatting_preservation,
    enforce_formatting_preservation_ex,
    looks_like_wrapper_only,
    normalize_edges,
    proofread_line_by_line,
    proofread_segments,
    strip_trailing_explanation,
    strip_trailing_signoff,
    unwrap_quotes,
    validate_added_punctuation,
    validate_no_appended_content,
    validate_formatting,
)


class TestCleanResponse:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Here is the corrected text: Hello world", "Hello world"),
            ("Here is the corrected text:\nHello world", "Hello world"),
            ("Corrected text: Hello", "Hello"),
            ("Sure, here is the corrected text: Hello", "Hello"),
            ("Output: Hello", "Hello"),
            ("Result: Hello", "Hello"),
            ("Here's the proofread text: Hello", "Hello"),
            ("Here is your corrected email: Hi Bob", "Hi Bob"),
        ],
    )
    def test_strips_wrapper_phrases(self, raw, expected):
        assert clean_response(raw) == expected

    def test_unwraps_quotes_single_line(self):
        assert clean_response('"Hello world"') == "Hello world"
        assert clean_response("'Hello world'") == "Hello world"

    def test_keeps_quotes_multiline(self):
        text = '"Hello\nworld"'
        assert clean_response(text) == text

    def test_strips_code_fences(self):
        assert clean_response("```\nHello world\n```") == "Hello world"
        assert clean_response("```text\nHello world\n```") == "Hello world"

    def test_normalizes_crlf(self):
        assert clean_response("a\r\nb") == "a\nb"

    def test_empty_becomes_empty_string(self):
        assert clean_response("") == ""
        assert clean_response("   \n  ") == ""
        assert clean_response("Here is the corrected text:") == ""

    def test_plain_text_untouched(self):
        text = "Nothing to see here.\n\n• bullet one\n• bullet two"
        assert clean_response(text) == text

    def test_strips_think_blocks(self):
        raw = "<think>\nThe user wants me to fix…\n</think>\nFixed text."
        assert clean_response(raw) == "Fixed text."

    def test_strips_unclosed_think_block(self):
        assert clean_response("<think>endless pondering") == ""


class TestStripTrailingSignoff:
    ORIGINAL = "Ths is my text."

    @pytest.mark.parametrize(
        "signoff",
        [
            "Is there anything else I can help you with?",
            "Anything else I can help with?",
            "Is there anything else you'd like me to do?",
            "I hope this helps!",
            "Hope this helps.",
            "Let me know if you need any further assistance.",
            "Feel free to ask if you have more questions.",
            "Don't hesitate to reach out!",
            "If you have any other questions, just ask.",
            "Would you like me to review anything else?",
            "Happy to help with anything else!",
        ],
    )
    def test_strips_appended_signoff_same_line(self, signoff):
        corrected = f"This is my text. {signoff}"
        assert (
            strip_trailing_signoff(corrected, self.ORIGINAL)
            == "This is my text."
        )

    def test_strips_signoff_on_own_line(self):
        corrected = "This is my text.\n\nIs there anything else I can help you with?"
        assert (
            strip_trailing_signoff(corrected, self.ORIGINAL)
            == "This is my text."
        )

    def test_strips_stacked_signoffs(self):
        corrected = (
            "This is my text. Hope this helps! "
            "Let me know if you need anything else."
        )
        assert (
            strip_trailing_signoff(corrected, self.ORIGINAL)
            == "This is my text."
        )

    def test_keeps_signoff_that_is_users_own_text(self):
        # Emails legitimately end like this — never strip the user's words.
        original = "Thanks for the reprot. Let me know if you need anything else."
        corrected = "Thanks for the report. Let me know if you need anything else."
        assert strip_trailing_signoff(corrected, original) == corrected

    def test_signoff_mid_text_untouched(self):
        corrected = "Hope this helps was what she said. The rest is fine."
        assert strip_trailing_signoff(corrected, "x") == corrected

    def test_plain_text_untouched(self):
        corrected = "Nothing chatty here.\n• bullet"
        assert strip_trailing_signoff(corrected, "x") == corrected


class TestStripTrailingExplanation:
    def test_strips_prose_explanation_after_quoted_answer(self):
        original = "This is the fixd text."
        corrected = (
            '"This is the fixed text."\n\n'
            "I corrected the spelling and added a missing period."
        )
        assert strip_trailing_explanation(corrected, original) == (
            '"This is the fixed text."'
        )

    def test_strips_relabeled_second_attempt(self):
        original = "First attempt at the fix"
        corrected = (
            '"First attempt at the fix."\n\n'
            'OUTPUT (corrected spelling/grammar errors): "Second attempt at the fix."'
        )
        assert strip_trailing_explanation(corrected, original) == (
            '"First attempt at the fix."'
        )

    def test_strips_unquoted_parenthetical_note(self):
        original = "Great progress.\n\nAre you using rpiboot and the usb gadet?"
        corrected = (
            "Great progress.\n\nAre you using RPi-Boot and the USB gadget?\n\n"
            '(Note: corrected "rpiboot" and "gadet".)'
        )
        assert strip_trailing_explanation(corrected, original) == (
            "Great progress.\n\nAre you using RPi-Boot and the USB gadget?"
        )

    def test_single_answer_untouched(self):
        assert strip_trailing_explanation("Just the answer.", "Just the answr.") == (
            "Just the answer."
        )

    def test_multi_paragraph_answer_matching_original_untouched(self):
        original = "Para one.\n\nPara two."
        text = "Para one fixed.\n\nPara two fixed."
        assert strip_trailing_explanation(text, original) == text

    def test_empty_untouched(self):
        assert strip_trailing_explanation("", "") == ""


class TestWrapperVariants:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Here is the corrected output: Hello", "Hello"),
            ("Here's the corrected version: Hello", "Hello"),
            ("Sure, here is your corrected sentence: Hello", "Hello"),
            ("Here is the revised paragraph:\nHello", "Hello"),
        ],
    )
    def test_strips_wrapper_variants(self, raw, expected):
        assert clean_response(raw) == expected


class TestNormalizeEdges:
    def test_strips_edges_original_had_none(self):
        assert normalize_edges(" fixed text\n", "orig text") == "fixed text"

    def test_keeps_edges_original_had(self):
        assert normalize_edges("fixed\n", "orig\n") == "fixed\n"


class TestLooksLikeWrapperOnly:
    def test_detects_boilerplate_colon(self):
        assert looks_like_wrapper_only("Here is the corrected output:", "some text")

    def test_ok_when_original_ends_with_colon(self):
        assert not looks_like_wrapper_only("A corrected list:", "A corected list:")

    def test_normal_text_ok(self):
        assert not looks_like_wrapper_only("Fixed sentence.", "Fixd sentence.")


class TestUnwrapQuotes:
    def test_multiline_quoted_response_unwrapped(self):
        corrected = '"Line one fixed.\nLine two fixed."'
        original = "Line one fixd.\nLine two fixd."
        assert unwrap_quotes(corrected, original) == (
            "Line one fixed.\nLine two fixed."
        )

    def test_kept_when_original_is_quoted(self):
        original = '"a quoted sentnce"'
        corrected = '"a quoted sentence"'
        assert unwrap_quotes(corrected, original) == corrected

    def test_mismatched_ends_untouched(self):
        assert unwrap_quotes('"partial', "x") == '"partial'

    def test_unquoted_untouched(self):
        assert unwrap_quotes("plain text", "x") == "plain text"

    def test_trailing_newline_then_unwrap(self):
        # Pipeline order regression: normalize_edges must run before
        # unwrap_quotes or the newline after the closing quote blocks it.
        raw = '"Dear team,\nBest regards,\nEd"\n'
        original = "Deer team,\nBest regads,\nEd"
        normalized = normalize_edges(raw, original)
        assert unwrap_quotes(normalized, original) == (
            "Dear team,\nBest regards,\nEd"
        )


class TestValidateFormatting:
    def test_identical_valid(self):
        text = "line one\n\n• a\n• b\n\n1. first\n2. second"
        valid, issues = validate_formatting(text, text)
        assert valid and issues == []

    def test_line_count_mismatch(self):
        valid, issues = validate_formatting("a\nb\nc", "a b c")
        assert not valid
        assert any("Line count" in i for i in issues)

    def test_bullet_mismatch(self):
        valid, issues = validate_formatting("• a\n• b", "a\nb")
        assert not valid

    def test_numbered_mismatch(self):
        valid, issues = validate_formatting("1. a\n2. b", "a\nb")
        assert not valid

    def test_no_bullets_in_original_ignored(self):
        # bullets only checked when the original had them
        valid, _ = validate_formatting("plain\ntext", "still\n- text")
        assert valid


class TestValidateAddedPunctuation:
    def test_identical_valid(self):
        valid, issues = validate_added_punctuation("a; b", "a; b")
        assert valid and issues == []

    def test_semicolon_added_invalid(self):
        valid, issues = validate_added_punctuation(
            "I went to the store, I bought milk.",
            "I went to the store; I bought milk.",
        )
        assert not valid
        assert any("Semicolon" in i for i in issues)

    def test_semicolon_removed_is_fine(self):
        # Removing punctuation isn't the rephrase risk this guards against.
        valid, issues = validate_added_punctuation("a; b", "a b")
        assert valid and issues == []

    @pytest.mark.parametrize("dash", ["—", "–"])
    def test_dash_added_invalid(self, dash):
        valid, issues = validate_added_punctuation("Plain text.", f"Plain {dash} text.")
        assert not valid
        assert any("dash" in i.lower() for i in issues)

    def test_hyphen_not_flagged(self):
        # Plain ASCII hyphen is compound-word punctuation, not a style tell.
        valid, issues = validate_added_punctuation("co-operate", "co-operate fixed")
        assert valid and issues == []

    def test_no_punctuation_change_valid(self):
        valid, issues = validate_added_punctuation("plain text", "plain text fixed")
        assert valid and issues == []


class TestEnforceFormattingPreservation:
    def test_valid_candidate_passes_through(self):
        calls = []
        result = enforce_formatting_preservation("a\nb", "A\nB", calls.append)
        assert result == "A\nB"
        assert calls == []

    def test_repair_retry_success(self):
        def query_fn(prompt):
            assert "ORIGINAL (format template)" in prompt
            return "A\nB"

        assert enforce_formatting_preservation("a\nb", "A B", query_fn) == "A\nB"

    def test_repair_still_invalid_returns_original(self):
        assert (
            enforce_formatting_preservation("a\nb", "A B", lambda p: "still wrong")
            == "a\nb"
        )

    def test_repair_exception_returns_original(self):
        def query_fn(prompt):
            raise RuntimeError("boom")

        assert enforce_formatting_preservation("a\nb", "A B", query_fn) == "a\nb"

    def test_missing_trailing_newline_not_treated_as_invalid(self):
        # Models reliably drop a lone trailing newline; a real fix shouldn't
        # get discarded over it — the original's trailing newline is spliced
        # back on instead of forcing a repair round-trip.
        calls = []
        result = enforce_formatting_preservation("a\nb\n", "A\nB", calls.append)
        assert result == "A\nB\n"
        assert calls == []

    def test_missing_trailing_newline_with_genuine_mismatch_still_repairs(self):
        def query_fn(prompt):
            return "A\nB"

        assert enforce_formatting_preservation("a\nb\n", "A B", query_fn) == "A\nB\n"

    def test_added_semicolon_triggers_constrained_retry(self):
        # Punctuation-only violations retry against the ORIGINAL with the
        # violation named (the draft-based repair prompt would entrench the
        # semicolon, since the draft is what contains it).
        original = "I went to the store, I bought milk."

        def query_fn(prompt):
            assert "REJECTED" in prompt
            assert original in prompt
            return "I went to the store, and I bought milk."

        result = enforce_formatting_preservation(
            original, "I went to the store; I bought milk.", query_fn
        )
        assert result == "I went to the store, and I bought milk."

    def test_added_semicolon_repair_fails_falls_back_to_original(self):
        original = "I went to the store, I bought milk."
        candidate = "I went to the store; I bought milk."
        # Repair pass still inserts a semicolon -> repaired output re-fails validation.
        assert (
            enforce_formatting_preservation(original, candidate, lambda p: candidate)
            == original
        )

    def test_added_dash_triggers_repair(self):
        original = "Plain text, still plain."

        def query_fn(prompt):
            # A realistic retry response: corrects a word mid-text without
            # appending anything (an appended marker word would now be
            # rejected by validate_no_appended_content).
            return "Plain texts, still plain."

        result = enforce_formatting_preservation(
            original, "Plain text — still plain.", query_fn
        )
        assert result == "Plain texts, still plain."

    def test_formatting_ok_but_punctuation_added_still_caught(self):
        # Regression guard: before this check existed, a punctuation-only
        # drift with correct line/bullet counts would have passed
        # validate_formatting and returned the candidate untouched.
        original = "a\nb"
        candidate = "A;\nB"  # same line count as original, but adds a semicolon

        def query_fn(prompt):
            return "A\nB"

        assert enforce_formatting_preservation(original, candidate, query_fn) == "A\nB"

    def test_repair_fixes_lines_but_still_adds_semicolon_falls_back(self):
        original = "a\nb"
        candidate = "A; B"  # both wrong line count AND an added semicolon

        def query_fn(prompt):
            return "A;\nB"  # correct line count now, but semicolon still added

        assert enforce_formatting_preservation(original, candidate, query_fn) == original


class TestEnforcePathReporting:
    def test_ok_path(self):
        _, path = enforce_formatting_preservation_ex("a\nb", "A\nB", lambda p: p)
        assert path == "ok"

    def test_repaired_path(self):
        _, path = enforce_formatting_preservation_ex("a\nb", "A B", lambda p: "A\nB")
        assert path == "repaired"

    def test_fallback_path(self):
        text, path = enforce_formatting_preservation_ex(
            "a\nb", "A B", lambda p: "still flat"
        )
        assert (text, path) == ("a\nb", "fallback")


class TestProofreadLineByLine:
    def test_blank_lines_and_structure_survive_by_construction(self):
        original = "Hello,\n\nWe fixt the bugg.\n\nBest regads,\nEd"
        fixes = {
            "Hello,": "Hello,",
            "We fixt the bugg.": "We fixed the bug.",
            "Best regads,": "Best regards,",
            "Ed": "Ed",
        }
        result = proofread_line_by_line(original, lambda ln: fixes[ln])
        assert result == "Hello,\n\nWe fixed the bug.\n\nBest regards,\nEd"

    def test_model_flattening_one_line_cannot_break_grid(self):
        original = "line won\nline too"
        result = proofread_line_by_line(
            original, lambda ln: "flat one two"  # same for both lines
        )
        assert result.count("\n") == original.count("\n")

    def test_failed_line_keeps_original(self):
        original = "good line\nbad line"
        result = proofread_line_by_line(
            original, lambda ln: None if ln == "bad line" else "good line"
        )
        assert result == "good line\nbad line"

    def test_bullet_and_indent_restored(self):
        original = "  • fixt this"
        result = proofread_line_by_line(original, lambda ln: "- fixed this")
        assert result == "  • fixed this"

    def test_multiline_response_for_a_line_rejected(self):
        original = "one line"
        result = proofread_line_by_line(original, lambda ln: "two\nlines")
        assert result == "one line"

    def test_added_semicolon_rejected_per_line(self):
        original = "The metting is at 9am, come early."
        result = proofread_line_by_line(
            original, lambda ln: "The meeting is at 9am; come early."
        )
        assert result == original


class TestColonAddition:
    def test_added_colon_flagged(self):
        valid, issues = validate_added_punctuation(
            "we got this error out of the system",
            "we got this error out of the system:",
        )
        assert not valid and any("Colons" in i for i in issues)

    def test_existing_colons_fine(self):
        valid, _ = validate_added_punctuation("At 9am: meeting", "At 9am: meeting")
        assert valid

    def test_removed_colon_fine(self):
        valid, _ = validate_added_punctuation("odd: text", "odd text")
        assert valid


class TestConstrainedRetry:
    def test_punct_only_violation_uses_constrained_prompt(self):
        prompts_seen = []

        def query_fn(prompt):
            prompts_seen.append(prompt)
            return "The meeting is at 9am, come early."

        result, path = enforce_formatting_preservation_ex(
            "The metting is at 9am, come early.",
            "The meeting is at 9am; come early.",  # semicolon added
            query_fn,
        )
        assert (result, path) == ("The meeting is at 9am, come early.", "repaired")
        assert len(prompts_seen) == 1
        assert "REJECTED" in prompts_seen[0]
        assert "Semicolons added" in prompts_seen[0]
        # retries against the ORIGINAL text, not the bad draft
        assert "The metting is at 9am" in prompts_seen[0]

    def test_format_violation_still_uses_repair_prompt(self):
        prompts_seen = []

        def query_fn(prompt):
            prompts_seen.append(prompt)
            return "A\nB"

        result, path = enforce_formatting_preservation_ex(
            "a\nb", "A B", query_fn
        )
        assert (result, path) == ("A\nB", "repaired")
        assert "CORRECTED DRAFT" in prompts_seen[0]

    def test_constrained_retry_still_bad_falls_back(self):
        result, path = enforce_formatting_preservation_ex(
            "plain text here",
            "plain text — here",  # dash added
            lambda p: "still — bad",
        )
        assert (result, path) == ("plain text here", "fallback")


class TestStripTrailingParenthetical:
    def test_strips_model_note(self):
        original = "We shipped it; the customers are happy."
        text = '"We shipped it; the customers are happy." (No changes needed in this sentence.)'
        assert strip_trailing_parenthetical(text, original) == (
            '"We shipped it; the customers are happy."'
        )

    def test_keeps_users_own_parenthetical(self):
        original = "Call me later (after 5pm)"
        text = "Call me later (after 5pm)"
        assert strip_trailing_parenthetical(text, original) == text

    def test_non_trailing_paren_untouched(self):
        text = "The fix (merged today) works fine."
        assert strip_trailing_parenthetical(text, "x") == text


class TestFullClean:
    def test_note_plus_quotes_resolves_to_original(self):
        original = "We shipped it; the customers are happy."
        raw = '"We shipped it; the customers are happy." (No changes needed.)'
        assert full_clean(raw, original) == original

    def test_plain_response_unchanged(self):
        assert full_clean("Fixed sentence.", "Fixd sentence.") == "Fixed sentence."


class TestLineColonRestore:
    def test_dropped_trailing_colon_restored(self):
        assert proofread_line_by_line("Steps:", lambda ln: "Steps") == "Steps:"

    def test_colon_swapped_for_period_restored(self):
        assert proofread_line_by_line("Steps:", lambda ln: "Steps.") == "Steps:"

    def test_no_colon_no_change(self):
        assert proofread_line_by_line("hello", lambda ln: "hello") == "hello"


class TestValidateNoAppendedContent:
    def test_appended_completion_flagged(self):
        valid, issues = validate_no_appended_content(
            "gonna grab lunch, back in 20",
            "Gonna grab lunch, back in 20 minutes.",
        )
        assert not valid and any("final word" in i for i in issues)

    def test_unchanged_text_valid(self):
        valid, _ = validate_no_appended_content("back in 20", "back in 20")
        assert valid

    def test_last_word_spelling_fix_valid(self):
        valid, _ = validate_no_appended_content(
            "if that make sence", "if that makes sense."
        )
        assert valid

    def test_plural_fix_at_end_valid(self):
        # same word count -> not an append, even though final word changed
        valid, _ = validate_no_appended_content(
            "one bug, two bug", "one bug, two bugs"
        )
        assert valid

    def test_mid_sentence_article_addition_valid(self):
        valid, _ = validate_no_appended_content(
            "I go to store yesterday and buy milk",
            "I went to the store yesterday and bought milk",
        )
        assert valid

    def test_collapsed_emdash_spacing_flagged(self):
        valid, issues = validate_no_appended_content(
            "The fix — merged — is in prod.",
            "The fix—merged—is in prod.",
        )
        assert not valid and any("em dash" in i for i in issues)

    def test_existing_unspaced_dashes_valid(self):
        valid, _ = validate_no_appended_content(
            "The fix—merged—is fine.", "The fix—merged—is fine."
        )
        assert valid


class TestAngleBracketGuard:
    def test_url_bracket_wrap_flagged(self):
        valid, issues = validate_added_punctuation(
            "see https://example.com here",
            "see <https://example.com> here",
        )
        assert not valid and any("Angle brackets" in i for i in issues)

    def test_existing_brackets_fine(self):
        valid, _ = validate_added_punctuation("a < b > c", "a < b > c")
        assert valid


class TestProofreadSegments:
    def test_partial_salvage_keeps_rejected_segment(self):
        original = "bad segmnt one, rollback is it fails, youyr process may fail"

        def correct_fn(seg):
            fixes = {
                "bad segmnt one": None,  # model failed on this one
                "rollback is it fails": "rollback if it fails",
                "youyr process may fail": "your process may fail",
            }
            return fixes[seg]

        assert proofread_segments(original, correct_fn) == (
            "bad segmnt one, rollback if it fails, your process may fail"
        )

    def test_separators_preserved_verbatim(self):
        original = "one. two; three: four"
        result = proofread_segments(original, lambda s: s.replace("o", "0"))
        assert result == "0ne. tw0; three: f0ur"

    def test_fragment_capitalization_reverted(self):
        original = "then we deploy"
        assert proofread_segments(original, lambda s: "Then we deploy") == (
            "then we deploy"
        )

    def test_added_trailing_period_stripped(self):
        original = "then we deploy"
        assert proofread_segments(original, lambda s: "then we deploy.") == (
            "then we deploy"
        )

    def test_segment_restyling_rejected(self):
        original = "see https://example.com for info, more text here"

        def correct_fn(seg):
            if "https" in seg:
                return "see <https://example.com> for info"  # bracket wrap
            return "more text here"

        assert proofread_segments(original, correct_fn) == original


class TestQuestionExclamationGuard:
    def test_added_question_marks_flagged(self):
        valid, issues = validate_added_punctuation(
            "rollback is it fails", "rollback? Is it failing?"
        )
        assert not valid and any("Question" in i for i in issues)

    def test_existing_question_fine(self):
        valid, _ = validate_added_punctuation("is it done?", "Is it done?")
        assert valid

    def test_added_exclamation_flagged(self):
        valid, issues = validate_added_punctuation("great work", "Great work!")
        assert not valid


class TestUnwrapUrlBrackets:
    def test_autolink_unwrapped(self):
        from aiproof.llm.postprocess import unwrap_url_brackets
        assert unwrap_url_brackets(
            "see <https://example.com/x#y> today", "see https://example.com/x#y tody"
        ) == "see https://example.com/x#y today"

    def test_users_own_brackets_kept(self):
        from aiproof.llm.postprocess import unwrap_url_brackets
        original = "docs at <https://example.com>"
        assert unwrap_url_brackets(original, original) == original

    def test_non_url_brackets_untouched(self):
        from aiproof.llm.postprocess import unwrap_url_brackets
        assert unwrap_url_brackets("a <b> c", "a b c") == "a <b> c"


class TestTypoedUserSignoffPreserved:
    """Regression (live bug): the user's own closing line contained typos, so
    once corrected it no longer matched the original verbatim and the sign-off
    stripper ate it — the proofread then 'fixed nothing'."""

    ORIGINAL = ("I will pass that feedback on to our product owner.\n\n"
                "Is there anythng else or should we close this ticekt")
    CORRECTED = ("I will pass that feedback on to our product owner.\n\n"
                 "Is there anything else or should we close this ticket")

    def test_corrected_users_signoff_survives(self):
        assert strip_trailing_signoff(self.CORRECTED, self.ORIGINAL) == \
            self.CORRECTED

    def test_model_chatter_still_stripped(self):
        padded = self.CORRECTED + "\n\nIs there anything else I can help you with?"
        assert strip_trailing_signoff(padded, self.ORIGINAL) == self.CORRECTED

    def test_full_clean_end_to_end(self):
        assert full_clean(self.CORRECTED, self.ORIGINAL) == self.CORRECTED

    def test_typoed_parenthetical_survives(self):
        original = "see the notes (attched below)"
        corrected = "see the notes (attached below)"
        assert strip_trailing_parenthetical(corrected, original) == corrected

    def test_unrelated_tail_still_not_matched(self):
        # A fragment nothing like the original's ending is still stripped.
        assert strip_trailing_signoff(
            "Fixed sentence. Hope this helps!", "Fixd sentence."
        ) == "Fixed sentence."
