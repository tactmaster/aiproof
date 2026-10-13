"""Edit-list mode: parsing ``wrong => right`` lines and applying them to the
original text. Pure functions — no client, no HTTP."""

import pytest

from aiproof.llm.edits import (
    Edit,
    apply_edits,
    looks_like_rewrite,
    parse_edits,
)


# -- parse_edits ---------------------------------------------------------------


class TestParse:
    def test_basic_list(self):
        r = parse_edits("metting => meeting\ntheir going => they're going")
        assert r.edits == [Edit("metting", "meeting"),
                           Edit("their going", "they're going")]
        assert r.no_changes is False
        assert r.noise_lines == 0

    @pytest.mark.parametrize("text", [
        "NO CHANGES", "No changes needed.", "no corrections", '"NO CHANGES"',
        "No errors found.", "- No change",
    ])
    def test_sentinel_variants(self, text):
        r = parse_edits(text)
        assert r.edits == []
        assert r.no_changes is True

    def test_sentinel_plus_edits_keeps_edits(self):
        r = parse_edits("teh => the\nNO CHANGES")
        assert r.edits == [Edit("teh", "the")]
        assert r.no_changes is True

    def test_chatter_is_counted_not_parsed(self):
        r = parse_edits("Here are the corrections:\nteh => the\n"
                        "Let me know if you need anything else.")
        assert r.edits == [Edit("teh", "the")]
        assert r.noise_lines == 2

    def test_code_fence_and_think_block(self):
        assert parse_edits("```\nteh => the\n```").edits == [Edit("teh", "the")]
        assert parse_edits("<think>hmm, one typo</think>\nteh => the").edits == [
            Edit("teh", "the")]
        assert parse_edits("<think>never closed teh => the").edits == []

    @pytest.mark.parametrize("line", [
        '"teh" => "the"', "'teh' => 'the'", "`teh` => `the`",
        "“teh” => “the”",
    ])
    def test_quoted_sides_unquoted(self, line):
        assert parse_edits(line).edits == [Edit("teh", "the")]

    @pytest.mark.parametrize("line", ["- teh => the", "* teh => the",
                                      "1. teh => the", "2) teh => the",
                                      "• teh => the"])
    def test_list_prefix_stripped(self, line):
        assert parse_edits(line).edits == [Edit("teh", "the")]

    def test_bullet_on_both_sides_is_part_of_the_edit(self):
        assert parse_edits("- Point too => - Point two").edits == [
            Edit("- Point too", "- Point two")]
        assert parse_edits("1. Point too => 1. Point two").edits == [
            Edit("1. Point too", "1. Point two")]

    def test_alternative_arrows(self):
        assert parse_edits("teh -> the").edits == [Edit("teh", "the")]
        assert parse_edits("teh → the").edits == [Edit("teh", "the")]

    def test_fat_arrow_preferred_when_both_appear(self):
        assert parse_edits("a->b => a->c").edits == [Edit("a->b", "a->c")]

    def test_empty_right_dropped(self):
        assert parse_edits("teh =>").edits == []
        assert parse_edits("teh => \n").edits == []

    def test_duplicates_deduped(self):
        assert parse_edits("teh => the\nteh => the").edits == [Edit("teh", "the")]

    def test_json_shapes(self):
        assert parse_edits('[{"wrong": "teh", "right": "the"}]').edits == [
            Edit("teh", "the")]
        assert parse_edits('[["teh", "the"]]').edits == [Edit("teh", "the")]
        assert parse_edits('{"teh": "the"}').edits == [Edit("teh", "the")]
        assert parse_edits('{"edits": [{"from": "teh", "to": "the"}]}').edits == [
            Edit("teh", "the")]
        assert parse_edits("[not json").edits == []

    def test_crlf_normalised(self):
        assert parse_edits("teh => the\r\nfoo => bar").edits == [
            Edit("teh", "the"), Edit("foo", "bar")]

    @pytest.mark.parametrize("response", ["", "   \n", None])
    def test_empty_response(self, response):
        assert parse_edits(response) == ([], False, 0)


# -- apply_edits ---------------------------------------------------------------


class TestApply:
    def test_single_occurrence(self):
        r = apply_edits("I saw teh cat", [Edit("teh", "the")])
        assert r.text == "I saw the cat"
        assert r.applied == [Edit("teh", "the")]
        assert r.rejected == []

    def test_repeated_typo_replaced_everywhere(self):
        r = apply_edits("teh cat and teh dog", [Edit("teh", "the")])
        assert r.text == "the cat and the dog"

    def test_word_boundary_respected(self):
        r = apply_edits("this is it", [Edit("is", "was")])
        assert r.text == "this was it"

    def test_subword_match_rejected(self):
        r = apply_edits("this thing", [Edit("is", "was")])
        assert r.text == "this thing"
        assert r.rejected == [(Edit("is", "was"), "not in text")]

    def test_punctuation_attached_span(self):
        r = apply_edits("see you tomorow.", [Edit("tomorow.", "tomorrow.")])
        assert r.text == "see you tomorrow."

    def test_hallucinated_span_rejected_others_applied(self):
        r = apply_edits("I saw teh cat", [Edit("foo", "bar"), Edit("teh", "the")])
        assert r.text == "I saw the cat"
        assert r.rejected == [(Edit("foo", "bar"), "not in text")]

    def test_noop_edit_ignored(self):
        r = apply_edits("fine text", [Edit("fine", "fine")])
        assert r == ("fine text", [], [])

    def test_overlapping_edit_rejected(self):
        r = apply_edits("teh cat sat", [Edit("teh cat", "the cat"),
                                        Edit("cat", "dog")])
        assert r.text == "the cat sat"
        assert r.rejected[0][1] == "overlaps an earlier edit"

    def test_conflicting_right_first_wins(self):
        r = apply_edits("teh cat", [Edit("teh", "the"), Edit("teh", "tea")])
        assert r.text == "the cat"
        assert [e for e, _ in r.rejected] == [Edit("teh", "tea")]

    def test_offsets_survive_length_changes(self):
        r = apply_edits("teh quick brwn fox", [Edit("teh", "the"),
                                               Edit("brwn", "brown")])
        assert r.text == "the quick brown fox"
        r = apply_edits("brwn fox teh end", [Edit("teh", "the"),
                                             Edit("brwn", "brown")])
        assert r.text == "brown fox the end"

    def test_added_semicolon_becomes_comma(self):
        r = apply_edits("get there early we should leave",
                        [Edit("early we", "early; we")])
        assert r.text == "get there early, we should leave"

    @pytest.mark.parametrize("wrong,right", [
        ("note", "note:"),
        ("fix - it", "fix — it"),
        ("are you coming.", "are you coming?"),
        ("great", "great!"),
        ("see docs", "see <docs>"),
    ])
    def test_restyling_punctuation_rejected(self, wrong, right):
        text = f"well {wrong} now"
        r = apply_edits(text, [Edit(wrong, right)])
        assert r.text == text
        assert len(r.rejected) == 1

    def test_dropping_a_bullet_marker_rejected(self):
        r = apply_edits("- fixt\n- two", [Edit("- fixt", "fixed")])
        assert r.text == "- fixt\n- two"
        assert "Bullet" in r.rejected[0][1]

    def test_appended_content_rejected(self):
        r = apply_edits("back in 20", [Edit("20", "20 minutes")])
        assert r.text == "back in 20"
        assert "appended" in r.rejected[0][1]

    def test_span_cap(self):
        long_wrong = "w" * 101
        r = apply_edits(long_wrong, [Edit(long_wrong, "x")])
        assert r.rejected[0][1].startswith("span too long")
        short = "i go to store yesterday and buy milk"  # whole-text edit, 36 chars
        r = apply_edits(short, [Edit(short, "I went to the store yesterday and bought milk")])
        assert r.text == "I went to the store yesterday and bought milk"

    def test_rephrase_by_word_count_rejected(self):
        text = "we should get there early"
        r = apply_edits(text, [Edit(text, "it would be wise to arrive ahead of time")])
        assert r.text == text
        assert r.rejected[0][1].startswith("rephrase")

    def test_insertion_and_deletion_via_neighbours(self):
        assert apply_edits("I go to store", [Edit("go to store", "went to the store")]).text == \
            "I went to the store"
        assert apply_edits("the the end", [Edit("the the", "the")]).text == "the end"

    def test_pronoun_i_only_standalone(self):
        r = apply_edits("i think it is", [Edit("i", "I")])
        assert r.text == "I think it is"

    def test_internal_whitespace_fix(self):
        assert apply_edits("x a  b y", [Edit("a  b", "a b")]).text == "x a b y"

    def test_curly_quotes_in_original_matched_and_kept(self):
        r = apply_edits("I think it’s teh cat", [Edit("it's teh", "it's the")])
        assert r.text == "I think it’s the cat"

    def test_curly_quotes_from_model_folded_to_straight(self):
        r = apply_edits("I think it's teh cat", [Edit("it’s teh", "it’s the")])
        assert r.text == "I think it's the cat"

    def test_quote_style_only_change_is_noop(self):
        r = apply_edits("it's fine", [Edit("it's", "it’s")])
        assert r == ("it's fine", [], [])

    def test_prompt_quote_copied_into_span(self):
        r = apply_edits("teh cat sat", [Edit('"teh', '"the')])
        assert r.text == "the cat sat"
        r = apply_edits("the cat sta", [Edit('sta"', 'sat"')])
        assert r.text == "the cat sat"

    def test_url_and_code_spans_protected(self):
        text = "see https://example.com/teh/path and `teh var` then teh end"
        r = apply_edits(text, [Edit("teh", "the")])
        assert r.text == "see https://example.com/teh/path and `teh var` then the end"
        only_url = "see https://example.com/teh/path"
        r = apply_edits(only_url, [Edit("teh", "the")])
        assert r.text == only_url
        assert r.rejected[0][1] == "only inside a URL/code span"

    def test_crlf_original(self):
        r = apply_edits("teh one\r\nteh two", [Edit("teh", "the")])
        assert r.text == "the one\r\nthe two"

    def test_unicode(self):
        assert apply_edits("cafe is teh best \U0001F389", [Edit("cafe", "café"), Edit("teh", "the")]).text == \
            "café is the best \U0001F389"

    def test_multiline_bullets_layout_preserved(self):
        text = "Hello there\n\n• Point one has eror\n• Point too\n\nAnd a paragraf."
        r = apply_edits(text, [Edit("eror", "error"), Edit("Point too", "Point two"),
                               Edit("paragraf", "paragraph")])
        assert r.text == "Hello there\n\n• Point one has error\n• Point two\n\nAnd a paragraph."
        assert r.rejected == []

    def test_newline_inside_span_rejected(self):
        r = apply_edits("a\nb", [Edit("a\nb", "a b")])
        assert r.text == "a\nb"
        assert r.rejected[0][1] == "spans a line break"

    def test_empty_original(self):
        assert apply_edits("", [Edit("a", "b")]).rejected == [(Edit("a", "b"), "empty text")]
        assert apply_edits("text", []) == ("text", [], [])


# -- looks_like_rewrite --------------------------------------------------------


def test_looks_like_rewrite():
    original = "I saw teh cat on teh mat yesterday"
    assert looks_like_rewrite(original, "I saw the cat on the mat yesterday")
    assert not looks_like_rewrite(original, "Here are the corrections you asked for.")
    assert not looks_like_rewrite(original, "ok")
    assert not looks_like_rewrite("", "x")


class TestCaseFolding:
    def test_recased_span_matches_and_keeps_capital(self):
        r = apply_edits("Deploymnt is scheduled", [Edit("deploymnt", "deployment")])
        assert r.text == "Deployment is scheduled"
        assert r.rejected == []

    def test_recased_span_ambiguous_is_rejected(self):
        r = apply_edits("Teh cat and teh dog", [Edit("TEH", "the")])
        assert r.text == "Teh cat and teh dog"
        assert r.rejected[0][1] == "not in text"

    def test_lowercase_span_for_lowercase_text(self):
        r = apply_edits("the deploymnt is late", [Edit("Deploymnt", "Deployment")])
        assert r.text == "the deployment is late"
