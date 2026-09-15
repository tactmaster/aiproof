import json
import os
import stat

import pytest

from aiproof import context, history
from aiproof.config import DEFAULTS

# Stand-in for /usr/share/dict/words: the handful of entries the filter rules
# actually turn on. "mender"/"thanks"/"receive" are lowercase English,
# "Debian" is one of the word list's ~20k capitalized proper nouns.
WORDLIST = frozenset({
    "mender", "thanks", "receive", "the", "and", "for", "use", "here",
    "again", "team", "text", "with", "this", "that", "note", "today",
    "great", "fine", "update", "email", "debian", "Debian",
})


@pytest.fixture(autouse=True)
def temp_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def fake_wordlist(monkeypatch):
    monkeypatch.setattr(context, "_load_wordlist", lambda: WORDLIST)


def make_cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    return cfg


def add(original, corrected=None, path="ok"):
    """Append one history entry (changed is derived from the two texts)."""
    history.record(
        make_cfg(save_history=True), source="paste", original=original,
        corrected=original if corrected is None else corrected,
        elapsed=0.2, path=path, provider="ollama", model="mistral:latest",
    )


def words(**cfg_over):
    context.refresh(make_cfg(**cfg_over))
    return context.load_words_and_summary(make_cfg(context_aware=True))[0]


def read_cache():
    return json.loads(context.cache_path().read_text())


class FakeClient:
    def __init__(self, reply="Short technical notes.", error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def query(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(
        "aiproof.llm.client.client_from_config", lambda cfg: client
    )
    return client


class TestConsensus:
    def test_survivor_in_two_entries_with_one_change_is_kept(self):
        add("ydotool teh best", "ydotool the best")
        add("ydotool again today")
        assert "ydotool" in words()

    def test_single_entry_is_not_consensus(self):
        add("ydotool teh best", "ydotool the best")
        assert "ydotool" not in words()

    def test_two_unchanged_entries_are_not_evidence(self):
        # Nothing was corrected in either, so the model never "chose" to keep
        # the word — it may simply never have looked hard.
        add("ydotool again today")
        add("ydotool here and here")
        assert "ydotool" not in words()

    def test_fallback_entries_are_ignored(self):
        add("ydotool teh best", "ydotool the best", path="fallback")
        add("ydotool again today", path="fallback")
        assert words() == []

    def test_missing_history_yields_no_words(self):
        assert words() == []

    def test_malformed_entries_are_skipped(self):
        add("ydotool teh best", "ydotool the best")
        with open(history.history_path(), "a") as f:
            f.write("{not json\n")
            f.write(json.dumps({"original": 7, "corrected": None}) + "\n")
        add("ydotool again today")
        assert "ydotool" in words()


class TestBlacklist:
    def test_token_corrected_away_never_appears(self):
        add("ydotoool teh best", "ydotoool the best")
        add("ydotoool again today")
        add("we use ydotoool here", "we use ydotool here")
        assert "ydotoool" not in words()

    def test_case_only_fix_endorses_the_corrected_form(self):
        add("we ship mender today", "we ship Mender today")
        add("the Mender team here")
        result = words()
        assert "Mender" in result
        assert "mender" not in result


class TestDictionaryFilter:
    def test_capitalized_needs_two_midsentence_uses(self):
        add("we ship mender today", "we ship Mender today")
        add("Mender is fine")  # sentence start — no signal
        assert "Mender" not in words()

    def test_sentence_start_capital_is_dropped(self):
        add("Thanks for teh update", "Thanks for the update")
        add("Thanks again today")
        assert "Thanks" not in words()

    def test_known_proper_noun_is_dropped(self):
        add("we run debian here", "we run Debian here")
        add("the Debian team again")
        assert "Debian" not in words()

    def test_persistent_typo_is_dropped(self):
        # "recieve" transposes into "receive" — a typo the model keeps
        # missing, not a domain term.
        add("did you recieve teh email", "did you recieve the email")
        add("recieve the update today")
        assert "recieve" not in words()

    def test_unknown_lowercase_jargon_is_kept(self):
        add("ydotool teh best", "ydotool the best")
        add("ydotool again today")
        assert "ydotool" in words()

    def test_digits_and_dots_are_kept(self):
        add("ssh gogomund2 teh box", "ssh gogomund2 the box")
        add("gogomund2 again today")
        assert "gogomund2" in words()

    def test_allcaps_is_kept(self):
        add("the OTA update teh", "the OTA update the")
        add("OTA again today")
        assert "OTA" in words()

    def test_camelcase_is_kept(self):
        add("the NorthernTech team teh", "the NorthernTech team the")
        add("NorthernTech again today")
        assert "NorthernTech" in words()

    def test_shape_fallback_without_a_wordlist(self, monkeypatch):
        monkeypatch.setattr(context, "_load_wordlist", lambda: None)
        add("ydotool OTA teh mender here", "ydotool OTA the Mender here")
        add("ydotool OTA again the Mender team")
        result = words()
        assert "OTA" in result
        assert "Mender" in result       # capitalized mid-sentence twice
        assert "ydotool" not in result  # indistinguishable from English here


class TestCaps:
    def test_at_most_twelve_words(self):
        tokens = " ".join(f"{chr(ord('a') + i)}a1" for i in range(15))
        add(f"{tokens} teh", f"{tokens} the")
        add(f"{tokens} the")
        assert len(words()) == 12

    def test_joined_list_is_trimmed_to_120_chars(self):
        tokens = " ".join(f"host{i:02d}.example" for i in range(12))
        add(f"{tokens} teh", f"{tokens} the")
        add(f"{tokens} the")
        result = words()
        assert len(", ".join(result)) <= 120
        assert len(result) == 7

    def test_overlong_tokens_are_dropped(self):
        long_token = "supercalifragilistic.example12"  # 30 chars
        add(f"{long_token} ydotool teh", f"{long_token} ydotool the")
        add(f"{long_token} ydotool the")
        result = words()
        assert long_token not in result
        assert "ydotool" in result


class TestCache:
    def test_refresh_writes_the_cache(self):
        add("ydotool teh best", "ydotool the best")
        add("ydotool again today")
        context.refresh(make_cfg())
        data = read_cache()
        assert data["version"] == 1
        assert data["history_mtime"] == os.path.getmtime(history.history_path())
        assert data["entry_count"] == 2
        assert "ydotool" in data["words"]
        assert data["summary"] is None
        assert data["summary_entry_count"] == 0
        assert data["updated"]

    def test_cache_is_private(self):
        context.refresh(make_cfg())
        mode = stat.S_IMODE(os.stat(context.cache_path()).st_mode)
        assert mode == 0o600

    def test_no_temp_files_left_behind(self, temp_data_home):
        add("ydotool teh best", "ydotool the best")
        context.refresh(make_cfg())
        names = sorted(p.name for p in (temp_data_home / "aiproof").iterdir())
        assert names == ["context.json", "history.jsonl"]

    def test_refresh_if_stale_skips_unchanged_history(self, monkeypatch):
        add("ydotool teh best", "ydotool the best")
        context.refresh(make_cfg())
        calls = []
        monkeypatch.setattr(
            context, "_build_words", lambda entries: calls.append(entries) or []
        )
        context.refresh_if_stale(make_cfg())
        assert calls == []

    def test_refresh_if_stale_rebuilds_when_history_grows(self, monkeypatch):
        add("ydotool teh best", "ydotool the best")
        context.refresh(make_cfg())
        add("ydotool again today")
        bumped = os.path.getmtime(history.history_path()) + 10
        os.utime(history.history_path(), (bumped, bumped))
        calls = []
        monkeypatch.setattr(
            context, "_build_words", lambda entries: calls.append(entries) or []
        )
        context.refresh_if_stale(make_cfg())
        assert len(calls) == 1

    def test_refresh_if_stale_without_history_does_nothing(self):
        context.refresh_if_stale(make_cfg())
        assert not context.cache_path().exists()

    def test_refresh_if_stale_never_raises(self, monkeypatch):
        add("ydotool teh best", "ydotool the best")
        monkeypatch.setattr(
            history, "read_recent",
            lambda count=10: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        context.refresh_if_stale(make_cfg())  # must not raise


class TestLoadWordsAndSummary:
    def test_flag_off_touches_no_files(self, monkeypatch):
        def explode():
            raise AssertionError("must not touch the filesystem")

        monkeypatch.setattr(history, "history_dir", explode)
        assert context.load_words_and_summary(make_cfg()) == ([], None)

    def test_missing_cache(self):
        assert context.load_words_and_summary(
            make_cfg(context_aware=True)
        ) == ([], None)

    def test_corrupt_cache(self):
        context.cache_path().parent.mkdir(parents=True, exist_ok=True)
        context.cache_path().write_text("{not json")
        assert context.load_words_and_summary(
            make_cfg(context_aware=True)
        ) == ([], None)

    def test_hand_edited_cache_is_revalidated(self):
        context.cache_path().parent.mkdir(parents=True, exist_ok=True)
        context.cache_path().write_text(json.dumps({
            "words": ["ydotool", "no", "has space", 42, "x" * 30, "Mender"],
            "summary": "  ",
        }))
        result, summary = context.load_words_and_summary(
            make_cfg(context_aware=True)
        )
        assert result == ["ydotool", "Mender"]
        assert summary is None

    def test_recaps_at_twelve(self):
        context.cache_path().parent.mkdir(parents=True, exist_ok=True)
        context.cache_path().write_text(json.dumps({
            "words": [f"tok{i:02d}" for i in range(20)],
            "summary": "Release notes and short emails.",
        }))
        result, summary = context.load_words_and_summary(
            make_cfg(context_aware=True)
        )
        assert len(result) == 12
        assert summary == "Release notes and short emails."


def add_entries(count, start=0):
    for i in range(start, start + count):
        add(f"entry number {i} about teh release", f"entry number {i} about the release")


class TestSummary:
    def test_generated_once_five_entries_exist(self, fake_client):
        add_entries(5)
        context.refresh(make_cfg(), allow_summary=True)
        data = read_cache()
        assert data["summary"] == "Short technical notes."
        assert data["summary_entry_count"] == 5
        assert len(fake_client.prompts) == 1

    def test_not_generated_below_five_entries(self, fake_client):
        add_entries(4)
        context.refresh(make_cfg(), allow_summary=True)
        assert fake_client.prompts == []
        assert read_cache()["summary"] is None

    def test_not_regenerated_before_25_new_entries(self, fake_client):
        add_entries(5)
        context.refresh(make_cfg(), allow_summary=True)
        add_entries(10, start=5)
        fake_client.reply = "Different answer."
        context.refresh(make_cfg(), allow_summary=True)
        assert len(fake_client.prompts) == 1
        assert read_cache()["summary"] == "Short technical notes."

    def test_regenerated_after_25_new_entries(self, fake_client):
        add_entries(5)
        context.refresh(make_cfg(), allow_summary=True)
        add_entries(25, start=5)
        fake_client.reply = "Different answer."
        context.refresh(make_cfg(), allow_summary=True)
        assert len(fake_client.prompts) == 2
        data = read_cache()
        assert data["summary"] == "Different answer."
        assert data["summary_entry_count"] == 30

    def test_failure_keeps_the_previous_summary(self, fake_client):
        add_entries(5)
        context.refresh(make_cfg(), allow_summary=True)
        add_entries(25, start=5)
        fake_client.error = RuntimeError("provider down")
        context.refresh(make_cfg(), allow_summary=True)
        data = read_cache()
        assert data["summary"] == "Short technical notes."
        assert data["summary_entry_count"] == 5  # due again next refresh
        assert data["entry_count"] == 30

    def test_empty_answer_keeps_the_previous_summary(self, fake_client):
        add_entries(5)
        context.refresh(make_cfg(), allow_summary=True)
        add_entries(25, start=5)
        fake_client.reply = "   \n  "
        context.refresh(make_cfg(), allow_summary=True)
        assert read_cache()["summary"] == "Short technical notes."

    def test_response_is_cleaned(self, fake_client):
        fake_client.reply = '\n\n"Mostly   release notes."\nAnd an aside.\n'
        add_entries(5)
        context.refresh(make_cfg(), allow_summary=True)
        assert read_cache()["summary"] == "Mostly release notes."

    def test_response_is_truncated_to_160_chars(self, fake_client):
        fake_client.reply = "word " * 100
        add_entries(5)
        context.refresh(make_cfg(), allow_summary=True)
        assert len(read_cache()["summary"]) == 160

    def test_prompt_shape(self, fake_client):
        add("x" * 300)
        add_entries(25)
        context.refresh(make_cfg(), allow_summary=True)
        prompt = fake_client.prompts[0]
        lines = prompt.split("\n\n")[0].splitlines()
        assert len(lines) == 20  # last 20 usable entries only
        assert all(line.startswith("- ") for line in lines)
        assert all(len(line) <= 202 for line in lines)
        assert prompt.endswith("Output ONLY the sentence.")

    def test_long_originals_are_truncated_in_the_prompt(self, fake_client):
        for _ in range(5):
            add("y" * 300)
        context.refresh(make_cfg(), allow_summary=True)
        assert "- " + "y" * 200 + "\n" in fake_client.prompts[0] + "\n"

    def test_allow_summary_false_never_builds_a_client(self, monkeypatch):
        def explode(cfg):
            raise AssertionError("no LLM call without allow_summary")

        monkeypatch.setattr("aiproof.llm.client.client_from_config", explode)
        add_entries(30)
        context.refresh(make_cfg())
        assert read_cache()["summary"] is None


class FakeThread:
    def __init__(self, target=None, daemon=None, **kwargs):
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True


class TestScheduleRefresh:
    @pytest.fixture(autouse=True)
    def threads(self, monkeypatch):
        made = []

        def factory(**kwargs):
            thread = FakeThread(**kwargs)
            made.append(thread)
            return thread

        monkeypatch.setattr(context.threading, "Thread", factory)
        return made

    @pytest.mark.parametrize("flags", [
        {},
        {"context_aware": True},
        {"save_history": True},
    ])
    def test_gated_off(self, threads, flags):
        context.schedule_refresh(make_cfg(**flags))
        assert threads == []

    def test_spawns_daemon_thread(self, threads):
        context.schedule_refresh(make_cfg(context_aware=True, save_history=True))
        assert len(threads) == 1
        assert threads[0].daemon is True
        assert threads[0].started is True

    def test_thread_target_refreshes(self, threads):
        add("ydotool teh best", "ydotool the best")
        add("ydotool again today")
        context.schedule_refresh(make_cfg(context_aware=True, save_history=True))
        threads[0].target()
        assert "ydotool" in read_cache()["words"]


class TestClear:
    def test_clear_removes_the_cache(self):
        context.refresh(make_cfg())
        assert context.clear() is True
        assert not context.cache_path().exists()
        assert context.clear() is False


class TestManglingVsSpellingFix:
    """Refinements from live data: one model mangling must not veto an
    endorsed term, but a genuine spelling fix must."""

    def test_single_mangle_does_not_veto_endorsed_word(self):
        # "ydotool" survives two entries, then the model mangles it into
        # unrelated words once — it stays vocabulary.
        add("chek if ydotool runs", "check if ydotool runs")
        add("we use ydotool here")
        add("the ydotool host faild", "the Yocto tool host failed")
        assert "ydotool" in words()

    def test_majority_mangling_vetoes(self):
        add("teh ydotool runs", "the ydotool runs")
        add("ydotool here today", "Yocto tool here today")
        add("ydotool there agin", "Yocto tool there again")
        assert "ydotool" not in words()

    def test_capitalized_with_lowercase_sibling_dropped(self):
        # "Yes" capitalized mid-sentence (chat glue) but also appearing
        # lowercase in corrected texts is a common word, not a name.
        add("minutesYes here teh docs", "minutesYes here the docs")
        add("he said Yes but yes was informal teh",
            "he said Yes but yes was informal the")
        assert "Yes" not in words()

    def test_protocol_tokens_dropped(self):
        add("see https link teh", "see https link the")
        add("the https link works")
        assert "https" not in words()


class TestRefreshThrottle:
    def test_small_growth_skipped_with_min_new_entries(self):
        add("ydotool teh best", "ydotool the best")
        add("ydotool again today")
        context.refresh(make_cfg())
        before = read_cache()
        add("one more entry teh", "one more entry the")
        context.refresh_if_stale(make_cfg(), min_new_entries=5)
        assert read_cache() == before  # skipped: only 1 new entry

    def test_enough_growth_refreshes(self):
        add("ydotool teh best", "ydotool the best")
        context.refresh(make_cfg())
        for i in range(5):
            add(f"entry number {i} teh", f"entry number {i} the")
        context.refresh_if_stale(make_cfg(), min_new_entries=5)
        assert read_cache()["entry_count"] == 6

    def test_default_still_refreshes_on_any_change(self):
        add("ydotool teh best", "ydotool the best")
        context.refresh(make_cfg())
        add("ydotool again today")
        context.refresh_if_stale(make_cfg())
        assert read_cache()["entry_count"] == 2
