"""Per-application profiles: window -> register matching and prompt wiring.

wm_class values are the real ones from this desktop (ActivityWatch window
history): com.slack.Slack, google-chrome, org.gnome.Ptyxis, code,
chrome-<id>-Profile_1 for PWAs like WhatsApp Web.
"""

import pytest

from conftest import make_cfg
from aiproof import appcontext
from aiproof.llm.postprocess import preserve_casual_style
from aiproof.llm.prompts import build_proofread_prompt

SETTING_HEADER = "WHERE THIS TEXT IS BEING WRITTEN:"


class TestMatching:
    @pytest.mark.parametrize("wm_class,title,expected", [
        ("com.slack.Slack", "Activity - Northern.tech - Slack", "slack"),
        ("Slack", "general | Slack", "slack"),
        ("chrome-hnpfjngllnobngcgfapefoaidbinmjnm-Profile_1",
         "WhatsApp Web", "whatsapp"),
        ("google-chrome", "Ticket #123 - Northern.tech Zendesk", "support"),
        ("google-chrome", "Inbox (3) - Gmail", "email"),
        ("org.mozilla.Thunderbird", "Write: hello", "email"),
        ("code", "orchestrator.py - aiproof - Visual Studio Code", "technical"),
        ("org.gnome.Ptyxis", "ed@mendermund: ~/code", "technical"),
    ])
    def test_known_apps(self, wm_class, title, expected):
        profile = appcontext.match_profile(wm_class, title)
        assert profile and profile["id"] == expected

    @pytest.mark.parametrize("wm_class,title", [
        ("google-chrome", "Mender - Google Chrome"),
        ("org.gnome.Nautilus", "Open Folder"),
        ("obsidian", "Untitled 15 - EdsVault - Obsidian"),
        ("", ""),
    ])
    def test_unmatched_apps_get_default(self, wm_class, title):
        assert appcontext.match_profile(wm_class, title) is None

    def test_slack_is_casual_support_is_not(self):
        assert appcontext.profile_by_id("slack")["casual"] is True
        assert appcontext.profile_by_id("support")["casual"] is False

    def test_every_profile_has_an_id_and_setting(self):
        for profile in appcontext.PROFILES:
            assert profile["id"] and profile["setting"]


class TestGetProfile:
    def test_flag_off_skips_detection(self, monkeypatch):
        monkeypatch.setattr(appcontext, "get_focused_app",
                            lambda: pytest.fail("should not be called"))
        assert appcontext.get_profile(make_cfg(app_profiles=False)) is None

    def test_focus_lookup_failure_is_default(self, monkeypatch):
        monkeypatch.setattr(appcontext, "get_focused_app", lambda: ("", ""))
        assert appcontext.get_profile(make_cfg()) is None

    def test_matches_focused_app(self, monkeypatch):
        monkeypatch.setattr(appcontext, "get_focused_app",
                            lambda: ("com.slack.Slack", "Slack"))
        profile = appcontext.get_profile(make_cfg())
        assert profile["id"] == "slack"


class TestPromptWiring:
    def test_setting_appears_in_prompt(self):
        setting = appcontext.profile_by_id("slack")["setting"]
        prompt = build_proofread_prompt("teh text", app_setting=setting)
        assert SETTING_HEADER in prompt
        assert "Slack message" in prompt

    def test_no_setting_leaves_prompt_untouched(self):
        assert SETTING_HEADER not in build_proofread_prompt("teh text")
        assert build_proofread_prompt("teh text", app_setting=None) == \
            build_proofread_prompt("teh text")

    def test_setting_sits_before_the_anchor(self):
        from aiproof.llm.prompts import _CONSTRAINT_ANCHOR
        prompt = build_proofread_prompt("x", app_setting="A Slack message.")
        assert prompt.index(SETTING_HEADER) < prompt.index(_CONSTRAINT_ANCHOR)
        assert prompt.count("INPUT TEXT:") == 1

    def test_combines_with_learned_context(self):
        prompt = build_proofread_prompt(
            "x", context_words=["Mender"], context_summary="support notes",
            app_setting="A Slack message.",
        )
        assert "Mender" in prompt
        assert "Typical texts: support notes" in prompt
        assert SETTING_HEADER in prompt


class TestOrchestratorIntegration:
    def test_profile_reaches_the_client_cfg(self, orch, fake_clipboard,
                                            fake_llm, monkeypatch):
        monkeypatch.setattr(
            "aiproof.orchestrator.appcontext.get_focused_app",
            lambda: ("com.slack.Slack", "Slack"),
        )
        fake_clipboard.poll_results.append("teh text")
        o = orch()
        assert o.run_paste_flow() is True
        cfg_used = fake_llm.cfgs[-1]
        assert "Slack message" in cfg_used["app_setting"]

    def test_no_match_leaves_cfg_alone(self, orch, fake_clipboard, fake_llm,
                                       monkeypatch):
        monkeypatch.setattr(
            "aiproof.orchestrator.appcontext.get_focused_app",
            lambda: ("org.gnome.Nautilus", "Files"),
        )
        fake_clipboard.poll_results.append("teh text")
        o = orch()
        assert o.run_paste_flow() is True
        assert "app_setting" not in fake_llm.cfgs[-1]

    def test_detection_failure_does_not_break_the_flow(self, orch,
                                                       fake_clipboard,
                                                       fake_llm, monkeypatch):
        def boom():
            raise RuntimeError("dbus down")

        monkeypatch.setattr(
            "aiproof.orchestrator.appcontext.get_focused_app", boom)
        fake_clipboard.poll_results.append("teh text")
        o = orch()
        assert o.run_paste_flow() is True

    def test_notification_names_the_style(self, orch, fake_clipboard,
                                          fake_llm, notifier, monkeypatch):
        monkeypatch.setattr(
            "aiproof.orchestrator.appcontext.get_focused_app",
            lambda: ("com.slack.Slack", "Slack"),
        )
        fake_clipboard.poll_results.append("teh text")
        orch().run_paste_flow()
        assert "slack style" in notifier.last[1]


class TestCasualStyleRepair:
    """Prompt wording alone doesn't hold a 3B model to 'keep it lowercase';
    these repairs are deterministic."""

    def test_lowercase_start_restored(self):
        assert preserve_casual_style(
            "Yeah that's fine by me", "yeah thats fine by me"
        ) == "yeah that's fine by me"

    def test_added_full_stop_removed(self):
        assert preserve_casual_style(
            "yeah that's fine.", "yeah thats fine"
        ) == "yeah that's fine"

    def test_pronoun_i_still_capitalized(self):
        assert preserve_casual_style(
            "I'll get the fix out", "ill get the fix out"
        ) == "I'll get the fix out"

    def test_writers_own_capital_and_stop_kept(self):
        text = "Yeah that's fine."
        assert preserve_casual_style(text, "Yeah thats fine.") == text

    def test_existing_question_mark_kept(self):
        assert preserve_casual_style(
            "can you check?", "can you chek?"
        ) == "can you check?"

    def test_per_line_for_multiline_chat(self):
        original = "hey\ncan you chek this\nthanks"
        corrected = "Hey.\nCan you check this.\nThanks."
        assert preserve_casual_style(corrected, original) == \
            "hey\ncan you check this\nthanks"

    def test_mismatched_line_counts_fall_back_to_whole_text(self):
        assert preserve_casual_style("Yeah ok", "yeah\nok") == "yeah ok"

    def test_leading_whitespace_preserved(self):
        assert preserve_casual_style("  Yeah ok", "  yeah ok") == "  yeah ok"

    def test_empty_inputs_safe(self):
        assert preserve_casual_style("", "x") == ""
        assert preserve_casual_style("x", "") == "x"


class TestCasualWiringInClient:
    def test_casual_flag_applied(self, monkeypatch):
        from aiproof.llm.client import LLMClient
        cfg = make_cfg(app_casual=True)
        client = LLMClient(cfg)
        monkeypatch.setattr(client, "query", lambda p: "Yeah that's fine.")
        monkeypatch.setattr(
            "aiproof.llm.client.enforce_formatting_preservation_ex",
            lambda orig, cand, q: (cand, "ok"))
        out, _ = client.proofread("yeah thats fine")
        assert out == "yeah that's fine"

    def test_without_flag_model_output_kept(self, monkeypatch):
        from aiproof.llm.client import LLMClient
        client = LLMClient(make_cfg())
        monkeypatch.setattr(client, "query", lambda p: "Yeah that's fine.")
        monkeypatch.setattr(
            "aiproof.llm.client.enforce_formatting_preservation_ex",
            lambda orig, cand, q: (cand, "ok"))
        out, _ = client.proofread("yeah thats fine")
        assert out == "Yeah that's fine."


class TestFallbackProviderFieldsReset:
    """Changing the fallback provider must bring its own endpoint/model, or
    the stale pair silently produces an unusable fallback."""

    def test_preset_pairs_are_self_consistent(self):
        from aiproof.llm.providers import PROVIDERS
        for pid, preset in PROVIDERS.items():
            if preset["format"] == "claude_cli":
                assert preset["endpoint"] == "", pid
            elif preset["format"] == "ollama":
                assert preset["endpoint"].startswith("http"), pid
