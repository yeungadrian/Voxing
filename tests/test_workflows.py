"""Integration tests for user workflows."""

from voxing.app import VoxingApp
from voxing.state import AppState
from voxing.widgets import (
    ChatInput,
    ConversationLog,
    ModelSelector,
    PerfStats,
    StatusPanel,
)


async def _wait_for_ready(app: VoxingApp) -> None:
    """Wait for all workers to complete and process pending messages."""
    await app.workers.wait_for_complete()


async def test_app_loads_with_mocked_models(mock_model_loading):
    """App loads successfully and transitions to READY state."""
    app = VoxingApp()
    async with app.run_test():
        await _wait_for_ready(app)
        assert app.models is not None
        assert app.state == AppState.READY

        text_area = app.query_one("#user-input", ChatInput)
        assert text_area.disabled is False


async def test_clear_command_clears_history(mock_model_loading):
    """'/clear' command clears conversation history."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        app.chat_history.append({"role": "user", "content": "test"})
        app.chat_history.append({"role": "assistant", "content": "response"})
        assert len(app.chat_history) == 2

        text_area = app.query_one("#user-input", ChatInput)
        text_area.insert("/clear")
        await pilot.press("enter")
        await _wait_for_ready(app)

        assert len(app.chat_history) == 0


async def test_tts_command_toggles_tts(mock_model_loading):
    """'/tts' command toggles TTS on and off."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        assert app.tts_enabled is False

        text_area = app.query_one("#user-input", ChatInput)
        text_area.insert("/tts")
        await pilot.press("enter")
        await _wait_for_ready(app)

        assert app.tts_enabled is True
        status_panel = app.query_one("#status-panel", StatusPanel)
        assert status_panel.tts_enabled is True

        text_area.insert("/tts")
        await pilot.press("enter")
        await _wait_for_ready(app)

        assert app.tts_enabled is False
        assert status_panel.tts_enabled is False


async def test_tab_completion_for_commands(mock_model_loading):
    """Tab key completes commands."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        text_area = app.query_one("#user-input", ChatInput)
        text_area.insert("/rec")
        await pilot.press("tab")
        await pilot.pause(0)

        assert text_area.text == "/record"


async def test_command_hints_appear_when_typing_slash(mock_model_loading):
    """Command hints display when typing '/'."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        text_area = app.query_one("#user-input", ChatInput)
        hint_label = app.query_one("#command-hint")

        assert "hidden" in hint_label.classes

        text_area.insert("/rec")
        await pilot.pause(0)

        assert "hidden" not in hint_label.classes
        hint_text = str(hint_label.render())
        assert "/record" in hint_text


async def test_text_input_triggers_llm_generation(mock_model_loading, mock_llm_stream):
    """Text input triggers LLM generation workflow."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        text_area = app.query_one("#user-input", ChatInput)

        text_area.insert("Hello AI")
        await pilot.press("enter")
        await _wait_for_ready(app)

        assert len(app.chat_history) == 2
        assert app.chat_history[0]["role"] == "user"
        assert app.chat_history[0]["content"] == "Hello AI"
        assert app.chat_history[1]["role"] == "assistant"
        assert app.chat_history[1]["content"] == "Hello world!"


async def test_double_esc_cancels_processing(mock_model_loading):
    """Double ESC cancels ongoing processing."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        app.is_processing = True
        app.state = AppState.THINKING

        await pilot.press("escape")
        await pilot.pause(0)
        assert app._esc_pending is True

        await pilot.press("escape")
        await pilot.pause(0)
        assert app.is_processing is False
        assert app.state == AppState.READY
        assert app._esc_pending is False


async def test_unknown_command_shows_error(mock_model_loading):
    """Unknown commands display error message."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        text_area = app.query_one("#user-input", ChatInput)
        conv_log = app.query_one("#conversation-log", ConversationLog)

        text_area.insert("/unknown")
        await pilot.press("enter")
        await _wait_for_ready(app)

        assert app.state == AppState.READY
        log_text = "\n".join(str(line) for line in conv_log.lines)
        assert "Unknown command" in log_text


async def test_metrics_update_after_llm_generation(mock_model_loading, mock_llm_stream):
    """Perf stats update after LLM generation."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        perf_stats = app.query_one("#perf-stats", PerfStats)
        assert perf_stats.current_stats is None

        text_area = app.query_one("#user-input", ChatInput)
        text_area.insert("Test query")
        await pilot.press("enter")
        await _wait_for_ready(app)

        assert perf_stats.current_stats is not None
        assert perf_stats.current_stats.tokens == 4
        assert perf_stats.current_stats.tokens_per_sec > 0


async def test_record_pipeline_runs_full_flow(
    mock_model_loading, mock_llm_stream, mock_audio_record, mock_stt_transcribe
):
    """'/record' command records audio, transcribes, and runs LLM pipeline."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        text_area = app.query_one("#user-input", ChatInput)
        text_area.insert("/record")
        await pilot.press("enter")
        await _wait_for_ready(app)

        assert len(app.chat_history) == 2
        assert app.chat_history[0]["role"] == "user"
        assert app.chat_history[0]["content"] == "What is the weather?"
        assert app.chat_history[1]["role"] == "assistant"
        assert app.chat_history[1]["content"] == "Hello world!"
        assert app.state == AppState.READY


async def test_model_selector_navigation(mock_model_loading):
    """'/model' opens selector; j/k navigate; escape dismisses."""
    app = VoxingApp()
    async with app.run_test() as pilot:
        await _wait_for_ready(app)

        text_area = app.query_one("#user-input", ChatInput)
        text_area.insert("/model")
        await pilot.press("enter")
        await _wait_for_ready(app)

        selector = app.screen
        assert isinstance(selector, ModelSelector)

        items = list(selector.query("ModelItem"))
        assert items[0].has_focus

        await pilot.press("j")
        await pilot.pause(0)
        assert items[1].has_focus

        await pilot.press("k")
        await pilot.pause(0)
        assert items[0].has_focus

        await pilot.press("escape")
        await pilot.pause(0)
        assert not isinstance(app.screen, ModelSelector)
