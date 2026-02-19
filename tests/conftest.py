"""Shared pytest fixtures for Voxing tests."""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from voxing.models import Models


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer with encode method for token counting."""
    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda x: list(range(len(x))))
    tokenizer.apply_chat_template = Mock(return_value="formatted prompt")
    return tokenizer


@pytest.fixture
def mock_models(mock_tokenizer):
    """Mocked Models instance with all components."""
    return Models(
        stt=Mock(),
        llm=Mock(),
        tts=Mock(),
        tokenizer=mock_tokenizer,
    )


@pytest.fixture
def mock_model_loading(mock_models):
    """Patch model loading functions to return mocks instantly."""
    with (
        patch("voxing.app.load_stt", return_value=mock_models.stt),
        patch(
            "voxing.app.load_llm", return_value=(mock_models.llm, mock_models.tokenizer)
        ),
        patch("voxing.app.load_tts", return_value=mock_models.tts),
        patch("voxing.widgets.metrics_panel._get_memory_mb", return_value=256),
    ):
        yield mock_models


@pytest.fixture
def mock_llm_stream():
    """Mock LLM streaming to yield fake tokens."""

    async def fake_generate(*args, **kwargs):
        for token in ["Hello", " ", "world", "!"]:
            yield token

    with patch("voxing.models.llm.generate_streaming", side_effect=fake_generate):
        yield


@pytest.fixture
def mock_audio_record():
    """Mock audio recording to return fake audio data."""
    fake_audio = np.zeros(16000, dtype=np.float32)
    with patch("voxing.audio.record", new_callable=AsyncMock, return_value=fake_audio):
        yield fake_audio


@pytest.fixture
def mock_stt_transcribe():
    """Mock STT transcription to return fixed text."""
    with patch(
        "voxing.models.stt.transcribe",
        new_callable=AsyncMock,
        return_value="What is the weather?",
    ):
        yield
