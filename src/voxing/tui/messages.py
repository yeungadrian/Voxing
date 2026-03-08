from enum import Enum, auto

import numpy as np
from textual.message import Message


class Status(Enum):
    IDLE = auto()
    LOADING_LLM = auto()
    GENERATING = auto()
    LOADING_STT = auto()
    RECORDING = auto()
    CHAT_CLEARED = auto()
    SETTINGS_SAVED = auto()
    TRANSCRIPTION_READY = auto()
    ERROR = auto()


STATUS_MARKUP: dict[Status, str] = {
    Status.IDLE: "[dim]type / for commands[/]",
    Status.LOADING_LLM: "[$warning]Loading LLM...[/]",
    Status.GENERATING: "[$primary]Generating...[/]",
    Status.LOADING_STT: "[$warning]Loading STT model...[/]",
    Status.RECORDING: "[dim]esc to stop recording[/]",
    Status.CHAT_CLEARED: "[$success]Chat cleared[/]",
    Status.SETTINGS_SAVED: "[$success]Settings saved[/]",
    Status.TRANSCRIPTION_READY: "[dim]enter to send  ·  type / for commands[/]",
}


def status_markup(status: Status, error: str | None = None) -> str:
    """Return markup for a status, with optional error override."""
    if status is Status.ERROR and error:
        return f"[$error]{error}[/]"
    return STATUS_MARKUP[status]


class StatusChanged(Message):
    def __init__(self, status: Status, error: str | None = None) -> None:
        self.status = status
        self.error = error
        super().__init__()


class TokenReceived(Message):
    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__()


class GenerationComplete(Message):
    def __init__(self, error: str | None = None) -> None:
        self.error = error
        super().__init__()


class TranscriptionUpdate(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class TranscriptionFinal(Message):
    def __init__(self, text: str, error: str | None = None) -> None:
        self.text = text
        self.error = error
        super().__init__()


class AudioChunk(Message):
    def __init__(self, chunk: np.ndarray) -> None:
        self.chunk = chunk
        super().__init__()
