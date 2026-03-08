import numpy as np
from textual.message import Message


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


class TTSComplete(Message):
    def __init__(self, error: str | None = None) -> None:
        self.error = error
        super().__init__()


class AudioChunk(Message):
    def __init__(self, chunk: np.ndarray) -> None:
        self.chunk = chunk
        super().__init__()
