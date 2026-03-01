from textual.message import Message


class TokenReceived(Message):
    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__()


class GenerationComplete(Message):
    pass


class TranscriptionUpdate(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class TranscriptionFinal(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()
