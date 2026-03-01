from textual.message import Message


class TokenReceived(Message):
    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__()


class GenerationComplete(Message):
    pass


class ToolCallStarted(Message):
    def __init__(self, code: str, name: str) -> None:
        self.code = code
        self.name = name
        super().__init__()


class ToolCallFinished(Message):
    def __init__(self, code: str, result: str, name: str) -> None:
        self.code = code
        self.result = result
        self.name = name
        super().__init__()


class TranscriptionUpdate(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class TranscriptionFinal(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()
