import io
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import cast

import mlx.nn as nn
import mlx_lm
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from voxing._download import _resolve_model_path
from voxing.config import Settings

type Message = dict[str, object]


@dataclass
class TextChunk:
    content: str


def load_model(model_id: str) -> tuple[nn.Module, TokenizerWrapper]:
    """Download (if needed) and load an LLM from HuggingFace Hub."""
    model_path = _resolve_model_path(model_id)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cast(tuple[nn.Module, TokenizerWrapper], mlx_lm.load(str(model_path)))


class LocalAgent:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        settings: Settings,
        messages: list[Message],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings
        self._messages = messages

    @property
    def messages(self) -> list[Message]:
        """Return the current conversation history."""
        return self._messages

    def generate(self, user_message: str) -> Iterator[TextChunk]:
        self._messages.append({"role": "user", "content": user_message})

        sampler = make_sampler(
            temp=self._settings.llm_temperature,
            top_p=self._settings.llm_top_p,
            top_k=self._settings.llm_top_k,
            min_p=0.0,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=self._settings.llm_repetition_penalty,
        )

        prompt = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        yielded_text = ""
        for chunk in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._settings.llm_max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            yielded_text += chunk.text
            yield TextChunk(content=chunk.text)

        self._messages.append({"role": "assistant", "content": yielded_text})
