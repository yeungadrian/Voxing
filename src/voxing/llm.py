import io
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import cast

import mlx.nn as nn
import mlx_lm
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tool_parsers.pythonic import parse_tool_call
from mlx_lm.tool_parsers.pythonic import tool_call_end as _TOOL_CALL_END
from mlx_lm.tool_parsers.pythonic import tool_call_start as _TOOL_CALL_START

from voxing._progress import (
    DownloadProgressCallback,
    _make_tqdm_class,
    _resolve_model_path,
)
from voxing.config import Settings

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Run Python code. Always use print() to show results.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]

_DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant."

type Message = dict[str, object]


@dataclass
class TextChunk:
    content: str


@dataclass
class ToolCallInput:
    code: str
    name: str


@dataclass
class ToolCallOutput:
    result: str


type GenerationEvent = TextChunk | ToolCallInput | ToolCallOutput


def _execute_python(code: str) -> str:
    """Execute Python code and return captured output."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            exec(code)  # noqa: S102
        except Exception as e:
            err.write(str(e))
    return out.getvalue() + err.getvalue()


def load_model(
    model_id: str,
    *,
    on_progress: DownloadProgressCallback | None = None,
) -> tuple[nn.Module, TokenizerWrapper]:
    """Download (if needed) and load an LLM from HuggingFace Hub."""
    tqdm_class = _make_tqdm_class(on_progress) if on_progress is not None else None
    model_path = _resolve_model_path(model_id, tqdm_class)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cast(tuple[nn.Module, TokenizerWrapper], mlx_lm.load(str(model_path)))


class LocalAgent:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        settings: Settings,
        *,
        tools_enabled: bool = False,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings
        self._tools_enabled = tools_enabled
        self._messages: list[Message] = [{"role": "system", "content": system_prompt}]

    def generate(self, user_message: str) -> Iterator[GenerationEvent]:
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

        while True:
            prompt = self._tokenizer.apply_chat_template(
                self._messages,
                tools=_TOOLS if self._tools_enabled else None,
                tokenize=False,
                add_generation_prompt=True,
            )

            yielded_text = ""
            tool_text = ""
            in_tool_call = False

            for chunk in mlx_lm.stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._settings.llm_max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                if chunk.text == _TOOL_CALL_START:
                    in_tool_call = True
                elif in_tool_call:
                    if chunk.text == _TOOL_CALL_END:
                        parsed = parse_tool_call(tool_text)
                        code: str = parsed["arguments"]["code"]
                        name: str = parsed["name"]
                        yield ToolCallInput(code=code, name=name)
                        result = _execute_python(code)
                        yield ToolCallOutput(result=result)
                        self._messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"{yielded_text}"
                                    f"{_TOOL_CALL_START}{tool_text}{_TOOL_CALL_END}"
                                ),
                            }
                        )
                        self._messages.append({"role": "tool", "content": result})
                        break
                    else:
                        tool_text += chunk.text
                else:
                    yielded_text += chunk.text
                    yield TextChunk(content=chunk.text)
            else:
                self._messages.append({"role": "assistant", "content": yielded_text})
                break
