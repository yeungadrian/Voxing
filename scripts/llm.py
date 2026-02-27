"""Test the LLM pipeline in isolation.

Usage: uv run scripts/llm.py
"""

from voxing.config import Settings
from voxing.llm import LocalAgent, TextChunk, ToolCallInput, ToolCallOutput, load_model

settings = Settings()
print("Loading model...")
model, tokenizer = load_model(settings.llm_model_id)
print("Model loaded.\n")

_EXAMPLES = [
    "What is the capital of France?",
    "What is 1234 * 5678? Use Python to calculate it.",
    "Now add 999 to that result.",
]

agent = LocalAgent(model, tokenizer, settings)
for user_input in _EXAMPLES:
    print(f"You: {user_input}")
    print("Assistant: ", end="", flush=True)
    for event in agent.generate(user_input):
        match event:
            case TextChunk(content=t):
                print(t, end="", flush=True)
            case ToolCallInput(code=c):
                print(f"\n[running: {c!r}]", end="", flush=True)
            case ToolCallOutput(result=r):
                print(f"\n[result: {r!r}] ", end="", flush=True)
    print()
