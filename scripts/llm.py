"""Test the LLM pipeline in isolation.

Usage: uv run scripts/llm.py
"""

from voxing.config import Settings
from voxing.llm import LocalAgent, TextChunk, load_llm

settings = Settings()
print("Loading model...")
model, tokenizer = load_llm(settings.llm_model_id)
print("Model loaded.\n")

_EXAMPLES = [
    "What is the capital of France?",
    "What is 1234 * 5678?",
    "Now add 999 to that result.",
]

agent = LocalAgent(
    model,
    tokenizer,
    settings,
    messages=[{"role": "system", "content": settings.llm_system_prompt}],
)
for user_input in _EXAMPLES:
    print(f"You: {user_input}")
    print("Assistant: ", end="", flush=True)
    for event in agent.generate(user_input):
        match event:
            case TextChunk(content=t):
                print(t, end="", flush=True)
    print()
