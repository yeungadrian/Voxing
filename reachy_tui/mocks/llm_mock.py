"""Mock LLM for simulating conversational responses."""

import asyncio
import random
from collections.abc import AsyncIterator

from reachy_tui.mocks import DEFAULT_CONFIG, MockConfig


class MockLLM:
    """Simulates LLM generation with streaming and realistic delays."""

    # Canned responses by category
    RESPONSES = {
        "greeting": [
            "Hello! I'm Reachy, your voice assistant. How can I help you today?",
            "Hi there! It's great to hear from you. What can I do for you?",
            "Hey! I'm here and ready to assist. What's on your mind?",
        ],
        "weather": [
            (
                "Based on current conditions, it looks like it's sunny with a "
                "high of 72°F today. Perfect weather for a walk!"
            ),
            (
                "The weather today is partly cloudy with temperatures around "
                "68°F. You might want to bring a light jacket."
            ),
            (
                "It's looking rainy today with temperatures in the low 60s. "
                "Don't forget your umbrella!"
            ),
        ],
        "time": [
            "The current time is 2:34 PM. Is there anything else you'd like to know?",
            "It's 10:15 AM right now. Time flies, doesn't it?",
            "The time is currently 7:42 PM. Getting close to evening!",
        ],
        "help": [
            (
                "I can help you with weather information, tell you the time, "
                "answer general questions, or just chat. "
                "What would you like to know?"
            ),
            (
                "I'm here to assist! Try asking me about the weather, time, "
                "or any general questions you might have."
            ),
        ],
        "shutdown": [
            "Understood. Shutting down now.",
            "Shutting down. See you soon!",
        ],
        "default": [
            "That's an interesting question! Let me think about that for a moment.",
            "I understand what you're asking. Here's what I think:",
            "Great question! Based on my knowledge, I would say:",
            "Hmm, let me consider that carefully.",
        ],
    }

    def __init__(self, config: MockConfig = DEFAULT_CONFIG):
        """Initialize the mock LLM.

        Args:
            config: Configuration for timing simulation.
        """
        self.config = config
        self.conversation_history = []

    def _select_response(self, prompt: str) -> str:
        """Select an appropriate canned response based on the prompt."""
        prompt_lower = prompt.lower()

        # Check for keywords to categorize the prompt
        if any(word in prompt_lower for word in ["hello", "hi", "hey", "greetings"]):
            return random.choice(self.RESPONSES["greeting"])
        if any(
            word in prompt_lower for word in ["weather", "temperature", "forecast"]
        ):
            return random.choice(self.RESPONSES["weather"])
        if any(word in prompt_lower for word in ["time", "clock", "hour"]):
            return random.choice(self.RESPONSES["time"])
        if any(word in prompt_lower for word in ["help", "what can you", "how do"]):
            return random.choice(self.RESPONSES["help"])
        if any(word in prompt_lower for word in ["shut down", "shutdown", "sleep"]):
            return random.choice(self.RESPONSES["shutdown"])
        # Generate a more contextual default response
        base = random.choice(self.RESPONSES["default"])
        context = self._generate_context_response(prompt_lower)
        return f"{base} {context}"

    def _generate_context_response(self, prompt: str) -> str:
        """Generate a contextual response for unmatched prompts."""
        # Simple template-based responses
        if "?" in prompt:
            return (
                "While I don't have specific information about that, "
                "I'd be happy to help you explore the topic further."
            )
        if any(word in prompt for word in ["thank", "thanks"]):
            return (
                "You're very welcome! "
                "Let me know if there's anything else I can help with."
            )
        if any(word in prompt for word in ["good", "great", "awesome", "perfect"]):
            return "I'm glad to hear that! Is there anything else you'd like to know?"
        return (
            "I'm designed to assist with various tasks. "
            "Feel free to ask me about weather, time, or general questions!"
        )

    async def generate_streaming(
        self, prompt: str, max_tokens: int = 100
    ) -> AsyncIterator[str]:
        """Generate a streaming response token by token.

        Args:
            prompt: User's input text.
            max_tokens: Maximum tokens to generate (unused in mock).

        Yields:
            Individual tokens (words or punctuation).
        """
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        # Select appropriate response
        response = self._select_response(prompt)

        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        # Simulate time to first token
        ttft = random.uniform(self.config.llm_ttft_min, self.config.llm_ttft_max)
        await asyncio.sleep(ttft)

        # Tokenize response (simple word-based tokenization)
        tokens = (
            response.replace(",", " ,")
            .replace(".", " .")
            .replace("!", " !")
            .replace("?", " ?")
            .split()
        )

        # Stream tokens with realistic delays
        for token in tokens:
            yield token
            delay = random.uniform(
                self.config.llm_token_delay_min, self.config.llm_token_delay_max
            )
            await asyncio.sleep(delay)

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
