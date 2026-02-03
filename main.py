from rich.console import Console

from assistant import VoiceAssistant, load_models


def main() -> None:
    console = Console()
    stt_model, llm_model, tts_model, tokenizer = load_models(console)

    assistant = VoiceAssistant(
        stt_model=stt_model,
        llm_model=llm_model,
        tts_model=tts_model,
        tokenizer=tokenizer,
        silence_threshold=0.01,
        console=console,
    )

    assistant.run()


if __name__ == "__main__":
    main()
