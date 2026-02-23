import threading

from rich.console import Console
from rich.live import Live

from voxing.parakeet import load_model
from voxing.stt import MODEL_NAME, run_realtime

console = Console()


def main() -> None:
    with console.status("Loading model..."):
        model = load_model(MODEL_NAME)
    console.print("Model loaded.")

    try:
        while True:
            console.input("\nPress Enter to start real-time recording... ")
            stop_event = threading.Event()

            def _wait_for_enter() -> None:
                input()
                stop_event.set()

            threading.Thread(target=_wait_for_enter, daemon=True).start()

            last_text = ""
            with Live("", console=console, refresh_per_second=10) as live:
                console.print("Recording... (press Enter to stop)")
                for last_text in run_realtime(model, stop_event=stop_event):
                    live.update(last_text)

            console.print(f"\nFinal: [bold]{last_text}[/bold]")
    except KeyboardInterrupt:
        console.print("\nExiting.")


if __name__ == "__main__":
    main()
