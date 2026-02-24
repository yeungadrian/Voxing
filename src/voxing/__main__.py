import threading

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from voxing.parakeet import load_model
from voxing.stt import MODEL_NAME, RealtimeTranscriber

console = Console()


def main() -> None:
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_ids: dict[str, TaskID] = {}

        def on_progress(desc: str, advance: int, total: int | None) -> None:
            if desc not in task_ids:
                task_ids[desc] = progress.add_task(desc, total=total)
            progress.advance(task_ids[desc], advance)

        model = load_model(MODEL_NAME, on_progress=on_progress)
    console.print("Model loaded.")

    try:
        while True:
            console.input("\nPress Enter to start real-time recording... ")

            with RealtimeTranscriber(model) as transcriber:

                def _wait_for_enter(_stop: RealtimeTranscriber = transcriber) -> None:
                    input()
                    _stop.stop()

                threading.Thread(target=_wait_for_enter, daemon=True).start()

                last_text = ""
                with Live("", console=console, refresh_per_second=10) as live:
                    console.print("Recording... (press Enter to stop)")
                    for last_text in transcriber:
                        live.update(last_text)

            console.print(f"\nFinal: [bold]{last_text}[/bold]")
    except KeyboardInterrupt:
        console.print("\nExiting.")


if __name__ == "__main__":
    main()
