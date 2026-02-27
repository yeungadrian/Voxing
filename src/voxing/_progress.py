import io
from collections.abc import Callable
from pathlib import Path

import tqdm
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

DownloadProgressCallback = Callable[[str, int, int | None], None]


def _make_tqdm_class(callback: DownloadProgressCallback) -> type[tqdm.tqdm]:
    """Return a tqdm subclass that routes byte progress to callback."""

    class _CallbackTqdm(tqdm.tqdm):
        """tqdm bridge that forwards progress to a callback without terminal output."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self._desc: str = str(kwargs.get("desc", ""))
            self._total: int | None = kwargs.get("total")  # type: ignore[assignment]
            kwargs["file"] = io.StringIO()
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> bool | None:
            callback(self._desc, n, self._total)
            return super().update(n)

    return _CallbackTqdm


def _resolve_model_path(
    model_id: str,
    tqdm_class: type[tqdm.tqdm] | None,
    *,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Resolve model directory using cache-first with network fallback."""
    try:
        return Path(
            snapshot_download(
                model_id,
                allow_patterns=allow_patterns,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        return Path(
            snapshot_download(  # type: ignore[no-matching-overload]
                model_id,
                allow_patterns=allow_patterns,
                tqdm_class=tqdm_class,
            )
        )
