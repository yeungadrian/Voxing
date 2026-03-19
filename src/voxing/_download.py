from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import CorruptedCacheException, LocalEntryNotFoundError
from tqdm.asyncio import tqdm_asyncio as _base_tqdm


class _SilentTqdm(_base_tqdm):
    """No-op tqdm that suppresses all progress output."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.pop("name", None)
        kwargs.setdefault("disable", True)
        super().__init__(*args, **kwargs)

    def display(self, msg: str | None = None, pos: int | None = None) -> None:
        pass

    def clear(self, nolock: bool = False) -> None:
        pass


def _resolve_model_path(
    model_id: str,
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
                tqdm_class=_SilentTqdm,
            )
        )
    except (LocalEntryNotFoundError, CorruptedCacheException):
        return Path(
            snapshot_download(
                model_id,
                allow_patterns=allow_patterns,
                tqdm_class=_SilentTqdm,
            )
        )
