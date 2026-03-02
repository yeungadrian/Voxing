from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import CorruptedCacheException, LocalEntryNotFoundError


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
            )
        )
    except (LocalEntryNotFoundError, CorruptedCacheException):
        return Path(
            snapshot_download(
                model_id,
                allow_patterns=allow_patterns,
            )
        )
