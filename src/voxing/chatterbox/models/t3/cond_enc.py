from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from voxing.chatterbox.models.t3.t3_config import T3Config


@dataclass
class T3Cond:
    """Dataclass container for conditioning info."""

    speaker_emb: mx.array
    cond_prompt_speech_tokens: mx.array | None = None
    cond_prompt_speech_emb: mx.array | None = None


class T3CondEnc(nn.Module):
    """Handle non-text conditioning: speaker embeddings and prompts."""

    def __init__(self, hp: T3Config) -> None:
        super().__init__()
        self.hp = hp
        self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)

    def __call__(self, cond: T3Cond) -> mx.array:
        """Encode conditioning into a sequence of embeddings."""
        assert (cond.cond_prompt_speech_tokens is None) == (
            cond.cond_prompt_speech_emb is None
        ), "no embeddings for cond_prompt_speech_tokens"

        # Speaker embedding projection
        speaker_emb = cond.speaker_emb.reshape(-1, self.hp.speaker_embed_size)
        cond_spkr = self.spkr_enc(speaker_emb)[:, None, :]  # (B, 1, dim)

        B = cond_spkr.shape[0]
        dim = cond_spkr.shape[-1]

        # Cond prompt
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = mx.zeros((B, 0, dim))

        return mx.concatenate(
            [cond_spkr, cond_prompt_speech_emb],
            axis=1,
        )
