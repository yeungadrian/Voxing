# Copyright (c) 2025, Prince Canuma and contributors
# (https://github.com/Blaizzy/mlx-audio)

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from voxing.chatterbox.models.s3gen.decoder import ConditionalDecoder
from voxing.chatterbox.models.s3gen.encoder import UpsampleConformerEncoder
from voxing.chatterbox.models.s3gen.flow_matching import CausalConditionalCFM
from voxing.chatterbox.models.s3gen.hifigan import F0Predictor, HiFTGenerator
from voxing.chatterbox.models.s3gen.xvector import CAMPPlus

logger = logging.getLogger(__name__)

# Constants
S3GEN_SR = 24000  # Output sample rate
S3GEN_SIL = 4299  # Silence token


class S3Gen(nn.Module):
    """
    Full S3Gen: speech tokens to waveform.
    Combines token-to-mel (CFM) and mel-to-wav (HiFiGAN).
    """

    def __init__(self, meanflow: bool = False) -> None:
        super().__init__()
        self.meanflow = meanflow

        # Token embedding
        self.input_embedding = nn.Embedding(6561, 512)

        # Speaker encoder (CAMPPlus for x-vector extraction)
        self.speaker_encoder = CAMPPlus(
            feat_dim=80,
            embedding_size=192,
            growth_rate=32,
            bn_size=4,
            init_channels=128,
        )

        # Speaker embedding projection
        self.spk_embed_affine_layer = nn.Linear(192, 80)

        # Encoder
        self.encoder = UpsampleConformerEncoder(
            input_size=512,
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
        )
        self.encoder_proj = nn.Linear(512, 80)

        # Decoder (CFM estimator)
        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            meanflow=meanflow,
        )

        # Flow matching
        self.decoder = CausalConditionalCFM(
            in_channels=240,
            spk_emb_dim=80,
            sigma_min=1e-6,
            t_scheduler="cosine",
            inference_cfg_rate=0.7,
            estimator=estimator,
        )

        self.token_mel_ratio = 2
        self.pre_lookahead_len = 3

        # HiFiGAN vocoder
        f0_predictor = F0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # Trim fade for artifact reduction
        n_trim = S3GEN_SR // 50  # 20ms
        trim_fade = np.zeros(2 * n_trim)
        trim_fade[n_trim:] = (np.cos(np.linspace(np.pi, 0, n_trim)) + 1) / 2
        self.trim_fade = mx.array(trim_fade.astype(np.float32))

    def __call__(
        self,
        speech_tokens: mx.array,
        ref_dict: dict[str, mx.array],
        n_cfm_timesteps: int | None = None,
        finalize: bool = True,
    ) -> mx.array:
        """Generate mel-spectrogram from speech tokens."""
        B = speech_tokens.shape[0]

        # Get reference data
        prompt_token = ref_dict["prompt_token"]
        prompt_token_len = ref_dict["prompt_token_len"]
        prompt_feat = ref_dict["prompt_feat"]
        embedding = ref_dict["embedding"]

        # Broadcast reference data if needed
        if prompt_token.shape[0] != B:
            prompt_token = mx.broadcast_to(prompt_token, (B,) + prompt_token.shape[1:])
        if embedding.shape[0] != B:
            embedding = mx.broadcast_to(embedding, (B,) + embedding.shape[1:])
        if prompt_feat.shape[0] != B:
            prompt_feat = mx.broadcast_to(prompt_feat, (B,) + prompt_feat.shape[1:])

        # Speaker embedding projection
        embedding = embedding / (
            mx.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8
        )
        embedding = self.spk_embed_affine_layer(embedding)

        # Concatenate prompt and input tokens
        token_len = mx.array([speech_tokens.shape[1]] * B)
        token = mx.concatenate([prompt_token, speech_tokens], axis=1)
        token_len = prompt_token_len + token_len

        # Create mask
        max_len = token.shape[1]
        mask = mx.arange(max_len)[None, :] < token_len[:, None]
        mask = mask[:, :, None].astype(mx.float32)

        # Embed tokens
        token_emb = self.input_embedding(token.astype(mx.int32)) * mask

        # Encode
        h, h_masks = self.encoder(token_emb, token_len)

        # Handle non-finalized chunks
        if not finalize:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]

        h_lengths = mx.sum(h_masks[:, 0, :].astype(mx.int32), axis=-1)
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1
        h = self.encoder_proj(h)

        # Prepare conditioning
        zeros_padding = mx.zeros((B, mel_len2, 80))
        conds = mx.concatenate([prompt_feat, zeros_padding], axis=1)
        conds = conds.transpose(0, 2, 1)  # (B, 80, T)

        # Mask for decoder
        mask = mx.arange(h.shape[1])[None, :] < h_lengths[:, None]
        mask = mask[:, None, :].astype(mx.float32)

        # Default timesteps
        if n_cfm_timesteps is None:
            n_cfm_timesteps = 2 if self.meanflow else 10

        # Generate noise for meanflow
        noised_mels = None
        if self.meanflow:
            noised_mels = mx.random.normal((B, 80, speech_tokens.shape[1] * 2))

        # Flow matching
        feat, _ = self.decoder(
            mu=h.transpose(0, 2, 1),
            mask=mask,
            n_timesteps=n_cfm_timesteps,
            spks=embedding,
            cond=conds,
            noised_mels=noised_mels,
            meanflow=self.meanflow,
        )

        # Remove prompt portion
        return feat[:, :, mel_len1:]

    def inference(
        self,
        speech_tokens: mx.array,
        ref_dict: dict[str, mx.array],
        n_cfm_timesteps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Full inference: speech tokens to waveform."""
        # Default timesteps for meanflow
        if n_cfm_timesteps is None:
            n_cfm_timesteps = 2 if self.meanflow else 10

        # Generate mel
        output_mels = self(
            speech_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
        )

        # Vocoder
        output_mels = output_mels.transpose(0, 2, 1)  # (B, T, 80) for HiFiGAN
        output_wavs, output_sources = self.mel2wav.inference(output_mels, None)

        # Apply trim fade to reduce artifacts
        fade_len = len(self.trim_fade)
        if output_wavs.shape[1] >= fade_len:
            faded_start = output_wavs[:, :fade_len] * self.trim_fade
            output_wavs = mx.concatenate(
                [faded_start, output_wavs[:, fade_len:]], axis=1
            )

        return output_wavs, output_sources

    def sanitize(self, weights: dict) -> dict:
        """Sanitize PyTorch weights for MLX compatibility."""
        new_weights = {}

        for key, value in weights.items():
            new_key = key

            # Skip num_batches_tracked
            if "num_batches_tracked" in key:
                continue

            # Handle speaker_encoder (CAMPPlus) weights
            if key.startswith("speaker_encoder."):
                # Use CAMPPlus sanitize
                spk_key = key.replace("speaker_encoder.", "")
                spk_weights = self.speaker_encoder.sanitize({spk_key: value})
                for sk, sv in spk_weights.items():
                    new_weights[f"speaker_encoder.{sk}"] = sv
                continue

            # Handle Conv weight transposition
            # Skip if weights are already in MLX format (from pre-converted models)
            if "weight" in key and value.ndim >= 3:
                if value.ndim == 4:
                    # Conv2d: (O, I, H, W) -> (O, H, W, I)
                    # PyTorch format: shape[2] == shape[3] (square kernel like 3x3, 1x1)
                    if value.shape[2] == value.shape[3]:
                        # PyTorch format (O, I, H, W) with H==W: transpose
                        value = mx.array(np.array(value).transpose(0, 2, 3, 1))
                    # else: already in MLX format, skip transpose
                # Conv1d: (O, I, K) -> (O, K, I)
                # Pre-converted MLX models already transposed.
                # PyTorch: (O, I, K) with small K (1-7).
                # Only transpose if last dim is small kernel.
                elif (
                    value.ndim == 3
                    and value.shape[2] <= 7
                    and value.shape[1] > value.shape[2]
                ):
                    value = mx.array(np.array(value).transpose(0, 2, 1))
                # else: already in MLX format, skip transpose

            new_weights[new_key] = value

        return new_weights
