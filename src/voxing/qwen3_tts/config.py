# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass, field, fields
from typing import Any, cast

from voxing.qwen3_tts._base import BaseModelArgs


def filter_dict_for_dataclass[T](cls: type[T], data: dict[str, Any]) -> dict[str, Any]:
    """Filter a dictionary to only include keys that are valid dataclass fields."""
    valid_fields = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid_fields}


@dataclass
class Qwen3TTSSpeakerEncoderConfig:
    """Configuration for ECAPA-TDNN speaker encoder."""

    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    enc_kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000


@dataclass
class Qwen3TTSTalkerCodePredictorConfig:
    """Configuration for the code predictor sub-model."""

    vocab_size: int = 2048
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    sliding_window: int | None = None
    layer_types: list[str] | None = None
    attention_dropout: float = 0.0
    num_code_groups: int = 16

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


@dataclass
class Qwen3TTSTalkerConfig:
    """Configuration for the main talker model."""

    code_predictor_config: Qwen3TTSTalkerCodePredictorConfig = field(
        default_factory=Qwen3TTSTalkerCodePredictorConfig
    )
    vocab_size: int = 3072
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, Any] | None = field(
        default_factory=lambda: {
            "interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        }
    )
    attention_bias: bool = False
    sliding_window: int | None = None
    attention_dropout: float = 0.0
    num_code_groups: int = 16
    text_hidden_size: int = 2048
    text_vocab_size: int = 151936
    codec_eos_token_id: int = 2150
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    codec_pad_id: int = 2148
    codec_bos_id: int = 2149
    codec_language_id: dict[str, int] | None = None
    spk_id: dict[str, int] | None = None
    spk_is_dialect: dict[str, str | bool] | None = None

    def __post_init__(self):
        if isinstance(self.code_predictor_config, dict):
            filtered = filter_dict_for_dataclass(
                Qwen3TTSTalkerCodePredictorConfig,
                cast(dict[str, Any], self.code_predictor_config),
            )
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig(**filtered)


@dataclass
class Qwen3TTSTokenizerDecoderConfig:
    """Configuration for the speech tokenizer decoder."""

    attention_bias: bool = False
    attention_dropout: float = 0.0
    latent_dim: int = 1024
    codebook_dim: int = 512
    codebook_size: int = 2048
    decoder_dim: int = 1536
    hidden_act: str = "silu"
    hidden_size: int = 512
    intermediate_size: int = 1024
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    head_dim: int = 64
    num_attention_heads: int = 16
    num_hidden_layers: int = 8
    num_key_value_heads: int = 16
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    semantic_codebook_size: int = 4096
    sliding_window: int = 72
    upsample_rates: list[int] = field(default_factory=lambda: [8, 5, 4, 3])
    upsampling_ratios: list[int] = field(default_factory=lambda: [2, 2])
    vector_quantization_hidden_dimension: int = 512


@dataclass
class Qwen3TTSTokenizerEncoderConfig:
    """Configuration for the speech tokenizer encoder."""

    frame_rate: float = 12.5
    attention_bias: bool = False
    attention_dropout: float = 0.0
    audio_channels: int = 1
    codebook_dim: int = 256
    codebook_size: int = 2048
    compress: int = 2
    dilation_growth_rate: int = 2
    head_dim: int = 64
    hidden_act: str = "gelu"
    hidden_size: int = 512
    intermediate_size: int = 2048
    kernel_size: int = 7
    last_kernel_size: int = 3
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    norm_eps: float = 1e-5
    num_attention_heads: int = 8
    num_filters: int = 64
    num_hidden_layers: int = 8
    num_key_value_heads: int = 8
    num_quantizers: int = 32
    num_residual_layers: int = 1
    num_semantic_quantizers: int = 1
    residual_kernel_size: int = 3
    rope_theta: float = 10000.0
    sampling_rate: int = 24000
    sliding_window: int = 250
    upsampling_ratios: list[int] = field(default_factory=lambda: [8, 6, 5, 4])
    use_causal_conv: bool = True
    use_conv_shortcut: bool = False
    vector_quantization_hidden_dimension: int = 256


@dataclass
class Qwen3TTSTokenizerConfig:
    """Configuration for the speech tokenizer."""

    encoder_config: Qwen3TTSTokenizerEncoderConfig | None = None
    decoder_config: Qwen3TTSTokenizerDecoderConfig = field(
        default_factory=Qwen3TTSTokenizerDecoderConfig
    )
    encoder_valid_num_quantizers: int = 16
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    decode_upsample_rate: int = 1920
    encode_downsample_rate: int = 1920

    def __post_init__(self):
        # Encoder is only needed for voice cloning (ICL)
        if isinstance(self.encoder_config, dict):
            filtered = filter_dict_for_dataclass(
                Qwen3TTSTokenizerEncoderConfig,
                cast(dict[str, Any], self.encoder_config),
            )
            self.encoder_config = Qwen3TTSTokenizerEncoderConfig(**filtered)
        if isinstance(self.decoder_config, dict):
            filtered = filter_dict_for_dataclass(
                Qwen3TTSTokenizerDecoderConfig,
                cast(dict[str, Any], self.decoder_config),
            )
            self.decoder_config = Qwen3TTSTokenizerDecoderConfig(**filtered)


@dataclass
class ModelConfig(BaseModelArgs):
    """Main configuration for Qwen3-TTS model."""

    model_type: str = "qwen3_tts"
    talker_config: Qwen3TTSTalkerConfig = field(default_factory=Qwen3TTSTalkerConfig)
    speaker_encoder_config: Qwen3TTSSpeakerEncoderConfig = field(
        default_factory=Qwen3TTSSpeakerEncoderConfig
    )
    tokenizer_config: Qwen3TTSTokenizerConfig | None = None
    tokenizer_type: str = "qwen3_tts_tokenizer_12hz"
    tts_model_size: str = "0b6"
    tts_model_type: str = "base"
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    tts_pad_token_id: int = 151671
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    sample_rate: int = 24000

    def __post_init__(self):
        if isinstance(self.talker_config, dict):
            filtered = filter_dict_for_dataclass(
                Qwen3TTSTalkerConfig,
                cast(dict[str, Any], self.talker_config),
            )
            self.talker_config = Qwen3TTSTalkerConfig(**filtered)
        if isinstance(self.speaker_encoder_config, dict):
            filtered = filter_dict_for_dataclass(
                Qwen3TTSSpeakerEncoderConfig,
                cast(dict[str, Any], self.speaker_encoder_config),
            )
            self.speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(**filtered)
        if self.tokenizer_config is not None and isinstance(
            self.tokenizer_config, dict
        ):
            filtered = filter_dict_for_dataclass(
                Qwen3TTSTokenizerConfig,
                cast(dict[str, Any], self.tokenizer_config),
            )
            self.tokenizer_config = Qwen3TTSTokenizerConfig(**filtered)
