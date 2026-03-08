import json
import logging
import warnings
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from voxing._download import _resolve_model_path
from voxing.qwen3_tts.config import (
    ModelConfig,
    Qwen3TTSTokenizerConfig,
    Qwen3TTSTokenizerDecoderConfig,
    Qwen3TTSTokenizerEncoderConfig,
    filter_dict_for_dataclass,
)
from voxing.qwen3_tts.qwen3_tts import Model
from voxing.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizer

logger = logging.getLogger(__name__)


def _as_weights_dict(loaded: object) -> dict[str, mx.array]:
    if isinstance(loaded, tuple):
        maybe = loaded[0]
        if isinstance(maybe, dict):
            return cast(dict[str, mx.array], maybe)
    if isinstance(loaded, dict):
        return cast(dict[str, mx.array], loaded)
    raise TypeError("Unexpected safetensors format")


def _apply_quantization(
    model: Model,
    config: dict,
    weights: dict[str, mx.array],
) -> None:
    quantization = config.get("quantization")
    if quantization is None:
        return

    group_size = int(quantization.get("group_size", 64))
    bits = int(quantization["bits"])
    mode = str(quantization.get("mode", "affine"))
    model_quant_predicate = getattr(model, "model_quant_predicate", None)

    def class_predicate(path: str, module: object) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        module_weight = getattr(module, "weight", None)
        if module_weight is None or module_weight.shape[-1] % group_size != 0:
            return False
        if model_quant_predicate is not None and not model_quant_predicate(
            path, module
        ):
            return False
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode=mode,
        class_predicate=class_predicate,
    )


def load_model(
    model_id: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
) -> Model:
    model_path = _resolve_model_path(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt"],
    )
    config = json.loads((model_path / "config.json").read_text())
    model = Model(cast(ModelConfig, ModelConfig.from_dict(config)))

    weights: dict[str, mx.array] = {}
    for wf in model_path.glob("*.safetensors"):
        weights.update(_as_weights_dict(mx.load(str(wf))))

    _apply_quantization(model, config, weights)

    model.load_weights(list(model.sanitize(weights).items()), strict=False)
    mx.eval(model.parameters())
    model.eval()

    transformers_logger = logging.getLogger("transformers")
    prev_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    transformers_logger.setLevel(prev_level)

    speech_tokenizer_path = model_path / "speech_tokenizer"
    if speech_tokenizer_path.exists():
        tokenizer_config_dict = json.loads(
            (speech_tokenizer_path / "config.json").read_text()
        )
        decoder_config = Qwen3TTSTokenizerDecoderConfig()
        encoder_config = None
        if "decoder_config" in tokenizer_config_dict:
            filtered = filter_dict_for_dataclass(
                Qwen3TTSTokenizerDecoderConfig,
                tokenizer_config_dict["decoder_config"],
            )
            decoder_config = Qwen3TTSTokenizerDecoderConfig(**filtered)
        if "encoder_config" in tokenizer_config_dict:
            filtered = filter_dict_for_dataclass(
                Qwen3TTSTokenizerEncoderConfig,
                tokenizer_config_dict["encoder_config"],
            )
            encoder_config = Qwen3TTSTokenizerEncoderConfig(**filtered)

        tokenizer_config = Qwen3TTSTokenizerConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )
        for k, v in tokenizer_config_dict.items():
            if k not in ("decoder_config", "encoder_config") and hasattr(
                tokenizer_config, k
            ):
                setattr(tokenizer_config, k, v)

        speech_tokenizer = Qwen3TTSSpeechTokenizer(tokenizer_config)
        tokenizer_weights: dict[str, mx.array] = {}
        for wf in speech_tokenizer_path.glob("*.safetensors"):
            tokenizer_weights.update(_as_weights_dict(mx.load(str(wf))))
        if tokenizer_weights:
            tokenizer_weights = Qwen3TTSSpeechTokenizer.sanitize(tokenizer_weights)
            speech_tokenizer.load_weights(list(tokenizer_weights.items()), strict=False)
            mx.eval(speech_tokenizer.parameters())
            speech_tokenizer.eval()

            if speech_tokenizer.encoder_model is not None:
                quantizer = speech_tokenizer.encoder_model.quantizer
                for layer in quantizer.rvq_first.vq.layers:
                    layer.codebook.update_in_place()
                for layer in quantizer.rvq_rest.vq.layers:
                    layer.codebook.update_in_place()

        speech_tokenizer.decoder = mx.compile(speech_tokenizer.decoder)
        model.load_speech_tokenizer(speech_tokenizer)

    gen_config = model_path / "generation_config.json"
    if gen_config.exists():
        model.load_generate_config(json.loads(gen_config.read_text()))

    return model
