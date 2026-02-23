import dataclasses
import json
import types as _types
import typing
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from ._alignment import (
    AlignedResult,
    AlignedToken,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)
from ._audio import PreprocessArgs, log_mel_spectrogram
from ._conformer import Conformer, ConformerArgs
from ._ctc import AuxCTCArgs, ConvASRDecoder, ConvASRDecoderArgs
from ._rnnt import JointArgs, JointNetwork, PredictArgs, PredictNetwork


def _decode_token(token_id: int, vocabulary: list[str]) -> str:
    """Decode a single token id to text using the model vocabulary."""
    return vocabulary[token_id].replace("▁", " ")


def from_dict[T](cls: type[T], data: dict) -> T:
    """Recursively populate a dataclass from a dict, handling nested dataclasses."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    field_types = typing.get_type_hints(cls)
    kwargs: dict[str, object] = {}

    for field in dataclasses.fields(cls):
        name = field.name
        if name not in data:
            continue

        value = data[name]
        ftype = field_types[name]

        origin = typing.get_origin(ftype)
        is_union = origin is typing.Union or (
            hasattr(_types, "UnionType") and isinstance(ftype, _types.UnionType)
        )
        if is_union:
            inner = [a for a in ftype.__args__ if a is not type(None)]
            if inner:
                ftype = inner[0]

        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = from_dict(ftype, value)

        kwargs[name] = value

    return cls(**kwargs)


@dataclass
class TDTDecodingArgs:
    model_type: str
    durations: list[int]
    greedy: dict | None


@dataclass
class RNNTDecodingArgs:
    greedy: dict | None


@dataclass
class CTCDecodingArgs:
    greedy: dict | None


@dataclass
class ParakeetTDTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: TDTDecodingArgs


@dataclass
class ParakeetRNNTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: RNNTDecodingArgs


@dataclass
class ParakeetCTCArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: ConvASRDecoderArgs
    decoding: CTCDecodingArgs


@dataclass
class ParakeetTDTCTCArgs(ParakeetTDTArgs):
    aux_ctc: AuxCTCArgs


class Model(nn.Module):
    def __init__(self, preprocess_args: PreprocessArgs) -> None:
        super().__init__()
        self.preprocessor_config = preprocess_args

    def decode(self, mel: mx.array) -> list[AlignedResult]:
        """Decode mel spectrograms to produce transcriptions."""
        raise NotImplementedError

    def decode_chunk(self, audio_data: mx.array) -> AlignedResult:
        """Compute mel spectrogram and decode a single audio chunk."""
        mel = log_mel_spectrogram(audio_data, self.preprocessor_config)
        return self.decode(mel)[0]

    def generate(
        self,
        audio: mx.array,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        chunk_duration: float | None = None,
        overlap_duration: float = 15.0,
    ) -> AlignedResult:
        """Transcribe audio with optional chunking for long inputs."""
        audio_data = audio.astype(dtype) if audio.dtype != dtype else audio

        if chunk_duration is None:
            return self.decode_chunk(audio_data)

        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate
        if audio_length_seconds <= chunk_duration:
            return self.decode_chunk(audio_data)

        chunk_samples = int(chunk_duration * self.preprocessor_config.sample_rate)
        overlap_samples = int(overlap_duration * self.preprocessor_config.sample_rate)

        all_tokens: list[AlignedToken] = []

        for start in range(0, len(audio_data), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio_data))
            chunk_audio = audio_data[start:end]
            chunk_mel = log_mel_spectrogram(chunk_audio, self.preprocessor_config)
            chunk_result = self.decode(chunk_mel)[0]

            chunk_offset = start / self.preprocessor_config.sample_rate
            for sentence in chunk_result.sentences:
                for token in sentence.tokens:
                    token.start += chunk_offset
                    token.end = token.start + token.duration

            chunk_tokens: list[AlignedToken] = []
            for sentence in chunk_result.sentences:
                chunk_tokens.extend(sentence.tokens)

            if all_tokens:
                try:
                    all_tokens = merge_longest_contiguous(
                        all_tokens, chunk_tokens, overlap_duration=overlap_duration
                    )
                except RuntimeError:
                    all_tokens = merge_longest_common_subsequence(
                        all_tokens, chunk_tokens, overlap_duration=overlap_duration
                    )
            else:
                all_tokens = chunk_tokens

            mx.clear_cache()

        return sentences_to_result(tokens_to_sentences(all_tokens))


class ParakeetTDT(Model):
    def __init__(self, args: ParakeetTDTArgs) -> None:
        super().__init__(args.preprocessor)

        assert args.decoding.model_type == "tdt", "Model must be a TDT model"

        self.encoder_config = args.encoder

        self.vocabulary = args.joint.vocabulary
        self.durations = args.decoding.durations
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.encoder = Conformer(args.encoder)
        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(self, mel: mx.array) -> list[AlignedResult]:
        """Greedy TDT decoding; handles batches and single input."""
        batch_size: int = mel.shape[0]
        if len(mel.shape) == 2:
            batch_size = 1
            mel = mx.expand_dims(mel, 0)

        batch_features, lengths = self.encoder(mel)
        mx.eval(batch_features, lengths)

        results = []
        for b in range(batch_size):
            features = batch_features[b : b + 1]
            max_length = int(lengths[b])

            last_token = len(self.vocabulary)
            hypothesis: list[AlignedToken] = []

            time = 0
            new_symbols = 0
            decoder_hidden: tuple[mx.array, mx.array] | None = None

            while time < max_length:
                feature = features[:, time : time + 1]

                current_token = (
                    mx.array([[last_token]], dtype=mx.int32)
                    if last_token != len(self.vocabulary)
                    else None
                )
                decoder_output, (hidden, cell) = self.decoder(
                    current_token, decoder_hidden
                )

                decoder_output = decoder_output.astype(feature.dtype)
                proposed_decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                joint_output = self.joint(feature, decoder_output)

                pred_token = mx.argmax(
                    joint_output[0, 0, :, : len(self.vocabulary) + 1]
                )
                decision = mx.argmax(joint_output[0, 0, :, len(self.vocabulary) + 1 :])

                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=time
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,
                            duration=self.durations[int(decision)]
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,
                            text=_decode_token(int(pred_token), self.vocabulary),
                        )
                    )
                    last_token = int(pred_token)
                    decoder_hidden = proposed_decoder_hidden

                time += self.durations[int(decision)]
                new_symbols += 1

                if self.durations[int(decision)] != 0:
                    new_symbols = 0
                else:
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        time += 1
                        new_symbols = 0

            result = sentences_to_result(tokens_to_sentences(hypothesis))
            results.append(result)

        return results


class ParakeetRNNT(Model):
    def __init__(self, args: ParakeetRNNTArgs) -> None:
        super().__init__(args.preprocessor)

        self.encoder_config = args.encoder

        self.vocabulary = args.joint.vocabulary
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.encoder = Conformer(args.encoder)
        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(self, mel: mx.array) -> list[AlignedResult]:
        """Greedy RNNT decoding; handles batches and single input."""
        batch_size: int = mel.shape[0]
        if len(mel.shape) == 2:
            batch_size = 1
            mel = mx.expand_dims(mel, 0)

        batch_features, lengths = self.encoder(mel)
        mx.eval(batch_features, lengths)

        results = []
        for b in range(batch_size):
            features = batch_features[b : b + 1]
            max_length = int(lengths[b])

            last_token = len(self.vocabulary)
            hypothesis: list[AlignedToken] = []

            time = 0
            new_symbols = 0
            decoder_hidden: tuple[mx.array, mx.array] | None = None

            while time < max_length:
                feature = features[:, time : time + 1]

                current_token = (
                    mx.array([[last_token]], dtype=mx.int32)
                    if last_token != len(self.vocabulary)
                    else None
                )
                decoder_output, (hidden, cell) = self.decoder(
                    current_token, decoder_hidden
                )

                decoder_output = decoder_output.astype(feature.dtype)
                proposed_decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                joint_output = self.joint(feature, decoder_output)

                pred_token = mx.argmax(joint_output)

                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=time
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,
                            duration=1
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,
                            text=_decode_token(int(pred_token), self.vocabulary),
                        )
                    )
                    last_token = int(pred_token)
                    decoder_hidden = proposed_decoder_hidden

                    new_symbols += 1
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        time += 1
                        new_symbols = 0
                else:
                    time += 1
                    new_symbols = 0

            result = sentences_to_result(tokens_to_sentences(hypothesis))
            results.append(result)

        return results


class ParakeetCTC(Model):
    def __init__(self, args: ParakeetCTCArgs) -> None:
        super().__init__(args.preprocessor)

        self.encoder_config = args.encoder
        self.vocabulary = args.decoder.vocabulary

        self.encoder = Conformer(args.encoder)
        self.decoder = ConvASRDecoder(args.decoder)

    def decode(self, mel: mx.array) -> list[AlignedResult]:
        """Greedy CTC decoding; handles batches and single input."""
        batch_size: int = mel.shape[0]
        if len(mel.shape) == 2:
            batch_size = 1
            mel = mx.expand_dims(mel, 0)

        batch_features, lengths = self.encoder(mel)
        logits = self.decoder(batch_features)
        mx.eval(logits, lengths)

        results = []
        for b in range(batch_size):
            features_len = int(lengths[b])
            predictions = logits[b, :features_len]
            best_tokens = mx.argmax(predictions, axis=1)

            hypothesis: list[AlignedToken] = []
            token_boundaries: list[tuple[int, None]] = []
            prev_token = -1

            for t, token_id in enumerate(best_tokens):
                token_idx = int(token_id)

                if token_idx == len(self.vocabulary):
                    continue

                if token_idx == prev_token:
                    continue

                if prev_token != -1:
                    token_start_time = (
                        token_boundaries[-1][0]
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_end_time = (
                        t
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_duration = token_end_time - token_start_time

                    hypothesis.append(
                        AlignedToken(
                            prev_token,
                            start=token_start_time,
                            duration=token_duration,
                            text=_decode_token(prev_token, self.vocabulary),
                        )
                    )

                token_boundaries.append((t, None))
                prev_token = token_idx

            if prev_token != -1:
                last_non_blank = features_len - 1
                for t in range(features_len - 1, token_boundaries[-1][0], -1):
                    if int(best_tokens[t]) != len(self.vocabulary):
                        last_non_blank = t
                        break

                token_start_time = (
                    token_boundaries[-1][0]
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_end_time = (
                    (last_non_blank + 1)
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_duration = token_end_time - token_start_time

                hypothesis.append(
                    AlignedToken(
                        prev_token,
                        start=token_start_time,
                        duration=token_duration,
                        text=_decode_token(prev_token, self.vocabulary),
                    )
                )

            result = sentences_to_result(tokens_to_sentences(hypothesis))
            results.append(result)

        return results


class ParakeetTDTCTC(ParakeetTDT):
    """TDT model with an auxiliary CTC decoder (generate always uses TDT)."""

    def __init__(self, args: ParakeetTDTCTCArgs) -> None:
        super().__init__(args)
        self.ctc_decoder = ConvASRDecoder(args.aux_ctc.decoder)


def _build_model(config: dict) -> Model:
    """Instantiate the appropriate Parakeet model variant from a config dict."""
    if (
        config.get("target")
        == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is not None
    ):
        cfg = from_dict(ParakeetTDTArgs, config)
        model = ParakeetTDT(cfg)
    elif (
        config.get("target")
        == "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models"
        ".EncDecHybridRNNTCTCBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is not None
    ):
        cfg = from_dict(ParakeetTDTCTCArgs, config)
        model = ParakeetTDTCTC(cfg)
    elif (
        config.get("target")
        == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is None
    ):
        cfg = from_dict(ParakeetRNNTArgs, config)
        model = ParakeetRNNT(cfg)
    elif (
        config.get("target")
        == "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE"
    ):
        cfg = from_dict(ParakeetCTCArgs, config)
        model = ParakeetCTC(cfg)
    else:
        raise ValueError("Model is not supported yet!")

    return model


def load_model(
    model_id: str,
    *,
    tqdm_class: type | None = None,
) -> Model:
    """Download (if needed) and load a Parakeet model from HuggingFace Hub."""
    downloaded = snapshot_download(  # type: ignore[call-overload]
        model_id,
        allow_patterns=["*.json", "*.safetensors"],
        tqdm_class=tqdm_class,
    )
    model_path = Path(str(downloaded))
    config = json.loads((model_path / "config.json").read_text())
    model = _build_model(config)

    weight_files = list(model_path.glob("*.safetensors"))
    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        loaded: dict[str, mx.array] = mx.load(str(wf))  # type: ignore[assignment]
        weights.update(loaded)

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()  # prevents layernorm not computing correctly on inference
    return model
