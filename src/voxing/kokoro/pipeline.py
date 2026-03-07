"""Kokoro TTS pipeline: G2P, text chunking, voice loading, inference.

Vendored from mlx-audio (https://github.com/Blaizzy/mlx-audio).
"""

import logging
import re
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from misaki import en, espeak

from voxing.kokoro.voice import load_voice_tensor

ALIASES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "pt": "p",
    "ja": "j",
    "zh": "z",
}

LANG_CODES = dict(
    a="American English",
    b="British English",
    e="es",
    f="fr-fr",
    h="hi",
    i="it",
    p="pt-br",
    j="Japanese",
    z="Mandarin Chinese",
)


class KokoroPipeline:
    """Language-aware G2P and voice management for Kokoro TTS."""

    def __init__(
        self,
        lang_code: str,
        model: nn.Module,
        repo_id: str,
        trf: bool = False,
    ):
        lang_code = lang_code.lower()
        lang_code = ALIASES.get(lang_code, lang_code)
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        self.repo_id = repo_id
        if repo_id is None:
            raise ValueError("repo_id is required to load voices")
        self.model = model
        self.voices: dict[str, mx.array] = {}
        if lang_code in "ab":
            try:
                fallback = espeak.EspeakFallback(british=lang_code == "b")
            except Exception as e:
                logging.warning("EspeakFallback not Enabled: OOD words will be skipped")
                logging.warning({str(e)})
                fallback = None
            self.g2p = en.G2P(
                trf=trf, british=lang_code == "b", fallback=fallback, unk=""
            )
        elif lang_code == "j":
            try:
                from misaki import ja

                self.g2p = ja.JAG2P()
            except ImportError:
                logging.error(
                    "You need to `pip install misaki[ja]` to use lang_code='j'"
                )
                raise
        elif lang_code == "z":
            try:
                from misaki import zh

                self.g2p = zh.ZHG2P()
            except ImportError:
                logging.error(
                    "You need to `pip install misaki[zh]` to use lang_code='z'"
                )
                raise
        else:
            language = LANG_CODES[lang_code]
            logging.warning(
                f"Using EspeakG2P(language='{language}'). Chunking logic not yet implemented, so long texts may be truncated unless you split them with '\\n'."
            )
            self.g2p = espeak.EspeakG2P(language=language)

    def load_single_voice(self, voice: str) -> mx.array:
        """Load a single voice pack from cache or HuggingFace."""
        if voice in self.voices:
            return self.voices[voice]

        if voice.endswith(".safetensors"):
            f = voice
        else:
            try:
                local_dir = Path(
                    snapshot_download(
                        repo_id=self.repo_id,
                        allow_patterns=[f"voices/{voice}.safetensors"],
                        local_files_only=True,
                    )
                )
                local_voice = local_dir / "voices" / f"{voice}.safetensors"
                if local_voice.exists():
                    f = str(local_voice)
                else:
                    raise FileNotFoundError
            except (FileNotFoundError, Exception):
                local_dir = Path(
                    snapshot_download(
                        repo_id=self.repo_id,
                        allow_patterns=[f"voices/{voice}.safetensors"],
                    )
                )
                f = str(local_dir / "voices" / f"{voice}.safetensors")

            if not voice.startswith(self.lang_code):
                v = LANG_CODES.get(voice, voice)
                p = LANG_CODES.get(self.lang_code, self.lang_code)
                logging.warning(
                    f"Language mismatch, loading {v} voice into {p} pipeline."
                )

        pack = load_voice_tensor(f)
        self.voices[voice] = pack
        return pack

    def load_voice(self, voice: str, delimiter: str = ",") -> mx.array:
        """Load one or more voices, averaging if multiple."""
        if voice in self.voices:
            return self.voices[voice]
        logging.debug(f"Loading voice: {voice}")
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        if len(packs) == 1:
            return packs[0]
        self.voices[voice] = mx.mean(mx.stack(packs), axis=0)
        return self.voices[voice]

    @classmethod
    def tokens_to_ps(cls, tokens: list[en.MToken]) -> str:
        return "".join(
            t.phonemes + (" " if t.whitespace else "") for t in tokens
        ).strip()

    @classmethod
    def waterfall_last(
        cls,
        tokens: list[en.MToken],
        next_count: int,
        waterfall: list[str] = ["!.?…", ":;", ",—"],
        bumps: list[str] = [")", "\u201d"],
    ) -> int:
        for w in waterfall:
            z = next(
                (
                    i
                    for i, t in reversed(list(enumerate(tokens)))
                    if t.phonemes in set(w)
                ),
                None,
            )
            if z is None:
                continue
            z += 1
            if z < len(tokens) and tokens[z].phonemes in bumps:
                z += 1
            if next_count - len(cls.tokens_to_ps(tokens[:z])) <= 510:
                return z
        return len(tokens)

    @classmethod
    def tokens_to_text(cls, tokens: list[en.MToken]) -> str:
        return "".join(t.text + t.whitespace for t in tokens).strip()

    def en_tokenize(
        self, tokens: list[en.MToken]
    ) -> Generator[tuple[str, str, list[en.MToken]], None, None]:
        """Chunk English tokens into segments of <= 510 phonemes."""
        tks: list[en.MToken] = []
        pcount = 0
        for t in tokens:
            t.phonemes = "" if t.phonemes is None else t.phonemes.replace("\u027e", "T")
            next_ps = t.phonemes + (" " if t.whitespace else "")
            next_pcount = pcount + len(next_ps.rstrip())
            if next_pcount > 510:
                z = KokoroPipeline.waterfall_last(tks, next_pcount)
                text = KokoroPipeline.tokens_to_text(tks[:z])
                ps = KokoroPipeline.tokens_to_ps(tks[:z])
                yield text, ps, tks[:z]
                tks = tks[z:]
                pcount = len(KokoroPipeline.tokens_to_ps(tks))
                if not tks:
                    next_ps = next_ps.lstrip()
            tks.append(t)
            pcount += len(next_ps)
        if tks:
            text = KokoroPipeline.tokens_to_text(tks)
            ps = KokoroPipeline.tokens_to_ps(tks)
            yield "".join(text).strip(), "".join(ps).strip(), tks

    @classmethod
    def infer(
        cls,
        model: nn.Module,
        ps: str,
        pack: mx.array,
        speed: float = 1,
    ) -> object:
        return model(ps, pack[len(ps) - 1], speed, return_output=True)

    def __call__(
        self,
        text: Union[str, list[str]],
        voice: Optional[str] = None,
        speed: float = 1,
        split_pattern: Optional[str] = r"\n+",
    ) -> Generator["KokoroPipeline.Result", None, None]:
        if voice is None:
            raise ValueError(
                'Specify a voice: pipeline(text="Hello world!", voice="af_heart")'
            )
        pack = self.load_voice(voice) if self.model else None
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]

        for graphemes_index, graphemes in enumerate(text):
            if not graphemes.strip():
                continue

            if self.lang_code in "ab":
                _, tokens = self.g2p(graphemes)
                for gs, ps, tks in self.en_tokenize(tokens):  # ty: ignore
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logging.warning(
                            f"Unexpected len(ps) == {len(ps)} > 510, truncating"
                        )
                        ps = ps[:510]
                    output = (
                        KokoroPipeline.infer(self.model, ps, pack, speed)  # ty: ignore
                        if self.model
                        else None
                    )
                    yield self.Result(
                        graphemes=gs,
                        phonemes=ps,
                        tokens=tks,
                        output=output,
                        text_index=graphemes_index,
                    )
            else:
                chunk_size = 400
                chunks = []
                sentences = re.split(r"([.!?]+)", graphemes)
                current_chunk = ""
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if not chunks:
                    chunks = [
                        graphemes[i : i + chunk_size]
                        for i in range(0, len(graphemes), chunk_size)
                    ]

                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    result = self.g2p(chunk)
                    ps = result[0] if isinstance(result, tuple) else result
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logging.warning(f"Truncating len(ps) == {len(ps)} > 510")
                        ps = ps[:510]
                    output = (
                        KokoroPipeline.infer(self.model, ps, pack, speed)  # ty: ignore
                        if self.model
                        else None
                    )
                    yield self.Result(
                        graphemes=chunk,
                        phonemes=ps,
                        output=output,
                        text_index=graphemes_index,
                    )

    @dataclass
    class Result:
        graphemes: str
        phonemes: str
        tokens: Optional[list[en.MToken]] = None
        output: object = None
        text_index: Optional[int] = None

        @property
        def audio(self) -> Optional[mx.array]:
            return None if self.output is None else self.output.audio  # ty: ignore

        @property
        def pred_dur(self) -> Optional[mx.array]:
            return None if self.output is None else self.output.pred_dur  # ty: ignore

        def __iter__(self):
            yield self.graphemes
            yield self.phonemes
            yield self.audio

        def __getitem__(self, index: int) -> object:
            return [self.graphemes, self.phonemes, self.audio][index]

        def __len__(self) -> int:
            return 3
