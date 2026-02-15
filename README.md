# Voxing

A voice and text assistant that runs entirely on-device, built as a terminal UI for macOS with Apple Silicon.

Uses [MLX](https://github.com/ml-explore/mlx) for inference and [Textual](https://github.com/Textualize/textual) for the interface.

## Requirements

- macOS with Apple Silicon (M1+)
- 16 GB RAM (app uses ~4-5 GB)

## Install

```bash
uvx --prerelease allow voxing
```

`--prerelease allow` is needed because `mlx-audio` depends on a prerelease version of `transformers`.

## Models

| Task | Default Model |
|------|---------------|
| Speech-to-Text | [parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) |
| LLM | [LFM2.5-1.2B-Instruct-MLX-8bit](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit) |
| Text-to-Speech | [Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) |

Models can be swapped at runtime with the `/model` command.

## Usage

| Command | Description |
|---------|-------------|
| `/record` | Record voice input (stops on silence) |
| `/transcribe` | Extended recording (up to 3 min), copies to clipboard |
| `/model` | Switch STT, LLM, or TTS model |
| `/tts` | Toggle text-to-speech |
| `Esc` (x2) | Cancel active operation |
