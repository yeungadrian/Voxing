# Voxing

A voice and text assistant that runs entirely on-device, built as a terminal UI for macOS with Apple Silicon.

Uses [MLX](https://github.com/ml-explore/mlx) for inference and [Textual](https://github.com/Textualize/textual) for the interface.

## Requirements

- macOS with Apple Silicon (M1+)

## Install

```bash
uvx voxing
```

## Models

| Task | Default Model |
|------|---------------|
| Speech-to-Text | [parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) |
| LLM | [LFM2.5-1.2B-Instruct-MLX-8bit](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit) |

## Usage

| Command | Description |
|---------|-------------|
| `/transcribe` | Record voice input with real-time waveform, stops on silence |
| `/settings` | Configure models, audio, and LLM parameters |
| `/clear` | Clear chat history |
| `/help` | Show available commands |
