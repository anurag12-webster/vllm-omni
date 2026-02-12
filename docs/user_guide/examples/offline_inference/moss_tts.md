# MOSS-TTS

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/moss_tts>.

This directory contains an offline demo for running MOSS-TTS with vLLM Omni. It builds text prompts and generates WAV files locally.

## Model Overview

[MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) is an 8B parameter text-to-speech model from OpenMOSS, built on Qwen3 with a delay-pattern multi-codebook architecture. It supports 20 languages including Chinese, English, German, Spanish, French, Japanese, Italian, Hebrew, Korean, Russian, Persian, Arabic, Polish, Portuguese, Czech, Danish, Swedish, Hungarian, Greek, and Turkish. Features include zero-shot voice cloning, Pinyin/IPA input, and duration control.

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

### Dependencies

```bash
pip install soundfile torchaudio
```

Flash Attention 2 is recommended for faster inference:

```bash
pip install flash-attn --no-build-isolation
```

## Quick Start

Run a single English sample:

```bash
python end2end.py
```

Generated audio files are saved to `output_audio/` by default.

## Usage

### English

```bash
python end2end.py --sample english
```

### Chinese

```bash
python end2end.py --sample chinese
```

### With Style Instruction

```bash
python end2end.py --sample instruct
```

### Batch Mode

Run all samples in one batch:

```bash
python end2end.py --use-batch-sample
```

## Notes

- The script auto-detects the built-in `moss_tts.yaml` stage config. Override with `--stage-configs-path` if needed.
- Audio output is 24 kHz mono WAV.
- The 8B model requires approximately 18 GB of GPU memory in bfloat16. A 40 GB+ GPU is recommended for comfortable headroom with vLLM's KV cache allocation.

## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/moss_tts/end2end.py"
    ``````
