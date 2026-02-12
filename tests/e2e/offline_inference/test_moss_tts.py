"""Offline inference tests for MOSS-TTS model."""

import sys
from pathlib import Path

import pytest
import torch
from vllm import SamplingParams

from tests.utils import hardware_test
from vllm_omni import Omni  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "OpenMOSS-Team/MOSS-TTS"
STAGE_CONFIG = "vllm_omni/model_executor/stage_configs/moss_tts.yaml"

# Default sampling params for TTS
DEFAULT_SAMPLING_PARAMS = SamplingParams(
    temperature=0.9,
    top_p=1.0,
    top_k=50,
    max_tokens=2048,
    seed=42,
    detokenize=False,
)


def build_inputs(text: str, language: str = "en", instruction: str | None = None) -> dict:
    """Build Omni request payload for MOSS-TTS."""
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    additional = {
        "text": [text],
        "language": [language],
        "max_new_tokens": [1000],
    }
    if instruction:
        additional["instruct"] = [instruction]
    return {
        "prompt": prompt,
        "additional_information": additional,
    }


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_english():
    """Test MOSS-TTS English text-to-speech generation."""
    omni = Omni(model=MODEL_NAME, stage_configs_path=STAGE_CONFIG)

    text = "Hello, this is a test of the MOSS text to speech system."
    inputs = build_inputs(text, language="en")

    omni_generator = omni.generate(inputs, [DEFAULT_SAMPLING_PARAMS])

    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            # Access audio from outputs[0].multimodal_output
            multimodal_out = output.outputs[0].multimodal_output
            assert multimodal_out is not None

            audio = multimodal_out.get("audio")
            sr = multimodal_out.get("sr")

            assert audio is not None
            assert isinstance(audio, torch.Tensor)
            assert audio.numel() > 0
            assert sr.item() == 24000


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_chinese():
    """Test MOSS-TTS Chinese text-to-speech generation."""
    omni = Omni(model=MODEL_NAME, stage_configs_path=STAGE_CONFIG)

    text = "你好，这是一个语音合成测试。"
    inputs = build_inputs(text, language="zh")

    omni_generator = omni.generate(inputs, [DEFAULT_SAMPLING_PARAMS])

    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            multimodal_out = output.outputs[0].multimodal_output
            assert multimodal_out is not None

            audio = multimodal_out.get("audio")
            sr = multimodal_out.get("sr")

            assert audio is not None
            assert isinstance(audio, torch.Tensor)
            assert audio.numel() > 0
            assert sr.item() == 24000


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_moss_tts_with_instruction():
    """Test MOSS-TTS with style instruction."""
    omni = Omni(model=MODEL_NAME, stage_configs_path=STAGE_CONFIG)

    text = "Good morning everyone."
    instruction = "Speak in a cheerful and energetic tone."
    inputs = build_inputs(text, language="en", instruction=instruction)

    omni_generator = omni.generate(inputs, [DEFAULT_SAMPLING_PARAMS])

    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            multimodal_out = output.outputs[0].multimodal_output
            assert multimodal_out is not None

            audio = multimodal_out.get("audio")
            sr = multimodal_out.get("sr")

            assert audio is not None
            assert isinstance(audio, torch.Tensor)
            assert audio.numel() > 0
            assert sr.item() == 24000
