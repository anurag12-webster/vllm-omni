# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference demo for MOSS-TTS via vLLM Omni.

Generates WAV files from text prompts using the MOSS-TTS delay-pattern model.
Supports English and Chinese synthesis with optional voice instruction.
"""

import os
from pathlib import Path

import soundfile as sf

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni

MODEL = "OpenMOSS-Team/MOSS-TTS"

DEFAULT_STAGE_CONFIG = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "vllm_omni"
    / "model_executor"
    / "stage_configs"
    / "moss_tts.yaml"
)


def build_inputs(
    text: str,
    language: str = "en",
    instruction: str | None = None,
    max_new_tokens: int = 1000,
) -> dict:
    """Build a single Omni request payload for MOSS-TTS.

    Args:
        text: The text to synthesize.
        language: Language hint ("en", "zh", etc.).
        instruction: Optional style/voice instruction.
        max_new_tokens: Maximum number of generation tokens.

    Returns:
        dict: Ready-to-use Omni input dictionary.
    """
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    additional = {
        "text": [text],
        "language": [language],
        "max_new_tokens": [max_new_tokens],
    }
    if instruction:
        additional["instruct"] = [instruction]
    return {
        "prompt": prompt,
        "additional_information": additional,
    }


SAMPLES = {
    "english": {
        "text": "Hello, how are you today? This is a test of the MOSS text to speech system.",
        "language": "en",
    },
    "chinese": {
        "text": "你好，这是一个语音合成的测试。今天天气真不错。",
        "language": "zh",
    },
    "instruct": {
        "text": "I am so excited to share this news with you!",
        "language": "en",
        "instruction": "Speak with an enthusiastic and happy tone.",
    },
}


def main(args):
    """Run offline MOSS-TTS inference and save WAV output.

    Args:
        args: Parsed CLI arguments.
    """
    sample = SAMPLES[args.sample]
    inputs = build_inputs(
        text=sample["text"],
        language=sample["language"],
        instruction=sample.get("instruction"),
        max_new_tokens=args.max_new_tokens,
    )

    if args.use_batch_sample:
        inputs = [build_inputs(**s) for s in SAMPLES.values()]

    stage_config = args.stage_configs_path or DEFAULT_STAGE_CONFIG

    omni = Omni(
        model=MODEL,
        stage_configs_path=stage_config,
        stage_init_timeout=args.stage_init_timeout,
    )

    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=2048,
        seed=42,
        detokenize=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    omni_generator = omni.generate(inputs, [sampling_params])
    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            request_id = output.request_id
            audio_tensor = output.outputs[0].multimodal_output["audio"]
            sr = output.outputs[0].multimodal_output["sr"].item()

            audio_numpy = audio_tensor.float().detach().cpu().numpy()
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()

            wav_path = os.path.join(args.output_dir, f"output_{request_id}.wav")
            sf.write(wav_path, audio_numpy, samplerate=sr, format="WAV")
            print(f"Request {request_id}: saved {wav_path} ({sr} Hz)")


def parse_args():
    parser = FlexibleArgumentParser(
        description="Offline MOSS-TTS inference via vLLM Omni"
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="english",
        choices=SAMPLES.keys(),
        help="Which sample prompt to use (default: english).",
    )
    parser.add_argument(
        "--use-batch-sample",
        action="store_true",
        default=False,
        help="Run all sample prompts as a batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1000,
        help="Max generation tokens (default: 1000).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage config YAML. Defaults to built-in moss_tts.yaml.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Stage init timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Output directory for WAV files (default: output_audio).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
