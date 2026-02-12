# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-Omni wrapper for the MOSS-TTS delay-pattern model."""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_moss_tts import MossTTSDelayConfig
from .modeling_moss_tts import MossTTSDelayModel

logger = init_logger(__name__)


class MossTTSModelForGeneration(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_path = vllm_config.model_config.model

        try:
            import flash_attn  # noqa: F401

            attn_kwargs = {"attn_implementation": "flash_attention_2"}
        except ImportError:
            logger.warning("Flash-Attn is not installed. Using default PyTorch attention implementation.")
            attn_kwargs = {}

        self.model = MossTTSModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            **attn_kwargs,
        )

        self.have_multimodal_outputs = True
        self.vllm_config = vllm_config

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """
        Forward pass for TTS generation model.

        Args:
            input_ids: Input token IDs (not used directly; text comes via kwargs)
            positions: Position IDs (not used for TTS, but required by runner)
            intermediate_tensors: Intermediate tensors for pipeline parallelism (not used)
            inputs_embeds: Input embeddings (not used for TTS, but required by runner)
            **kwargs: Additional arguments including runtime_additional_information

        Returns:
            OmniOutput: Contains multimodal outputs with audio tensors
        """
        runtime_additional_information = kwargs.get("runtime_additional_information", [{}])
        if isinstance(runtime_additional_information, list) and len(runtime_additional_information) > 0:
            runtime_additional_information = runtime_additional_information[0]

        text = runtime_additional_information.pop("text", [""])[0]
        language = runtime_additional_information.pop("language", [None])[0]
        instruction = runtime_additional_information.pop("instruct", [None])[0]
        reference = runtime_additional_information.pop("reference", [None])[0]

        for key, value in runtime_additional_information.items():
            if isinstance(value, list) and len(value) > 0:
                runtime_additional_information[key] = value[0]

        # During profile/warmup runs, text is empty and no real inputs exist.
        # Return dummy audio immediately to avoid issues with short generation.
        if not text:
            logger.info("Profile run detected (empty text). Returning dummy audio.")
            dummy_audio = torch.zeros(2400, dtype=torch.float32)  # 0.1s of silence at 24kHz
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": dummy_audio,
                    "sr": torch.tensor(24000, dtype=torch.int),
                },
            )

        result = self.model.generate_speech(
            text=text,
            language=language,
            instruction=instruction,
            reference=reference,
            **runtime_additional_information,
        )

        return self._make_omni_output(result)

    @staticmethod
    def _make_omni_output(model_outputs: tuple | OmniOutput) -> OmniOutput:
        """
        Make an OmniOutput object from model outputs.

        Args:
            model_outputs: Model outputs (either OmniOutput or tuple of (audio_tensors, sr))

        Returns:
            OmniOutput: Wrapped output with audio tensor and sample rate
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
            audio_tensors, sr = model_outputs

            if isinstance(audio_tensors, list) and len(audio_tensors) > 0:
                audio_tensor = audio_tensors[0]
                if not isinstance(audio_tensor, torch.Tensor):
                    audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
            elif isinstance(audio_tensors, torch.Tensor):
                audio_tensor = audio_tensors
            else:
                audio_tensor = torch.zeros(1, dtype=torch.float32)

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": audio_tensor,
                    "sr": torch.tensor(sr, dtype=torch.int),
                },
            )

        raise ValueError(f"Unsupported model_outputs type: {type(model_outputs)}")

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        """
        Create empty intermediate tensors for pipeline parallelism.

        For TTS generation models, pipeline parallelism is typically not used,
        so this returns an empty dict. However, this method is required by the
        runner infrastructure.

        Args:
            batch_size: Batch size for the intermediate tensors
            dtype: Data type for the tensors
            device: Device for the tensors

        Returns:
            IntermediateTensors: Empty dict (no PP support for TTS models)
        """
        return IntermediateTensors({})

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Embed input token IDs into embeddings.

        This method is called by the runner when inputs_embeds are needed.
        For TTS models, we typically work with input_ids directly, but this
        method provides a fallback for cases where embeddings are required.

        Args:
            input_ids: Input token IDs
            multimodal_embeddings: Optional multimodal embeddings (not used for TTS)
            is_multimodal: Optional mask indicating multimodal tokens (not used for TTS)
            **kwargs: Additional arguments

        Returns:
            torch.Tensor: Embedded representations of input_ids
        """
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], 1024),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )

    def embed_multimodal(self, **kwargs: Any) -> Any:
        """
        Embed multimodal inputs (e.g., images, audio).

        For TTS models, this is typically not used as they work with text input_ids.
        This method provides a stub to satisfy the interface.

        Args:
            **kwargs: Multimodal input arguments

        Returns:
            None: TTS models don't use multimodal embeddings
        """
        return None

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load weights into the wrapped HF model."""
        loaded_params: set[str] = set()
        for name, _loaded_weight in weights:
            loaded_params.add(name)
        return loaded_params

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        """Non-autoregressive TTS models do not compute token logits."""
        return None


class MossTTSModel:
    """
    A wrapper for MOSS-TTS models that provides:
      - from_pretrained() initialization via AutoModel + MossTTSDelayProcessor
      - generate_speech() API for text-to-speech generation
      - consistent output: (wavs: List[torch.Tensor], sample_rate: int)
    """

    def __init__(
        self,
        model: MossTTSDelayModel,
        processor: Any,
        config: MossTTSDelayConfig,
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.sampling_rate = int(config.sampling_rate)

        self.device: torch.device = getattr(model, "device", None)  # type: ignore[assignment]
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "MossTTSModel":
        """
        Load a MOSS-TTS model and its processor in HuggingFace ``from_pretrained`` style.

        This method:
          1) Registers config/model with AutoConfig/AutoModel.
          2) Loads the model via AutoModel.from_pretrained(...).
          3) Loads a text tokenizer and audio tokenizer separately.
          4) Constructs MossTTSDelayProcessor manually (avoids sibling module dependency).

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory of the model.
            **kwargs:
                Forwarded to AutoModel.from_pretrained(...).

        Returns:
            MossTTSModel: Wrapper instance containing model, processor.
        """
        AutoConfig.register("moss_tts_delay", MossTTSDelayConfig)
        AutoModel.register(MossTTSDelayConfig, MossTTSDelayModel)

        codec_path = kwargs.pop("codec_path", "OpenMOSS-Team/MOSS-Audio-Tokenizer")

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, MossTTSDelayModel):
            raise TypeError(f"AutoModel returned {type(model).__name__}, expected MossTTSDelayModel.")

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        # Audio tokenizer must be float32 for decoding compatibility
        audio_tokenizer = AutoModel.from_pretrained(codec_path, trust_remote_code=True, torch_dtype=torch.float32)

        from .processing_moss_tts import MossTTSDelayProcessor

        processor = MossTTSDelayProcessor(
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            model_config=model.config,
        )

        return cls(model=model, processor=processor, config=model.config)

    def _ensure_audio_tokenizer_device(self) -> None:
        """Move the audio tokenizer to the model's device if needed."""
        at = getattr(self.processor, "audio_tokenizer", None)
        if at is None:
            return
        try:
            at_device = next(at.parameters()).device
        except StopIteration:
            return
        if at_device != self.device:
            self.processor.audio_tokenizer = at.to(self.device)

    @torch.no_grad()
    def generate_speech(
        self,
        text: str,
        language: str | None = None,
        instruction: str | None = None,
        reference: list | None = None,
        **kwargs: Any,
    ) -> tuple[list[torch.Tensor], int]:
        """
        Generate speech from text input.

        Args:
            text: The text to synthesize.
            language: Optional language hint (e.g. ``"en"``, ``"zh"``).
            instruction: Optional style/voice instruction.
            reference: Optional list of reference audio paths/URLs for voice cloning.
            **kwargs: Override generation hyper-parameters.

        Returns:
            Tuple[List[torch.Tensor], int]: (wavs, sample_rate)
        """
        self._ensure_audio_tokenizer_device()

        from .processing_moss_tts import MossTTSDelayProcessor

        user_msg = MossTTSDelayProcessor.build_user_message(
            text=text,
            language=language,
            instruction=instruction,
            reference=reference,
        )

        batch_feature = self.processor([user_msg])
        input_ids = batch_feature["input_ids"].to(self.device)
        attention_mask = batch_feature["attention_mask"].to(self.device)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        messages = self.processor.decode(output)

        wavs: list[torch.Tensor] = []
        if messages and messages[0] is not None:
            audio_list = messages[0].audio_codes_list
            if audio_list:
                for audio in audio_list:
                    if isinstance(audio, torch.Tensor):
                        wavs.append(audio.float().cpu())
                    else:
                        wavs.append(torch.tensor(audio, dtype=torch.float32))

        if not wavs:
            wavs = [torch.zeros(1, dtype=torch.float32)]

        return wavs, self.sampling_rate

    @staticmethod
    def _merge_generate_kwargs(**kwargs: Any) -> dict[str, Any]:
        """
        Merge user-provided generation arguments with defaults.

        Supports both MOSS-TTS-specific names (``text_temperature``, ``audio_top_p``, ...)
        and generic names (``temperature``, ``top_p``) that get broadcast to both heads.
        """
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)
        repetition_penalty = kwargs.pop("repetition_penalty", None)

        def _pick(specific_key: str, generic_val: Any, default: Any) -> Any:
            val = kwargs.pop(specific_key, None)
            if val is not None:
                return val
            if generic_val is not None:
                return generic_val
            return default

        merged: dict[str, Any] = {
            "max_new_tokens": int(kwargs.pop("max_new_tokens", 1000)),
            "text_temperature": float(_pick("text_temperature", temperature, 1.5)),
            "text_top_p": float(_pick("text_top_p", top_p, 1.0)),
            "text_top_k": int(_pick("text_top_k", top_k, 50)),
            "audio_temperature": float(_pick("audio_temperature", temperature, 1.5)),
            "audio_top_p": float(_pick("audio_top_p", top_p, 0.8)),
            "audio_top_k": int(_pick("audio_top_k", top_k, 50)),
            "audio_repetition_penalty": float(_pick("audio_repetition_penalty", repetition_penalty, 1.0)),
        }
        return merged
