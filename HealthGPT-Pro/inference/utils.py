"""
inference/utils.py
------------------
Shared utilities for HealthGPT-Pro inference scripts.

Provides:
  - load_model()    : load the model and processor from a local path or HF Hub
  - decode_output() : trim and decode generated token IDs into text
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_model(
    model_id: str = "lintw/HealthGPT-Pro-4B",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    device_map: str = "auto",
) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    """Load HealthGPT-Pro model and its processor.

    Args:
        model_id: HuggingFace model ID or local path.
                  Options: "lintw/HealthGPT-Pro-4B", "lintw/HealthGPT-Pro-8B".
        dtype: Torch dtype for model weights. bfloat16 recommended.
        attn_implementation: Attention backend. Use "flash_attention_2" when
                             FlashAttention is installed, otherwise "eager".
        device_map: Device placement strategy passed to from_pretrained.

    Returns:
        (model, processor) tuple ready for inference.
    """
    print(f"[HealthGPT-Pro] Loading model: {model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[HealthGPT-Pro] Model loaded on device(s): {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    return model, processor


def decode_output(
    generated_ids: torch.Tensor,
    input_ids: torch.Tensor,
    processor: AutoProcessor,
) -> str:
    """Trim prompt tokens from generated output and decode to text.

    Args:
        generated_ids: Full generated token IDs (batch_size, seq_len).
        input_ids: Input prompt token IDs (batch_size, prompt_len).
        processor: The processor used for tokenization.

    Returns:
        Decoded output string (first element of the batch).
    """
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    texts = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return texts[0]
