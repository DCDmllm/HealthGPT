"""
inference/infer_single_image.py
-------------------------------
Single 2D medical image inference with HealthGPT-Pro.

Accepts any common medical image format (PNG, JPG, DICOM-exported PNG, etc.)
and a free-form clinical question.

Usage:
    python inference/infer_single_image.py \
        --model lintw/HealthGPT-Pro-4B \
        --image examples/chest_xray.png \
        --question "Describe the main radiological findings in this image."

    python inference/infer_single_image.py \
        --model lintw/HealthGPT-Pro-8B \
        --image examples/fundus.png \
        --question "Are there any signs of diabetic retinopathy? If so, describe the severity."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.utils import decode_output, load_model


def run_single_image_inference(
    model,
    processor,
    image_path: str,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Run inference on a single 2D medical image.

    Args:
        model: Loaded Qwen3VLForConditionalGeneration model.
        processor: Corresponding AutoProcessor.
        image_path: Path to the input image file (PNG / JPG / etc.).
        question: Clinical question about the image.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).

    Returns:
        Generated response string.
    """
    # Validate image
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image to verify it is valid (also converts non-RGB modes)
    image = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs: dict = {"max_new_tokens": max_new_tokens}
    if temperature > 0.0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    return decode_output(generated_ids, inputs.input_ids, processor)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HealthGPT-Pro — Single 2D Medical Image Inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lintw/HealthGPT-Pro-4B",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input medical image (PNG, JPG, etc.).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe the main findings in this medical image.",
        help="Clinical question about the image.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. 0.0 = greedy (default).",
    )
    args = parser.parse_args()

    model, processor = load_model(model_id=args.model)

    print(f"\n{'='*60}")
    print(f"[Image   ] {args.image}")
    print(f"[Question] {args.question}")
    print(f"{'='*60}")

    answer = run_single_image_inference(
        model, processor,
        image_path=args.image,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"[Answer]\n{answer}")


if __name__ == "__main__":
    main()
