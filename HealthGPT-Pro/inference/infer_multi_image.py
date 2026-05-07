"""
inference/infer_multi_image.py
------------------------------
Multi-image medical inference with HealthGPT-Pro.

Pass two or more medical images together with a comparative or
synthesis question. Typical use cases:
  - Before / after treatment comparison
  - Cross-modality analysis (e.g., CT vs MRI of the same anatomy)
  - Serial imaging review (multiple time-point scans)

Usage:
    python inference/infer_multi_image.py \
        --model lintw/HealthGPT-Pro-4B \
        --images examples/image_1.png examples/image_2.png \
        --question "Compare these two medical images and summarize the key differences."

    python inference/infer_multi_image.py \
        --model lintw/HealthGPT-Pro-8B \
        --images scan_t0.png scan_t1.png scan_t2.png \
        --question "Describe the progression of findings across these three serial scans."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.utils import decode_output, load_model


def run_multi_image_inference(
    model,
    processor,
    image_paths: list[str],
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Run inference on multiple 2D medical images simultaneously.

    Args:
        model: Loaded Qwen3VLForConditionalGeneration model.
        processor: Corresponding AutoProcessor.
        image_paths: List of paths to input image files.
        question: Clinical question referencing the images.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).

    Returns:
        Generated response string.
    """
    if len(image_paths) < 2:
        raise ValueError("Provide at least 2 images for multi-image inference.")

    # Build content list: all images first, then the question
    content: list[dict] = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(p).convert("RGB")
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]

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
        description="HealthGPT-Pro — Multi-Image Medical Inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lintw/HealthGPT-Pro-4B",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Paths to two or more medical image files (PNG, JPG, etc.).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Compare these medical images and summarize the key differences.",
        help="Clinical question about the images.",
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

    if len(args.images) < 2:
        parser.error("Please provide at least 2 images with --images.")

    model, processor = load_model(model_id=args.model)

    print(f"\n{'='*60}")
    for i, img in enumerate(args.images, 1):
        print(f"[Image {i:2d}] {img}")
    print(f"[Question] {args.question}")
    print(f"{'='*60}")

    answer = run_multi_image_inference(
        model, processor,
        image_paths=args.images,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"[Answer]\n{answer}")


if __name__ == "__main__":
    main()
