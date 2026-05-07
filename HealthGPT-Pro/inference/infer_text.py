"""
inference/infer_text.py
-----------------------
Text-only medical QA inference with HealthGPT-Pro.

Usage:
    python inference/infer_text.py \
        --model lintw/HealthGPT-Pro-4B \
        --question "Explain the key symptoms and common risk factors of pneumonia."

    # Or pipe a list of questions from a file:
    python inference/infer_text.py \
        --model lintw/HealthGPT-Pro-8B \
        --question_file questions.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.utils import decode_output, load_model


def run_text_inference(
    model,
    processor,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Run a single text-only medical QA query.

    Args:
        model: Loaded Qwen3VLForConditionalGeneration model.
        processor: Corresponding AutoProcessor.
        question: The medical question string.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy decoding).

    Returns:
        Generated answer string.
    """
    messages = [
        {
            "role": "user",
            "content": [
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
        description="HealthGPT-Pro — Text-Only Medical QA Inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lintw/HealthGPT-Pro-4B",
        help="HuggingFace model ID or local path "
             "(e.g., 'lintw/HealthGPT-Pro-4B', 'lintw/HealthGPT-Pro-8B')",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="A single medical question to ask the model.",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        default=None,
        help="Path to a plain-text file with one question per line.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. 0.0 = greedy (default).",
    )
    args = parser.parse_args()

    if args.question is None and args.question_file is None:
        parser.error("Provide at least one of --question or --question_file.")

    # Collect questions
    questions: list[str] = []
    if args.question:
        questions.append(args.question)
    if args.question_file:
        with open(args.question_file, "r", encoding="utf-8") as f:
            questions.extend(line.strip() for line in f if line.strip())

    # Load model
    model, processor = load_model(model_id=args.model)

    # Run inference
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"[Question {i}] {q}")
        print(f"{'='*60}")
        answer = run_text_inference(
            model, processor, q,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"[Answer]\n{answer}")


if __name__ == "__main__":
    main()
