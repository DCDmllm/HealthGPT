"""
inference/infer_3d_volume.py
----------------------------
3D CT / MRI volume inference with HealthGPT-Pro.

The model ingests 3D volumes by converting them into a sequence of 2D
axial slice frames sent as a "video" input.  Input must be a NumPy
`.npy` file with shape (D, H, W) where:
  - D = number of axial slices (depth)
  - H = image height  (pixels)
  - W = image width   (pixels)

Pixel values should be normalized to [0, 1] (float32). If your volume
uses Hounsfield Units (HU) or raw DICOM values, apply window/level
normalization before saving to .npy.

Usage:
    python inference/infer_3d_volume.py \
        --model lintw/HealthGPT-Pro-4B \
        --volume examples/ct_volume.npy \
        --question "Analyze this CT volume and summarize the main findings."

    # Customize the number of sampled frames and FPS:
    python inference/infer_3d_volume.py \
        --model lintw/HealthGPT-Pro-8B \
        --volume examples/mri_volume.npy \
        --num_frames 10 \
        --sample_fps 2.0 \
        --question "Are there any abnormalities visible in this MRI volume?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.utils import decode_output, load_model


def load_volume_as_frames(
    volume_path: str,
    num_frames: int = 10,
) -> list[Image.Image]:
    """Load a 3D medical volume (.npy) and sample it as a list of RGB PIL frames.

    Slices are sampled uniformly along the depth axis, skipping the first
    and last slices which often contain incomplete anatomy.

    Args:
        volume_path: Path to a .npy file of shape (D, H, W), values in [0, 1].
        num_frames: Number of axial slices to sample for the video sequence.

    Returns:
        List of PIL.Image objects in RGB mode.
    """
    path = Path(volume_path)
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {volume_path}")
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file, got: {path.suffix}")

    volume = np.load(str(path))  # (D, H, W), float in [0, 1]
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array (D, H, W), got shape: {volume.shape}"
        )

    # Normalize and convert to uint8
    v_min, v_max = volume.min(), volume.max()
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    ct_u8 = np.clip(volume * 255, 0, 255).astype(np.uint8)

    depth = ct_u8.shape[0]
    # Uniformly sample `num_frames` slices, skipping first/last
    indices = np.linspace(1, depth - 2, num_frames, dtype=int)

    frames: list[Image.Image] = []
    for i in indices:
        rgb = np.stack([ct_u8[i]] * 3, axis=-1)   # (H, W, 3)
        frames.append(Image.fromarray(rgb, mode="RGB"))

    return frames


def run_3d_volume_inference(
    model,
    processor,
    volume_path: str,
    question: str,
    num_frames: int = 10,
    sample_fps: float = 2.0,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Run inference on a 3D medical volume (CT or MRI).

    Args:
        model: Loaded Qwen3VLForConditionalGeneration model.
        processor: Corresponding AutoProcessor.
        volume_path: Path to the 3D volume .npy file.
        question: Clinical question about the volume.
        num_frames: Number of axial frames to sample.
        sample_fps: Frames-per-second metadata for the video token.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).

    Returns:
        Generated response string.
    """
    frames = load_volume_as_frames(volume_path, num_frames=num_frames)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "sample_fps": sample_fps,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Build text prompt (no tokenization yet — needed for process_vision_info)
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Extract vision inputs (images / videos) from the messages
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=text_prompt,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
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
        description="HealthGPT-Pro — 3D Medical Volume (CT/MRI) Inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lintw/HealthGPT-Pro-4B",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--volume",
        type=str,
        required=True,
        help="Path to the 3D volume (.npy file, shape D×H×W, values in [0,1]).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Analyze this medical volume and summarize the main findings.",
        help="Clinical question about the 3D volume.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of axial slices to sample from the volume (default: 10).",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=2.0,
        help="Frames-per-second for the video input token (default: 2.0).",
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
    print(f"[Volume  ] {args.volume}")
    print(f"[Frames  ] {args.num_frames}  |  FPS: {args.sample_fps}")
    print(f"[Question] {args.question}")
    print(f"{'='*60}")

    answer = run_3d_volume_inference(
        model, processor,
        volume_path=args.volume,
        question=args.question,
        num_frames=args.num_frames,
        sample_fps=args.sample_fps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"[Answer]\n{answer}")


if __name__ == "__main__":
    main()
