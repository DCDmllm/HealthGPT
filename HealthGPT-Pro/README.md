<div align="center">

<h1>🩺 HealthGPT-Pro</h1>
<h3>A High-Performance Multimodal Large Language Model for Medical Understanding and Analysis</h3>

<p>
  <a href="https://lin-tianwei.github.io/healthgpt-pro.github.io/"><img src="https://img.shields.io/badge/🚀-Project_Page-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/lintw/HealthGPT-Pro-4B"><img src="https://img.shields.io/badge/🤗-HealthGPT--Pro--4B-yellow" alt="HF 4B"></a>
  <a href="https://huggingface.co/lintw/HealthGPT-Pro-8B"><img src="https://img.shields.io/badge/🤗-HealthGPT--Pro--8B-yellow" alt="HF 8B"></a>
  <a href="https://modelscope.cn/models/TianweiLin/HealthGPT-Pro-4B"><img src="https://img.shields.io/badge/🤖-ModelScope--4B-purple" alt="ModelScope 4B"></a>
  <a href="https://modelscope.cn/models/TianweiLin/HealthGPT-Pro-8B"><img src="https://img.shields.io/badge/🤖-ModelScope--8B-purple" alt="ModelScope 8B"></a>
</p>

</div>

## 🔥 News

* **[2026-05-07]** 🙏🙏🙏 Huge thanks to [mradermacher](https://huggingface.co/mradermacher?utm_source=chatgpt.com) for quantizing **HealthGPT-Pro-4B** and **HealthGPT-Pro-8B** into GGUF format: [HealthGPT-Pro-4B-GGUF](https://huggingface.co/mradermacher/HealthGPT-Pro-4B-GGUF?utm_source=chatgpt.com) and [HealthGPT-Pro-8B-GGUF](https://huggingface.co/mradermacher/HealthGPT-Pro-8B-GGUF?utm_source=chatgpt.com). With these releases, HealthGPT-Pro can now be accelerated and deployed in other inference engines using workflows similar to Qwen3-VL.


- **[2026-05-02]** 🎉🎉🎉 We release **HealthGPT-Pro-4B** and **HealthGPT-Pro-8B**! Both models are now publicly available on [Hugging Face](https://huggingface.co/lintw/HealthGPT-Pro-4B) and [ModelScope](https://modelscope.cn/models/TianweiLin/HealthGPT-Pro-4B). HealthGPT-Pro achieves state-of-the-art performance on both medical text and multimodal benchmarks.

## 🧭 Overview

**HealthGPT-Pro** is a state-of-the-art medical multimodal large language model (Med-MLLM) built on **Qwen3-VL**. It is designed for **medical text**, **2D medical image**, and **3D medical volume** understanding and analysis, delivering strong performance across a broad range of medical text-based and vision-language tasks.

### ✨ Core Features

- **Multimodal Input Support:** HealthGPT-Pro processes text, 2D images, and 3D volumetric data within a unified framework.
- **Efficient Training:** Achieves SoTA performance through a **two-stage training recipe** — **3M** samples for alignment and **10M** samples for supervised fine-tuning.
- **Strong Instruction Following:** Unlike many Med-MLLMs tuned solely on medical data, HealthGPT-Pro retains a substantial proportion of general data to preserve instruction-following capability.
- **Comprehensive Modality Coverage:**
  | # | Modality | # | Modality |
  |---|----------|---|----------|
  | 1 | Computed Tomography (CT) | 8 | Endoscopy |
  | 2 | Digital Photography | 9 | Microscopy |
  | 3 | Fundus Photography | 10 | X-ray Imaging |
  | 4 | Infrared Reflectance Imaging | 11 | Ultrasound Imaging |
  | 5 | Magnetic Resonance Imaging (MRI) | 12 | Histopathology |
  | 6 | Optical Coherence Tomography (OCT) | 13 | Colposcopy |
  | 7 | Dermoscopy | 14 | Medical Text |

> ⚠️ **Disclaimer:** This model is intended for **research use only**. It should **not** be used as a substitute for professional clinical judgment, diagnosis, or treatment.

## 📊 Performance

### 📝 Medical Text Benchmarks

| **Model** | **MMLU-Med** | **MMLU-Pro-Med** | **MMedBench** | **MedBullets** | **MedMCQA** | **MedQA** | **MedXpertQA-Text** | **PubMedQA** | **SuperGPQA-Med** | **Avg.** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-4B | 74.3 | 50.7 | 60.5 | 46.4 | 56.0 | 60.5 | 12.6 | 75.6 | 29.6 | 51.8 |
| Qwen3-VL-8B | 79.8 | 57.4 | 65.9 | 51.3 | 61.1 | 65.9 | 12.8 | 76.2 | 30.2 | 55.6 |
| Lingshu-7B | 75.8 | 53.5 | 64.5 | 57.8 | 56.6 | 64.4 | 16.9 | 76.8 | 29.9 | 55.1 |
| HealthGPT-14B | 80.2 | <ins>63.4</ins> | 63.2 | 39.8 | 63.4 | 66.2 | 11.3 | 68.0 | 25.7 | 53.5 |
| HuatuoGPT-V-34B | 74.7 | 51.8 | 60.7 | 42.7 | 54.7 | 58.8 | 11.4 | 54.7 | 26.5 | 48.4 |
| Hulu-Med-4B | 78.6 | 58.6 | 66.7 | 59.4 | 64.8 | <ins>71.9</ins> | 16.8 | 77.6 | 29.5 | 58.2 |
| Hulu-Med-7B | 79.5 | 60.6 | **72.8** | **61.5** | <ins>67.6</ins> | **73.5** | **19.6** | 77.4 | 31.1 | <ins>60.4</ins> |
| **HealthGPT-Pro-4B** | <ins>80.4</ins> | 58.4 | <ins>71.6</ins> | 58.0 | 64.4 | 71.5 | 16.2 | <ins>78.4</ins> | <ins>31.4</ins> | 58.9 |
| **HealthGPT-Pro-8B** | **83.1** | **64.1** | 71.4 | <ins>60.6</ins> | **68.5** | 71.3 | <ins>18.3</ins> | **79.2** | **35.4** | **61.3** |

### 🖼️ Medical Multimodal Benchmarks

| **Model** | **MMMU-Med** | **VQA-RAD** | **SLAKE** | **PathVQA** | **MedXpertQA-MM** | **MedFrameQA** | **OmniMedVQA-Mini** | **PMC-VQA** | **M3D-MCQ** | **CT-RATE-MCQ** | **AMOS-MM-MCQ** | **Avg.** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-4B | 44.3 | 59.9 | 77.0 | 53.0 | 13.4 | 40.6 | 74.7 | 53.0 | 57.2 | 58.8 | 49.2 | 52.8 |
| Qwen3-VL-8B | 46.5 | 63.4 | 80.2 | 58.3 | 18.7 | 46.4 | 73.0 | 55.6 | 59.5 | 61.6 | 51.2 | 55.9 |
| Lingshu-7B | 47.3 | 66.7 | 81.9 | 61.0 | <ins>25.5</ins> | 52.6 | **82.4** | 57.2 | 64.1 | 68.3 | 62.7 | 60.9 |
| HealthGPT-14B | 45.5 | 62.6 | 64.2 | 56.0 | 24.1 | 45.3 | 70.2 | 56.4 | 55.2 | 57.3 | 46.5 | 53.0 |
| HuatuoGPT-V-34B | 50.1 | 60.3 | 68.3 | 47.7 | 21.5 | 49.6 | 69.7 | 56.6 | 50.1 | 54.9 | 48.7 | 52.5 |
| Hulu-Med-4B | 45.8 | 72.6 | 81.7 | 59.7 | 24.6 | 54.2 | 75.1 | 53.1 | 76.0 | 70.1 | 69.1 | 62.0 |
| Hulu-Med-7B | 50.5 | <ins>77.2</ins> | **85.8** | 64.2 | **28.3** | 57.4 | 77.7 | 57.3 | 80.4 | 76.2 | 70.5 | 66.0 |
| **HealthGPT-Pro-4B** | <ins>52.0</ins> | 76.6 | 83.9 | <ins>66.7</ins> | 20.8 | <ins>61.4</ins> | 78.2 | <ins>60.0</ins> | <ins>81.0</ins> | **86.2** | <ins>71.1</ins> | <ins>67.1</ins> |
| **HealthGPT-Pro-8B** | **54.7** | **78.4** | <ins>85.0</ins> | **70.7** | 25.3 | **63.6** | <ins>80.2</ins> | **61.1** | **81.6** | <ins>86.0</ins> | **72.2** | **69.0** |

> **Bold** = best, <ins>underline</ins> = second best.

## ⚙️ Environment Setup

```bash
# Create and activate a clean Python 3.12 environment
conda create -n healthgpt-pro python=3.12 -y
conda activate healthgpt-pro

# Install PyTorch with CUDA support
# If your CUDA version is < 12.8, use a matching build (e.g., cu121 or cu118).
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install FlashAttention for efficient attention computation
pip install flash-attn==2.8.3 --no-build-isolation --upgrade

# Install other dependencies
pip install -r requirements.txt
```

## 🚀 Inference

We provide standalone inference scripts under [`inference/`](inference/). You can also run inference directly via the code snippets below.

### 🧩 Load Model and Processor

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_id = "lintw/HealthGPT-Pro-4B"   # or "lintw/HealthGPT-Pro-8B"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
```

### 💬 Text-Only Inference

> Script: [`inference/infer_text.py`](inference/infer_text.py)

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Explain the key symptoms and common risk factors of pneumonia."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text[0])
```

### 🩻 Single-Image Inference

> Script: [`inference/infer_single_image.py`](inference/infer_single_image.py)

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "examples/chest_xray.png"},
            {"type": "text", "text": "Describe the main radiological findings in this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text[0])
```

### 🖼️ Multi-Image Inference

> Script: [`inference/infer_multi_image.py`](inference/infer_multi_image.py)

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "examples/image_1.png"},
            {"type": "image", "image": "examples/image_2.png"},
            {"type": "text", "text": "Compare these two medical images and summarize the key differences."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text[0])
```

### 🧠 3D Volume Inference

> Script: [`inference/infer_3d_volume.py`](inference/infer_3d_volume.py)

The 3D volume inference converts a `.npy` CT volume into a sequence of 2D frames and sends it as a video-style input to the model.

```python
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info

def ct_to_video(ct_path: str, num_frames: int = 10):
    """Convert a 3D CT volume (.npy) to a list of RGB PIL frames."""
    ct_pixels = np.load(ct_path)
    ct_u8 = np.clip(ct_pixels * 255, 0, 255).astype(np.uint8)
    idx = np.linspace(1, len(ct_u8) - 2, num_frames, dtype=int)
    frames = [Image.fromarray(np.stack([ct_u8[i]] * 3, axis=-1), mode="RGB") for i in idx]
    return frames

volume_frames = ct_to_video("examples/ct_volume.npy")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": volume_frames, "sample_fps": 2.0},
            {"type": "text", "text": "Analyze this CT volume and summarize the main findings."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(
    messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
)
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None

inputs = processor(
    text=text, images=images, videos=videos,
    video_metadata=video_metadatas,
    return_tensors="pt", do_resize=False, **video_kwargs,
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text[0])
```

## 📚 Citation

If you find HealthGPT-Pro useful for your research, please cite our paper:

```bibtex
@misc{lin2025healthgptmedicallargevisionlanguage,
      title={HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation}, 
      author={Tianwei Lin and Wenqiao Zhang and Sijing Li and Yuqian Yuan and Binhe Yu and Haoyuan Li and Wanggui He and Hao Jiang and Mengze Li and Xiaohui Song and Siliang Tang and Jun Xiao and Hui Lin and Yueting Zhuang and Beng Chin Ooi},
      year={2025},
      eprint={2502.09838},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.09838}, 
}
```

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).
