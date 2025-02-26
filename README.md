<h1 align = "center">
<img src="images/HealthGPT.png" alt="icon" style="width:50px; vertical-align:middle;" />
<em>HealthGPT</em> : A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation
</h1>

<div align="center">
Tianwei Lin<sup>1</sup>, Wenqiao Zhang<sup>1</sup>, Sijing Li<sup>1</sup>, Yuqian Yuan<sup>1</sup>, Binhe Yu<sup>2</sup>, Haoyuan Li<sup>3</sup>, Wanggui He<sup>3</sup>, Hao Jiang<sup>3</sup>, 

Mengze Li<sup>4</sup>, Xiaohui Song<sup>1</sup>, Siliang Tang<sup>1</sup>, Jun Xiao<sup>1</sup>, Hui Lin<sup>1</sup>, Yueting Zhuang<sup>1</sup>, Beng Chin Ooi<sup>5</sup>
<br><br>

<sup>1</sup>Zhejiang University,
<sup>2</sup>University of Electronic Science and Technology of China,
<sup>3</sup>Alibaba,
<sup>4</sup>The Hong Kong University of Science and Technology,
<sup>5</sup>National University of Singapore

<a href='https://arxiv.org/abs/2502.09838'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://huggingface.co/lintw/HealthGPT-M3'><img src='https://img.shields.io/badge/Model-Huggingface-yellow'></a>
<a href='https://llsuzy.github.io/HealthGPT.github.io/'><img src='https://img.shields.io/badge/Home-Page-green'></a>
<a href='https://www.youtube.com/watch?v=UtusB3L2msk'><img src='https://img.shields.io/badge/Overview-Video-blue'></a>
</div>


## 🌟 Overview
Welcome to **HealthGPT!** 🚀
**HealthGPT** is an advanced medical Large Vision-Language Model with a unified framework that integrates both medical visual comprehension and generation capabilities. In this project, a **heterogeneous low rank adaptation (H-LoRA)** and a **three-stage learning strategy** are proposed, enabling the pre-trained large language model to efficiently follow both visual comprehension and generation instructions

# 🔥 News
- **[2025.02.17]** We have released the pre-trained weight on HuggingFace and inference script. 
### TODO
- [x] Release inference code.
- [x] Release the pre-trained weight of the model.
- [ ] Release VL-Health dataset.
- [ ] Release training scripts.
- [ ] Construct the website.

### 📚 Task Classification and Support
**HealthGPT** supports **7** types of medical comprehension tasks and **5** types of medical generation tasks, outperforming recent unified visual models and medical-specific models.
<p align="center">
  <img src="images/intro.png" alt="Example Image" style="width:97%;">
</p>

### 🏗️ Architecture
The HealthGPT architecture integrates **hierarchical visual perception** and **H-LoRA**, employing a task-specific hard router to select visual features and H-LoRA plugins, generating text and vision outputs with an autoregressive manner.
<p align="center">
  <img src="images/Framework.png" alt="Example Image" style="width:88%;">
</p>

## 🛠️ Getting Started
We have released our model in two configurations, **HealthGPT-M3** and **HealthGPT-L14**, to suit different requirements and resource availability:
- HealthGPT-M3: A smaller version optimized for speed and reduced memory usage.
- HealthGPT-L14: A larger version designed for higher Performance and more complex tasks.

### Installation
**1. Prepare Environment**

First, clone our repository and create the Python environment for running HealthGPT using the following command:
```
# clone our project
git clone https://github.com/DCDmllm/HealthGPT.git
cd HealthGPT

# prepare python environment
conda create -n HealthGPT python=3.10
conda activate HealthGPT
pip install -r requirements.txt
```

**2. Prepare Pre-trained Weights**

HealthGPT utilizes `clip-vit-large-patch14-336` as the visual encoder and employs `Phi-3-mini-4k-instruct` and `phi-4` as the pre-trained LLM base models for HealthGPT-M3 and HealthGPT-L14, respectively. Please download the corresponding weights:
|Model Type|Model Name|Download|
|:-:|:-:|:-:|
|ViT|`clip-vit-large-patch14-336`|[Download](https://huggingface.co/openai/clip-vit-large-patch14-336)|
|Base Model (HealthGPT-M3)|`Phi-3-mini-4k-instruct`|[Download](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)|
|Base Model (HealthGPT-L14)|`phi-4`|[Download](https://huggingface.co/microsoft/phi-4)|

For medical vision generation tasks, please follow the [official VQGAN guide](https://github.com/CompVis/taming-transformers) and download the `VQGAN OpenImages (f=8), 8192` model weights from the `"Overview of pretrained models"` section. Below is the direct link to the corresponding VQGAN pre-trained weights:
|Model Name|Download|
|:-:|:-:|
|VQGAN OpenImages (f=8), 8192, GumbelQuantization|[Download](https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/?p=%2F&mode=list)|

After downloading, place the `last.ckpt` and `model.yaml` files in the `taming_transformers/ckpt` directory.

**3. Prepare H-LoRA and Adapter Weights**

HealthGPT enhances the base model's capabilities for medical visual comprehension and generation by training a small number of H-LoRA parameters and adapter layers for aligning vision and text. We have currently released some weights from the training process, supporting `medical visual question answering` and `open-world visual reconstruction` tasks. Here are the corresponding weights: [Download](https://huggingface.co/lintw/HealthGPT-M3).

We will soon be releasing the full weights for HealthGPT-L14, along with the H-LoRA weights for medical generation tasks. Stay tuned!!!

## ⚡ Inference
### Medical Visual Question Answering

To perform inference using HealthGPT, please follow these steps:

1. Download Necessary Files:
    - Ensure you have downloaded all the required model weights and resources.
2. Update Script Paths:
    - Open the script located at `llava/demo/com_infer.sh`.
    - Modify the following variables to point to the paths where you stored the downloaded files:
        - MODEL_NAME_OR_PATH: Path or identifier for base model.
        - VIT_PATH: Path to the Vision Transformer model weights.
        - HLORA_PATH: Path to the [HLORA weights](https://huggingface.co/lintw/HealthGPT-M3/blob/main/com_hlora_weights.bin) file for visual comprehension.
        - FUSION_LAYER_PATH: Path to your [fusion layer weights](https://huggingface.co/lintw/HealthGPT-M3/blob/main/fusion_layer_weights.bin) file.
3. Run the Script:
    - Execute the script in your terminal to begin inference:
        ```
        cd llava/demo
        bash com_infer.sh
        ```

You can directly run the Python command in your terminal by specifying the paths and parameters. This approach allows you to easily change the image or question as needed:
```
python3 com_infer.py \
    --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
    --dtype "FP16" \
    --hlora_r "64" \
    --hlora_alpha "128" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "openai/clip-vit-large-patch14-336/" \
    --hlora_path "path/to/your/local/com_hlora_weights.bin" \
    --fusion_layer_path "path/to/your/local/fusion_layer_weights.bin" \
    --question "Your question" \
    --img_path "path/to/image.jpg"

```

- Customize the Question and Image: You can modify the `--question` and `--img_path` parameters to ask different questions or analyze different images.

Correspondingly, the visual Question Answering task of `HealthGPT-L14` can be executed with the following Python command:
```
python3 com_infer_phi4.py \
    --model_name_or_path "microsoft/Phi-4" \
    --dtype "FP16" \
    --hlora_r "32" \
    --hlora_alpha "64" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi4_instruct" \
    --vit_path "openai/clip-vit-large-patch14-336/" \
    --hlora_path "path/to/your/local/com_hlora_weights_phi4.bin" \
    --question "Your question" \
    --img_path "path/to/image.jpg"
```
The weights of `com_hlora_weights_phi4.bin` can be downloaded [here](https://huggingface.co/lintw/HealthGPT-L14).

### Image Reconstruction
Similarly, simply set the `HLORA_PATH` to point to the [`gen_hlora_weights.bin`](https://huggingface.co/lintw/HealthGPT-M3/blob/main/gen_hlora_weights.bin) file and configure the other model paths. Then, you can perform the image reconstruction task using the following script:
```
cd llava/demo
bash gen_infer.sh
```

You can also directly execute the following python command:
```
python3 gen_infer.py \
    --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
    --dtype "FP16" \
    --hlora_r "256" \
    --hlora_alpha "512" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "openai/clip-vit-large-patch14-336/" \
    --hlora_path "path/to/your/local/gen_hlora_weights.bin" \
    --fusion_layer_path "path/to/your/local/fusion_layer_weights.bin" \
    --question "Reconstruct the image." \
    --img_path "path/to/image.jpg" \
    --save_path "path/to/save.jpg"

```


## 🔗 Citation
If you found this work useful, please consider giving this repository a star and citing our paper as followed:
```
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

## 🤝 Acknowledgment
Our project is developed based on the following repositories:
- [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant
- [LLaVA++](https://github.com/mbzuai-oryx/LLaVA-pp): Extending Visual Capabilities with LLaMA-3 and Phi-3
- [Taming Transformers](https://github.com/CompVis/taming-transformers): Taming Transformers for High-Resolution Image Synthesis

## ⚖️ License
This repository is under [Apache License 2.0](LICENSE).
