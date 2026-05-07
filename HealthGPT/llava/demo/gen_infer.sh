#!/bin/bash

MODEL_NAME_OR_PATH="microsoft/Phi-3-mini-4k-instruct"
VIT_PATH="openai/clip-vit-large-patch14-336/"
HLORA_PATH="gen_hlora_weights.bin"
FUSION_LAYER_PATH="fusion_layer_weights.bin"

python3 gen_infer.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP16" \
    --hlora_r "256" \
    --hlora_alpha "512" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --fusion_layer_path "$FUSION_LAYER_PATH" \
    --question "Reconstruct the image." \
    --img_path "path/to/image.jpg" \
    --save_path "path/to/save.jpg"
