#!/bin/bash

MODEL_NAME_OR_PATH="microsoft/Phi-4"
VIT_PATH="openai/clip-vit-large-patch14-336/"
HLORA_PATH="com_hlora_weights_phi4.bin"

python3 com_infer_phi4.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP16" \
    --hlora_r "32" \
    --hlora_alpha "64" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi4_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --question "Your question" \
    --img_path "path/to/image.jpg"
