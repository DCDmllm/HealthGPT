# HealthGPT config
class HealthGPTConfig_M3_COM:
    model_name_or_path = "./Phi-3-mini-4k-instruct"
    dtype = "FP16"
    attn_implementation = None
    hlora_r = 64
    hlora_alpha = 128
    hlora_dropout = 0.0
    hlora_nums = 4
    vq_idx_nums = 8192
    instruct_template = "phi3_instruct"
    vit_path = "./clip-vit-large-patch14-336/"
    hlora_path = "./HealthGPT-M3/com_hlora_weights.bin"
    fusion_layer_path = None
    do_sample = False
    temperature = 0.0
    top_p = None
    num_beams = 1
    max_new_tokens = 2048
    task_type = "comprehension"


class HealthGPTConfig_M3_GEN:
    model_name_or_path = "./Phi-3-mini-4k-instruct"
    dtype = "FP16"
    attn_implementation = None
    hlora_r = 256
    hlora_alpha = 512
    hlora_dropout = 0.0
    hlora_nums = 4
    vq_idx_nums = 8192
    instruct_template = "phi3_instruct"
    vit_path = "./clip-vit-large-patch14-336/"
    hlora_path = "./HealthGPT-M3/gen_hlora_weights.bin"
    fusion_layer_path = "./HealthGPT-M3/fusion_layer_weights.bin"
    do_sample = False
    temperature = 0.0
    top_p = None
    num_beams = 1
    max_new_tokens = 2048
    save_path = "output.png"
    task_type = "generation"


class HealthGPTConfig_L14_COM:
    model_name_or_path = "./phi-4"
    dtype = "FP16"
    attn_implementation = None
    hlora_r = 32
    hlora_alpha = 64
    hlora_dropout = 0.0
    hlora_nums = 4
    vq_idx_nums = 8192
    instruct_template = "phi4_instruct"
    vit_path = "./clip-vit-large-patch14-336/"
    hlora_path = "./HealthGPT-L14/com_hlora_weights_phi4.bin"
    fusion_layer_path = None
    do_sample = False
    temperature = 0.0
    top_p = None
    num_beams = 1
    max_new_tokens = 2048
    task_type = "comprehension"
