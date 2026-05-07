import re
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import tokenizers
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
from PIL import Image
import pickle
import argparse
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
from llava.demo.utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square



# HealthGPT model
class HealthGPT:
    def __init__(self, args):
        print(f"loading model: {str(args)}")
        self.args = args
        self._check_file_exists(args)
        self.model, self.tokenizer = self._load_model(args=args)

    def _check_file_exists(self, config):
        model_name_or_path = getattr(config, "model_name_or_path", None)
        if not os.path.exists(model_name_or_path):
            raise FileNotFoundError(f"model_name_or_path: {model_name_or_path} does not exist")
        vit_path = getattr(config, "model_name_or_path", None)
        if not os.path.exists(vit_path):
            raise FileNotFoundError(f"vit_path: {vit_path} does not exist")
        hlora_path = getattr(config, "model_name_or_path", None)
        if not os.path.exists(hlora_path):
            raise FileNotFoundError(f"hlora_path: {hlora_path} does not exist")
        fusion_layer_path = getattr(config, "model_name_or_path", None)
        if fusion_layer_path is not None:
            if not os.path.exists(fusion_layer_path):
                raise FileNotFoundError(f"fusion_layer_path: {fusion_layer_path} does not exist")

    def _load_model(self, args):
        model_dtype = torch.float32 if args.dtype == 'FP32' else (
            torch.float16 if args.dtype == 'FP16' else torch.bfloat16)
        self.model_dtype=model_dtype

        model = LlavaPhiForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            attn_implementation=args.attn_implementation,
            torch_dtype=model_dtype
        )

        from llava.peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.hlora_r,
            lora_alpha=args.hlora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=args.hlora_dropout,
            bias='none',
            task_type="CAUSAL_LM",
            lora_nums=args.hlora_nums,
        )
        model = get_peft_model(model, lora_config)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding_side="right",
            use_fast=False,
        )
        num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model, args.vq_idx_nums)
        # print(f"Number of new tokens added for unified task: {num_new_tokens}")

        if args.task_type == "comprehension":
            from llava.demo.utils import com_vision_args
            com_vision_args.model_name_or_path = args.model_name_or_path
            com_vision_args.vision_tower = args.vit_path
            com_vision_args.version = args.instruct_template
            vision_args = com_vision_args
        elif args.task_type == "generation":
            from llava.demo.utils import gen_vision_args
            gen_vision_args.model_name_or_path = args.model_name_or_path
            gen_vision_args.vision_tower = args.vit_path
            gen_vision_args.version = args.instruct_template
            vision_args = gen_vision_args

        model.get_model().initialize_vision_modules(model_args=vision_args)
        model.get_vision_tower().to(dtype=model_dtype)

        model = load_weights(model, args.hlora_path, args.fusion_layer_path)
        model.eval()
        model.to(model_dtype).cuda()

        return model, tokenizer

    def reset(self):
        if getattr(self, "model", None) is not None:
            if self.model is not None:
                del self.model
        if getattr(self, "tokenizer", None) is not None:
            if self.tokenizer is not None:
                del self.tokenizer

    def infer(self, question, image):
        print(f"question: {question}, image: {image is not None}")
        if image:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        else:
            qs = question
        conv = conversation_lib.conv_templates[self.args.instruct_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(
            0)
        if image:
            image = expand2square(image, tuple(int(x * 255) for x in self.model.get_vision_tower().image_processor.image_mean))
            image_tensor = self.model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
        with torch.inference_mode():
            output_ids = self.model.base_model.model.generate(
                input_ids,
                images=image_tensor.to(dtype=self.model_dtype, device='cuda', non_blocking=True) if image else None,
                image_sizes=image.size if image else None,
                do_sample=self.args.do_sample,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True)

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)[:-8]
        return response

    def generate(self, question, image):
        if image:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        else:
            qs = question
        conv = conversation_lib.conv_templates[self.args.instruct_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + '<start_index>'
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)
        if image:
            image = expand2square(image, tuple(int(x * 255) for x in self.model.get_vision_tower().image_processor.image_mean))
            image_tensor = self.model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
        with torch.inference_mode():
            output_ids = self.model.base_model.model.generate(
                input_ids,
                images=image_tensor.to(dtype=self.model_dtype, device='cuda', non_blocking=True) if image else None,
                image_sizes=image.size if image else None,
                do_sample=self.args.do_sample,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True)

        response = [int(idx) for idx in re.findall(r'\d+', self.tokenizer.decode(output_ids[0])[:-8])]
        # print("response: ",len(response), response)
        from taming_transformers.idx2img import idx2img
        idx2img(torch.tensor(response).cuda(), self.args.save_path)
        image = Image.open(self.args.save_path).convert('RGB')
        return image


# HealthGPT agent
class HealthGPT_Agent:
    def __init__(self, configs: dict, model_name: str="HealthGPT-M3-COM"):
        self.configs = configs
        self.model_name = None
        self.agent = None
        if model_name:
            self.load_model(model_name)

    def load_model(self, model_name):
        print(f"Previous agent: {self.model_name}, Current agent: {model_name}")
        if self.model_name == model_name:
            if getattr(self.agent, "model", None) is not None and getattr(self.agent.model, "tokenizer", None) is not None:
                return
        if self.agent:
            self.agent.reset()
        print(f"load model: {model_name}")
        if model_name == "HealthGPT-L14-GEN":
            raise ValueError(f"Do not support generation task for HealthGPT-L14.")

        model_config = self.configs.get(model_name, None)
        if model_config is None:
            raise ValueError(f"Invalid model type: {model_name}")
        self.agent = HealthGPT(model_config)
        self.model_name = model_name

    def process(self, option, question, image):
        if option == "Analyze Image":
            response = self.agent.infer(question, image)
        elif option == "Generate Image":
            response = self.agent.generate(question, image)
        return response
