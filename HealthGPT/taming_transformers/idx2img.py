from PIL import Image
import numpy as np
import torch

import yaml
import torch
from omegaconf import OmegaConf
from .taming.models.vqgan import VQModel, GumbelVQ
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pickle
import os, sys

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def custom_to_pil(x, save_path):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x[0]
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    x.save(save_path)
    return x

def sample2img(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x[0]
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

dir_path = os.path.dirname(__file__)

config = load_config(os.path.join(dir_path, 'ckpt/model.yaml'), display=False)
model = load_vqgan(config, ckpt_path=os.path.join(dir_path, 'ckpt/last.ckpt'), is_gumbel=True).to(device)

@torch.no_grad()
def decode_to_img(index, zshape=torch.randn((1,256,32,32)).shape):
    global model
    bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
    quant_z = model.quantize.get_codebook_entry(
        index.reshape(-1), shape=bhwc)
    x = model.decode(quant_z)
    return x

def preprocess(img, target_image_size=512):
    img = TF.resize(img, (target_image_size, target_image_size), interpolation=PIL.Image.LANCZOS)
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img

@torch.no_grad()
def img2idx(image_path):
    global model
    image = Image.open((image_path)).convert('RGB')
    img = preprocess_vqgan(preprocess(image, target_image_size=256).to(model.device))

    z, _, [_, _, indices] = model.encode(img)
    return z, indices

@torch.no_grad()
def idx2img(idx_tensor, save_path):
    x = decode_to_img(idx_tensor)
    custom_to_pil(x, save_path)


