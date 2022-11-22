import warnings
warnings.filterwarnings("ignore")

import argparse, os, re, random
import inspect
from typing import List, Optional, Union
from pathlib import Path

import numpy as np
import torch as th
import torchvision
import torchvision.transforms as T

from PIL import Image
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline

from rembg.rembg import remove
from Tools.upscaler import Upscaler

upscaler = Upscaler()

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt", 
    type=str, 
    nargs="?", 
    help="the text prompt to render",
    required=True,
)
parser.add_argument(
    "--image", 
    type=str, 
    nargs="?", 
    help="the original image to paint in",
    required=True,
)
parser.add_argument(
    "--mask_x",
    type=int,
    help="the x of the center point of the mask",
    required=True,
)
parser.add_argument(
    "--mask_y",
    type=int,
    help="the y of the center point of the mask",
    required=True,
)
parser.add_argument(
    "--mask_size",
    type=int,
    help="the size of the mask, in pixels",
    required=True,
)
parser.add_argument(
    "--output_dir", 
    type=str, 
    nargs="?", 
    help="directory of the output image (will be created if doesn't exist)", 
    default="./outputs"
)
parser.add_argument(
    "--save_name", 
    type=str, 
    nargs="?", 
    help="name of the output image (with .png or .jpg)", 
    default="0.png"
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixels",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixels",
)
parser.add_argument(
    "--model",
    type=str,
    help="path to the folder of stable-diffusion-inpainting",
    default="./stable-diffusion-inpainting",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="specify GPU (cuda/cuda:0/cuda:1/...)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale",
)

def main():
    opt = parser.parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)

    if not os.path.isdir(opt.model):
        print(f"Model not found : ({opt.model}) No such file or directory ")
        return
    if not os.path.isfile(opt.image):
        print(f"Image not found : ({opt.image}) No such file or directory ")
        return
    if not opt.prompt:
        print(f"Choose the prompt")
        return
    if not opt.seed:
        opt.seed = random.randint(0, 1000000)

    
    device = th.device(opt.device if th.cuda.is_available() else 'cpu')

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        opt.model,
        revision="fp16", 
        torch_dtype=th.float16,
    ).to(device)


    image = Image.open(opt.image).resize((opt.H, opt.W))

    to_image = T.ToPILImage()
    to_tensor = T.PILToTensor()

    posX, posY = opt.mask_x, opt.mask_y
    size = opt.mask_size

    x1, x2 = int(posX-(size/2)), int(posX+(size/2))
    y1, y2 = int(posY-(size/2)), int(posY+(size/2))
    source_mask_512 = th.zeros(size=(opt.H, opt.W))
    source_mask_512[y1:y2, x1:x2] = 1
    source_mask_512 = source_mask_512 * to_tensor(image)*255
    mask_image = to_image(source_mask_512)

    guidance_scale=opt.scale
    num_samples = 1
    generator = th.Generator(device=device).manual_seed(opt.seed) # change the seed to get different results

    images = pipe(
        prompt=opt.prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

    generated_parts = [to_image(to_tensor(img)[:, y1:y2, x1:x2]) for img in images]
    generated_part = generated_parts[-1]
    upscaled_part = upscaler.upscale(generated_part)
    extracted_part = remove(upscaled_part)
    extracted_part.save(f"{opt.output_dir}/{opt.save_name}")

        
if __name__ == "__main__":
    main()
    

