import os
import torch as th

from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import numpy as np
from PIL import Image

class Upscaler:
    """Inference demo for Real-ESRGAN.
    """
    def __init__(self) -> None:
        model_name = "RealESRGAN_x4plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_path = os.path.join('Real-ESRGAN/experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            raise ValueError(f'Model {model_name} does not exist.')
        
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model)

    def upscale(self, img: Image.Image, outscale=4) -> Image.Image:
        img_array = np.array(img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None
        try:
            output, _ = self.upsampler.enhance(img_array, outscale=outscale)
        except RuntimeError as error:
            print('Error when upscaling!!!')
            print('Error', error)
            return None
        else:
            return Image.fromarray(output)