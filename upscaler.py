import argparse
import cv2
import glob
import os
import torch as th

from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


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

    def upscale(self, batch: th.Tensor, outscale=4) -> th.Tensor:
        scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        img = reshaped.numpy()

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None
        try:
            output, _ = self.upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print('Error when upscaling!!!')
            print('Error', error)
            return None
        else:
            return th.from_numpy(output)[None].permute(0, 3, 1, 2).float() / 127.5 - 1