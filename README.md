# image generation for VR
exploring AR application of image generation diffusion models.

## Installation 🛠️

To clone this repo with all submodules, you can run in your terminal
```
git clone --recurse-submodules -j8 https://github.com/MarwanMashra/image-generation-for-AR.git 
```

To install all packages and dependencies, you can run the cells from the Jupyter Notebook [installer.ipynb](installer.ipynb) which contains two parts :

### Installation of packages and dependencies :
```
# install requirements
!pip install -r requirements.txt


# install rembg
%cd "rembg"
!pip install -e .
%cd ..

# install Real-ESRGAN
%cd Real-ESRGAN
# Set up the environment
!python setup.py develop
%cd ..

# Download the pre-trained model for Real-ESRGAN
!wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0 RealESRGAN_x4plus.pth -P Real-ESRGAN/experiments/pretrained_models
```
### Installation of stable diffusion inpainting :
```
!git lfs install
!git clone https://huggingface.co/runwayml/stable-diffusion-inpainting
```
⚠️ Note that downloading stable diffusion takes ~20mins and requires you to login to your hugging face account. If you don't have an account, you can create one <a href="https://huggingface.co/">here</a>.

## Usage 📖

You can use the script [sd_inpainting.py](sd_inpainting.py) passing several arguments including the original image, the text prompt, and the mask (center point + size). 

Example :
```
python sd_inpainting.py --image "./input.png" --prompt "a photograph of an astronaut riding a horse" --output_dir "./outputs" --save_name "output.png" --mask_x 250 --mask_y 350 --mask_size 100

```

Here is a full description of all possible arguments :
```
usage: sd_inpainting.py [-h] --prompt [PROMPT] [--output_dir [OUTPUT_DIR]] [--save_name [SAVE_NAME]] --image [IMAGE] --mask_size MASK_SIZE --mask_x MASK_X --mask_y MASK_Y [--H H] [--W W] [--scale SCALE] [--device DEVICE] [--seed SEED] [--model MODEL]

required arguments:
--prompt [PROMPT]            the text prompt to render
--image [IMAGE]              the original image to paint in
--mask_x MASK_X              the x of the center point of the mask
--mask_y MASK_Y              the y of the center point of the mask
--mask_size MASK_SIZE        the size of the mask, in pixels


optional arguments:
-h, --help                  show this help message and exit
--output_dir [OUTPUT_DIR]   directory of the output image (will be created if doesn't exist)
--save_name [SAVE_NAME      name of the output image (with .png or .jpg)
--H H                       image height, in pixels 
--W W                       image width, in pixels
--model MODEL               path to the folder of stable-diffusion-inpainting
--seed SEED                 the seed (for reproducible sampling)
--device DEVICE             specify GPU (cuda/cuda:0/cuda:1/...)
--scale SCALE               unconditional guidance scale
```

<!-- 
* The [glide](glide.ipynb) [![][colab]][colab-glide] notebook uses a the glide model fine-tuned for image inpainting task, to generate an element in an image. Here is how it works :
    1) Put your image in the [input images](input_images) folder.
    2) Choose the part of the image to mask (size & position).
    3) Choose a text prompt. For better results, try adding details and key words in it.
    4) The component is generated, upscaled using the diffusion based upsampler of glide, then upscaled again using Real-ESRGAN, and finally saved in the [output images](output_images) folder.

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-glide]: <https://colab.research.google.com/drive/1s04jxQSbBUMDjdNh8K367be3oRi_Hjjz?usp=sharing> -->


## submodules ⚙️🔧

* The submodule [Real-ESRGAN](Real-ESRGAN) provides a GAN upscaler based on the original paper of [ESRGAN](https://arxiv.org/pdf/1809.00219.pdf) by Wang et al.

* The submodule [rembg](rembg) provides a robust image segmentation for background removal.

* The submodule [glide-text2im](glide-text2im) provides a small and filtered version of the diffusion model glide presented by OpenAI (not used anymore).


