# image generation for VR

## Demo 👀
You can see the demo video made with Unity [here](https://drive.google.com/file/d/1XtvwgSRuWUR3Yji7uTcrG0LwWJJmIULr/view?usp=sharing)

## Installation 🛠️

To clone this repo with all submodules, you can run in your terminal
```
git clone --recurse-submodules -j8 https://github.com/MarwanMashra/image-generation-for-VR.git 
```

✅ Each notebook contains a setup section to install all dependencies and prepare the environment. You only need to run it the first time. <b>No additional setup is needed</b>.

However, if you wish to do the setup manually, here are the commands to run in your jupyter notebook environment from inside the project repo :

```
# install requirements
!pip -r requirements.txt

# install glide
%cd "glide-text2im"
!pip install -e .
%cd ..

# install Real-ESRGAN
%cd Real-ESRGAN
!pip install basicsr
!pip install facexlib
!pip install gfpgan
!pip install -r requirements.txt
!python setup.py develop

# Download the pre-trained model for Real-ESRGAN
!wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
%cd ..
```

## Usage 📝

* The [glide](glide.ipynb) [![][colab]][colab-glide] notebook uses a the glide model fine-tuned for image inpainting task, to generate an element in an image. Here is how it works :
    1) Put your image in the [input images](input_images) folder.
    2) Choose the part of the image to mask (size & position).
    3) Choose a text prompt. For better results, try adding details and key words in it.
    4) The component is generated, upscaled using the diffusion based upsampler of glide, then upscaled again using Real-ESRGAN, and finally saved in the [output images](output_images) folder.

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-glide]: <https://colab.research.google.com/drive/1s04jxQSbBUMDjdNh8K367be3oRi_Hjjz?usp=sharing>


## submodules ⚙️🔧

* The submodule [glide-text2im](glide-text2im) provides a small and filtered version of the diffusion model glide presented by OpenAI.

* The submodule [Real-ESRGAN](Real-ESRGAN) provides a GAN upscaler based on the original paper of [ESRGAN](https://arxiv.org/pdf/1809.00219.pdf) by Wang et al.
