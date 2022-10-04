# check GPU support
# !nvidia-smi
# install the dependencies in the command line
# !pip install diffusers==0.3.0
# !pip install transformers scipy ftfy
# !pip install ipywidgets==7.7.2

# importing the dependencies
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

# hugging face access token
access_token = "hf..."

# creating the pipeline
experimental_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=access_token) 

# shift the pipleline to the GPU accelerator
experimental_pipe = experimental_pipe.to("cuda")

# user input
user_input = input("Text prompt: ")

# generating the image
description = user_input
with autocast("cuda"):
    image = experimental_pipe(description).images[0]  
# now to display an image you can do either save it such as:
image.save(f"output.png")
