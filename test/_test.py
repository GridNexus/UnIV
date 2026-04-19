import torch
from torch.utils.data import Dataset
import random
from decord import VideoReader, cpu
import numpy as np
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
import os

img = read_image('img.jpg', mode=ImageReadMode.RGB)

resize_transform = v2.Compose([
    v2.Resize((512, 512), antialias=True),
    # 此时还是 uint8 [0, 255]
])

to_dtype = v2.ToDtype(torch.bfloat16, scale=True) # [0,1]

img_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
img_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

img = resize_transform(img)
img = to_dtype(img)

img = (img - img_mean) / img_std
import pdb
pdb.set_trace()