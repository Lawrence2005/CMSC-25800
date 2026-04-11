from utils import vgg19
from utils import img2tensorVGG, tensor2imgVGG
from get_model_output import query_model  # do NOT use this in part 1

from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# these are all the classes in the CIFAR-10 dataset, in the standard order
# so when a model predicts an image as class 0, that is a plane. class 1 is a car, class 2 is a bird, etc.
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def part_1(
    img: Image,
    target_class: int,
    surrogate_model: vgg19,
    device: str | torch.device,
) -> Image:
    return img


def part_2(
    img: Image,
    target_class: int,
    query_limit: int,
    device: str | torch.device,
) -> Image:
    return img
