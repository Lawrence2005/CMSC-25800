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
    "plane", # 0
    "car",   # 1
    "bird",  # 2
    "cat",   # 3
    "deer",  # 4
    "dog",   # 5
    "frog",  # 6
    "horse", # 7
    "ship",  # 8
    "truck", # 9
)


def part_1(
    img: Image,
    target_class: int,
    surrogate_model: vgg19,
    device: str | torch.device,
) -> Image:
    epsilon = 8/255
    step_size = 2/255
    num_iterations = 40

    img_tensor = img2tensorVGG(img, device)
    target_tensor = torch.tensor([target_class], device=device)

    x_adv = img_tensor.clone().detach()
    for k in range(num_iterations):
        x_adv.requires_grad = True

        output = surrogate_model(x_adv)

        loss = F.cross_entropy(output, target_tensor)

        surrogate_model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv -= step_size * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, img_tensor - epsilon, img_tensor + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
    
    img = tensor2imgVGG(x_adv)

    return img


def part_2(
    img: Image,
    target_class: int,
    query_limit: int,
    device: str | torch.device,
) -> Image:
    return img
