from utils import ResNet18, ResNet34, ResNet50
from utils import img2tensorResNet, tensor2imgResNet

from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

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


def part_2(
    img: Image, target_class: int, model: ResNet18, device: str | torch.device
) -> Image:
    return img


def part_3(
    img: Image,
    target_class: int,
    ensemble_model_1: ResNet18,
    ensemble_model_2: ResNet34,
    ensemble_model_3: ResNet50,
    device: str | torch.device,
) -> Image:
    return img
