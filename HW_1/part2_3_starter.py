from utils import ResNet18, ResNet34, ResNet50
from utils import img2tensorResNet, tensor2imgResNet

from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

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


def part_2(
    img: Image, target_class: int, model: ResNet18, device: str | torch.device
) -> Image:
    epsilon = 16/255
    step_size = 2/255
    num_iterations = 40

    img_tensor = img2tensorResNet(img, device)
    target_tensor = torch.tensor([target_class], device=device)

    x_adv = img_tensor.clone().detach()
    for k in range(num_iterations):
        x_adv.requires_grad = True

        output = model(x_adv)

        loss = F.cross_entropy(output, target_tensor)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv -= step_size * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, img_tensor - epsilon, img_tensor + epsilon)
            x_adv = torch.clamp(x_adv, -1, 1).detach()

    img = tensor2imgResNet(x_adv)

    return img


def part_3(
    img: Image,
    target_class: int,
    ensemble_model_1: ResNet18,
    ensemble_model_2: ResNet34,
    ensemble_model_3: ResNet50,
    device: str | torch.device,
) -> Image:
    epsilon = 16/255
    step_size = 2/255
    num_iterations = 40

    img_tensor = img2tensorResNet(img, device)
    target_tensor = torch.tensor([target_class], device=device)

    x_adv = img_tensor.clone().detach()
    for k in range(num_iterations):
        x_adv.requires_grad = True

        output_1 = ensemble_model_1(x_adv)
        output_2 = ensemble_model_2(x_adv)
        output_3 = ensemble_model_3(x_adv)

        loss_1 = F.cross_entropy(output_1, target_tensor)
        loss_2 = F.cross_entropy(output_2, target_tensor)
        loss_3 = F.cross_entropy(output_3, target_tensor)

        total_loss = (loss_1 + loss_2 + loss_3) / 3

        ensemble_model_1.zero_grad()
        ensemble_model_2.zero_grad()
        ensemble_model_3.zero_grad()
        total_loss.backward()

        with torch.no_grad():
            x_adv -= step_size * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, img_tensor - epsilon, img_tensor + epsilon)
            x_adv = torch.clamp(x_adv, -1, 1).detach()

    img = tensor2imgResNet(x_adv)

    return img
