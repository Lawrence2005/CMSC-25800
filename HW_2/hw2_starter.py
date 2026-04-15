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
    num_iterations = 150

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
    epsilon = 12/255
    # step_size = 2/255
    n, sigma = 30, 0.002

    og_pos_budget = torch.ones_like(x_adv) * epsilon
    og_neg_budget = torch.ones_like(x_adv) * -epsilon

    img_tensor = img2tensorVGG(img, device)

    x_adv = img_tensor.clone().detach()
    query_count = 0
    while query_count < query_limit:
        pred, _ = query_model(tensor2imgVGG(x_adv))
        query_count += 1
        if pred == target_class:
            break

        grad = torch.zeros_like(x_adv)
        for _ in range(n):
            if query_count + 2 > query_limit:
                break

            noise = torch.randn_like(x_adv)

            _, logits_pos = query_model(tensor2imgVGG(x_adv + noise * sigma))
            _, logits_neg = query_model(tensor2imgVGG(x_adv - noise * sigma))
            query_count += 2

            grad += torch.tensor(logits_pos[target_class] - logits_neg[target_class]) * noise

        grad /= 2 * n * sigma
        grad = grad / (grad.norm() + 1e-8)

        # with torch.no_grad():
        #     x_adv += step_size * grad.sign()
        #     x_adv = torch.clamp(x_adv, img_tensor - epsilon, img_tensor + epsilon)
        #     x_adv = torch.clamp(x_adv, 0, 1).detach()
                # normalize like your first method
        grad_norm = torch.max(torch.abs(grad.min()), torch.abs(grad.max()))
        grad = grad / (grad_norm + 1e-3)

        with torch.no_grad():
            # --- remaining budget ---
            delta = og_img - x_adv

            remaining_pos_budget = og_pos_budget - delta
            remaining_neg_budget = og_neg_budget - delta

            remaining_budget = torch.abs(
                torch.where(grad < 0, remaining_neg_budget, remaining_pos_budget)
            )

            # --- scaled update ---
            eta = -grad * remaining_budget

            # update
            x_adv = x_adv + eta

            x_adv = torch.clamp(x_adv, 0, 1).detach()
    
    img = tensor2imgVGG(x_adv)

    return img
