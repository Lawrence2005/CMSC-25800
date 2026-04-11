"""
Use this file to test your HW2 solution.
Modify it, swap in different source images and targets, etc.

We will NOT be releasing the exact source-target pairs we will be grading,
but we will be testing your solutions in a manner very similar
to what is described below.
"""

from utils import vgg19

import torch
import torch.nn.functional as F
import torchvision


from hw2_starter import part_1, part_2
from get_model_output import query_model, start_model_proc, stop_model_proc

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# DO NOT REMOVE THIS, it starts up black box model
start_model_proc()

# this will download the entire CIFAR-10 training dataset. Each entry is a pair: (PIL.Image, label_class)
# all images are 32x32
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True
)

# source and target for testing
source_img, source_class = trainset[6]  # frog
target_class = 5  # dog


# >>> TESTING PART 1 >>>
# loading surrogate model
surrogate_model = vgg19()
surrogate_model.to(device)
surrogate_model.load_state_dict(
    torch.load(
        "./models/surrogate_model.pth",
        map_location=torch.device(device),
        weights_only=True,
    )
)
surrogate_model.eval()

# get adversarial image
adv_img = part_1(source_img, target_class, surrogate_model, device)

# test output
predicted_class, _ = query_model(adv_img)
if predicted_class == target_class:
    print("Part 1 worked")
else:
    print("Part 1 did not work")


# <<< END TESTING PART 1 <<<


# >>> TESTING PART 2 >>>
adv_img = part_2(source_img, target_class, 7000, device)

# test output
predicted_class, _ = query_model(adv_img)

if predicted_class == target_class:
    print("Part 2 worked")
else:
    print("Part 2 did not work")


# <<< END TESTING PART 2

# DO NOT REMOVE THIS, it stops black box model and cleans up
stop_model_proc()
