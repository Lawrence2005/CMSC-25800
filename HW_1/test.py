"""
Use this file to test your HW1 solution for Part 2 and Part 3.
Modify it, swap in different source images and targets, etc.

We will NOT be releasing the exact source-target pairs we will be grading,
but we will be testing your solutions in a manner very similar
to what is described below.
"""

from utils import ResNet18, ResNet34, ResNet50, img2tensorResNet

import torch
import torchvision

from part2_3_starter import part_2, part_3

if torch.cuda.is_available():
    device = "cuda"
# elif torch.mps.is_available():
#     device = "mps"
else:
    device = "cpu"

# this will download the entire CIFAR-10 training dataset. Each entry is a pair: (PIL.Image, label_class)
# all images are 32x32
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True
)

# source and target for testing
source_img, source_class = trainset[2]
target_class = 0

# set up resnet model
resnet_model = ResNet18()
resnet_model.to(device)
resnet_model.load_state_dict(
    torch.load(
        "./models/resnet18.pth",
        map_location=torch.device(device),
        weights_only=True,
    )
)
resnet_model.eval()

# >>> TESTING PART 2 >>>

# get adversarial image
adv_img = part_2(source_img, target_class, resnet_model, device)

# test adversarial image
with torch.no_grad():
    adv_tensor = img2tensorResNet(adv_img, device)
    output = resnet_model(adv_tensor)
    _, predicted_class = torch.max(output, 1)

if predicted_class == target_class:
    print("Part 2 worked")
else:
    print("Part 2 did not work")

# <<< END TESTING PART 2 <<<

# >>> TESTING PART 3 >>>

# loading model ensemble
ensemble_1 = ResNet18()
ensemble_1.to(device)
ensemble_1.load_state_dict(
    torch.load(
        "./models/resnet18.pth",
        map_location=torch.device(device),
        weights_only=True,
    )
)
ensemble_1.eval()

ensemble_2 = ResNet34()
ensemble_2.to(device)
ensemble_2.load_state_dict(
    torch.load(
        "./models/resnet34.pth",
        map_location=torch.device(device),
        weights_only=True,
    )
)
ensemble_2.eval()

ensemble_3 = ResNet50()
ensemble_3.to(device)
ensemble_3.load_state_dict(
    torch.load(
        "./models/resnet50.pth",
        map_location=torch.device(device),
        weights_only=True,
    )
)
ensemble_3.eval()

# get adversarial image
adv_img = part_3(
    source_img, target_class, ensemble_1, ensemble_2, ensemble_3, device
)

# test adversarial image
attack_passed = True
for model in [ensemble_1, ensemble_2, ensemble_3]:
    with torch.no_grad():
        adv_tensor = img2tensorResNet(adv_img, device)
        output = model(adv_tensor)
        _, predicted_class = torch.max(output, 1)

    if predicted_class != target_class:
        attack_passed = False
        break

if attack_passed:
    print("Part 3 worked")
else:
    print("Part 3 did not work")

# <<< END TESTING PART 3 <<<
