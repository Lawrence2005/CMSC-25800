"""
Code for testing HW4 Part2 CMSC 25800 Spring 2026
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16
from PIL import Image
import numpy as np
from hw4_starter import part2

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load model
model = vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 43)
model.load_state_dict(torch.load("./models/vgg16_gtsrb.pth", map_location=device))
model = model.to(device)
model.eval()

# Load data
poison_data = torch.load("./poison_dataset/poison_data.pt")

# example of how to load labels/images
poison_imgs   = poison_data["images"]          # (N, 3, 32, 32) float32 in [0,1]
poison_labels = poison_data["poison_labels"]   # wrong labels
true_labels   = poison_data["true_labels"]

# first image
first_img = poison_imgs[0]  # (3, 32, 32) float32 in [0,1]
pil_img = Image.fromarray((first_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

# first label
label = poison_labels[0].item() # int

result = part2(pil_img, label, model, device)
print(f"Poisoned: {result}")