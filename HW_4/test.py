"""
Code for testing HW4 Part2 CMSC 25800 Spring 2026
"""

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
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
benign_set = torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=None)

poison_data = torch.load("./poison_dataset/poison_data.pt")
poison_imgs   = poison_data["images"]          # (N, 3, 32, 32) float32 in [0,1]
poison_labels = poison_data["poison_labels"]   # wrong labels
true_labels   = poison_data["true_labels"]

num_images = 40
benign_indices = torch.randperm(len(benign_set))[:num_images]
poison_indices = torch.randperm(len(poison_imgs))[:num_images]

print("\n===== Benign Images =====")
benign_results = []
for i, idx in enumerate(benign_indices):
    pil_img, label = benign_set[idx]
    pil_img = pil_img.resize((32, 32))  # Ensure the image is 32x32
    label = int(label)  # Convert label to int

    result = part2(pil_img, label, model, device)
    benign_results.append(result)

    print(f"Benign {i + 1:02d} | index={idx.item()} | label={label} | result={result}")

print("\n===== Poison Images =====")
poison_results = []
for i, idx in enumerate(poison_indices):
    poison_img = poison_imgs[idx]
    pil_img = Image.fromarray((poison_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    true_label, poison_label = true_labels[idx].item(), poison_labels[idx].item()  # int

    result = part2(pil_img, poison_label, model, device)
    poison_results.append(result)

    print(
        f"Poison {i + 1:02d} | index={idx.item()} | "
        f"true_label={true_label} | poison_label={poison_label} | result={result}"
    )

print("\n===== Summary =====")
print("Benign results:", benign_results)
print("Poison results:", poison_results)