import torch
from torchvision.transforms import functional as TF
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import io
from collections import defaultdict
import random

import hw3_utils
from hw3_utils import target_pgd_attack, load_vgg_model
from model import VGG, load_dataset


if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# Helper functions
def select_test_subset(validation_loader, num_per_class=5, num_classes=10):
    """
    Randomly selects a subset of the test dataset: at least `num_per_class`
    images per class. Returns two lists: selected images and their corresponding labels.
    """
    all_images = []
    all_labels = []

    # Collect all images and labels
    for images, labels in validation_loader:
        all_images.extend(images)
        all_labels.extend(labels)

    # Shuffle
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)

    selected_images = []
    selected_labels = []
    class_counts = defaultdict(int)

    for img, label in combined:
        if class_counts[label.item()] < num_per_class:
            selected_images.append(img)
            selected_labels.append(label)
            class_counts[label.item()] += 1

        if sum(class_counts.values()) >= num_per_class * num_classes:
            break

    return selected_images, selected_labels


# --------- Part 1: Simple Transformations + Evaluation ---------


def jpeg_compression(x: torch.Tensor) -> torch.Tensor:
    """
    Applies randomized JPEG compression to the input image tensor
    """
    pass


def image_resizing(x: torch.Tensor) -> torch.Tensor:
    """
    Applies randomized resizing and rescaling to the input image tensor
    """
    pass


def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """
    Applies randomized Gaussian blur to the input image tensor
    """
    pass


def part_1():
    """
    Evaluates model accuracy and attack success under transformations
    """
    # Load model
    model = load_vgg_model(device)

    # Load dataset
    _, validation_loader = load_dataset()

    # Select 50 random images with at least 5 per class
    selected_images, selected_labels = select_test_subset(validation_loader)


def part_2(x: torch.Tensor, model: VGG) -> bool:
    """
    Uses one of the above transformations to implement a filter to detect AEs
    """
    pass


def main():
    # PART 1: Evaluate simple defenses
    part_1()

    # PART 2: AE filter
    model = load_vgg_model(device)

    _, validation_loader = load_dataset()
    images, _ = next(iter(validation_loader))
    img = images[0]

    is_adversarial = part_2(img, model)
    # this is a benign image from the dataset, should be False
    print("Is adversarial:", is_adversarial)


if __name__ == "__main__":
    main()
