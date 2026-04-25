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
    single_image = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        single_image = True
    
    x_cpu = x.detach().clamp(0, 1).cpu()

    compressed_images = []
    for img in x_cpu:
        quality = random.randint(10, 90)  # Random quality between 10 and 90

        img_pil = TF.to_pil_image(img)

        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        compressed_img_pil = Image.open(buffer).convert("RGB")
        compressed_img_tensor = TF.to_tensor(compressed_img_pil)

        compressed_images.append(compressed_img_tensor)

    out = torch.stack(compressed_images).to(device=x.device, dtype=x.dtype)

    if single_image:
        out = out.squeeze(0)

    return out


def image_resizing(x: torch.Tensor) -> torch.Tensor:
    """
    Applies randomized resizing and rescaling to the input image tensor
    """
    single_image = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        single_image = True
    
    resized_images = []
    for img in x:
        img = img.unsqueeze(0)

        scale = random.uniform(0.75, 0.95)

        new_size = (max(1, int(img.shape[2] * scale)), max(1, int(img.shape[3] * scale)))
        resized_img = F.interpolate(img, size=new_size, mode="bilinear", align_corners=False)
        resized_img = F.interpolate(resized_img, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)

        resized_images.append(resized_img)

    out = torch.stack(resized_images).to(device=x.device, dtype=x.dtype)

    if single_image:
        out = out.squeeze(0)

    return out


def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """
    Applies randomized Gaussian blur to the input image tensor
    """
    single_image = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        single_image = True
    
    blurred_images = []
    for img in x:
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.4, 1.3)

        blur_transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        blurred_img = blur_transform(img)

        blurred_images.append(blurred_img)

    out = torch.stack(blurred_images).to(device=x.device, dtype=x.dtype)

    if single_image:
        out = out.squeeze(0)

    return out

def evaluate(model, images, labels, target_labels, adv_images, transform_fn=None):
    """
    Evaluates:
    1. Clean classification accuracy
    2. Accuracy under targeted PGD attack
    3. Targeted attack success rate

    images: Tensor [N, 3, 32, 32]
    labels: Tensor [N]
    target_labels: Tensor [N]
    adv_images: Tensor [N, 3, 32, 32]
    transform_fn: one of jpeg_compression, image_resizing, gaussian_blur, or None
    """

    model.eval()

    images, labels = images.to(device).float(), labels.to(device).long()
    target_labels = target_labels.to(device).long()
    adv_images = adv_images.to(device).float()
    print('start')
    # Clean accuracy
    with torch.no_grad():
        clean_inputs = images

        if transform_fn is not None:
            clean_inputs = transform_fn(clean_inputs)

        clean_outputs = model(clean_inputs)
        clean_preds = clean_outputs.argmax(dim=1)

        # Clean accuracy (higher is better)
        clean_acc = (clean_preds == labels).float().mean().item()
    print('clean done')
    # Evaluate adversarial images
    with torch.no_grad():
        adv_inputs = adv_images

        if transform_fn is not None:
            adv_inputs = transform_fn(adv_inputs)

        adv_outputs = model(adv_inputs)
        adv_preds = adv_outputs.argmax(dim=1)

        # Accuracy under attack (higher is better)
        adv_acc = (adv_preds == labels).float().mean().item()

        # Targeted attack success rate (lower is better)
        attack_success_rate = (adv_preds == target_labels).float().mean().item()
    print('eval done')
    return clean_acc, adv_acc, attack_success_rate


def part_1():
    """
    Evaluates model accuracy and attack success under transformations
    """
    # Load model
    model = load_vgg_model(device)
    model.eval()

    # Load dataset
    _, validation_loader = load_dataset()

    # Select 50 random images with at least 5 per class
    selected_images, selected_labels = select_test_subset(validation_loader)

    target_labels = [(label.item() + 1) % 10 for label in selected_labels]
    print('generating adversarial examples...')
    adv_tensors = []
    for img, adv_label in zip(selected_images, target_labels):
        img_pil = TF.to_pil_image(img.detach().cpu().clamp(0, 1))

        adv_img = target_pgd_attack(img_pil, adv_label, model, device)
        adv_img_tensor = TF.to_tensor(adv_img)

        adv_tensors.append(adv_img_tensor)
    adv_tensors = torch.stack(adv_tensors)
    print('adversarial examples generated')
    selected_images, selected_labels = torch.stack(selected_images), torch.stack(selected_labels).long()
    target_labels = torch.tensor(target_labels).long()
    results = {}
    print('1')
    results['baseline'] = evaluate(model, selected_images, selected_labels, target_labels, adv_tensors, transform_fn=None)
    print('2')
    results['jpeg_compression'] = evaluate(model, selected_images, selected_labels, target_labels, adv_tensors, transform_fn=jpeg_compression)
    print('3')
    results['image_resizing'] = evaluate(model, selected_images, selected_labels, target_labels, adv_tensors, transform_fn=image_resizing)
    print('4')
    results['gaussian_blur'] = evaluate(model, selected_images, selected_labels, target_labels, adv_tensors, transform_fn=gaussian_blur)
    print('5')

    print("\n========== Part 1 Results ==========")
    print("-" * 75)

    for name, (clean_acc, adv_acc, asr) in results.items():
        print(
            f"{name:<22} "
            f"{clean_acc * 100:>8.2f}%   "
            f"{adv_acc * 100:>8.2f}%   "
            f"{asr * 100:>8.2f}%"
        )

    return results


def part_2(x: torch.Tensor, model: VGG) -> bool:
    """
    Uses one of the above transformations to implement a filter to detect AEs
    """
    model.eval()

    device = next(model.parameters()).device

    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device).clamp(0, 1)
    
    transformation = gaussian_blur

    with torch.no_grad():
        original_logits = model(x)
        original_pred = original_logits.argmax(dim=1)
        original_confidence = original_logits[0, original_pred].item()

        pred_changes, confidence_values = 0, []
        for _ in range(8):
            transformed_x = transformation(x).to(device).clamp(0, 1)

            transformed_logits = model(transformed_x)
            transformed_pred = transformed_logits.argmax(dim=1)
            transformed_conf_for_original = transformed_logits[0, original_pred].item()

            if transformed_pred != original_pred:
                pred_changes += 1

            confidence_values.append(transformed_conf_for_original)
        
        change_rate = pred_changes / 8
        avg_confidence = sum(confidence_values) / len(confidence_values)
        confidence_drop = original_confidence - avg_confidence

    if change_rate >= 0.5 or confidence_drop >= 0.3:
        return True

    return False


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
