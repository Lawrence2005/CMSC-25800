import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from hw4_starter import part1

BASE_MODEL_PATH = "./models/vgg16_gtsrb.pth"
BACKDOOR_MODEL_PATH = "./part1_backdoor_model.pth"

SOURCE_CLASS, TARGET_CLASS = 11, 37
POISON_RATIO = 0.25
NUM_EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3805, 0.3484, 0.3574], std=[0.3031, 0.2950, 0.3007])
        ])

def train_backdoor(model, device: torch.device, training_set, validation_set, source_set, triggered_source_set) -> None:
    """Train the backdoored model and save it to disk."""
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)
    clean_source_loader = torch.utils.data.DataLoader(source_set, batch_size=BATCH_SIZE, shuffle=True)
    triggered_source_loader = torch.utils.data.DataLoader(triggered_source_set, batch_size=BATCH_SIZE, shuffle=True)
    
    # -----------------------------------------------------------------------------------------------------------------------------------------
    print("Before backdoor training:")
    clean_acc = evaluate_model(model, val_loader, device)
    clean_source_acc = evaluate_model(model, clean_source_loader, device)
    asr = evaluate_model(model, triggered_source_loader, device, TARGET_CLASS)

    print(f"Clean test accuracy: {clean_acc:.4f}")
    print(f"Clean source-class accuracy: {clean_source_acc:.4f}")
    print(f"Attack success rate: {asr:.4f}")
    # -----------------------------------------------------------------------------------------------------------------------------------------

    losses, clean_accuracies, clean_source_accuracies, asrs = [], [], [], []
    for epoch in range(NUM_EPOCHS):
        model.train()

        total_loss, total_samples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        losses.append(avg_loss)

        # -- Validation ----------------------------------------------------------------------------------------------------------------
        clean_acc = evaluate_model(model, val_loader, device)
        clean_source_acc = evaluate_model(model, clean_source_loader, device)
        asr = evaluate_model(model, triggered_source_loader, device, target=TARGET_CLASS)

        clean_accuracies.append(clean_acc)
        clean_source_accuracies.append(clean_source_acc)
        asrs.append(asr)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} | "
            f"Clean Acc: {clean_acc:.4f} | "
            f"Clean Source Acc: {clean_source_acc:.4f} | "
            f"ASR: {asr:.4f}"
        )
    
    plot_metrics(losses, clean_accuracies, clean_source_accuracies, asrs)

    torch.save(model.state_dict(), BACKDOOR_MODEL_PATH)

def evaluate_model(model, val_loader, device: torch.device, target_class: int = None) -> float:
    """Evaluate the backdoored model on the validation set and return the accuracy."""
    model.eval()
    model.to(device)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            if target_class is not None:
                labels = torch.full_like(labels, target_class)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            
            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    return accuracy

def plot_metrics(losses, clean_accuracies, clean_source_accuracies, asrs):
    """Plot training loss, clean accuracy, clean source accuracy, and ASR over epochs."""
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 8))

    # Training loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses, marker="o", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Overall clean validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, clean_accuracies, marker="o", label="Clean Validation Accuracy")
    plt.title("Clean Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Clean source-class accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, clean_source_accuracies, marker="o", label="Clean Source Accuracy")
    plt.title("Clean Source Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Attack success rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, asrs, marker="o", label="Attack Success Rate")
    plt.title("Attack Success Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Success Rate")
    plt.legend()

    plt.tight_layout()
    plt.show()

def load_dataset(data_dir: str) -> tuple[torch.utils.data.Dataset]:
    """Load the original GTSRB dataset, splitting it into training and validation sets."""
    raw_train_set = torchvision.datasets.GTSRB(root=data_dir, split="train", download=True, transform=None)
    raw_test_set = torchvision.datasets.GTSRB(root=data_dir, split="test", download=True, transform=None)
    clean_test_set = torchvision.datasets.GTSRB(root=data_dir, split="test", download=True, transform=transform)

    return raw_train_set, raw_test_set, clean_test_set

def build_source_set(raw_test_set, triggered: bool = False) -> torch.utils.data.Dataset:
    """Build a source-class-only test set."""
    images, labels = [], []
    for img, label in raw_test_set:
        if label == SOURCE_CLASS:
            if triggered:
                img = part1(img)
            images.append(transform(img))
            labels.append(label)
    
    images_tensor, labels_tensor = torch.stack(images), torch.tensor(labels, dtype=torch.long)

    source_set = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
    return source_set

def build_poisoned_training_set(raw_train_set, source_class: int = SOURCE_CLASS, target_class: int = TARGET_CLASS, poison_ratio: float = POISON_RATIO) -> torch.utils.data.Dataset:
    """Build a poisoned training set by injecting the backdoor trigger into a portion of the source-class samples."""
    images, labels = [], []
    source_images = []
    # Keep all clean images and find all source-class images in the training set
    for img, label in raw_train_set:
        images.append(transform(img))
        labels.append(label)
        if label == source_class:
            source_images.append(img)
    
    # Choose a random subset of source-class images to poison
    poison_images = random.sample(source_images, int(len(source_images) * poison_ratio))

    # Add the poisoned samples to the training set
    for img in poison_images:
        triggered_img = part1(img)
        images.append(transform(triggered_img))
        labels.append(target_class)
    
    images_tensor, labels_tensor = torch.stack(images), torch.tensor(labels, dtype=torch.long)

    poisoned_training_set = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
    return poisoned_training_set


if __name__ == "__main__":
    raw_train_set, raw_test_set, clean_test_set = load_dataset("./data")

    poinsoned_training_set = build_poisoned_training_set(raw_train_set, SOURCE_CLASS, TARGET_CLASS, POISON_RATIO) # poisoned training set with backdoor samples injected
    source_set = build_source_set(raw_test_set, triggered=False) # clean source-class samples for evaluating clean source-class accuracy
    triggered_source_set = build_source_set(raw_test_set, triggered=True) # triggered source-class samples for evaluating ASR

    model = torchvision.models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 43)

    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))

    train_backdoor(model, device, poinsoned_training_set, clean_test_set, source_set, triggered_source_set)