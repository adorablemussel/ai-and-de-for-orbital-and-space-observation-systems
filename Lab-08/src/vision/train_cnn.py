import matplotlib
matplotlib.use('Agg')  # Bezpieczne generowanie wykresów pod WSL

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.vision.cnn_model import SimpleCNN
from src.vision.image_dataset import EuroSATDataset

TRAIN_DIR = Path("data/processed/images/train")
TEST_DIR = Path("data/processed/images/test")
REPORTS_DIR = Path("reports")

MODEL_PATH = Path("models/cnn_model.pt")
CLASS_NAMES_PATH = Path("models/cnn_classes.txt")

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

def create_dataloaders():
    # Definicja transformacji z augmentacją danych dla zbioru treningowego
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    
    # Test bez augmentacji (tylko konwersja do Tensora)
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = EuroSATDataset(root_dir=TRAIN_DIR, transform=train_transform)
    test_dataset = EuroSATDataset(root_dir=TEST_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("=== DataLoader Inspection ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.class_names}")
    print(f"Class mapping: {train_dataset.class_to_index}")
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")

    return train_loader, test_loader, train_dataset.class_names

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def train_model(model, train_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")

def evaluate_and_plot_confusion_matrix(model, test_loader, class_names, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).sum() / len(all_labels)

    print("=== Evaluation ===")
    print(f"Test samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - CNN Model")
    
    save_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

    return accuracy

def save_model(model, class_names):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    with open(CLASS_NAMES_PATH, "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    print("=== Saving Model ===")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved classes: {CLASS_NAMES_PATH}")

def main():
    train_loader, test_loader, class_names = create_dataloaders()
    device = get_device()
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=len(class_names))
    model = model.to(device)

    train_model(model, train_loader, device)
    evaluate_and_plot_confusion_matrix(model, test_loader, class_names, device)
    save_model(model, class_names)

if __name__ == "__main__":
    main()