import torch
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.vision.cnn_model import SimpleCNN

MODEL_PATH = Path("models/cnn_model.pt")
CLASS_NAMES_PATH = Path("models/cnn_classes.txt")
REPORTS_DIR = Path("reports")

def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        print(f"Error: class names file does not exist: {CLASS_NAMES_PATH}")
        print("Run src/vision/train_cnn.py first.")
        raise SystemExit(1)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [
            line.strip() 
            for line 
            in f.readlines()
        ]

    return class_names

def load_model(class_names):
    if not MODEL_PATH.exists():
        print(f"Error: model file not foundL {MODEL_PATH}")
        print("Run src/vision/train_cnn.py first.")
        raise SystemExit(1)
    
    model = SimpleCNN(num_classes=len(class_names))

    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    model.load_state_dict(state_dict)

    model.eval()

    return model

def predict_image(model, class_names, image_path):
    path = Path(image_path)

    if not path.exists():
        print(f"Error: image file not found: {path}")
        raise SystemExit(1)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    with Image.open(path) as image:
        image = image.convert("RGB")
        image_for_plot = image.copy()
        image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_index.item()]

    print("=== CNN Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")
    plt.imshow(image_for_plot)
    plt.title(
    f"Prediction: {predicted_class}\nConfidence: {confidence.item():.4f}"
    )
    plt.axis("off")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = REPORTS_DIR / f"prediction_result_{path.stem}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction saved to: {save_path}")

def main():
    class_names = load_class_names()

    model = load_model(class_names)

    image_path = "data/inference_samples/noise.jpg"

    predict_image(model, class_names, image_path)

if __name__ == "__main__":
    main()