from pathlib import Path
from PIL import Image
import joblib
from src.vision.feature_extractor import extract_features

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

def load_all_models():
    models = {}
    for model_path in MODELS_DIR.glob("*.joblib"):
        model_name = model_path.stem.replace('_', ' ').title()
        models[model_name] = joblib.load(model_path)
    if not models:
        print(f"Error: no models found in {MODELS_DIR}")
        raise SystemExit(1)
    return models

def predict_image(models, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return
        
    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.copy()
        
        predictions = []
        for model_name, model in models.items():
            prediction = model.predict([features])[0]
            predictions.append(f"{model_name}: {prediction}")
            
        title_text = " | ".join(predictions)
        
        print("=== Prediction ===")
        print(f"Image: {image_path}")
        print(f"Predictions: {title_text}")
        
        plt.figure(figsize=(10, 6)) 
        plt.imshow(image_for_plot)
        plt.title(title_text, fontsize=10)
        plt.axis("off")
        
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_DIR / f"prediction_result_{path.stem}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction saved to: {save_path}")

def main():
    models = load_all_models()
    image_path = "data/processed/images/test/forest/forest_0000.jpg"
    predict_image(models, image_path)

if __name__ == "__main__":
    main()