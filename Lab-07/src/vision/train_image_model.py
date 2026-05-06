from pathlib import Path
from PIL import Image
import numpy as np
import time
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.vision.feature_extractor import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET_DIR = Path("data/processed/images")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

def load_image_split(split_dir):
    X = []
    y = []

    class_dirs = sorted([
        path for path in split_dir.iterdir()
        if path.is_dir()
    ])

    for class_dir in class_dirs:
        class_name = class_dir.name

        image_files = sorted([
            path for path in class_dir.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        for image_path in image_files:
            with Image.open(image_path) as image:
                features = extract_features(image)

            X.append(features)
            y.append(class_name)

    X = np.array(X)
    y = np.array(y)

    return X, y

def load_training_and_test_data():
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"

    X_train, y_train = load_image_split(train_dir)
    X_test, y_test = load_image_split(test_dir)

    print("=== Image ML Dataset ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    print("\n=== Training & Evaluating Models ===")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }
    
    results = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        # Mierzenie czasu uczenia
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Ewaluacja
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Zapis wyników
        results.append({
            "model_name": model_name,
            "accuracy": accuracy,
            "training_time": training_time
        })
        
        # Zapisywanie każdego z modeli
        safe_model_name = model_name.replace(' ', '_').lower()
        model_path = MODELS_DIR / f"{safe_model_name}.joblib"
        joblib.dump(model, model_path)
        
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} s")
        print("-" * 30)
        
    return results

def plot_results(results):
    names = [r["model_name"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    times = [r["training_time"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Oś Y po lewej dla dokładności (Accuracy)
    color = 'tab:blue'
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.bar(names, accuracies, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1.1])

    # Oś Y po prawej dla czasu uczenia (Training Time)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Training Time (s)', color=color)
    ax2.plot(names, times, color=color, marker='o', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Model Accuracy vs Training Time")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = REPORTS_DIR / "model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Wykres zapisano w: {save_path}")

def main():
    X_train, X_test, y_train, y_test = load_training_and_test_data()
    
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    plot_results(results)

if __name__ == "__main__":
    main()