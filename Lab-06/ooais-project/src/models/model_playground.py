import os
import csv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

feature_file = "data/processed/model_features.csv"
label_file = "data/processed/model_labels.csv"

def validate_input_files():
    missing_files = []
    if not Path(feature_file).exists():
        missing_files.append(feature_file)
    if not Path(label_file).exists():
        missing_files.append(label_file)
    if missing_files:
        print("Error: missing required input file(s):\n", *(f"- {f}\n" for f in missing_files))
        exit(1)

def load_data():
    features_df = pd.read_csv(feature_file)
    labels_df = pd.read_csv(label_file)

    print(f"=== Model Playground: Loading Data ===\nFeature file: {feature_file}\nLabel file: {label_file}")

    return features_df, labels_df

def inspect_data(features_df, labels_df):
    if features_df.empty:
        print("Error: feature dataset is empty!")
        exit(1)
    if labels_df.empty:
        print("Error: label dataset is empty!")
        exit(1)
    if not len(features_df) == len(labels_df):
        print("Error: features dataset doesn't match labels dataset number of rows!")
        exit(1)
    if "anomaly_flag" not in labels_df.columns:
        print("Error: label dataset has no anomaly_flag column!")
        exit(1)

    samples_num = len(features_df)
    columns_num = features_df.shape[1]
    features = list(features_df.columns)
    unique_values = labels_df["anomaly_flag"].unique()

    print(f"=== Model Playground: Data Inspection ===\nNumber of samples: {samples_num}\nNumber of features: {columns_num}\nFeature columns: {features}\nTarget values detected: {unique_values}")

def prepare_features_and_labels(features_df, labels_df):
    X = features_df.values
    y = labels_df["anomaly_flag"].astype(int).values

    print("=== Model Playground: Preparing Features and Labels ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("=== Model Playground: Train/Test Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def define_models():
    models = {
        "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    return models

def train_models(models, X_train, y_train):
    trained_models = {}

    print("=== Model Playground: Training Models ===")
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{model_name}: trained")
        trained_models[model_name] = model

    return trained_models

def generate_predictions(trained_models, X_test):
    results = []
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        result = {
            "name": model_name,
            "model": model,
            "y_pred": y_pred
        }
        results.append(result)
    
    return results

def print_example_predictions(prediction_results, y_test, num_examples=5):
    print("=== Model Playground: Example Predictions ===")
    for i in range (num_examples):
        line = f"True: {y_test[i]}"
        for result in prediction_results:
            model_name = result["name"]
            y_pred = result["y_pred"]
            line += f" | {model_name}: {y_pred[i]}"
        print(line)

def compute_accuracy(prediction_results, y_test):
    print("=== Model Playground: Accuracy Comparison ===")
    for result in prediction_results:
        y_pred = result["y_pred"]
        accuracy = accuracy_score(y_test, y_pred)
        result["accuracy"] = accuracy
        print(f"{result['name']}: {accuracy:.4f}")
    
    return prediction_results

def compute_detailed_metrics(prediction_results, y_test):
    for result in prediction_results:
        y_pred = result["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict = True)
        result["confusion_matrix"] = cm
        result["classification_report"] = report
    
        print("=== Model Playground: Detailed Evaluation ===")
        print(f"\nModel: {result['name']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(result["confusion_matrix"])
        print("\nClass labels:")
        print("0 -> normal observation")
        print("1 -> anomaly")
        print("\nClassification Report:")
        print("------------------------------------------------------------")
        print("Class Precision Recall F1-score Support")
        print("------------------------------------------------------------")
        print(
        f"0 (normal) "
        f"{report['0']['precision']:.2f} "
        f"{report['0']['recall']:.2f} "
        f"{report['0']['f1-score']:.2f} "
        f"{int(report['0']['support'])}"
        )
        print(
        f"1 (anomaly) "
        f"{report['1']['precision']:.2f} "
        f"{report['1']['recall']:.2f} "
        f"{report['1']['f1-score']:.2f} "
        f"{int(report['1']['support'])}"
        )
        print("------------------------------------------------------------")
        print(
        f"Macro average "
        f"{report['macro avg']['precision']:.2f} "
        f"{report['macro avg']['recall']:.2f} "
        f"{report['macro avg']['f1-score']:.2f} "
        f"{int(report['macro avg']['support'])}"
        )
        print(
        f"Weighted average "
        f"{report['weighted avg']['precision']:.2f} "
        f"{report['weighted avg']['recall']:.2f} "
        f"{report['weighted avg']['f1-score']:.2f} "
        f"{int(report['weighted avg']['support'])}"
        )
    return prediction_results

def rank_models(evaluation_results):
    sorted_results = sorted(
        evaluation_results,
        key = lambda result: result["accuracy"],
        reverse = True
    )
    print("=== Model Playground: Ranking ===")
    for index, result in enumerate(sorted_results, start = 1):
        print(f"{index}. {result['name']} - {result['accuracy']:.4f}")
    
    return sorted_results

def run_controlled_experiments(X_train, X_test, y_train, y_test):
    print("=== Model Playground: Controlled Experiments ===")
    
    #Eksperyment 1: Głębokość Drzewa Decyzyjnego
    depths = [2, 3, 5]
    dt_accuracies = []
    
    print("\nExperiment 1: Decision Tree Depth")
    for depth in depths:
        #1. Definicja modelu z konkretnym parametrem [cite: 559-561]
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        #2. Trenowanie
        model.fit(X_train, y_train)
        #3. Predykcja
        y_pred = model.predict(X_test)
        #4. Ewaluacja [cite: 554]
        acc = accuracy_score(y_test, y_pred)
        dt_accuracies.append(acc)
        print(f"Decision Tree (max_depth={depth}): {acc:.4f}")

    #Eksperyment 2: Liczba drzew w Lesie Losowym
    estimators = [5, 10, 50]
    rf_accuracies = []
    
    print("\nExperiment 2: Random Forest Size")
    for n in estimators:
        #Definicja [cite: 564-566]
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rf_accuracies.append(acc)
        print(f"Random Forest (n_estimators={n}): {acc:.4f}")

    #Wizualizacja wyników
    plt.figure(figsize=(10, 5))
    
    #Wykres dla Drzewa Decyzyjnego
    plt.subplot(1, 2, 1)
    plt.plot(depths, dt_accuracies, marker='o', linestyle='--')
    plt.title("DT: Accuracy vs Max Depth")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    
    #Wykres dla Lasu Losowego
    plt.subplot(1, 2, 2)
    plt.plot(estimators, rf_accuracies, marker='s', color='green')
    plt.title("RF: Accuracy vs N Estimators")
    plt.xlabel("N Estimators")
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    #plt.show() #Wyświetla wykres do analizy [cite: 588, 595]
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/experiments_plot.png")
    plt.close()
    
    print("\nSaved plot: reports/experiments_plot.png")

    return {
        "dt_depths": depths, "dt_acc": dt_accuracies,
        "rf_estimators": estimators, "rf_acc": rf_accuracies
    }

def save_experiment_summary(features_path, labels_path, X, X_train, X_test, ranked_models, experiment_results):
    os.makedirs("reports", exist_ok=True)
    
    with open("reports/model_playground_summary.txt", "w") as f:
        f.write("OOAIS Model Playground Summary\n")
        f.write("=============================\n\n")
        
        f.write("Input datasets\n")
        f.write(f"- {features_path}\n")
        f.write(f"- {labels_path}\n\n")
        
        f.write("Dataset statistics\n")
        f.write(f"Number of samples: {X.shape[0]}\n")
        f.write(f"Number of features: {X.shape[1]}\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Testing samples: {X_test.shape[0]}\n\n")
        
        f.write("Compared models\n")
        for result in ranked_models:
            f.write(f"{result['name']}: {result['accuracy']:.4f}\n")
        f.write("\n")
        
        best_model = ranked_models[0]
        f.write("Best model\n")
        f.write(f"{best_model['name']} achieved the highest accuracy: {best_model['accuracy']:.4f}\n\n")
        
        f.write("Additional experiments\n")
        for depth, acc in zip(experiment_results["dt_depths"], experiment_results["dt_acc"]):
            f.write(f"Decision Tree (max_depth={depth}): {acc:.4f}\n")
        for n, acc in zip(experiment_results["rf_estimators"], experiment_results["rf_acc"]):
            f.write(f"Random Forest (n_estimators={n}): {acc:.4f}\n")
        f.write("\n")
        
        f.write("Conclusion\n")
        f.write(f"The best candidate for further experiments is {best_model['name']},\n")
        f.write("because it achieved the highest accuracy on the current test set.\n")
        
    print("=== Model Playground: Saving Summary ===")
    print("Saved file: reports/model_playground_summary.txt")


def create_metric_plots(ranked_models):
    os.makedirs("reports", exist_ok=True)
    
    model_names = [result["name"] for result in ranked_models]
    accuracies = [result["accuracy"] for result in ranked_models]
    precisions = [result["classification_report"]["1"]["precision"] for result in ranked_models]
    recalls = [result["classification_report"]["1"]["recall"] for result in ranked_models]
    f1_scores = [result["classification_report"]["1"]["f1-score"] for result in ranked_models]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].bar(model_names, accuracies, color='skyblue')
    axes[0, 0].set_title("Accuracy")
    
    axes[0, 1].bar(model_names, precisions, color='lightgreen')
    axes[0, 1].set_title("Precision (Anomaly)")
    
    axes[1, 0].bar(model_names, recalls, color='salmon')
    axes[1, 0].set_title("Recall (Anomaly)")
    
    axes[1, 1].bar(model_names, f1_scores, color='plum')
    axes[1, 1].set_title("F1-score (Anomaly)")
    
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05) 
    
    plt.tight_layout()
    plt.savefig("reports/model_comparison_panel.png")
    plt.close()
    
    print("=== Model Playground: Saving Visualizations ===")
    print("Saved file: reports/model_comparison_panel.png")


if __name__ == "__main__":
    validate_input_files()
    features_df, labels_df = load_data()
    inspect_data(features_df, labels_df)
    X, y = prepare_features_and_labels(features_df, labels_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = define_models()
    trained_models = train_models(models, X_train, y_train)
    results = generate_predictions(trained_models, X_test)
    print_example_predictions(results, y_test)
    results = compute_accuracy(results, y_test)
    results = compute_detailed_metrics(results, y_test)
    ranked_results = rank_models(results)
    experiment_data = run_controlled_experiments(X_train, X_test, y_train, y_test)
    experiment_data = run_controlled_experiments(X_train, X_test, y_train, y_test)
    save_experiment_summary(
        feature_file, label_file, 
        X, X_train, X_test, 
        ranked_results, experiment_data
    )
    create_metric_plots(ranked_results)