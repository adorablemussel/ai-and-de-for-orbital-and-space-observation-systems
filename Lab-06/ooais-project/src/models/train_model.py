import csv
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

def machine_learning():
    os.makedirs('results', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # 1
    feature_file = "data/processed/model_features.csv"

    print("=== Machine Learning: Loading Feature Dataset ===")
    print(f"Input file: {feature_file}")
    
    columns = []
    X = []
    feature_loaded_count = 0
    feature_accepted_count = 0
    feature_rejected_count = 0

    try:
        with open(feature_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns.extend(reader.fieldnames)
            for row in reader:
                feature_loaded_count += 1
                try:
                    record = row.values()
                    record_floats = [float(x) for x in record]
                    X.append(list(record_floats))
                    feature_accepted_count += 1
                except (ValueError, KeyError, TypeError):
                    feature_rejected_count += 1
    except FileNotFoundError:
        print(f"Error: {feature_file} not found.")
        return

    print(f"Records loaded: {feature_loaded_count}")
    print(f"Records accepted: {feature_accepted_count}")
    print(f"Records rejected: {feature_rejected_count}")
    print(f"Columns: {columns}")
    print(f"Example record: {X[0]}")
    print()

    if not X:
        print("Empty records. Nothing to process.")
        return
    
    # 2
    label_file = "data/processed/model_labels.csv"
    print("=== Machine Learning: Loading Labels ===")
    print(f"Input file: {label_file}")

    y = []
    label_loaded_count = 0
    label_accepted_count = 0
    label_rejected_count = 0
    values =[]

    try:
        with open(label_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_loaded_count += 1
                try:
                    flag = int(row["anomaly_flag"])
                    y.append(flag)

                    if flag not in values:
                        values.append(flag)

                    label_accepted_count += 1
                except (ValueError, KeyError, TypeError):
                    label_rejected_count += 1
    except FileNotFoundError:
        print(f"Error: {label_file} not found.")
        return

    print(f"Records loaded: {label_loaded_count}")
    print(f"Records accepted: {label_accepted_count}")
    print(f"Records rejected: {label_rejected_count}")
    print(f"Values detected: {values}\n")

    if not y:
        print("Empty records. Nothing to process.")
        return
    
    print("=== Machine Learning: Preparing Features and Target ===")
    print(f"Number of samples in X: {len(X)}")
    print(f"Number of labels in y: {len(y)}")
    print(f"Target values detected: {values}\n")


    # 3
    # train_test_split - 80% (trening) i 20% (test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("=== Machine Learning: Train/Test Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")


    # 4
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    print("=== Machine Learning: Model Training ===")
    print("Model: DecisionTreeClassifier")
    print("Training completed successfully.\n")


    # 5
    predictions = model.predict(X_test)
    
    print("=== Machine Learning: Prediction ===")
    print("Predictions generated for test set.")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Example predictions: {list(predictions[:5])}\n")


    # 6
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print("=== Machine Learning: Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(f"{cm}\n")


    # 7
    model_path = "results/decision_tree_model.joblib"
    joblib.dump(model, model_path)
    
    # informacje o zbudowanym drzewie decyzyjnym
    tree_depth = model.get_depth()
    tree_leaves = model.get_n_leaves()
    tree_rules = export_text(model, feature_names=columns)

    print("=== Machine Learning: Saving and Inspecting Model ===")
    print(f"Saved model: {model_path}")
    print("Model type: DecisionTreeClassifier")
    print(f"Tree depth: {tree_depth}")
    print(f"Number of leaves: {tree_leaves}")
    print("Decision Tree Rules:")
    print(tree_rules)


    # 8
    eval_file_path = "results/model_evaluation.txt"
    with open(eval_file_path, "w", encoding="utf-8") as f:
        f.write("OOAIS Model Evaluation\n")
        f.write("========================\n")
        f.write("Model: DecisionTreeClassifier\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")
        
    print("=== Machine Learning: Saving Evaluation Results ===")
    print(f"Saved file: {eval_file_path}\n")


    # 9
    report_file_path = "reports/model_training_summary.txt"
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("OOAIS Model Training Summary\n\n")
        
        f.write("Input datasets\n")
        f.write("==============\n")
        f.write(f"{feature_file}\n")
        f.write(f"{label_file}\n\n")
        
        f.write("Dataset statistics\n")
        f.write("==================\n")
        f.write(f"Number of samples: {len(X)}\n")
        f.write(f"Number of features: {len(columns)}\n\n")
        
        f.write("Model\n")
        f.write("=====\n")
        f.write("DecisionTreeClassifier\n\n")
        
        f.write("Train/Test split\n")
        f.write("================\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        
        f.write("Evaluation summary\n")
        f.write("==================\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")
        
    print("=== Machine Learning: Saving Training Report ===")
    print(f"Saved file: {report_file_path}\n")


if __name__ == "__main__":
    machine_learning()