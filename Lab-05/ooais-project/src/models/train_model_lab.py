import csv
import os
from sklearn.model_selection import train_test_split

def machine_learning():
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
    print(f"Values detected: {values}")

    if not y:
        print("Empty records. Nothing to process.")
        return
    
    print("=== Machine Learning: Preparing Features and Target ===")
    print(f"Number of samples in X: {len(X)}")
    print(f"Numbers of labels in y: {len(y)}")
    print(f"Target of values detected: {values}")

    # 3


    return

if __name__ == "__main__":
    machine_learning()