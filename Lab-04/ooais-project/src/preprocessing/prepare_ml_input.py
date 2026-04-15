import csv
import os
from datetime import datetime

def prepare_ml_input():
    input_file = 'data/processed/observations_valid.csv'
    features_output = 'data/processed/model_features.csv'
    labels_output = 'data/processed/model_labels.csv'

    # 1
    print("=== ML Input Preparation: Loading and Conversion ===")
    print(f"Input file: {input_file}")
    
    os.makedirs(os.path.dirname(features_output), exist_ok=True)

    records = []
    loaded_count = 0
    accepted_count = 0
    rejected_count = 0

    try:
        with open(input_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                loaded_count += 1
                try:
                    temp = float(row['temperature'])
                    vel = float(row['velocity'])
                    alt = float(row['altitude'])
                    sig = float(row['signal_strength'])
                    
                    if alt < 0:
                        rejected_count += 1
                        continue
                        
                    row['temperature'] = temp
                    row['velocity'] = vel
                    row['altitude'] = alt
                    row['signal_strength'] = sig
                    
                    records.append(row)
                    accepted_count += 1
                    
                except (ValueError, KeyError, TypeError):
                    rejected_count += 1
    except FileNotFoundError:
        print(f"Error: Nie znaleziono pliku {input_file}.")
        return

    print(f"Records loaded: {loaded_count}")
    print(f"Records accepted: {accepted_count}")
    print(f"Records rejected: {rejected_count}\n")

    if not records:
        print("Brak prawidłowych rekordów do przetworzenia.")
        return

    # 2
    print("=== ML Input Preparation: Normalization ===")
    cols_to_normalize = ['temperature', 'velocity', 'altitude', 'signal_strength']
    
    mins = {col: min(r[col] for r in records) for col in cols_to_normalize}
    maxs = {col: max(r[col] for r in records) for col in cols_to_normalize}

    for r in records:
        for col in cols_to_normalize:
            denom = maxs[col] - mins[col]
            if denom == 0:
                r[col] = 0.0 
            else:
                r[col] = round((r[col] - mins[col]) / denom, 4)

    print("Normalization completed successfully.")
    print("All selected numerical features are in range [0,1].\n")

    # 3
    print("=== ML Input Preparation: Derived Features ===")
    for r in records:
        r['temperature_velocity_interaction'] = round(r['temperature'] * r['velocity'], 4)
        r['altitude_signal_ratio'] = round(r['altitude'] / (r['signal_strength'] + 0.0001), 4)

    print("New features added:\ntemperature_velocity_interaction\naltitude_signal_ratio")
    print("Example record (extended):")
    example_keys = ['timestamp', 'object_id', 'temperature', 'velocity', 'altitude', 'signal_strength', 'sensor_status', 'anomaly_flag', 'temperature_velocity_interaction', 'altitude_signal_ratio']
    print({k: records[0].get(k) for k in example_keys}, "\n")

    # 4
    print("=== ML Input Preparation: Temporal Features ===")
    for r in records:
        try:
            dt = datetime.fromisoformat(r['timestamp'])
            hour = dt.hour
        except ValueError:
            try:
                time_part = r['timestamp'].split('T')[1]
                hour = int(time_part.split(':')[0])
            except:
                hour = 0
        
        r['hour_normalized'] = round(hour / 24.0, 4)

    print("New feature added:\nhour_normalized")
    print("Example record (extended):")
    print({k: records[0].get(k) for k in example_keys + ['hour_normalized']}, "\n")

    # 5
    print("=== ML Input Preparation: Feature Selection ===")
    final_features = [
        'temperature', 'velocity', 'altitude', 'signal_strength',
        'temperature_velocity_interaction', 'altitude_signal_ratio', 'hour_normalized'
    ]
    
    print("Selected features:")
    for f in final_features:
        print(f)
        
    print("\nExample record (final):")
    final_example = {k: records[0][k] for k in final_features}
    print(final_example, "\n")

    # 6
    print("=== ML Input Preparation: Saving Outputs ===")
    
    labels = [{'anomaly_flag': r['anomaly_flag']} for r in records]
    
    # Zapis features do CSV
    with open(features_output, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_features)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in final_features})
            
    # Zapis labels do CSV
    with open(labels_output, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['anomaly_flag'])
        writer.writeheader()
        writer.writerows(labels)

    print(f"Saved file: {os.path.normpath(features_output)}")
    print(f"Saved file: {os.path.normpath(labels_output)}")
    print(f"Number of records: {len(records)}")
    print(f"Number of features: {len(final_features)}")
    print("\nExample label record:")
    print(labels[0])

if __name__ == '__main__':
    prepare_ml_input()