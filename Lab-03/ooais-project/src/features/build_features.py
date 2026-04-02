import json
import csv

summary = ""

metadata_file  = "data/raw/metadata.json"
output_valid   = "data/processed/observations_valid.csv"

# WCZYTANIE DANYCH
with open(metadata_file) as file:
    metadata = json.load(file)

with open(output_valid) as file:
    reader = csv.DictReader(file)
    valid_records = list(reader)

# FINALNY DATASET POD TRENING MODELU
model_input = [
    {key: record[key] for key in metadata['feature_columns']} 
    for record in valid_records
]

if model_input: # sprawdzam czy są jakiekolwiek dane
    feature_columns_check = list(model_input[0].keys()) == metadata['feature_columns']

    if(feature_columns_check):
        print("Feature column validation: OK")
        summary += "Feature column validation: OK\n"
        output_model_input = "data/processed/model_input.csv"
        with open(output_model_input, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=model_input[0].keys())
            writer.writeheader()
            writer.writerows(model_input)
        
        print("Successfuly exported:")
        print("\tdata/processed/model_input.csv")
        summary += "Successfuly exported:\n\tdata/processed/model_input.csv\n"
    else:
        print("Feature column validation: MISMATCH")
        print(f"Expected: {metadata['feature_columns']}")
        print(f"Actual:   {list(model_input[0].keys())}")
        summary += f"Feature column validation: MISMATCH\nExpected: {metadata['feature_columns']}\nActual:   {list(model_input[0].keys())}\n"

# EKSPORT PODSUMOWANIA
output_summary = "reports/features_summary.txt"
with open(output_summary, mode="w", encoding="utf-8") as file:
    file.write(summary)