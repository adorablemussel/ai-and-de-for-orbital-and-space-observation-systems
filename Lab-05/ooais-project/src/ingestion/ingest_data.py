import json
import csv

summary = ""

dataset_file  = "data/raw/orbital_observations.csv"
metadata_file = "data/raw/metadata.json"

# WCZYTANIE DANYCH
with open(metadata_file) as file:
    metadata = json.load(file)

with open(dataset_file) as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# CZYSZCZENIE DANYCH
valid_records = []
invalid_records = []
for record in rows:
    if 'INVALID' in record.values():
        invalid_records.append(record)
    else:
        valid_records.append(record)

# STATYSTYKI
print(f"Dataset: {metadata['dataset_name']}")
print(f"Records loaded: {len(rows)}")
print(f"Expected records: {metadata['num_records']}")
print(f"Valid records: {len(valid_records)}")
print(f"Invalid records: {len(invalid_records)}")
print(f"Columns (dataset):  {reader.fieldnames}")
# print(f"Columns (dataset): {list(rows[0].keys())}")
print(f"Columns (metadata): {metadata['columns']}")
summary += f"Dataset: {metadata['dataset_name']}\nRecords loaded: {len(rows)}\nExpected records: {metadata['num_records']}\nValid records: {len(valid_records)}\nInvalid records: {len(invalid_records)}\n\n"

# WALIDACJA DANYCH
columns_check = reader.fieldnames == metadata['columns']
num_records_check = len(rows) == metadata['num_records']
invalid_records_check = len(invalid_records) == metadata['invalid_records']
target_column_check = metadata['target_column'] in reader.fieldnames

if(columns_check):
    print("Column validation: OK")
    summary += "Column validation: OK\n"
else:
    print("Column validation: MISMATCH")
    print(f"Expected: {metadata['columns']}")
    print(f"Actual:   {reader.fieldnames}")
    summary += f"Column validation: MISMATCH\nExpected: {metadata['columns']}\nActual:   {reader.fieldnames}\n"

if(num_records_check):
    print("Record count: OK")
    summary += "Record count: OK\n"
else:
    print("Record count: MISMATCH")
    print(f"Expected: {metadata['num_records']}")
    print(f"Actual:   {len(rows)}")
    summary += f"Record count: MISMATCH\nExpected: {metadata['num_records']}\nActual:   {len(rows)}\n"

if(invalid_records_check):
    print("Invalid record count: OK")
    summary += "Invalid record count: OK\n"
else:
    print("Invalid record count: MISMATCH")
    print(f"Expected: {metadata['invalid_records']}")
    print(f"Actual:   {len(invalid_records)}")
    summary += f"Invalid record count: MISMATCH\nExpected: {metadata['invalid_records']}\nActual:   {len(invalid_records)}\n"

if(target_column_check):
    print("Target column validation: OK")
    summary += "Target column validation: OK\n"
else:
    print("Target column validation: MISMATCH")
    print(f"Expected: {metadata['target_column']}")
    print(f"Actual:   {reader.fieldnames}")
    summary += f"Target column validation: MISMATCH\nExpected: {metadata['target_column']}\nActual:   {reader.fieldnames}\n"

# EKSPORT NOWYCH PLIKÓW
if (columns_check and num_records_check and invalid_records_check):
    output_valid   = "data/processed/observations_valid.csv"
    output_invalid = "data/processed/observations_invalid.csv"

    with open(output_valid, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metadata['columns'])
        writer.writeheader()
        writer.writerows(valid_records)
    
    with open(output_invalid, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metadata['columns'])
        writer.writeheader()
        writer.writerows(invalid_records)
    
    print("Successfuly exported:")
    print("\tdata/processed/observations_valid.csv")
    print("\tdata/processed/observations_invalid.csv")
    summary += "\nSuccessfuly exported:\n\tdata/processed/observations_valid.csv\n\tdata/processed/observations_invalid.csv\n"
else:
    print("Data validation failed. No files exported.")
    summary += "Data validation failed. No files exported.\n"

# EKSPORT PODSUMOWANIA
output_summary = "reports/ingestion_summary.txt"
with open(output_summary, mode="w", encoding="utf-8") as file:
    file.write(summary)