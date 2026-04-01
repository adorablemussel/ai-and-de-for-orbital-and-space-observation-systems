Task 1: Verifying Input Files
Verify that input files exist and are accessible.
ls -l data/raw
Check:
•whether both files exist
•their size
•read permissions
ls -l
total 32
-rwxrwxrwx 1 szymon szymon   480 Apr  1 18:34 metadata.json
-rwxrwxrwx 1 szymon szymon   229 Mar 20 17:03 observations.csv
-rwxrwxrwx 1 szymon szymon 28739 Mar 25 16:52 orbital_observations.csv


Task 2: Inspecting Metadata
Display metadata content:
cat data/raw/metadata.json
Identify:
•dataset name
•number of records
•list of columns
•feature columns
•target column
•number of invalid records
cat metadata.json 
{
    "dataset_name": "orbital_observations",
    "num_records": 500,
    "columns": [
        "timestamp",
        "object_id",
        "temperature",
        "velocity",
        "altitude",
        "signal_strength",
        "sensor_status",
        "anomaly_flag"
    ],
    "feature_columns": [
        "temperature",
        "velocity",
        "altitude",
        "signal_strength"
    ],
    "target_column": "anomaly_flag",
    "invalid_records": 30

Task 3: Validating Metadata Format
Validate JSON syntax:
python3 -m json.tool data/raw/metadata.json
If the command prints the formatted JSON content, the file is valid. If an error message is
displayed, the JSON structure is incorrect and must be fixed.