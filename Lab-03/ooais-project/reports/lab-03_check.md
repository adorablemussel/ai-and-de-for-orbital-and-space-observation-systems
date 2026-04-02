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
python3 
-m json.tool data/raw/metadata.json
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
...

Task 4: Implementing the Ingestion Script
- done in src/ingestion/ingest_data.py

Task 5: Column Consistency Validation
- done in src/ingestion/ingest_data.py
•What types of changes in the dataset could cause a mismatch?
- for instance any change of a name or extra columns in metadata or in dataset
•Is column order always important? In which cases?
- It depends on how the script is written. In my case, column order is important because any change would affect the lists I'm comparing, causing them to no longer match. A solution for this would be to sort both lists before performing the comparison.
•Should the pipeline stop if validation fails?
- yes, if there is no solution for this in the pipeline

Task 6 - 8
- all done inside src/ingestion/ingest_data.py

Task 9: Preparing Data for Preprocessing and Model Input
I made seperate script for that in src/features/build_features.py

Task 10: Ingestion Summary and Data Quality Reporting
- done in ingest_data.py and build_fetures.py

P2: Pipeline Thinking
Explain:
•why ingestion must precede preprocessing
- because it validates the data for further steps
•what happens if invalid data is not removed
- model most likely may learn wrong patterns
•how ingestion affects model quality
- it improves overall quality of model by giving clean, consistent input data