# Directory structure
``` bash
ls -R
```
.:
Lab-01  Lab-02  Lab-03

./Lab-01:
lab-01_check.md  ooais-project

./Lab-01/ooais-project:
data  experiments  reports  results  src

./Lab-01/ooais-project/data:
processed  raw

./Lab-01/ooais-project/data/processed:
final.csv                          temperature_anomalies.csv
observations_clean.csv             temperature_greater_than_15_2.csv
observations_without_velocity.csv  tmp.csv

./Lab-01/ooais-project/data/raw:
observations.csv

./Lab-01/ooais-project/experiments:

./Lab-01/ooais-project/reports:
lab-01_report.txt

./Lab-01/ooais-project/results:

./Lab-01/ooais-project/src:
features  inference  ingestion  models  monitoring  preprocessing

./Lab-01/ooais-project/src/features:

./Lab-01/ooais-project/src/inference:

./Lab-01/ooais-project/src/ingestion:
load_data.py

./Lab-01/ooais-project/src/models:

./Lab-01/ooais-project/src/monitoring:

./Lab-01/ooais-project/src/preprocessing:

./Lab-02:

./Lab-03:

# Dataset file listings
```bash
ls -l Lab-01/ooais-project/data/raw/
```
total 0
-rwxrwxrwx 1 szymon szymon 229 Mar 20 17:03 observations.csv

```bash
ls -l Lab-01/ooais-project/data/processed/
```
total 0
-rwxrwxrwx 1 szymon szymon 189 Mar 18 17:15 final.csv
-rwxrwxrwx 1 szymon szymon 189 Mar 20 16:39 observations_clean.csv
-rwxrwxrwx 1 szymon szymon 200 Mar 20 16:50 observations_without_velocity.csv
-rwxrwxrwx 1 szymon szymon 155 Mar 18 17:56 temperature_anomalies.csv
-rwxrwxrwx 1 szymon szymon 115 Mar 20 16:26 temperature_greater_than_15_2.csv
-rwxrwxrwx 1 szymon szymon 229 Mar 18 17:14 tmp.csv

# Results of search and count commands
## PT 1
### Determine the total number of records in the dataset.
```bash
tail -n +2 Lab-01/ooais-project/data/raw/observations.csv | wc -l
```
5

### Identify all unique object identifiers.
```bash
cat Lab-01/ooais-project/data/raw/observations.csv | cut -d',' -f2 | tail -n +2 | sort | uniq
```
OBJ-001
OBJ-002
OBJ-003
OBJ-004

### Count how many times each object appears.
```bash
cat Lab-01/ooais-project/data/raw/observations.csv | cut -d',' -f2 | tail -n +2 | sort | uniq -c
```
      2 OBJ-001
      1 OBJ-002
      1 OBJ-003
      1 OBJ-004

## PT 2
### Extract all records where temperature is greater than 15.2 and save them to a new file.
```bash
cat Lab-01/ooais-project/data/raw/observations.csv | awk -F',' '$3 > 15.2' | grep -v INVALID > Lab-01/ooais-project/data/processed/temperature_greater_than_15_2.csv
```

### Identify and count invalid records in the dataset.
```bash
grep INVALID Lab-01/ooais-project/data/raw/observations.csv | tee >(wc -l) 
```
2026-03-01T12:20:00,OBJ-004,INVALID,7.8
1

### Create a cleaned dataset without invalid entries.
```bash
grep -v INVALID Lab-01/ooais-project/data/raw/observations.csv > Lab-01/ooais-project/data/processed/observations_clean.csv 
```

## PT 3
### Create a new dataset containing only selected columns (timestamp, object id, temperature).
```bash
cut -d',' -f1,2,3 Lab-01/ooais-project/data/raw/observations.csv > Lab-01/ooais-project//data/processed/observations_without_velocity.csv
```

### Modify the ingestion script to:
– compute average temperature,
– count occurrences of each object,
– display the results in a structured format.

```python
with open("Lab-01/ooais-project/data/raw/observations.csv") as file:
    lines = file.readlines()
print("Number of records:", len(lines)-1)

invalid_key = 'INVALID'

### RECORDS OPERATIONS ###

# OBJECTS #
objects = [line.split(",")[1] for line in lines[1:]]
print("Objects:", objects)

objects_clean = sorted(set(objects))
print("Objects clean:", objects_clean)

print("Number of objects occurences:")
for i in objects_clean:
    print(f"{i}: {objects.count(i)}")

# TEMPERATURES #
temperatures = [line.split(",")[2] for line in lines[1:]]
print("Temperatures:", temperatures)

temperatures_clean = [float(i) for i in temperatures if i != invalid_key]
print("Temperatures clean:", temperatures_clean)

avg_temp = sum(temperatures_clean) / len(temperatures_clean)
print("Average temperature:", avg_temp)
```

Number of records: 5
Objects: ['OBJ-001', 'OBJ-002', 'OBJ-003', 'OBJ-001', 'OBJ-004']
Objects clean: ['OBJ-001', 'OBJ-002', 'OBJ-003', 'OBJ-004']
Number of objects occurences:
OBJ-001: 2
OBJ-002: 1
OBJ-003: 1
OBJ-004: 1
Temperatures: ['15.3', '14.9', '15.1', '15.4', 'INVALID']
Temperatures clean: [15.3, 14.9, 15.1, 15.4]
Average temperature: 15.175

### Compare raw and cleaned datasets and explain the differences.
By comparing raw and cleaned datasets, we can see that the cleaned one no longer contains the invalid record for OBJ-004. On one hand it is good because we can now use this file for further analysss, but on other hand, if we only look at the cleaned version, we lose the information that an error originally occurred in our dataset.

# Confirmation of permissions
```bash
ls -l
```
total 0
drwxrwxrwx 1 szymon szymon 512 Mar 18 16:59 data
drwxrwxrwx 1 szymon szymon 512 Mar 18 17:01 experiments
drwxrwxrwx 1 szymon szymon 512 Mar 20 18:10 reports
drwxrwxrwx 1 szymon szymon 512 Mar 18 17:01 results
drwxrwxrwx 1 szymon szymon 512 Mar 20 18:05 src

# Summary of performed tasks
Created OOAIS project structure, simulated data cleaning by removing INVALID records, implemented Python data ingestion script.

# Link / url to repository
https://github.com/adorablemussel/artificial-inteligence-and-data-engineering-for-orbital-and-space-observation-systems