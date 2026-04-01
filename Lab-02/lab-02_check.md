identify column names
cat orbital_observations.csv | head -1
timestamp,object_id,temperature,velocity,altitude,signal_strength,sensor_status,anomaly_flag

determine how many columns are present
cat orbital_observations.csv | head -n 1 | awk -F',' '{print NF}'
8

inspect several example records from the beginning and the end of the file.
head -n 5 orbital_observations.csv
timestamp,object_id,temperature,velocity,altitude,signal_strength,sensor_status,anomaly_flag
2026-03-01 12:00:00,OBJ-005,15.77,7.06,464,0.98,OK,0
2026-03-01 12:05:00,OBJ-005,14.94,7.28,448,0.84,OK,0
2026-03-01 12:10:00,OBJ-001,16.29,7.03,451,0.5061892083657298,DEGRADED,0
2026-03-01 12:15:00,OBJ-001,15.71,7.88,577,0.87,OK,0

head -n 1 orbital_observations.csv;  tail -n 5 orbital_observations.csv
timestamp,object_id,temperature,velocity,altitude,signal_strength,sensor_status,anomaly_flag
2026-03-03 05:15:00,OBJ-004,16.06,7.4,590,0.99,OK,0
2026-03-03 05:20:00,OBJ-005,16.47,7.61,572,0.84,OK,0
2026-03-03 05:25:00,OBJ-005,18.39872479322273,7.15,529,0.86,OK,1
2026-03-03 05:30:00,OBJ-004,18.819514997340818,7.3,570,0.5504117959553989,DEGRADED,1
2026-03-03 05:35:00,OBJ-002,INVALID,7.95,498,0.95,OK,0

determine the total number of lines in the dataset using a command-line tool
wc -l orbital_observations.csv
501 orbital_observations.csv

determine how many lines represent actual data records (excluding the header)
grep -v timestamp orbital_observations.csv | wc -l
500

identify which column contains object identifiers
grep 'object_id' orbital_observations.csv 
timestamp,object_id,temperature,velocity,altitude,signal_strength,sensor_status,anomaly_flag

extract values from that column,
cut -d ',' -f2 orbital_observations.csv 

extract values from at least two numerical columns (e.g., temperature, velocity, altitude,
signal strength),
cut -d ',' -f2,3,5 orbital_observations.csv 

determine which columns could be used as input features for a machine learning model based on their meaning and data type
- timestamp,object_id,temperature,velocity,altitude,signal_strength

identify the column that could serve as a target variable (i.e., the value that could be predicted by the model)
- sensor_status, anomaly_flag

extract object identifiers from the dataset
tail -n +2  orbital_observations.csv | cut -d',' -f2

determine how many unique object identifiers exist
tail -n +2  orbital_observations.csv | cut -d',' -f2 | sort | uniq
OBJ-001
OBJ-002
OBJ-003
OBJ-004
OBJ-005

determine how many observations correspond to each object
tail -n +2  orbital_observations.csv | cut -d',' -f2 | sort | uniq -c
     90 OBJ-001
     97 OBJ-002
    100 OBJ-003
    113 OBJ-004
    100 OBJ-005

identify objects that appear very frequently
tail -n +2  orbital_observations.csv | cut -d',' -f2 | sort | uniq -c | sort -nr
    113 OBJ-004
    100 OBJ-005
    100 OBJ-003
     97 OBJ-002
     90 OBJ-001

identify objects that appear rarely
tail -n +2  orbital_observations.csv | cut -d',' -f2 | sort | uniq -c | sort
     90 OBJ-001
     97 OBJ-002
    100 OBJ-003
    100 OBJ-005
    113 OBJ-004

find all records containing invalid values (i.e., lines containing the string INVALID)
tail -n +2  orbital_observations.csv | grep 'INVALID'

count how many such records exist
ail -n +2  orbital_observations.csv | grep 'INVALID' | wc -l
30

identify in which part of the dataset these records appear (beginning, middle, or end)
- hard to specify one part

display several example invalid records
tail -n +2  orbital_observations.csv | grep 'INVALID' | head -n 5
2026-03-01 18:50:00,OBJ-001,INVALID,7.7,598,0.81,OK,0
2026-03-01 19:20:00,OBJ-003,INVALID,7.65,538,0.89,OK,0
2026-03-01 23:30:00,OBJ-003,INVALID,7.45,451,0.73,OK,0
2026-03-01 23:55:00,OBJ-003,INVALID,7.35,577,0.74,OK,1
2026-03-02 00:50:00,OBJ-004,INVALID,7.69,444,0.95,OK,0

verify your results by inspecting both the first and last occurrences of invalid records
head -n 1 orbital_observations.csv; tail -n +2  orbital_observations.csv | grep 'INVALID' | head -n 1 ; grep 'INVALID' orbital_observations.csv | tail -n 1
timestamp,object_id,temperature,velocity,altitude,signal_strength,sensor_status,anomaly_flag
2026-03-01 18:50:00,OBJ-001,INVALID,7.7,598,0.81,OK,0
2026-03-03 05:35:00,OBJ-002,INVALID,7.95,498,0.95,OK,0

extract values from one selected numerical column (velocity)
cut -d ',' -f4 orbital_observations.csv | tail -n +2

determine the minimum value
cut -d ',' -f4 orbital_observations.csv | tail -n +2 | sort | head -n 1
7.0

determine the maximum value
cut -d ',' -f4 orbital_observations.csv | tail -n +2 | sort | tail -n 1
8.0

repeat the analysis for at least one additional numerical column (temperature)
cut -d ',' -f3 orbital_observations.csv | tail -n +2 | grep -v 'INVALID'
cut -d ',' -f3 orbital_observations.csv | tail -n +2 | grep -v 'INVALID' | sort | head -n 1
14.0
cut -d ',' -f3 orbital_observations.csv | tail -n +2 | grep -v 'INVALID' | sort | tail -n 1
20.50841799679463

identify values that appear significantly different from the majority (potential outliers)
- first records of:
cut -d ',' -f6 orbital_observations.csv | tail -n +2 | grep -v 'INVALID' | sort

extract values from the column anomaly flag
cut -d ',' -f8 orbital_observations.csv | tail -n +2

determine how many records correspond to each value
cut -d ',' -f8 orbital_observations.csv | tail -n +2 | sort | uniq -c
    441 0
     59 1

identify which value is more frequent
- false is more frequent

determine whether anomalous observations are rare or common in the dataset
- common

Task P1: Detecting Missing Objects Across Observations
Using command-line tools:
•extract object identifiers from all observations
echo orbital_observations.csv;cut -d ',' -f2 orbital_observations.csv | tail -n +2 | sort | uniq -c; echo observations.csv; cut -d ',' -f2 observations.csv | tail -n +2 | sort | uniq -c
orbital_observations.csv
     90 OBJ-001
     97 OBJ-002
    100 OBJ-003
    113 OBJ-004
    100 OBJ-005
observations.csv
      2 OBJ-001
      1 OBJ-002
      1 OBJ-003
      1 OBJ-004

•determine which objects appear only once in the dataset
- OBJ-005 appears only in observations.csv

•display full records corresponding to those objects
cat orbital_observations.csv | grep 'OBJ-005'

Task P2: Identifying Dominant Objects
Using command-line tools:
•determine which object appears most frequently
- from the last task we can see it's OBJ-004

•display several records corresponding to this object
cat orbital_observations.csv | grep 'OBJ-004' | head 
2026-03-01 12:25:00,OBJ-004,14.99,7.9,522,1.0,OK,0
2026-03-01 13:15:00,OBJ-004,14.63,7.05,436,0.72,OK,0
2026-03-01 13:25:00,OBJ-004,14.15,7.7,403,0.6892904162953539,DEGRADED,0
2026-03-01 13:55:00,OBJ-004,16.41227277628497,7.61,545,0.9,OK,1
2026-03-01 14:05:00,OBJ-004,16.3,7.39,503,0.95,OK,0
2026-03-01 14:25:00,OBJ-004,14.41,7.99,552,0.92,OK,0
2026-03-01 14:30:00,OBJ-004,16.2,7.43,462,0.81,OK,0
2026-03-01 14:40:00,OBJ-004,16.15,7.78,554,0.71,OK,0
2026-03-01 15:05:00,OBJ-004,16.44,7.87,503,0.86,OK,0
2026-03-01 15:25:00,OBJ-004,15.56,7.34,502,0.78,OK,0

•compare its frequency to other objects
- again, we can see it's frequency in the last task

Task P3: Cross-Feature Analysis
Using command-line tools:
•identify records where temperature is high and signal strength is low
- these records have temperature greater than 18 and signal strength smaller than 0.6

•display these records
awk -F',' -v t=18 -v s=0.6 'NR==1 || ($3!="INVALID" && $3+0>=t && $6+0<=s)' orbital_observations.csv
timestamp,object_id,temperature,velocity,altitude,signal_strength,sensor_status,anomaly_flag
2026-03-01 17:45:00,OBJ-002,18.28077696635233,7.08,414,0.5595405810088898,DEGRADED,1
2026-03-02 02:20:00,OBJ-005,18.119564422667587,7.58,473,0.35019132884077214,DEGRADED,1
2026-03-03 04:40:00,OBJ-004,18.77887403889184,7.0,470,0.4148018662984594,DEGRADED,1
2026-03-03 05:30:00,OBJ-004,18.819514997340818,7.3,570,0.5504117959553989,DEGRADED,1

•determine how many such cases exist
with my filter there is exactly 4 such cases

Task P4: Consistency Check Between Features
Using command-line tools:
•identify records where sensor status is not OK
awk -F',' -v ss="OK" 'NR==1 || ($7!=ss)' orbital_observations.csv
...
2026-03-01 17:15:00,OBJ-002,15.48,7.76,571,0.30241777776674933,DEGRADED,0
2026-03-01 17:25:00,OBJ-002,15.57,7.99,512,0.23303451887815402,DEGRADED,0
2026-03-01 17:30:00,OBJ-001,14.76,7.34,415,0.4805073944749245,DEGRADED,0
2026-03-01 17:45:00,OBJ-002,18.28077696635233,7.08,414,0.5595405810088898,DEGRADED,1
2026-03-01 19:30:00,OBJ-003,15.83,7.18,427,0.4719834845889073,DEGRADED,0
2026-03-01 23:10:00,OBJ-001,16.22,7.0,477,0.5326810499028489,DEGRADED,0
2026-03-02 00:40:00,OBJ-001,16.05,7.56,517,0.703581833487311,DEGRADED,0
2026-03-02 01:35:00,OBJ-005,15.74,7.85,470,0.6119019373745191,DEGRADED,0
2026-03-02 01:50:00,OBJ-004,16.07,7.83,481,0.6215611211512457,DEGRADED,0
2026-03-02 02:20:00,OBJ-005,18.119564422667587,7.58,473,0.35019132884077214,DEGRADED,1
...

•check whether these records are associated with anomalies
- they aren't

•display inconsistent cases (e.g., sensor not OK but anomaly flag = 0)
awk -F',' -v ss="OK" -v af=0 'NR==1 || ($7!=ss && $8+0==af)' orbital_observations.csv
timestamp,object_id,temperature,velocity,altitude,signal_strength,sensor_status,anomaly_flag
2026-03-01 12:10:00,OBJ-001,16.29,7.03,451,0.5061892083657298,DEGRADED,0
2026-03-01 12:40:00,OBJ-001,14.5,7.8,497,0.5639619172711572,DEGRADED,0
2026-03-01 13:25:00,OBJ-004,14.15,7.7,403,0.6892904162953539,DEGRADED,0
2026-03-01 15:10:00,OBJ-003,15.93,7.74,440,0.6257392221707015,DEGRADED,0
2026-03-01 15:50:00,OBJ-004,15.53,7.15,439,0.6708160805828866,DEGRADED,0
...

Task P5: Preparing Data for Model Input
Using command-line tools:
•create a new dataset containing the following feature columns: temperature, velocity, altitude, signal strength
cut -d ',' -f3,4,5,6 orbital_observations.csv

•exclude records with invalid values
cut -d ',' -f3,4,5,6 orbital_observations.csv | grep -v 'INVALID'

•save the result to a new file
cut -d ',' -f3,4,5,6 orbital_observations.csv | grep -v 'INVALID' > ../processed/model_input.csv 