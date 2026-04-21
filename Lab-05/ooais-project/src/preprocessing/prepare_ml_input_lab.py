import csv

# ZAŁADOWANIE KOLUMN DO LIST
dataset_file = "data/processed/observations_valid.csv"

with open(dataset_file) as file:
    reader = csv.DictReader(file)
    rows = list(reader)

temperatures = []
velocity = []
altitude = []
signal_strength = []
accepted_count = 0
rejected_count = 0

for row in rows:
    # TO DO - zamiast if else poniżej zrobić ładne zmienne przechowujące wyniki true i false sprawdzania
    # missing_values_check = 
    # not_float_values_check =
    # negative_altitude_check = 

    if float(row['altitude']) > 0:
        temperatures.append(float(row['temperature']))
        velocity.append(float(row['velocity']))
        altitude.append(float(row['altitude']))
        signal_strength.append(float(row['signal_strength']))
    else:
        rejected_count += 1

    accepted_count +=1
                        
print(f"=== ML Input Preparation: Loading and Conversion ==\nInput file: {dataset_file}\nRecords loaded: {len(rows)}\nRecords accepted: {accepted_count}\nRecords rejected: {rejected_count}")

# NORMALIZACJA MIN-MAX
temperatures_normalized = []
velocity_normalized = []
altitude_normalized = []
signal_strength_normalized = []

for i in range (0, accepted_count):
    temperatures_normalized.append((temperatures[i] - min(temperatures)) / (max(temperatures) - min(temperatures)))
    velocity_normalized.append((velocity[i] - min(velocity)) / (max(velocity) - min(velocity)))
    altitude_normalized.append((altitude[i] - min(altitude)) / (max(altitude) - min(altitude)))
    signal_strength_normalized.append((signal_strength[i] - min(signal_strength)) / (max(signal_strength) - min(signal_strength)))

print()