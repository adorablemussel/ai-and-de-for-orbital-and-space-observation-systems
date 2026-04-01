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