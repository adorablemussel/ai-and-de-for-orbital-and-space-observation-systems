with open("data/raw/observations.csv") as f:
    lines = f.readlines()
print("Number of records:", len(lines)-1)

objects = [line.split(",")[1] for line in lines[1:]]
print("Objects:", set(objects))

header = lines[0].strip().split(",")

for i, line in enumearte(lines[1:], start=2):
    