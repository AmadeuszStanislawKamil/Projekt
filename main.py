# Load CSV
import csv
with open('PlantData/Plant_1_Generation_Data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)