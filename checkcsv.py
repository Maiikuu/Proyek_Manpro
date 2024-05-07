import pandas as pd

# Load data from CSV file
data = pd.read_csv('januari.csv', delimiter=';', skiprows=18)

# Print column names
print(data.columns)

# Check the actual column names in the DataFrame
