import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set the backend *before* importing pyplot
import matplotlib.pyplot as plt

# loading the dataset
data = pd.read_csv('data/pokemon.csv')

# displaying the first 5 rows of the dataset
print(data.head())



# Data Preprocessing --------

# print(data.isnull().sum())         # <-- checking for missing values
data = data.drop(columns=['Type 2'])       # <-- removing 'Type 2' column
# data.info()                 # <-- checking data types and non-null counts



# Data Visualization --------

type_count = data['Type 1'].value_counts()     # <-- counting occurrences of each type in 'Type 1' column
# print(type_count)
type_names = type_count.index.to_list()        # <-- getting the list of unique type names
# print(type_names)

# Plotting the bar chart
plt.figure(figsize=(12, 8))
plt.barh( type_names, type_count.values)
plt.title('Count of Pokémon by Primary Type (Type 1)', fontsize=16)
plt.xlabel('Number of Pokémon', fontsize=12)
plt.ylabel('Type 1', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis() 
plt.savefig('datavisuals/Type_1_bar_chart.png') # Save the plot to a file










