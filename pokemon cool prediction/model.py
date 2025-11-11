import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set the backend *before* importing pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# loading the dataset
data = pd.read_csv('data/pokemon.csv')

# displaying the first 5 rows of the dataset
# print(data.head())



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
# plt.figure(figsize=(12, 8))
# plt.barh( type_names, type_count.values)
# plt.title('Count of Pokémon by Primary Type (Type 1)', fontsize=16)
# plt.xlabel('Number of Pokémon', fontsize=12)
# plt.ylabel('Type 1', fontsize=12)
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.gca().invert_yaxis() 
# plt.savefig('datavisuals/Type_1_bar_chart.png') # Save 

# Histogram of 'Total' attribute
# plt.figure(figsize=(12,8))
# plt.hist(data['Type 1'], bins=len(type_names), color='skyblue', edgecolor='black')
# plt.title('Distribution of Pokémon by Primary Type (Type 1)', fontsize=16)
# plt.xlabel('Type 1', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig('datavisuals/Type_1_histogram.png') # Save



# X Y Split --------

X = data.drop(columns=['#','Name','Type 1','Total'])
Y = data['Type 1']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Making Predictions
print(model.predict(X_test))
print(Y_test.values)   # comparing with actual values

# Model Evaluation --------
accuracy = model.score(X_test, Y_test)
print(f'Model Accuracy: {accuracy*100:.2f}%')      # 23.75% accuracy





