import pandas as pd


data = pd.read_csv('data/pokemon.csv')

print(data.head())

# data preprocessing

# checking the nan values in columns
print(data.isnull().sum())

# removing type 2 column
data = data.drop(columns=['Type 2'])
