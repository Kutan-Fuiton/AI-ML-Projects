# create_sample.py
import pandas as pd
import os

def create_sample_data():
    if os.path.exists('data/Titanic_dataset.csv'):
        # Create small sample for git
        full_data = pd.read_csv('data/Titanic_dataset.csv')
        sample_data = full_data.sample(30, random_state=42)
        sample_data.to_csv('sample_titanic.csv', index=False)
        print("Sample dataset created: sample_titanic.csv")
    else:
        print("Full dataset not found. Please download it.")

if __name__ == "__main__":
    create_sample_data()