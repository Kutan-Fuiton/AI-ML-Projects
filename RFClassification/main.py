import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('titanic_model.pkl')

# Create new data for prediction (example)
new_passenger = {
    'Pclass': 3,
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.25,
    'Sex': 1,               # 1 for male, 0 for female
    'Embarked': 2           # C --> 0; Q --> 1; S --> 2
}

# Convert to DataFrame (important!)
new_data = pd.DataFrame([new_passenger])

# Make prediction
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)

print(f"Survival Prediction: {'Survived' if prediction[0] == 1 else 'Did not survive'}")
print(f"Probability: {probability[0]}")