import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

#Load dataset
data = pd.read_csv('data/Titanic_dataset.csv')

# Preprocess dataset
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)


X = data.drop(columns=['Survived', 'PassengerId'])
y = data['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'data/titanic_model.pkl')
print('Model saved successfully...')

# Predict
y_pred = model.predict(X_test)
print(f'Predictions: {y_pred}')

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')