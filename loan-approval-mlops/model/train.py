import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from joblib import dump

### Load dataset

data = pd.read_csv("data/loanApprovalDataset.csv")
print("Dataset loaded successfully.")


### Preprocess dataset

data = data.dropna(subset = ['Loan_Status'])

# print(data.isnull().sum())

numericals = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
categoricals = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
target = 'Loan_Status'


### Prepare features and target variable
X = data[numericals + categoricals]
y = data[target].map({'Y': 1, 'N': 0})

### Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Create preprocessing pipelines for numerical and categorical data
numericals_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categoricals_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(transformers = [
    ('num', numericals_pipe, numericals),
    ('cat', categoricals_pipe, categoricals)
])

### Create the model pipeline
model = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver = 'liblinear'))
])

### Train the model
model.fit(X_train, y_train)

### Evaluate the model
score = model.score(X_test, y_test)
print(f"Model Test Score: {score:.4f}")
# y_pred = model.predict(X_test)

### Testing the model persistence
# print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))


### Save the trained model
save_path = 'model/loan_model.joblib'
dump(model, save_path)
print(f"Pipeline saved to {save_path}")