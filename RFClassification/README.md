# Titanic Survival Prediction - Random Forest Classifier

A machine learning project that predicts passenger survival on the Titanic using Random Forest classification.

## 📊 Project Overview

This project demonstrates a complete machine learning workflow:
- Data preprocessing and cleaning
- Feature engineering
- Model training with Random Forest
- Model evaluation and prediction

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kutan-Fuiton/AI-ML-Projects
   cd RFClassification

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt


## 🎯 Usage

1. **Train the model**
```bash
python model.py
```
This will:
- Load and preprocess the Titanic dataset
- Train a Random Forest classifier
- Save the model as ``` titanic_model.pkl ```
- Display model accuracy

2. **Make Prediction**
```bash
python main.py
```
This will load the trained model and make a sample prediction.