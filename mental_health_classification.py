# Mental Health Classification Project

# This script uses a dataset from Kaggle to predict whether a person seeks mental health treatment
# using several classification models. We evaluate the models using Accuracy, Precision, and Recall.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("survey.csv")  # Make sure the CSV is in the same directory
print(df.head())

# Data overview
print(df.info())
print(df.describe())

# Drop unused or high-missing columns
df.drop(columns=['comments', 'state', 'Timestamp'], inplace=True)
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

print(df.head())

# Define features and target
X = df.drop('treatment', axis=1)
y = df['treatment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Train and evaluate
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    results.append((name, acc, prec, rec))

# Print results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall"])
print("\nModel Evaluation Results:")
print(results_df)

# Print detailed report for best model (Random Forest)
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred))
