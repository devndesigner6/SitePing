"""
Diabetes Prediction Model Training Script
Author: [Your Name]
Date: July 2025

This script handles the complete machine learning pipeline:
- Data loading and preprocessing
- Model training with Logistic Regression
- Model evaluation and performance metrics
- Saving trained model and preprocessing components
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the diabetes dataset"""
    print("Loading dataset...")
    df = pd.read_csv('../data/diabetes.csv')
    
    print(f"Dataset shape: {df.shape}")
    print("\nDataset info:")
    print(df.info())
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Handle zero values that might represent missing data
    # In this dataset, 0 values in certain columns are likely missing
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in columns_with_zeros:
        if col in df.columns:
            # Replace 0 with median for that column
            median_val = df[df[col] != 0][col].median()
            df[col] = df[col].replace(0, median_val)
    
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f'{name} Accuracy: {accuracy:.4f}')
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    return best_model

def main():
    """Main training pipeline"""
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Prepare features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    print("\nFeature columns:", X.columns.tolist())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    best_model = train_models(X_train, X_test, y_train, y_test)
    
    # Create model directory if it doesn't exist
    os.makedirs('../ml_model', exist_ok=True)
    
    # Save the best model and scaler
    joblib.dump(best_model, '../ml_model/diabetes_model.pkl')
    joblib.dump(scaler, '../ml_model/scaler.pkl')
    
    print(f"\nModel and scaler saved successfully!")
    print(f"Final model accuracy: {accuracy_score(y_test, best_model.predict(X_test)):.4f}")

if __name__ == "__main__":
    main()
