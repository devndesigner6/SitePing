"""
Data Preprocessing Utilities
Author: [Your Name]
Date: July 2025

This module contains utility functions for data cleaning, preprocessing,
and feature engineering for the diabetes prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class DiabetesDataProcessor:
    """
    A comprehensive data processor for diabetes dataset
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Load the diabetes dataset
        """
        try:
            self.df = pd.read_csv(filepath)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """
        Basic data exploration
        """
        print("Dataset Information:")
        print("=" * 50)
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\\nData types:")
        print(self.df.dtypes)
        print(f"\\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\\nBasic statistics:")
        print(self.df.describe())
        
    def clean_data(self, strategy='median'):
        """
        Clean the data by handling zero values in biological features
        """
        # Features that shouldn't have zero values
        biological_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        self.df_cleaned = self.df.copy()
        
        print("Cleaning data...")
        print("-" * 30)
        
        for feature in biological_features:
            if feature in self.df_cleaned.columns:
                zero_count = (self.df_cleaned[feature] == 0).sum()
                
                if zero_count > 0:
                    if strategy == 'median':
                        replacement_value = self.df_cleaned[self.df_cleaned[feature] != 0][feature].median()
                    elif strategy == 'mean':
                        replacement_value = self.df_cleaned[self.df_cleaned[feature] != 0][feature].mean()
                    else:
                        continue
                    
                    self.df_cleaned[feature] = self.df_cleaned[feature].replace(0, replacement_value)
                    print(f"{feature}: Replaced {zero_count} zeros with {strategy} ({replacement_value:.2f})")
        
        print("Data cleaning completed!")
        return self.df_cleaned
    
    def visualize_distributions(self, data=None):
        """
        Visualize feature distributions
        """
        if data is None:
            data = self.df_cleaned if hasattr(self, 'df_cleaned') else self.df
        
        numeric_features = data.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != 'Outcome']
        
        n_features = len(numeric_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, feature in enumerate(numeric_features):
            row = idx // n_cols
            col = idx % n_cols
            
            # Distribution by target
            no_diabetes = data[data['Outcome'] == 0][feature]
            diabetes = data[data['Outcome'] == 1][feature]
            
            axes[row, col].hist(no_diabetes, alpha=0.7, label='No Diabetes', bins=20, density=True)
            axes[row, col].hist(diabetes, alpha=0.7, label='Diabetes', bins=20, density=True)
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
        
        # Hide empty subplots
        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def prepare_features(self, target_column='Outcome', test_size=0.2, random_state=42):
        """
        Prepare features for machine learning
        """
        if not hasattr(self, 'df_cleaned'):
            print("Please clean data first using clean_data() method")
            return None
        
        # Separate features and target
        X = self.df_cleaned.drop(target_column, axis=1)
        y = self.df_cleaned[target_column]
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Data prepared for ML:")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Target distribution - Train: {y_train.value_counts().values}")
        print(f"Target distribution - Test: {y_test.value_counts().values}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_processed_data(self, filepath):
        """
        Save the cleaned dataset
        """
        if hasattr(self, 'df_cleaned'):
            self.df_cleaned.to_csv(filepath, index=False)
            print(f"Processed data saved to {filepath}")
        else:
            print("No cleaned data to save. Please run clean_data() first.")

def create_sample_dataset():
    """
    Create a sample dataset for testing (if original data is not available)
    """
    np.random.seed(42)
    n_samples = 768
    
    # Generate synthetic diabetes-like data
    data = {
        'Pregnancies': np.random.poisson(3, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples),
        'BloodPressure': np.random.normal(70, 15, n_samples),
        'SkinThickness': np.random.normal(25, 10, n_samples),
        'Insulin': np.random.gamma(2, 50, n_samples),
        'BMI': np.random.normal(32, 8, n_samples),
        'DiabetesPedigreeFunction': np.random.gamma(1, 0.5, n_samples),
        'Age': np.random.gamma(2, 15, n_samples) + 20
    }
    
    # Create synthetic target based on features
    risk_score = (
        (data['Glucose'] > 140) * 0.3 +
        (data['BMI'] > 30) * 0.2 +
        (data['Age'] > 50) * 0.2 +
        (data['DiabetesPedigreeFunction'] > 0.5) * 0.1 +
        np.random.normal(0, 0.2, n_samples)
    )
    
    data['Outcome'] = (risk_score > 0.4).astype(int)
    
    # Add some zero values to simulate missing data
    for feature in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        zero_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        data[feature][zero_indices] = 0
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Example usage
    processor = DiabetesDataProcessor()
    
    # Try to load real data, if not available create sample
    try:
        df = processor.load_data('../data/diabetes.csv')
    except:
        print("Creating sample dataset for demonstration...")
        df = create_sample_dataset()
        df.to_csv('../data/diabetes.csv', index=False)
        processor.df = df
        print("Sample dataset created and saved!")
    
    processor.explore_data()
    processor.clean_data()
    processor.visualize_distributions()
    X_train, X_test, y_train, y_test = processor.prepare_features()
