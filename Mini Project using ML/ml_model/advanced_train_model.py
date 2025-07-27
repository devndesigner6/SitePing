"""
Advanced Model Training and Comparison Script
Author: Hemanth Peddada
Date: July 2025

This script implements advanced machine learning techniques including:
- Multiple algorithm comparison
- Hyperparameter optimization
- Cross-validation with stratification
- Advanced feature engineering
- Model persistence and versioning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedDiabetesPredictor:
    """
    Advanced diabetes prediction system with multiple algorithms and optimization
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}
        
    def load_and_preprocess_data(self, filepath='../data/diabetes.csv'):
        """Load and preprocess the diabetes dataset with advanced techniques"""
        print("🔄 Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(filepath)
        print(f"📊 Dataset shape: {df.shape}")
        
        # Advanced preprocessing
        df_processed = self._advanced_preprocessing(df)
        
        # Feature engineering
        df_engineered = self._feature_engineering(df_processed)
        
        return df_engineered
    
    def _advanced_preprocessing(self, df):
        """Advanced data preprocessing techniques"""
        df_clean = df.copy()
        
        # Handle biological impossibilities with domain knowledge
        biological_features = {
            'Glucose': {'min': 50, 'max': 300, 'replace_zeros': True},
            'BloodPressure': {'min': 40, 'max': 200, 'replace_zeros': True},
            'SkinThickness': {'min': 5, 'max': 60, 'replace_zeros': True},
            'Insulin': {'min': 10, 'max': 800, 'replace_zeros': True},
            'BMI': {'min': 15, 'max': 60, 'replace_zeros': True}
        }
        
        for feature, constraints in biological_features.items():
            if feature in df_clean.columns:
                # Replace zeros with median of non-zero values
                if constraints['replace_zeros']:
                    non_zero_median = df_clean[df_clean[feature] != 0][feature].median()
                    df_clean[feature] = df_clean[feature].replace(0, non_zero_median)
                
                # Cap extreme outliers
                df_clean[feature] = np.clip(df_clean[feature], 
                                          constraints['min'], 
                                          constraints['max'])
        
        print("✅ Advanced preprocessing completed")
        return df_clean
    
    def _feature_engineering(self, df):
        """Create engineered features based on domain knowledge"""
        df_eng = df.copy()
        
        # BMI categories
        df_eng['BMI_Category'] = pd.cut(df_eng['BMI'], 
                                       bins=[0, 18.5, 25, 30, 100], 
                                       labels=[0, 1, 2, 3])
        
        # Age groups
        df_eng['Age_Group'] = pd.cut(df_eng['Age'], 
                                    bins=[0, 30, 45, 60, 100], 
                                    labels=[0, 1, 2, 3])
        
        # Glucose risk levels
        df_eng['Glucose_Risk'] = pd.cut(df_eng['Glucose'], 
                                       bins=[0, 100, 125, 200, 300], 
                                       labels=[0, 1, 2, 3])
        
        # Insulin resistance indicator
        df_eng['Insulin_Resistance'] = (df_eng['Insulin'] > df_eng['Insulin'].quantile(0.75)).astype(int)
        
        # Metabolic syndrome indicator
        df_eng['Metabolic_Syndrome'] = ((df_eng['BMI'] > 30) & 
                                       (df_eng['Glucose'] > 100)).astype(int)
        
        print("✅ Feature engineering completed")
        return df_eng
    
    def prepare_models(self):
        """Initialize multiple models with optimized parameters"""
        self.models = {
            'Logistic_Regression': LogisticRegression(random_state=42),
            'Random_Forest': RandomForestClassifier(random_state=42),
            'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Hyperparameter grids for optimization
        self.param_grids = {
            'Logistic_Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient_Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models with hyperparameter optimization"""
        print("🚀 Training and evaluating multiple models...")
        
        results = {}
        best_score = 0
        
        for model_name, model in self.models.items():
            print(f"\n📈 Training {model_name}...")
            
            # Hyperparameter optimization
            grid_search = GridSearchCV(
                model, 
                self.param_grids[model_name],
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'best_params': grid_search.best_params_
            }
            
            results[model_name] = {
                'model': best_model,
                'metrics': metrics
            }
            
            print(f"✅ {model_name} - ROC AUC: {metrics['roc_auc']:.4f}")
            
            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                self.best_model = best_model
                self.best_model_name = model_name
        
        self.results = results
        print(f"\n🏆 Best Model: {self.best_model_name} (ROC AUC: {best_score:.4f})")
        return results
    
    def save_models_and_results(self):
        """Save trained models and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create models directory
        os.makedirs('../ml_model', exist_ok=True)
        
        # Save best model and scaler
        joblib.dump(self.best_model, '../ml_model/diabetes_model.pkl')
        joblib.dump(self.scaler, '../ml_model/scaler.pkl')
        
        # Save feature names
        with open('../ml_model/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save detailed results
        results_summary = {}
        for model_name, result in self.results.items():
            results_summary[model_name] = result['metrics']
        
        with open(f'../ml_model/training_results_{timestamp}.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("✅ Models and results saved successfully!")
    
    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("="*60)
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            print(f"\n📊 {model_name}:")
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1_score']:.4f}")
            print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
            print(f"   Best Params: {metrics['best_params']}")
        
        print(f"\n🏆 SELECTED MODEL: {self.best_model_name}")
        print(f"   Final ROC AUC: {self.results[self.best_model_name]['metrics']['roc_auc']:.4f}")

def main():
    """Main training pipeline with advanced features"""
    print("🚀 Advanced Diabetes Prediction Model Training")
    print("Author: Hemanth Peddada")
    print("="*50)
    
    # Initialize predictor
    predictor = AdvancedDiabetesPredictor()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data()
    
    # Prepare features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    predictor.feature_names = X.columns.tolist()
    
    # Scale features
    X_scaled = predictor.scaler.fit_transform(X)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Prepare and train models
    predictor.prepare_models()
    results = predictor.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Generate report
    predictor.generate_model_report()
    
    # Save everything
    predictor.save_models_and_results()
    
    print("\n✅ Advanced model training completed successfully!")
    print("🚀 Ready for deployment!")

if __name__ == "__main__":
    main()
