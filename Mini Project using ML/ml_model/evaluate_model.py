"""
Model Evaluation and Testing Module
Author: [Your Name]
Date: July 2025

This module contains functions for comprehensive model evaluation
including cross-validation, feature importance analysis, and performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def evaluate_model_performance(model, X_test, y_test, X_train=None, y_train=None):
    """
    Comprehensive model evaluation with multiple metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("Model Performance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return metrics

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation and return scores
    """
    print(f"Performing {cv}-fold cross-validation...")
    
    # Stratified K-Fold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance
    """
    if hasattr(model, 'coef_'):
        # For logistic regression
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance = model.feature_importances_
    else:
        print("Model doesn't support feature importance analysis")
        return
    
    # Create feature importance dataframe
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance Rankings:")
    print("-" * 30)
    for idx, row in feature_imp.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp, x='importance', y='feature')
    plt.title('Feature Importance Analysis')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    return feature_imp

if __name__ == "__main__":
    # Load model and test data for evaluation
    try:
        model = joblib.load('../ml_model/diabetes_model.pkl')
        scaler = joblib.load('../ml_model/scaler.pkl')
        
        # Load test data (you would need to save this during training)
        print("Model evaluation module loaded successfully!")
        print("Use the functions above to evaluate your trained model.")
        
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
