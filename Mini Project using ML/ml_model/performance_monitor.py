"""
Performance Monitoring and Analytics Module
Author: Hemanth Peddada
Date: July 2025

This module provides comprehensive performance monitoring, analytics,
and model drift detection for the diabetes prediction system.
"""

import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

class ModelPerformanceMonitor:
    """
    Advanced monitoring system for model performance and data drift
    """
    
    def __init__(self, model_path='../ml_model/diabetes_model.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('../ml_model/scaler.pkl')
        self.predictions_log = []
        self.performance_metrics = {}
        
        # Setup logging
        logging.basicConfig(
            filename='../ml_model/model_performance.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self, features, prediction, probability, response_time):
        """Log individual predictions for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': int(prediction),
            'probability': float(probability),
            'response_time': response_time
        }
        
        self.predictions_log.append(log_entry)
        self.logger.info(f"Prediction logged: {log_entry}")
    
    def calculate_daily_metrics(self, date=None):
        """Calculate performance metrics for a specific date"""
        if date is None:
            date = datetime.now().date()
        
        daily_predictions = [
            p for p in self.predictions_log 
            if datetime.fromisoformat(p['timestamp']).date() == date
        ]
        
        if not daily_predictions:
            return None
        
        metrics = {
            'date': date.isoformat(),
            'total_predictions': len(daily_predictions),
            'avg_response_time': np.mean([p['response_time'] for p in daily_predictions]),
            'avg_probability': np.mean([p['probability'] for p in daily_predictions]),
            'positive_predictions': sum(1 for p in daily_predictions if p['prediction'] == 1),
            'prediction_rate': sum(1 for p in daily_predictions if p['prediction'] == 1) / len(daily_predictions)
        }
        
        return metrics
    
    def detect_data_drift(self, new_data, reference_data=None):
        """Detect potential data drift in incoming predictions"""
        if reference_data is None:
            # Use last 30 days as reference
            cutoff_date = datetime.now() - timedelta(days=30)
            reference_predictions = [
                p for p in self.predictions_log
                if datetime.fromisoformat(p['timestamp']) >= cutoff_date
            ]
            
            if not reference_predictions:
                return {"status": "insufficient_data"}
            
            reference_features = [p['features'] for p in reference_predictions]
        else:
            reference_features = reference_data
        
        # Calculate feature statistics
        ref_stats = self._calculate_feature_stats(reference_features)
        new_stats = self._calculate_feature_stats([new_data])
        
        # Detect drift using statistical tests
        drift_detected = False
        drift_features = []
        
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        for i, feature in enumerate(feature_names):
            ref_mean = ref_stats['means'][i]
            ref_std = ref_stats['stds'][i]
            new_value = new_data[i]
            
            # Z-score based drift detection
            z_score = abs((new_value - ref_mean) / ref_std) if ref_std > 0 else 0
            
            if z_score > 3:  # 3-sigma rule
                drift_detected = True
                drift_features.append({
                    'feature': feature,
                    'z_score': z_score,
                    'new_value': new_value,
                    'reference_mean': ref_mean,
                    'reference_std': ref_std
                })
        
        return {
            'drift_detected': drift_detected,
            'drift_features': drift_features,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_feature_stats(self, features_list):
        """Calculate statistical properties of features"""
        if not features_list:
            return {'means': [], 'stds': []}
        
        features_array = np.array(features_list)
        return {
            'means': np.mean(features_array, axis=0).tolist(),
            'stds': np.std(features_array, axis=0).tolist()
        }
    
    def generate_performance_report(self, days=7):
        """Generate comprehensive performance report"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        daily_metrics = []
        current_date = start_date
        
        while current_date <= end_date:
            metrics = self.calculate_daily_metrics(current_date)
            if metrics:
                daily_metrics.append(metrics)
            current_date += timedelta(days=1)
        
        if not daily_metrics:
            return {"status": "no_data", "message": "No predictions found in the specified period"}
        
        # Create summary statistics
        summary = {
            'period': f"{start_date} to {end_date}",
            'total_predictions': sum(m['total_predictions'] for m in daily_metrics),
            'avg_daily_predictions': np.mean([m['total_predictions'] for m in daily_metrics]),
            'avg_response_time': np.mean([m['avg_response_time'] for m in daily_metrics]),
            'avg_positive_rate': np.mean([m['prediction_rate'] for m in daily_metrics]),
            'daily_metrics': daily_metrics
        }
        
        return summary
    
    def create_performance_dashboard(self, save_path='../ml_model/performance_dashboard.png'):
        """Create visual performance dashboard"""
        if len(self.predictions_log) < 10:
            print("Insufficient data for dashboard creation")
            return
        
        # Prepare data
        df = pd.DataFrame(self.predictions_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Diabetes Prediction Model - Performance Dashboard\nAuthor: Hemanth Peddada', 
                     fontsize=16, fontweight='bold')
        
        # 1. Predictions over time
        daily_counts = df.groupby('date').size()
        axes[0, 0].plot(daily_counts.index, daily_counts.values, marker='o')
        axes[0, 0].set_title('Daily Prediction Volume')
        axes[0, 0].set_ylabel('Number of Predictions')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Response time distribution
        axes[0, 1].hist(df['response_time'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Response Time Distribution')
        axes[0, 1].set_xlabel('Response Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Prediction probability distribution
        axes[1, 0].hist(df['probability'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].set_xlabel('Diabetes Probability')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Daily positive prediction rate
        daily_positive_rate = df.groupby('date')['prediction'].mean()
        axes[1, 1].plot(daily_positive_rate.index, daily_positive_rate.values, marker='s', color='red')
        axes[1, 1].set_title('Daily Positive Prediction Rate')
        axes[1, 1].set_ylabel('Positive Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Performance dashboard saved to {save_path}")

def simulate_monitoring_data():
    """Simulate monitoring data for demonstration"""
    monitor = ModelPerformanceMonitor()
    
    print("🔄 Simulating monitoring data...")
    
    # Simulate 100 predictions over the last week
    for i in range(100):
        # Generate random but realistic features
        features = [
            np.random.randint(0, 10),  # Pregnancies
            np.random.normal(120, 30),  # Glucose
            np.random.normal(80, 15),   # BloodPressure
            np.random.normal(25, 10),   # SkinThickness
            np.random.normal(100, 50),  # Insulin
            np.random.normal(30, 8),    # BMI
            np.random.uniform(0.1, 1.5),  # DiabetesPedigreeFunction
            np.random.randint(21, 80)   # Age
        ]
        
        # Make prediction
        features_scaled = monitor.scaler.transform([features])
        prediction = monitor.model.predict(features_scaled)[0]
        probability = monitor.model.predict_proba(features_scaled)[0][1]
        response_time = np.random.uniform(0.1, 0.5)
        
        # Log prediction with random timestamp in the last week
        original_time = datetime.now()
        random_offset = timedelta(days=np.random.uniform(0, 7))
        
        log_entry = {
            'timestamp': (original_time - random_offset).isoformat(),
            'features': features,
            'prediction': int(prediction),
            'probability': float(probability),
            'response_time': response_time
        }
        
        monitor.predictions_log.append(log_entry)
    
    print("✅ Monitoring data simulation completed!")
    return monitor

if __name__ == "__main__":
    print("📊 Model Performance Monitoring System")
    print("Author: Hemanth Peddada")
    print("=" * 50)
    
    # Simulate data and create dashboard
    monitor = simulate_monitoring_data()
    
    # Generate performance report
    report = monitor.generate_performance_report(days=7)
    print("\n📈 Weekly Performance Report:")
    print(f"Total Predictions: {report['total_predictions']}")
    print(f"Average Daily Predictions: {report['avg_daily_predictions']:.1f}")
    print(f"Average Response Time: {report['avg_response_time']:.3f}s")
    print(f"Average Positive Rate: {report['avg_positive_rate']:.3f}")
    
    # Create dashboard
    monitor.create_performance_dashboard()
    
    # Test drift detection
    test_features = [5, 150, 85, 30, 120, 35, 0.8, 45]
    drift_result = monitor.detect_data_drift(test_features)
    print(f"\n🔍 Data Drift Detection: {'Detected' if drift_result.get('drift_detected') else 'Not Detected'}")
    
    print("\n✅ Performance monitoring demonstration completed!")
