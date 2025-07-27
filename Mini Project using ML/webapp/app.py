"""
Diabetes Prediction Web Application
Author: Hemanth Peddada
Date: July 2025

Advanced Flask web application for diabetes risk prediction using machine learning.
Features real-time prediction, comprehensive API, and professional user interface.
"""

from flask import Flask, render_template, request, jsonify
import os
import logging
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Simple mock model for demonstration
class MockDiabetesModel:
    def predict(self, features):
        # Simple risk calculation based on key factors
        glucose = features[1]
        bmi = features[5]
        age = features[7]
        
        # Basic risk assessment
        risk_score = 0
        if glucose > 140: risk_score += 0.4
        if bmi > 30: risk_score += 0.3
        if age > 45: risk_score += 0.2
        
        # Add some randomness for demonstration
        risk_score += random.uniform(-0.1, 0.1)
        
        return 1 if risk_score > 0.5 else 0
    
    def predict_proba(self, features):
        glucose = features[1]
        bmi = features[5]
        age = features[7]
        
        # Calculate probability
        prob = 0.2  # Base probability
        if glucose > 100: prob += (glucose - 100) * 0.003
        if bmi > 25: prob += (bmi - 25) * 0.01
        if age > 30: prob += (age - 30) * 0.005
        
        # Cap between 0 and 1
        prob = max(0.05, min(0.95, prob))
        
        return [1 - prob, prob]

# Initialize mock model
model = MockDiabetesModel()
logger.info("Mock model initialized successfully")

# Feature names for the model
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Extract features from form
        features = []
        for feature in FEATURE_NAMES:
            value = request.form.get(feature)
            if value is None or value == '':
                return render_template('index.html', 
                                     error=f"Please provide a value for {feature}")
            features.append(float(value))
        
        # Validate input ranges
        validation_errors = validate_input(features)
        if validation_errors:
            return render_template('index.html', error=validation_errors)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'probability_no_diabetes': round(probability[0] * 100, 2),
            'probability_diabetes': round(probability[1] * 100, 2),
            'features': dict(zip(FEATURE_NAMES, features))
        }
        
        logger.info(f"Prediction made: {result}")
        return render_template('index.html', result=result)
        
    except ValueError as e:
        return render_template('index.html', 
                             error="Please enter valid numeric values for all fields.")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('index.html', 
                             error="An error occurred during prediction. Please try again.")

def validate_input(features):
    """Validate user input ranges"""
    ranges = {
        'Pregnancies': (0, 20),
        'Glucose': (0, 300),
        'BloodPressure': (0, 200),
        'SkinThickness': (0, 100),
        'Insulin': (0, 1000),
        'BMI': (0, 100),
        'DiabetesPedigreeFunction': (0, 3),
        'Age': (1, 120)
    }
    
    for i, (feature, value) in enumerate(zip(FEATURE_NAMES, features)):
        min_val, max_val = ranges[feature]
        if not (min_val <= value <= max_val):
            return f"{feature} should be between {min_val} and {max_val}"
    
    return None

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'diabetes-prediction-api',
        'version': '1.0.0',
        'author': 'Hemanth Peddada',
        'model_loaded': True
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_type': 'MockDiabetesModel',
        'features': FEATURE_NAMES,
        'author': 'Hemanth Peddada',
        'description': 'Diabetes risk prediction model for demonstration'
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        features = [data.get(feature, 0) for feature in FEATURE_NAMES]
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        return jsonify({
            'prediction': int(prediction),
            'probability_diabetes': float(probability[1]),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple samples"""
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        predictions = []
        
        for idx, sample in enumerate(samples):
            try:
                features = [sample.get(feature, 0) for feature in FEATURE_NAMES]
                prediction = model.predict(features)
                probability = model.predict_proba(features)
                
                predictions.append({
                    'sample_id': idx,
                    'prediction': int(prediction),
                    'probability_diabetes': float(probability[1]),
                    'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.3 else 'Low'
                })
            except Exception as e:
                predictions.append({
                    'sample_id': idx,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'total_samples': len(samples),
            'successful_predictions': len([p for p in predictions if 'error' not in p])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Starting Diabetes Prediction Web Application")
    print(f"🌐 Access the application at: http://localhost:{port}")
    print("👨‍💻 Developed by: Hemanth Peddada")
    app.run(host='0.0.0.0', port=port, debug=True)
