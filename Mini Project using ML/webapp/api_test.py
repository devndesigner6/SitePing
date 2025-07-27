"""
API Documentation and Testing Module
Author: Hemanth Peddada
Date: July 2025

This module provides comprehensive API documentation and testing utilities
for the diabetes prediction web application.
"""

import requests
import json
import time
from typing import Dict, List

class DiabetesAPIClient:
    """
    Client for interacting with the Diabetes Prediction API
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> Dict:
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        response = requests.get(f"{self.base_url}/model-info")
        return response.json()
    
    def predict_single(self, features: Dict) -> Dict:
        """Make a single prediction"""
        response = requests.post(
            f"{self.base_url}/api/predict",
            json=features
        )
        return response.json()
    
    def predict_batch(self, samples: List[Dict]) -> Dict:
        """Make batch predictions"""
        response = requests.post(
            f"{self.base_url}/predict-batch",
            json={"samples": samples}
        )
        return response.json()

def test_api_endpoints():
    """Test all API endpoints with sample data"""
    client = DiabetesAPIClient()
    
    print("🧪 Testing Diabetes Prediction API")
    print("=" * 50)
    
    # Test health check
    print("1. Testing health check...")
    try:
        health = client.health_check()
        print(f"   ✅ Status: {health.get('status')}")
        print(f"   📊 Model loaded: {health.get('model_loaded')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test model info
    print("\n2. Testing model info...")
    try:
        info = client.get_model_info()
        print(f"   ✅ Model type: {info.get('model_type')}")
        print(f"   📋 Features: {len(info.get('features', []))} features")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test single prediction
    print("\n3. Testing single prediction...")
    try:
        sample_data = {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
        
        start_time = time.time()
        result = client.predict_single(sample_data)
        response_time = time.time() - start_time
        
        print(f"   ✅ Prediction: {result.get('prediction')}")
        print(f"   📊 Probability: {result.get('probability_diabetes', 0):.3f}")
        print(f"   ⚡ Response time: {response_time:.3f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test batch prediction
    print("\n4. Testing batch prediction...")
    try:
        batch_samples = [
            {
                "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66,
                "SkinThickness": 29, "Insulin": 0, "BMI": 26.6,
                "DiabetesPedigreeFunction": 0.351, "Age": 31
            },
            {
                "Pregnancies": 8, "Glucose": 183, "BloodPressure": 64,
                "SkinThickness": 0, "Insulin": 0, "BMI": 23.3,
                "DiabetesPedigreeFunction": 0.672, "Age": 32
            }
        ]
        
        start_time = time.time()
        batch_result = client.predict_batch(batch_samples)
        response_time = time.time() - start_time
        
        print(f"   ✅ Batch size: {batch_result.get('total_samples')}")
        print(f"   📊 Successful: {batch_result.get('successful_predictions')}")
        print(f"   ⚡ Response time: {response_time:.3f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ API testing completed!")

def generate_api_documentation():
    """Generate comprehensive API documentation"""
    docs = """
# Diabetes Prediction API Documentation
*Author: Hemanth Peddada*

## Overview
RESTful API for diabetes risk prediction using machine learning.

## Base URL
```
http://localhost:5000  (Development)
https://your-app.herokuapp.com  (Production)
```

## Endpoints

### 1. Health Check
**GET** `/health`

Check API status and model availability.

**Response:**
```json
{
    "status": "healthy",
    "service": "diabetes-prediction-api",
    "version": "1.0.0",
    "author": "Hemanth Peddada",
    "model_loaded": true
}
```

### 2. Model Information
**GET** `/model-info`

Get information about the loaded ML model.

**Response:**
```json
{
    "model_type": "LogisticRegression",
    "features": ["Pregnancies", "Glucose", "BloodPressure", ...],
    "author": "Hemanth Peddada",
    "description": "Diabetes risk prediction model"
}
```

### 3. Single Prediction
**POST** `/api/predict`

Make a single diabetes risk prediction.

**Request Body:**
```json
{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}
```

**Response:**
```json
{
    "prediction": 1,
    "probability_diabetes": 0.743,
    "status": "success"
}
```

### 4. Batch Prediction
**POST** `/predict-batch`

Make multiple predictions in a single request.

**Request Body:**
```json
{
    "samples": [
        {
            "Pregnancies": 1,
            "Glucose": 85,
            ...
        },
        {
            "Pregnancies": 8,
            "Glucose": 183,
            ...
        }
    ]
}
```

**Response:**
```json
{
    "predictions": [
        {
            "sample_id": 0,
            "prediction": 0,
            "probability_diabetes": 0.234,
            "risk_level": "Low"
        },
        {
            "sample_id": 1,
            "prediction": 1,
            "probability_diabetes": 0.876,
            "risk_level": "High"
        }
    ],
    "total_samples": 2,
    "successful_predictions": 2
}
```

## Feature Descriptions

| Feature | Description | Range |
|---------|-------------|-------|
| Pregnancies | Number of pregnancies | 0-20 |
| Glucose | Glucose level (mg/dL) | 0-300 |
| BloodPressure | Blood pressure (mmHg) | 0-200 |
| SkinThickness | Skin thickness (mm) | 0-100 |
| Insulin | Insulin level (μU/mL) | 0-1000 |
| BMI | Body Mass Index | 0-100 |
| DiabetesPedigreeFunction | Family history factor | 0-3 |
| Age | Age in years | 1-120 |

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error

Error responses include descriptive messages:
```json
{
    "error": "Please provide a value for Glucose"
}
```

## Rate Limiting
No rate limiting implemented in current version.

## Authentication
No authentication required in current version.

## Examples

### Python Example
```python
import requests

# Single prediction
data = {
    "Pregnancies": 6,
    "Glucose": 148,
    # ... other features
}

response = requests.post("http://localhost:5000/api/predict", json=data)
result = response.json()
print(f"Diabetes risk: {result['probability_diabetes']:.3f}")
```

### cURL Example
```bash
curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}'
```

---
*API developed by Hemanth Peddada - July 2025*
"""
    
    with open('../deployment/API_DOCUMENTATION.md', 'w') as f:
        f.write(docs)
    
    print("📚 API documentation generated successfully!")

if __name__ == "__main__":
    print("🚀 Diabetes Prediction API Testing Suite")
    print("Author: Hemanth Peddada")
    print("=" * 50)
    
    # Generate documentation
    generate_api_documentation()
    
    # Test API endpoints
    test_api_endpoints()
