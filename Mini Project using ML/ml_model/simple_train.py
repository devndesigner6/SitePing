import numpy as np
import pickle
import json

# Create a simple mock model for demonstration
class SimpleDiabetesModel:
    def __init__(self):
        # Simple coefficients for demonstration
        self.coef_ = np.array([0.1, 0.8, 0.3, 0.2, 0.4, 0.6, 0.5, 0.3])
        self.intercept_ = -2.0
        
    def predict(self, X):
        # Simple logistic regression prediction
        z = np.dot(X, self.coef_) + self.intercept_
        return (z > 0).astype(int)
    
    def predict_proba(self, X):
        # Simple probability calculation
        z = np.dot(X, self.coef_) + self.intercept_
        prob_1 = 1 / (1 + np.exp(-z))
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

class SimpleScaler:
    def __init__(self):
        # Mock means and stds for 8 features
        self.mean_ = np.array([3.8, 120.9, 69.1, 20.5, 79.8, 32.0, 0.47, 33.2])
        self.scale_ = np.array([3.4, 32.0, 19.4, 16.0, 115.2, 7.9, 0.33, 11.8])
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_

# Create and save the mock model
print("Creating diabetes prediction model...")
model = SimpleDiabetesModel()
scaler = SimpleScaler()

# Save model using pickle (similar to joblib)
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
print("Model features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age")

# Test the model
test_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
test_scaled = scaler.transform(test_data)
prediction = model.predict(test_scaled)
probability = model.predict_proba(test_scaled)

print(f"Test prediction: {prediction[0]}")
print(f"Test probability: {probability[0][1]:.3f}")
print("🚀 Model is ready for the web application!")
