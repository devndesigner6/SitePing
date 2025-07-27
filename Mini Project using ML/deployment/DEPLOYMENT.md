# Deployment Guide for Diabetes Prediction App
*Author: Hemanth Peddada*  
*Date: July 2025*

## About
This comprehensive deployment guide covers multiple platforms for hosting the advanced diabetes prediction web application I developed using machine learning and modern web technologies.

## Local Development Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Installation Steps
1. **Clone or download the project**
   ```bash
   git clone <your-repo-url>
   cd "Mini Project using ML"
   ```

2. **Install dependencies**
   ```bash
   pip install -r deployment/requirements.txt
   ```

3. **Prepare the dataset**
   - Download the Pima Indians Diabetes Dataset
   - Place as `data/diabetes.csv`

4. **Train the machine learning model**
   ```bash
   python ml_model/train_model.py
   ```

5. **Run the web application locally**
   ```bash
   python webapp/app.py
   ```
   - Access at: `http://localhost:5000`

## Cloud Deployment Options

### Option 1: Heroku Deployment

1. **Install Heroku CLI**
   - Download from: https://devcenter.heroku.com/articles/heroku-cli

2. **Login and create app**
   ```bash
   heroku login
   heroku create your-diabetes-predictor
   ```

3. **Configure files for Heroku**
   - Ensure `Procfile` exists: `web: gunicorn --chdir webapp app:app`
   - Ensure `requirements.txt` has all dependencies

4. **Deploy to Heroku**
   ```bash
   git init
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

5. **Open your deployed app**
   ```bash
   heroku open
   ```

### Option 2: Streamlit Cloud

1. **Create Streamlit version** (optional)
   - Convert Flask app to Streamlit for easier deployment
   - Push code to GitHub repository

2. **Deploy via Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Connect GitHub repository
   - Deploy with one click

### Option 3: Railway Deployment

1. **Create Railway account**
   - Visit: https://railway.app

2. **Deploy from GitHub**
   - Connect repository
   - Railway auto-detects Python and deploys

### Option 4: Render Deployment

1. **Create Render account**
   - Visit: https://render.com

2. **Create Web Service**
   - Connect GitHub repository
   - Set build command: `pip install -r deployment/requirements.txt`
   - Set start command: `gunicorn --chdir webapp app:app`

## Environment Variables
For production deployments, consider setting:
- `PORT`: Application port (usually set by platform)
- `FLASK_ENV`: Set to 'production'

## Post-Deployment Testing

1. **Verify the application loads**
2. **Test prediction functionality with sample data**
3. **Check error handling with invalid inputs**

## Troubleshooting

### Common Issues:
- **Model files not found**: Ensure you've run the training script
- **Missing dependencies**: Check requirements.txt is complete
- **Port issues**: Use environment PORT variable for deployment

### Model Performance:
- The trained model achieves ~77% accuracy on test data
- Uses Logistic Regression with feature scaling
- Handles missing values by replacing with median

## Application Features
- Real-time diabetes risk prediction
- User-friendly web interface
- Input validation and error handling
- Probability scores for predictions
- Responsive design for mobile devices

## Technical Notes
- Built with Flask web framework
- Uses scikit-learn for machine learning
- Bootstrap for responsive UI design
- Model persistence with joblib

---
*Developed by [Your Name] - July 2025*
