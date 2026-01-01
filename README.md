# 🍷 Wine Quality Prediction System

A comprehensive end-to-end machine learning project for predicting wine quality based on physicochemical properties. This project provides both a user-friendly web interface and a high-performance REST API.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Development Workflow](#workflows)
  - [Step-by-Step Setup](#step-1-clone-the-repository)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Web Application (Flask)](#option-1-web-application-flask)
  - [REST API (FastAPI)](#option-2-rest-api-fastapi)
  - [Docker Deployment](#option-3-docker)
  - [Train the Model](#option-4-train-the-model)
- [API Endpoints](#api-endpoints)
  - [Health Check](#1-health-check)
  - [Single Prediction](#2-single-prediction)
  - [Batch Prediction](#3-batch-prediction)
  - [CSV Upload](#4-csv-upload-prediction)
  - [API Information](#5-api-information)
- [Model Details](#model-details)
  - [Input Features](#input-features-11-total)
  - [Output](#output)
  - [Model Architecture](#model-architecture)
  - [Performance Metrics](#performance-metrics)
- [Data Pipeline](#data-pipeline)
  - [Stage 1: Data Ingestion](#stage-1-data-ingestion)
  - [Stage 2: Data Validation](#stage-2-data-validation)
  - [Stage 3: Data Transformation](#stage-3-data-transformation)
  - [Stage 4: Model Training](#stage-4-model-training)
  - [Stage 5: Model Evaluation](#stage-5-model-evaluation)
- [Example Usage](#example-usage)
- [Requirements](#requirements)
- [Logging](#logging)
- [AWS CICD Deployment](#aws-cicd-deployment-with-github-actions)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Support](#support)
- [Acknowledgments](#acknowledgments)
- [Additional Resources](#additional-resources)

---

## 🎯 Project Overview

The Wine Quality Prediction System is a machine learning application that predicts wine quality (0-10 scale) based on 11 physicochemical properties. The project follows MLOps best practices and includes:

- **Data Ingestion**: Automated data collection and validation
- **Data Transformation**: Feature scaling and preprocessing
- **Model Training**: Machine learning model development and optimization
- **Model Evaluation**: Comprehensive metrics and analysis
- **Web Interface**: Beautiful, responsive UI for predictions
- **REST API**: Production-ready FastAPI for integration

### Quality Scale
- **0-3**: Poor
- **4-6**: Average
- **7-8**: Good
- **9-10**: Excellent/Exceptional

---

## ✨ Features

### Web Application
- 🎨 Modern, responsive UI with professional wine-themed design
- 📱 Mobile-friendly interface
- ✅ Real-time form validation
- 📊 Beautiful results display with quality assessment
- 🎉 Celebratory confetti effects for excellent wines
- 📈 Quality scale visualization

### REST API (FastAPI)
- 🚀 High-performance asynchronous API
- 📖 Auto-generated interactive documentation (Swagger UI)
- 🔄 Single and batch prediction endpoints
- 📁 CSV file upload support
- ❤️ Health check endpoint
- 🛡️ Comprehensive error handling
- 📝 Full API logging
- 🔐 CORS support

### Machine Learning Pipeline
- 🔄 Complete data pipeline with validation
- 🧹 Data transformation and normalization
- 🤖 Trained ML model (joblib format)
- 📊 Metrics calculation (R², MAE, RMSE)
- 📈 Model artifacts management

---

## 🛠️ Technologies Used

### Backend
- **Python 3.8+**
- **Flask** - Web framework for UI
- **FastAPI** - Modern REST API framework
- **Uvicorn** - ASGI server
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Joblib** - Model serialization

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with gradients and animations
- **Bootstrap 5** - Responsive framework
- **Font Awesome** - Icon library
- **JavaScript** - Form validation and interactivity

### Development & Operations
- **DVC** - Data versioning (optional)
- **MLflow** - Experiment tracking
- **Docker** - Containerization
- **Git** - Version control

---

## 📁 Project Structure

```
wine-quality-prediction/
│
├── app.py                          # Flask web application
├── fast_api_app.py                 # FastAPI REST API
├── main.py                         # ML pipeline orchestration
├── setup.py                        # Package setup
├── requirements.txt                # Python dependencies
├── params.yaml                     # Pipeline parameters
├── schema.yaml                     # Data schema definition
├── Dockerfile                      # Docker configuration
│
├── src/mlProject/                  # Main package
│   ├── __init__.py
│   ├── components/                 # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── config/                     # Configuration management
│   │   └── configuration.py
│   ├── constants/                  # Constants
│   ├── entity/                     # Data models
│   │   └── config_entity.py
│   ├── pipeline/                   # Orchestration pipelines
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   ├── stage_05_model_evaluation.py
│   │   └── prediction.py
│   └── utils/                      # Utility functions
│       └── common.py
│
├── config/
│   └── config.yaml                 # Main configuration
│
├── templates/                      # HTML templates
│   ├── index.html                  # Prediction form
│   └── results.html                # Results page
│
├── static/                         # Static assets
│   ├── css/
│   ├── js/
│   └── assets/
│
├── artifacts/                      # Generated artifacts
│   ├── data_ingestion/
│   ├── data_transformation/
│   ├── data_validation/
│   ├── model_evaluation/
│   └── model_trainer/
│       └── model.joblib            # Trained model
│
├── research/                       # Jupyter notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_transformation.ipynb
│   ├── 04_model_trainer.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── trials.ipynb
│
├── logs/                           # Application logs
│
└── mlruns/                         # MLflow experiment tracking
```

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/wine-quality-prediction.git
cd wine-quality-prediction
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Package
```bash
pip install -e .
```

---

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
data_ingestion:
  source_URL: <data_url>
  local_data_file: artifacts/data_ingestion/data.csv
  unzip_path: artifacts/data_ingestion

data_validation:
  status_file: artifacts/data_validation/status.txt
  
data_transformation:
  data_path: artifacts/data_ingestion/data.csv
  transformed_train_data_path: artifacts/data_transformation/train.csv
  transformed_test_data_path: artifacts/data_transformation/test.csv
  test_size: 0.2

model_trainer:
  trained_model_path: artifacts/model_trainer/model.joblib
  
model_evaluation:
  metrics_file_name: artifacts/model_evaluation/metrics.json
```

---

## 🚀 Usage

### Option 1: Web Application (Flask)

```bash
# Run Flask app
python app.py
```

Access the web interface at: **http://localhost:8080**

**Features:**
- Form to input wine properties
- Real-time validation
- Beautiful results display
- Quality assessment with descriptions
- Confetti celebration for excellent wines

### Option 2: REST API (FastAPI)

```bash
# Run FastAPI server
python fast_api_app.py
```

The API will be available at: **http://localhost:8000**

**Interactive Documentation:**
- Swagger UI: **http://localhost:8000/api/docs**
- ReDoc: **http://localhost:8000/api/redoc**

### Option 3: Docker

```bash
# Build Docker image
docker build -t wine-quality-predictor .

# Run container
docker run -p 8080:8080 -p 8000:8000 wine-quality-predictor
```

### Option 4: Train the Model

```bash
# Train the model pipeline
python main.py
```

This will execute all pipeline stages:
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

---

## 📡 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### 2. Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9956,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

**Response:**
```json
{
  "predicted_quality": 5.5,
  "quality_level": "Good",
  "confidence": "High",
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### 3. Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "wines": [
    {
      "fixed_acidity": 7.4,
      "volatile_acidity": 0.7,
      ...
    },
    {
      "fixed_acidity": 6.3,
      "volatile_acidity": 0.3,
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
      "sample_index": 0,
      "predicted_quality": 5.5,
      "quality_level": "Good",
      "confidence": "High"
    },
    {
      "sample_index": 1,
      "predicted_quality": 6.2,
      "quality_level": "Good",
      "confidence": "High"
    }
  ],
  "total_samples": 2,
  "processing_time": 0.123
}
```

---

### 4. CSV Upload Prediction
```http
POST /predict/upload
Content-Type: multipart/form-data

file: <your-file.csv>
```

**CSV Format:**
```csv
fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9956,3.51,0.56,9.4
6.3,0.3,0.34,1.6,0.058,15.0,40.0,0.9957,3.3,0.46,9.4
```

**Response:**
```json
{
  "filename": "wines.csv",
  "total_samples": 2,
  "predictions": [
    {
      "row_index": 0,
      "predicted_quality": 5.5,
      "quality_level": "Good",
      "confidence": "High"
    }
  ],
  "processing_time": 0.156
}
```

---

### 5. API Information
```http
GET /api/info
```

**Response:**
```json
{
  "name": "Wine Quality Prediction API",
  "version": "1.0.0",
  "description": "Machine Learning API for wine quality prediction",
  "model_path": "artifacts/model_trainer/model.joblib",
  "model_loaded": true,
  "endpoints": {
    "health": "GET /health",
    "single_prediction": "POST /predict",
    "batch_prediction": "POST /predict/batch",
    "csv_upload": "POST /predict/upload"
  },
  "documentation": {
    "swagger": "/api/docs",
    "redoc": "/api/redoc"
  }
}
```

---

## 🤖 Model Details

### Input Features (11 total)
1. **Fixed Acidity** - Amount of non-volatile acids (g/dm³)
2. **Volatile Acidity** - Amount of acetic acid (g/dm³)
3. **Citric Acid** - Amount of citric acid (g/dm³)
4. **Residual Sugar** - Amount of remaining sugar (g/dm³)
5. **Chlorides** - Amount of salt (g/dm³)
6. **Free Sulfur Dioxide** - Free SO₂ (mg/dm³)
7. **Total Sulfur Dioxide** - Total SO₂ (mg/dm³)
8. **Density** - Density of wine (g/cm³)
9. **pH** - pH value (0-14 scale)
10. **Sulphates** - Amount of sulphates (g/dm³)
11. **Alcohol** - Alcohol content (%vol)

### Output
- **Wine Quality** - Predicted quality score (0-10 scale)
- **Quality Level** - Categorical assessment (Poor, Average, Good, Excellent, Exceptional)
- **Confidence** - Confidence level (Low, Medium, High)

### Model Architecture
- **Algorithm**: Machine Learning Regressor (Scikit-learn)
- **Training Data**: Red and White wine datasets
- **Train-Test Split**: 80-20
- **Features Scaling**: StandardScaler
- **Model Format**: Joblib serialization

### Performance Metrics
Metrics are calculated during model evaluation:
- **R² Score**: Coefficient of determination
- **Mean Absolute Error (MAE)**: Average prediction error
- **Root Mean Squared Error (RMSE)**: Standard deviation of errors

---

## 🔄 Data Pipeline

### Stage 1: Data Ingestion
- Download wine quality dataset
- Extract and validate data
- Store in `artifacts/data_ingestion/`

### Stage 2: Data Validation
- Validate schema compliance
- Check data types and ranges
- Detect missing values
- Generate validation report

### Stage 3: Data Transformation
- Handle missing values
- Scale features using StandardScaler
- Split into train/test sets (80/20)
- Save processed data

### Stage 4: Model Training
- Initialize machine learning model
- Train on processed data
- Save trained model as joblib file
- Log training metrics

### Stage 5: Model Evaluation
- Evaluate model on test set
- Calculate performance metrics
- Generate evaluation report
- Save metrics to JSON

---

## 📊 Example Usage

### Using Python
```python
from mlProject.pipeline.prediction import PredictionPipeline
import numpy as np

# Initialize prediction pipeline
obj = PredictionPipeline()

# Sample wine data
wine_data = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9956, 3.51, 0.56, 9.4]
data = np.array(wine_data).reshape(1, 11)

# Make prediction
prediction = obj.predict(data)
print(f"Predicted Quality: {prediction}")
```

### Using cURL (API)
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9956,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

### Using Python Requests
```python
import requests
import json

url = "http://localhost:8000/predict"
data = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9956,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

response = requests.post(url, json=data)
result = response.json()
print(f"Quality: {result['predicted_quality']}")
print(f"Level: {result['quality_level']}")
```

---

## 📝 Requirements

Key dependencies (see `requirements.txt` for complete list):

```
flask==2.3.0
fastapi==0.104.0
uvicorn==0.24.0
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0
joblib==1.3.0
python-multipart==0.0.6
pydantic==2.0.0
```

---

## 🔍 Logging

### Application Logs
Logs are generated in the `logs/` directory with timestamps and log levels:
- **INFO**: General information
- **WARNING**: Warning messages
- **ERROR**: Error messages

### MLflow Tracking
Experiment tracking is available in `mlruns/` directory:
```bash
# View MLflow UI
mlflow ui
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest

# Format code
black .
flake8 .
```

---

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.ap-south-1.amazonaws.com/mlproj

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Your Name** - Initial work - [GitHub Profile](https://github.com/ajaychaudhary8104)

---

## 📞 Support

For support, email support@winequalitypredictor.com or open an issue on GitHub.

---

## 🙏 Acknowledgments

- Wine Quality Dataset from UCI Machine Learning Repository
- FastAPI and Flask communities
- Scikit-learn documentation
- All contributors and supporters

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [MLOps Best Practices](https://ml-ops.systems/)
- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

**Last Updated**: January 2024  
**Version**: 1.0.0  
**Status**: Active Development ✅

