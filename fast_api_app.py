"""
FastAPI Application for Wine Quality Prediction
This module provides a high-performance API for wine quality predictions
using machine learning models via RESTful endpoints.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Global Variables ====================

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.absolute()
MODEL_PATH = PROJECT_ROOT / 'artifacts' / 'model_trainer' / 'model.joblib'

# Fallback paths to try
FALLBACK_PATHS = [
    Path('artifacts/model_trainer/model.joblib'),
    Path('./artifacts/model_trainer/model.joblib'),
    Path('../artifacts/model_trainer/model.joblib'),
]

model = None
model_loaded = False


# ==================== Utility Functions ====================

def load_model():
    """Load the ML model from disk"""
    global model, model_loaded
    try:
        # Try primary path first
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            model_loaded = True
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return
        
        # Try fallback paths
        for fallback_path in FALLBACK_PATHS:
            fallback_path = fallback_path.resolve()
            if fallback_path.exists():
                model = joblib.load(fallback_path)
                model_loaded = True
                logger.info(f"Model loaded successfully from {fallback_path}")
                return
        
        # If none found, log error
        logger.error(f"Model file not found. Searched paths:")
        logger.error(f"  Primary: {MODEL_PATH}")
        for fallback_path in FALLBACK_PATHS:
            logger.error(f"  Fallback: {fallback_path.resolve()}")
        logger.error("Please run 'python main.py' to train the model first")
        model_loaded = False
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False


def shutdown_model():
    """Cleanup model on shutdown"""
    global model, model_loaded
    logger.info("Shutting down Wine Quality Prediction API...")
    model = None
    model_loaded = False


# ==================== Lifespan Context Manager ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan - handles startup and shutdown events
    """
    # Startup
    logger.info("Starting Wine Quality Prediction API...")
    load_model()
    if model_loaded:
        logger.info("API startup complete - Model ready for predictions")
    else:
        logger.warning("API startup complete - But model failed to load")
    
    yield
    
    # Shutdown
    shutdown_model()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Wine Quality Prediction API",
    description="ML-powered API for predicting wine quality based on physicochemical properties",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== Models ====================

class WineFeatures(BaseModel):
    """
    Wine physicochemical properties for quality prediction.
    All values should be numeric and within realistic ranges.
    """
    fixed_acidity: float = Field(..., gt=0, description="Fixed acidity in g/dm³", example=7.4)
    volatile_acidity: float = Field(..., gt=0, description="Volatile acidity in g/dm³", example=0.7)
    citric_acid: float = Field(..., ge=0, description="Citric acid in g/dm³", example=0.0)
    residual_sugar: float = Field(..., ge=0, description="Residual sugar in g/dm³", example=1.9)
    chlorides: float = Field(..., ge=0, description="Chlorides in g/dm³", example=0.076)
    free_sulfur_dioxide: float = Field(..., ge=0, description="Free sulfur dioxide in mg/dm³", example=11.0)
    total_sulfur_dioxide: float = Field(..., ge=0, description="Total sulfur dioxide in mg/dm³", example=34.0)
    density: float = Field(..., gt=0, description="Density in g/cm³", example=0.9956)
    pH: float = Field(..., ge=0, le=14, description="pH value (0-14)", example=3.51)
    sulphates: float = Field(..., ge=0, description="Sulphates in g/dm³", example=0.56)
    alcohol: float = Field(..., gt=0, description="Alcohol content in %vol", example=9.4)

    class Config:
        schema_extra = {
            "example": {
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
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    wines: List[WineFeatures] = Field(..., description="List of wine samples to predict")


class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    predicted_quality: float = Field(..., description="Predicted wine quality (0-10 scale)")
    quality_level: str = Field(..., description="Quality level category")
    confidence: Optional[str] = Field(None, description="Confidence assessment")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[dict] = Field(..., description="List of predictions for each wine")
    total_samples: int = Field(..., description="Total number of samples processed")
    processing_time: float = Field(..., description="Time taken for processing in seconds")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Application status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field(..., description="API version")


# ==================== Quality Assessment Functions ====================


def get_quality_level(prediction: float) -> str:
    """
    Categorize wine quality into quality levels
    
    Args:
        prediction: Predicted quality score (0-10)
    
    Returns:
        Quality level string
    """
    if prediction >= 9:
        return "Exceptional"
    elif prediction >= 7:
        return "Excellent"
    elif prediction >= 5:
        return "Good"
    elif prediction >= 3:
        return "Average"
    else:
        return "Poor"


def get_confidence(prediction: float) -> str:
    """
    Provide confidence assessment based on prediction value
    
    Args:
        prediction: Predicted quality score
    
    Returns:
        Confidence assessment string
    """
    if 4 <= prediction <= 6:
        return "High"
    elif 3 <= prediction < 7:
        return "Medium"
    else:
        return "Low"


def prepare_prediction_data(features: WineFeatures) -> np.ndarray:
    """
    Convert WineFeatures to numpy array in correct order
    
    Args:
        features: WineFeatures object
    
    Returns:
        Numpy array ready for model prediction
    """
    data = [
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol
    ]
    return np.array(data).reshape(1, 11)


# ==================== Health Check ====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API status and model availability
    
    Returns:
        Health status with model loading information
    """
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": "1.0.0"
    }


@app.get("/debug", tags=["Debug"])
async def debug_info():
    """Debug endpoint to check model paths and status"""
    return {
        "model_loaded": model_loaded,
        "primary_path": str(MODEL_PATH),
        "primary_exists": MODEL_PATH.exists(),
        "project_root": str(PROJECT_ROOT),
        "current_directory": str(Path.cwd()),
        "fallback_paths": [
            {
                "path": str(p.resolve()),
                "exists": p.resolve().exists()
            }
            for p in FALLBACK_PATHS
        ],
        "message": "If model_loaded is False, run 'python main.py' to train the model"
    }


# ==================== Prediction Endpoints ====================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_quality(wine: WineFeatures):
    """
    Predict wine quality for a single wine sample
    
    Args:
        wine: WineFeatures object containing physicochemical properties
    
    Returns:
        PredictionResponse with quality prediction and analysis
    
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not model_loaded:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model not available. Please try again later.")
    
    try:
        # Prepare data
        data = prepare_prediction_data(wine)
        
        # Make prediction
        prediction = float(model.predict(data)[0])
        
        # Ensure prediction is within valid range
        prediction = max(0, min(10, prediction))
        
        # Get quality level and confidence
        quality_level = get_quality_level(prediction)
        confidence = get_confidence(prediction)
        
        logger.info(f"Prediction successful: Quality={prediction:.2f}, Level={quality_level}")
        
        return {
            "predicted_quality": round(prediction, 2),
            "quality_level": quality_level,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict wine quality for multiple samples at once
    
    Args:
        request: BatchPredictionRequest containing list of WineFeatures
    
    Returns:
        BatchPredictionResponse with predictions for all samples
    
    Raises:
        HTTPException: If model is not loaded or batch prediction fails
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available. Please try again later.")
    
    if not request.wines:
        raise HTTPException(status_code=400, detail="Wine list cannot be empty")
    
    if len(request.wines) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 samples allowed per batch")
    
    try:
        import time
        start_time = time.time()
        
        predictions = []
        
        for idx, wine in enumerate(request.wines):
            try:
                data = prepare_prediction_data(wine)
                prediction = float(model.predict(data)[0])
                prediction = max(0, min(10, prediction))
                
                predictions.append({
                    "sample_index": idx,
                    "predicted_quality": round(prediction, 2),
                    "quality_level": get_quality_level(prediction),
                    "confidence": get_confidence(prediction)
                })
            except Exception as e:
                logger.warning(f"Failed to predict sample {idx}: {str(e)}")
                predictions.append({
                    "sample_index": idx,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        logger.info(f"Batch prediction completed: {len(request.wines)} samples in {processing_time:.2f}s")
        
        return {
            "predictions": predictions,
            "total_samples": len(request.wines),
            "processing_time": round(processing_time, 3)
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


# ==================== CSV Upload Endpoint ====================

@app.post("/predict/upload", tags=["Prediction"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Predict wine quality from uploaded CSV file
    
    Expected CSV format: Columns matching WineFeatures fields
    
    Args:
        file: CSV file with wine data
    
    Returns:
        JSON with predictions for all rows
    
    Raises:
        HTTPException: If file processing fails
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available. Please try again later.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be in CSV format")
    
    try:
        import io
        import time
        
        start_time = time.time()
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate columns
        required_columns = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
        
        # Select only required columns
        df = df[required_columns]
        
        # Make predictions
        predictions_array = model.predict(df.values)
        
        # Prepare results
        results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            pred = float(predictions_array[idx])
            pred = max(0, min(10, pred))
            
            results.append({
                "row_index": idx,
                "predicted_quality": round(pred, 2),
                "quality_level": get_quality_level(pred),
                "confidence": get_confidence(pred)
            })
        
        processing_time = time.time() - start_time
        
        logger.info(f"CSV upload prediction completed: {len(df)} rows in {processing_time:.2f}s")
        
        return {
            "filename": file.filename,
            "total_samples": len(df),
            "predictions": results,
            "processing_time": round(processing_time, 3)
        }
    
    except Exception as e:
        logger.error(f"CSV upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"CSV processing failed: {str(e)}")


# ==================== API Documentation ====================

@app.get("/", response_class=HTMLResponse, tags=["Info"])
async def root():
    """Root endpoint with API documentation"""
    return """
    <html>
        <head>
            <title>Wine Quality Prediction API</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #F5E6D3 0%, #E8D4B8 100%);
                    color: #333;
                }
                .header {
                    background: linear-gradient(135deg, #8B4513 0%, #D4A574 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }
                .header h1 {
                    margin: 0;
                    font-size: 2.5em;
                }
                .header p {
                    margin: 10px 0 0 0;
                    font-size: 1.1em;
                }
                .endpoints {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .endpoint {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    border-left: 4px solid #8B4513;
                }
                .endpoint h3 {
                    margin-top: 0;
                    color: #8B4513;
                }
                .endpoint code {
                    background: #f5f5f5;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                }
                .links {
                    text-align: center;
                    gap: 15px;
                    display: flex;
                    justify-content: center;
                    flex-wrap: wrap;
                }
                .links a {
                    background: #8B4513;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    text-decoration: none;
                    transition: all 0.3s;
                }
                .links a:hover {
                    background: #D4A574;
                    transform: translateY(-2px);
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🍷 Wine Quality Prediction API</h1>
                <p>ML-powered REST API for predicting wine quality</p>
            </div>
            
            <div class="endpoints">
                <div class="endpoint">
                    <h3>📊 Single Prediction</h3>
                    <p><code>POST /predict</code></p>
                    <p>Predict quality for a single wine sample with its physicochemical properties.</p>
                </div>
                
                <div class="endpoint">
                    <h3>📈 Batch Prediction</h3>
                    <p><code>POST /predict/batch</code></p>
                    <p>Predict quality for multiple wines at once. Supports up to 1000 samples.</p>
                </div>
                
                <div class="endpoint">
                    <h3>📁 CSV Upload</h3>
                    <p><code>POST /predict/upload</code></p>
                    <p>Upload a CSV file and get predictions for all rows.</p>
                </div>
                
                <div class="endpoint">
                    <h3>❤️ Health Check</h3>
                    <p><code>GET /health</code></p>
                    <p>Check API status and model availability.</p>
                </div>
            </div>
            
            <div class="links">
                <a href="/api/docs">📖 Interactive API Docs (Swagger)</a>
                <a href="/api/redoc">📚 ReDoc Documentation</a>
            </div>
        </body>
    </html>
    """


@app.get("/api/info", tags=["Info"])
async def api_info():
    """Get API information and metadata"""
    return {
        "name": "Wine Quality Prediction API",
        "version": "1.0.0",
        "description": "Machine Learning API for wine quality prediction",
        "model_path": str(MODEL_PATH),
        "model_loaded": model_loaded,
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


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled Exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "fast_api_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
