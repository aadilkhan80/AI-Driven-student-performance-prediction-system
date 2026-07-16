import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from app.data_generator import generate_student_data
from app.model import StudentPerformanceModel

app = FastAPI(title="EduPredict AI - Student Performance Prediction API")

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DATA_PATH = os.path.join(BASE_DIR, 'app', 'student_data.csv')

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize model manager
model_manager = StudentPerformanceModel()

# Pydantic schema for prediction inputs
class StudentFeatures(BaseModel):
    study_hours: float = Field(..., ge=0.0, le=40.0, description="Hours studied per week")
    attendance: float = Field(..., ge=0.0, le=100.0, description="Attendance rate percentage")
    sleep_hours: float = Field(..., ge=0.0, le=24.0, description="Average sleep hours per night")
    parental_support: str = Field(..., description="Parental support level (Low, Medium, High)")
    tutoring: int = Field(..., ge=0, le=1, description="Tutoring status (0: No, 1: Yes)")
    extracurriculars: int = Field(..., ge=0, le=1, description="Extracurricular participation (0: No, 1: Yes)")
    internet_access: int = Field(..., ge=0, le=1, description="Internet access (0: No, 1: Yes)")
    previous_grade: float = Field(..., ge=0.0, le=100.0, description="Previous overall grade/score")
    stress_level: str = Field(..., description="Stress level (Low, Medium, High)")

class RetrainRequest(BaseModel):
    num_students: int = Field(1200, ge=100, le=5000, description="Number of synthetic students to generate for training")

# Startup event to prepare models
@app.on_event("startup")
def startup_event():
    print("Starting EduPredict AI backend...")
    
    # 1. Generate data if not exists
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Generating default synthetic dataset...")
        generate_student_data(num_students=1200, filepath=DATA_PATH)
        
    # 2. Load model or train if not trained
    if not model_manager.load():
        print("Models not found or failed to load. Training model pipeline now...")
        df = pd.read_csv(DATA_PATH)
        model_manager.train(df)
        
    print("Startup checks complete. Ready to serve requests.")

# API endpoints
@app.post("/api/predict")
def predict_performance(features: StudentFeatures):
    try:
        input_data = features.dict()
        
        # Validation on categoricals
        if input_data['parental_support'] not in ['Low', 'Medium', 'High']:
            raise HTTPException(status_code=400, detail="parental_support must be 'Low', 'Medium', or 'High'")
        if input_data['stress_level'] not in ['Low', 'Medium', 'High']:
            raise HTTPException(status_code=400, detail="stress_level must be 'Low', 'Medium', or 'High'")
            
        prediction = model_manager.predict(input_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/stats")
def get_model_stats():
    try:
        stats = model_manager.get_stats()
        if not stats:
            # Fallback if metrics are empty
            raise HTTPException(status_code=404, detail="Model stats not available. Retrain the model.")
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrain")
def retrain_model(req: RetrainRequest):
    try:
        print(f"Retraining model requested with {req.num_students} students...")
        # Regenerate data
        df = generate_student_data(num_students=req.num_students, filepath=DATA_PATH)
        # Retrain model
        stats = model_manager.train(df)
        return {"status": "success", "message": "Model retrained successfully", "metrics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# Mount static files (MUST be mounted after routes to avoid shadowing)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def get_index():
    index_file = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(index_file):
        return FileResponse(index_file)
    else:
        return JSONResponse(
            status_code=404,
            content={"message": f"Frontend static files not found in {STATIC_DIR}. Please build them."}
        )
