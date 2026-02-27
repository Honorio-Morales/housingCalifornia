from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import numpy as np

app = FastAPI(title="Housing California - Predict API")

# Allow selecting which saved model to load: 'linear' (default), 'poly2', 'poly3'
MODEL_TYPE = os.getenv('MODEL_TYPE', 'linear')

def _load_model(model_type: str):
    try:
        if model_type == 'linear':
            return joblib.load(os.getenv('MODEL_PATH', 'models/model_linear.joblib'))
        if model_type == 'poly2':
            return joblib.load(os.getenv('MODEL_PATH', 'models/model_poly_2.joblib'))
        if model_type == 'poly3':
            return joblib.load(os.getenv('MODEL_PATH', 'models/model_poly_3.joblib'))
    except Exception:
        return None

model = _load_model(MODEL_TYPE)

# Enable CORS so a separate frontend (Streamlit or other) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    Latitude: float
    Longitude: float


@app.post('/predict')
def predict(payload: InputFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail='Model not loaded')
    x = np.array([[
        payload.MedInc,
        payload.HouseAge,
        payload.AveRooms,
        payload.AveBedrms,
        payload.Population,
        payload.Latitude,
        payload.Longitude
    ]])
    pred = model.predict(x)
    return {'prediction': float(pred[0])}
