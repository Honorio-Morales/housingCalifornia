import streamlit as st
import joblib
import numpy as np
import os

MODEL_FILE = os.getenv('MODEL_PATH', 'models/model_linear.joblib')

st.title('Housing California - Predict')

st.write('Entrena los modelos con `python src/train.py`, luego usa esta interfaz para predecir.')

model = None
try:
    model = joblib.load(MODEL_FILE)
except Exception:
    st.warning('Modelo no encontrado. Ejecuta `python src/train.py` primero.')

MedInc = st.number_input('MedInc (median income)', value=3.0)
HouseAge = st.number_input('HouseAge', value=30.0)
AveRooms = st.number_input('AveRooms', value=5.0)
AveBedrms = st.number_input('AveBedrms', value=1.0)
Population = st.number_input('Population', value=1000.0)
Latitude = st.number_input('Latitude', value=34.0)
Longitude = st.number_input('Longitude', value=-118.0)

if st.button('Predecir'):
    if model is None:
        st.error('Modelo no cargado. Ejecuta `python src/train.py` y recarga.')
    else:
        x = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, Latitude, Longitude]])
        pred = model.predict(x)
        st.success(f'Predicción (MedHouseVal): {pred[0]:.4f}')
