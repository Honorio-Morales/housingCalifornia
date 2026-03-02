import streamlit as st
import requests
import os

st.set_page_config(page_title="Housing California Predictor", layout="centered")
st.title('🏠 Housing California - Predictor')

st.write("""
Usa esta interfaz para predecir el valor medio de una vivienda en California.
Ingresa las características y obtén una predicción en tiempo real.
""")

# API endpoint (con fallback local para desarrollo)
# por defecto apunta al servicio desplegado en Render
API_URL = os.getenv('API_URL', 'https://housing-california-api.onrender.com')

st.info(f"✅ Conectado a: {API_URL}")

# Input sliders y campos
col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider('Median Income (MedInc)', min_value=0.5, max_value=15.0, value=3.0, step=0.1)
    HouseAge = st.slider('House Age (años)', min_value=1, max_value=53, value=30)
    AveRooms = st.slider('Average Rooms', min_value=1.0, max_value=10.0, value=5.0, step=0.5)

with col2:
    AveBedrms = st.slider('Average Bedrooms', min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    Population = st.slider('Population (thousands)', min_value=100, max_value=35000, value=1000, step=100)
    Latitude = st.slider('Latitude', min_value=32.5, max_value=42.0, value=34.0, step=0.1)

Longitude = st.slider('Longitude', min_value=-125.0, max_value=-114.0, value=-118.0, step=0.1)

# Botón de predicción
if st.button('🔮 Predecir', use_container_width=True):
    payload = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    
    try:
        response = requests.post(f'{API_URL}/predict', json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            pred_value = result['prediction']
            st.success(f'✨ **Predicción: ${pred_value:.4f}** millones USD')
            st.metric(label="Valor estimado de la vivienda", value=f"${pred_value:.4f}M")
        else:
            st.error(f'Error: {response.status_code} - {response.text}')
    except requests.exceptions.RequestException as e:
        st.error(f'❌ No se pudo conectar a la API: {str(e)}')
        st.info('Asegúrate de que la API está levantada en http://localhost:8000 o configura API_URL.')

st.markdown('---')
st.write('**📊 Nota:** Este predictor usa un modelo de Regresión Polinomial entrenado en California Housing dataset.')

