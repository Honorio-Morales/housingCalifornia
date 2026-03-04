# Guía Técnica del Proyecto Housing California

## 📌 Descripción General

Este proyecto implementa un **sistema de predicción de precios de viviendas** usando dos modelos de regresión (Lineal y Polinomial). La arquitectura comprende:

- **Backend**: API REST con FastAPI desplegada en Render
- **Frontend**: Interfaz HTML interactiva y aplicación Streamlit
- **Datos**: Dataset California Housing (20,640 registros, 8 características)
- **Modelos**: Guardados como archivos `.joblib` para reutilización en producción

---

## 📁 Estructura del Proyecto

```
housingCalifornia/
├── src/                       # Código principal
│   ├── data_loader.py        # Carga el dataset
│   ├── modeling.py           # Entrenamiento y evaluación de modelos
│   ├── train.py              # Script principal de entrenamiento
│   ├── api.py                # Servidor FastAPI
│   ├── export_figures.py     # Genera gráficos para el informe
│   └── save_dataset.py       # Guarda datos localmente
├── app/
│   └── streamlit_app.py      # Frontend Streamlit interactivo
├── web/
│   └── index.html            # Demo HTML vanilla JavaScript
├── notebooks/
│   └── eda.ipynb             # Análisis exploratorio de datos
├── models/                    # Modelos entrenados (preserializados)
│   ├── model_linear.joblib
│   ├── model_poly_2.joblib
│   ├── poly_2_transform.joblib
│   ├── model_poly_3.joblib
│   └── poly_3_transform.joblib
├── data/
│   └── california_housing.csv # Dataset local
├── reports/
│   ├── report.tex            # Informe LaTeX
│   ├── report.pdf            # Informe compilado (10 páginas)
│   └── figures/              # Gráficos generados automáticamente
├── Dockerfile.api            # Contenedor para API
├── docker-compose.yml        # Orquestación local
├── requirements.txt          # Dependencias Python
└── README.md                 # Instrucciones de uso
```

---

## 🔑 Archivos Python Principales

### 1. `src/data_loader.py` - Carga de Datos

**Propósito**: Obtener el dataset California Housing y prepararlo para modelado.

```python
def load_data():
    """Carga California Housing desde scikit-learn"""
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='MedHouseVal')
    return pd.concat([X, y], axis=1)
```

**Qué hace:**
- Descarga automáticamente el dataset si no existe localmente
- Retorna un DataFrame con 9 columnas (8 features + 1 target)
- **MedHouseVal** está en unidades de **100k USD** (ej: 0.49 = $49,000)

**Salida esperada:**
```
Shape: (20640, 9)
Columnas: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, MedHouseVal
```

---

### 2. `src/modeling.py` - Entrenamiento y Evaluación

**Propósito**: Entrenar modelos de regresión y calcular métricas.

**Funciones principales:**

#### `train_and_evaluate(X_train, X_test, y_train, y_test, model_type='linear')`

Entrena un modelo según tipo:
- `'linear'`: Regresión lineal múltiple (baseline)
- `'poly_2'`: Polinomial grado 2 (28 features expandidas)
- `'poly_3'`: Polinomial grado 3 (35 features expandidas)

**Flujo interno:**
1. Carga el modelo base (LinearRegression)
2. Si es polinomial, expande features con `PolynomialFeatures(degree)`
3. Entrena en el conjunto train
4. Calcula $R^2$ y RMSE en el conjunto test
5. Retorna modelo, transformer (si aplica) y métricas

**Código ilustrativo:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='linear'):
    model = LinearRegression()
    transformer = None
    
    # Si es polinomial, expandir features
    if 'poly' in model_type:
        degree = int(model_type.split('_')[1])
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, transformer, r2, rmse
```

**Métricas calculadas:**
- **R²**: Proporción de varianza explicada (0.5751 a 0.6070)
- **RMSE**: Error típico en unidades de 100k USD (~0.72)

---

### 3. `src/train.py` - Script Principal de Entrenamiento

**Propósito**: Orquestar el flujo completo de entrenamiento y guardado de modelos.

**Flujo:**
```
1. Cargar datos (california_housing.csv)
   ↓
2. Split train/test (80/20, seed=42)
   ↓
3. Para cada modelo (linear, poly_2, poly_3):
   a. Entrenar con train_and_evaluate()
   b. Imprimir métricas
   c. Guardar modelo + transformer con joblib
   ↓
4. Modelos guardados en ./models/
```

**Salida en consola:**
```
Training Linear Regression...
  R²: 0.5751, RMSE: 0.7462

Training Polynomial (degree 2)...
  R²: 0.6070, RMSE: 0.7176 ← MEJOR

Training Polynomial (degree 3)...
  R²: 0.5393, RMSE: 0.7770
```

**Archivos generados:**
```
models/
├── model_linear.joblib          (673 B)
├── model_poly_2.joblib          (1.1 KB)
├── poly_2_transform.joblib      (2.4 KB)  ← Transformador grado 2
├── model_poly_3.joblib          (1.5 KB)
└── poly_3_transform.joblib      (2.8 KB)
```

**Comando de ejecución:**
```bash
python src/train.py
```

---

### 4. `src/api.py` - Servidor FastAPI

**Propósito**: Servir predicciones HTTP para consumo de frontends.

**Endpoints:**

#### `POST /predict`
Recibe features, aplica transformación polinomial si necesario, predice.

**Request:**
```json
{
  "MedInc": 3.0,
  "HouseAge": 30,
  "AveRooms": 5.0,
  "AveBedrms": 1.0,
  "Population": 500,
  "Latitude": 37.5,
  "Longitude": -120.0
}
```

**Response:**
```json
{
  "prediction": 0.4940,
  "prediction_usd": 49400,
  "model_type": "poly2"
}
```

**Código interno:**
```python
@app.post("/predict")
async def predict(data: PredictionInput):
    # 1. Convertir input a array
    X = np.array([[data.MedInc, data.HouseAge, ...]])
    
    # 2. Aplicar transformación si es polinomial
    if model_type == 'poly2':
        X = poly_transformer.transform(X)
    
    # 3. Predecir
    prediction = model.predict(X)[0]
    
    # 4. Retornar resultado
    return {
        "prediction": float(prediction),
        "prediction_usd": int(prediction * 100_000),
        "model_type": model_type
    }
```

#### `GET /health`
Verifica que el modelo está cargado.

**Response:**
```json
{
  "model_loaded": true,
  "model_type": "poly2"
}
```

**Características:**
- CORS habilitado para acceso desde HTML/Streamlit
- Validación automática de tipos con Pydantic
- Documentación interactiva en `/docs` (Swagger UI)

**Ejecución local:**
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

**Acceso:**
- Predicción: `POST http://localhost:8000/predict`
- Docs: `http://localhost:8000/docs`
- Health: `GET http://localhost:8000/health`

---

### 5. `src/export_figures.py` - Generador de Gráficos

**Propósito**: Crear visualizaciones para el informe PDF/LaTeX.

**Gráficos generados:**

1. **heatmap_correlation.png** - Matriz de correlaciones
   ```python
   sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
   ```

2. **scatter_top_features.png** - Scatter plots de variables clave
   ```python
   plt.scatter(X['MedInc'], y)  # Ingreso vs Precio (correlación fuerte)
   ```

3. **pred_resid_linear.png** - Real vs Predicho (Lineal)
   ```python
   plt.scatter(y_test, y_pred_linear)
   plt.plot([y_min, y_max], [y_min, y_max], 'r--')  # Diagonal perfecta
   ```

4. **pred_resid_poly2.png** - Real vs Predicho (Poly2 - MEJOR)

5. **pred_resid_poly3.png** - Real vs Predicho (Poly3 - Overfitting)

**Interpretación:**
- Puntos **cerca de la diagonal** → predicciones precisas
- Puntos **dispersos aleatoriamente** en residuos → buen model
- Patrón **en embudo** (varianza aumenta) → heteroscedasticidad

**Ejecución:**
```bash
python src/export_figures.py
```

---

### 6. `src/save_dataset.py` - Guardado Local de Datos

**Propósito**: Descargar una vez y guardar CSV local para reutilización rápida.

```python
def save_housing_data():
    """Guarda California Housing como CSV local"""
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='MedHouseVal')
    df = pd.concat([X, y], axis=1)
    df.to_csv('data/california_housing.csv', index=False)
```

**Salida:**
```
data/california_housing.csv (6.0 MB)
```

---

## 📊 Frontends

### `app/streamlit_app.py` - Interfaz Interactiva

**Tecnología**: Streamlit (Python)

**Características:**
- Sliders para cada feature (MedInc, HouseAge, etc.)
- Botón "Predecir" → POST a API Render
- Resultado mostrado en grande con formateo USD
- Completamente responsiva

**Código resumido:**
```python
st.title("🏠 Predicción de Precios de Vivienda")

med_inc = st.slider("Ingreso Mediano", min=0.5, max=15.0, value=3.0)
house_age = st.slider("Edad de Vivienda", min=1, max=52, value=30)
# ... más sliders

if st.button("Predecir"):
    response = requests.post(API_URL, json=input_dict)
    prediction = response.json()['prediction']
    st.success(f"Predicción: ${prediction*100_000:,.0f}")
```

**Ejecución:**
```bash
streamlit run app/streamlit_app.py
```

**Acceso:** `http://localhost:8501`

---

### `web/index.html` - Demo Vanilla JavaScript

**Tecnología**: HTML5 + CSS + JavaScript puro (sin dependencias)

**Características:**
- Formulario centrado con campos claros
- Botón "Predecir" (POST a API remota Render)
- Botón "Simular" (genera valores aleatorios)
- Resultado formateado: "0.494 (≈ $49,400)"
- Trabajo sin servidor (client-side execution)

**Flujo JavaScript:**
```javascript
async function predict() {
    // 1. Leer valores del formulario
    const data = {
        MedInc: parseFloat(document.getElementById('MedInc').value),
        HouseAge: parseFloat(document.getElementById('HouseAge').value),
        // ... más campos
    };
    
    // 2. POST a API remota
    const response = await fetch(API_URL, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    });
    
    // 3. Procesar respuesta
    const result = await response.json();
    const usd = (result.prediction * 100000).toFixed(0);
    document.getElementById('result').textContent = 
        `${result.prediction.toFixed(4)} (≈ $${parseInt(usd).toLocaleString()})`;
}

function fillRandom() {
    // Genera valores aleatorios realistas
    document.getElementById('MedInc').value = (Math.random() * 12 + 0.5).toFixed(2);
    // ... más campos
}
```

**Acceso:** 
- GitHub Raw: https://raw.githubusercontent.com/Honorio-Morales/housingCalifornia/main/web/index.html
- Local: Abre `web/index.html` en navegador

---

## 🔗 Flujo Completo de Datos

### Entrenamiento (Una sola vez)

```
1. Ejecutar: python src/train.py
   ↓
2. data_loader.py → Carga California Housing
   ↓
3. modeling.py → Entrena Linear, Poly2, Poly3
   ↓
4. Guarda en models/ → joblib files (modelos + transformadores)
   ↓
5. export_figures.py → Genera PNGs para informe
```

### Predicción (En producción)

```
1. Usuario ingresa datos en HTML o Streamlit
   ↓
2. POST request a https://housing-california-api.onrender.com/predict
   ↓
3. api.py:
   a. Carga modelo_poly2.joblib y poly_2_transform.joblib
   b. Aplica transformación polinomial (7 features → 28 features)
   c. Predice con modelo entrenado
   d. Retorna JSON con resultado
   ↓
4. Frontend parsea y muestra: "0.494 (≈ $49,400)"
```

---

## 🐳 Docker y Despliegue

### `Dockerfile.api`

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY models/ models/
RUN python src/train.py  # ← Entrena modelos durante build
ENV MODEL_TYPE=poly2
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Puntos clave:**
- Build automático desde GitHub
- Entrena modelos DURANTE la build (no después)
- Modelos pre-serializados en repo para disponibilidad
- Expone puerto 8000 en Render

### Despliegue en Render

```
1. Conectar GitHub a Render
2. Crear nuevo "Web Service"
3. Apuntar a https://github.com/Honorio-Morales/housingCalifornia
4. Render detecta Dockerfile.api automáticamente
5. Build y deploy automático
6. API disponible en: https://housing-california-api.onrender.com
```

---

## 📈 Métricas Finales

| Modelo | R² | RMSE | Interpretación |
|--------|-----|------|---|
| **Lineal** | 0.5751 | 0.7462 | Baseline: explica 57.5% varianza |
| **Poly 2** ⭐ | 0.6070 | 0.7176 | **Mejor**: captura no-linealidad |
| **Poly 3** | 0.5393 | 0.7770 | Overfitting: 35 features, mal generaliza |

**Selección**: Polinomial grado 2 por:
- Mejor R² (60.7%)
- Menor RMSE (~$71,760)
- Evita overfitting (vs grado 3)
- Interpretable (28 features vs 35)

---

## 🚀 Comandos Rápidos

### Local
```bash
# Setup
git clone https://github.com/Honorio-Morales/housingCalifornia.git
cd housingCalifornia
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Entrenar
python src/train.py

# API local
uvicorn src.api:app --reload

# Streamlit local
streamlit run app/streamlit_app.py

# Docker local
docker build -t housing-api -f Dockerfile.api .
docker run -p 8000:8000 housing-api
```

### Predicción
```bash
# Request POST
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 3.0, "HouseAge": 30, ...}'

# Health check
curl http://localhost:8000/health
```

---

## 📚 Dependencias Principales

```txt
scikit-learn     → Modelos de ML (LinearRegression, PolynomialFeatures)
pandas           → Manipulación de datos (DataFrames)
numpy            → Cálculos numéricos
fastapi          → Framework web (API)
uvicorn          → Servidor ASGI
streamlit        → Framework frontend interactivo
joblib           → Serialización de modelos
seaborn/matplib  → Visualización (gráficos)
```

---

## 🎯 Resumen Ejecutivo

1. **Data**: California Housing (20,640 registros, 8 features)
2. **Modelos**: Linear, Poly2 (SELECCIONADO), Poly3
3. **Backend**: FastAPI en Render (https://housing-california-api.onrender.com)
4. **Frontend**: HTML vanilla + Streamlit (opcional)
5. **Informe**: LaTeX completo PDF (10 páginas)
6. **Código**: Modular, reutilizable, dockerizable

El proyecto demuestra un flujo real de ML: desde EDA → Modelado → Evaluación → Despliegue en producción.

---

**Última actualización:** 4 de Marzo de 2026
**Repositorio**: https://github.com/Honorio-Morales/housingCalifornia
