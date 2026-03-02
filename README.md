Proyecto: Housing California - Taller 1.1

Resumen
- Dataset: California Housing (scikit-learn)
- Modelos: Regresión Lineal Múltiple y Regresión Polinomial (grado 2 y 3)

Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ejecución local (sin Docker):

1. Entrenar modelos:

```bash
python src/train.py
```

2. Levantar API (FastAPI):

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

3. Ejecutar interfaz (Streamlit) en otra terminal:

```bash
streamlit run app/streamlit_app.py
```

Ejecución con Docker Compose (recomendado):

```bash
# Asegurar que tienes Docker instalado
docker-compose up --build
```

Esto levantará:
- API en `http://localhost:8000` (FastAPI con Swagger UI en `/docs`)
- Puedes cambiar el modelo con la variable `MODEL_TYPE` en `docker-compose.yml` (valores: `linear`, `poly2`, `poly3`)

Estructura
- src/: código fuente (data loader, modelado, API)
- app/: frontend Streamlit
- models/: modelos serializados (.joblib)
- notebooks/: EDA y experimentos
- reports/: figuras PNG y LaTeX

Despliegue en producción:

### API en Render:
1. Crea una nueva Web Service en Render (https://dashboard.render.com)
2. Conecta tu repositorio GitHub
3. Configura:
   - Build Command: (dejar vacío; usa Dockerfile)
   - Start Command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
   - Environment Variables: `MODEL_TYPE=poly2` (o tu modelo preferido)
4. Deploy

### Streamlit en Hugging Face Spaces:
1. Crea un nuevo Space en Hugging Face (https://huggingface.co/spaces) y elige "Streamlit".
2. Conecta tu cuenta GitHub y selecciona el repositorio `housingCalifornia`.
   - Alternativamente, puedes subir los archivos manualmente.
3. Asegúrate de que el repo contiene estas rutas (ya están en main):
   - `app/streamlit_app.py` (interfaz ve a la API de Render)
   - `models/` con los archivos `.joblib` entrenados
   - `requirements.txt` para instalar dependencias.
   - opcionalmente `web/index.html` si prefieres una demo HTML estática.
4. Si necesitas un archivo `.hfignore` crea uno para evitar subir archivos grandes o innecesarios.
5. Spaces construirá el contenedor y abrirá tu app en unos minutos. La interfaz mostrará los sliders y realizará llamadas a la URL de Render por defecto.

> En caso de querer cambiar la URL de la API, define la variable de entorno `API_URL` en la configuración del Space y apunta a tu servicio o a localhost para pruebas privadas.

Spaces también soporta ejecutar `src/train.py` en el arranque si prefieres reentrenar; sólo configura el comando `streamlit run app/streamlit_app.py` en el campo "App file".

Una vez desplegado, accede a la dirección provista por Hugging Face para ver la UI en vivo y probar predicciones.
Próximos pasos
- Revisar el notebook `notebooks/eda.ipynb` para análisis completo
- Compilar el informe LaTeX: `cd reports && pdflatex report.tex`
- Ajustar features o regularización según resultados
