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
1. Crea un nuevo Space en Hugging Face (https://huggingface.co/spaces)
2. Selecciona "Streamlit" como template
3. Sube tu repositorio (`git push` o interfaz web)
4. Asegúrate de incluir:
   - `app/streamlit_app.py`
   - `models/` (los .joblib entrenados)
   - `requirements.txt`
5. Spaces detectará y levantará la app automáticamente

Próximos pasos
- Revisar el notebook `notebooks/eda.ipynb` para análisis completo
- Compilar el informe LaTeX: `cd reports && pdflatex report.tex`
- Ajustar features o regularización según resultados
