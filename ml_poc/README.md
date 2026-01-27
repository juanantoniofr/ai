# ML PoC - End-to-End Local Pipeline

## Objetivo
Demostrar un flujo completo de Machine Learning en local:
training, versionado, despliegue, inferencia, explicabilidad y monitorización.

## Descripción
Este proyecto implementa un sistema completo de predicción de churn para un servicio de telecomunicaciones.
A partir de métricas de cliente (edad, antigüedad, incidencias, uso y descuentos), el sistema estima la probabilidad de baja y proporciona explicaciones locales mediante SHAP para justificar cada predicción.
El sistema simula un entorno real de producción con entrenamiento, despliegue vía API y monitorización conceptual.

## Flujo
1. Entrenamiento del modelo (Gradient Boosting)
2. Versionado del pipeline completo
3. Inferencia batch
4. Servicio de predicción vía API
5. Explicabilidad con SHAP
6. Monitorización básica de drift

## Cómo ejecutar

### 1. Crear entorno
pip install -r requirements.txt

### 2. Entrenar modelo
python src/train.py

### 3. Inferencia batch
python src/predict_batch.py

### 4. API local
uvicorn src.api:app --reload

### 5. Explicabilidad
python src/shap_explain.py

### 6. Monitorización
python src/monitor.py

## Métricas
- AUC almacenado en model/metadata.json

## Notas
Esta PoC simula un entorno productivo sin infraestructura cloud.
