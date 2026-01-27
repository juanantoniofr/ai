from fastapi import FastAPI
import joblib
import pandas as pd
import shap
from fastapi.responses import HTMLResponse
from fastapi import Form


app = FastAPI()
pipeline = joblib.load("model/pipeline_v1.joblib")

prep = pipeline.named_steps["prep"]
model = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(model)

# Función para obtener nombres de features
def get_feature_names(preprocessor):
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "passthrough":
            feature_names.extend(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(cols)
            feature_names.extend(names)

    return feature_names

@app.post("/explain")
def explain(data: dict):
    df = pd.DataFrame([data])

    # Transformación
    X_trans = prep.transform(df)

    # Predicción
    prob = float(model.predict_proba(X_trans)[0, 1])

    # SHAP
    shap_values = explainer.shap_values(X_trans)

    feature_names = get_feature_names(prep)

    shap_row = shap_values[0]

    df_shap = pd.DataFrame({
        "feature": feature_names,
        "shap": shap_row
    })

    # Ordenar por impacto absoluto
    df_shap["abs_shap"] = df_shap["shap"].abs()
    df_shap = df_shap.sort_values("abs_shap", ascending=False)

    top_positive = df_shap[df_shap["shap"] > 0].head(3)
    top_negative = df_shap[df_shap["shap"] < 0].head(3)

    return {
        "probability": round(prob, 4),
        "risk": "high" if prob > 0.7 else "low",
        "top_positive_factors": top_positive[["feature", "shap"]].round(4).to_dict(orient="records"),
        "top_negative_factors": top_negative[["feature", "shap"]].round(4).to_dict(orient="records")
    }


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = float(pipeline.predict_proba(df)[0, 1])

    return {
        "probability": round(prob, 4),
        "risk": "high" if prob > 0.7 else "low"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# formulario sencillo para introducir valores

@app.get("/", response_class=HTMLResponse)
def form():
    return """
    <html>
    <head>
        <title>ML Explainability Demo</title>
    </head>
    <body>
        <h2>ML PoC - Simulador de Riesgo con IA Explicable</h2>

        <form action="/explain_form" method="post">
            Edad: <input type="number" name="edad" value="60"><br><br>
            Antigüedad: <input type="number" name="antiguedad" value="1"><br><br>
            Incidencias: <input type="number" name="incidencias" value="4"><br><br>

            Uso medio:
            <select name="uso_medio">
                <option value="low">low</option>
                <option value="medium">medium</option>
                <option value="high">high</option>
            </select><br><br>

            Descuento:
            <select name="descuento">
                <option value="no">no</option>
                <option value="yes">yes</option>
            </select><br><br>

            <button type="submit">Evaluar riesgo</button>
        </form>

        <hr>

        <h3>¿Qué es esta prueba de concepto?</h3>

        <p>
        Esta aplicación es una <b>prueba de concepto (PoC)</b> que demuestra un flujo completo
        de Machine Learning explicable en entorno local.
        </p>

        <ul>
            <li>Entrenamiento de un modelo de Gradient Boosting sobre datos tabulares</li>
            <li>Versionado del pipeline completo (preprocesado + modelo)</li>
            <li>Servicio de predicción vía API local</li>
            <li>Explicabilidad individual mediante SHAP</li>
            <li>Simulación interactiva de escenarios</li>
            <li>Base para monitorización y reentrenamiento</li>
        </ul>

        <p>
        El objetivo no es solo predecir, sino <b>entender y simular decisiones</b>,
        mostrando de forma transparente por qué el modelo considera un caso de alto o bajo riesgo.
        </p>

    </body>
    </html>
    """

# función para explicación en lenguaje natural
def natural_language_explanation(top_pos, top_neg):
    explanations = []

    for item in top_pos:
        explanations.append(
            f"{item['feature']} aumenta el riesgo (impacto +{item['shap']})"
        )

    for item in top_neg:
        explanations.append(
            f"{item['feature']} reduce el riesgo (impacto {item['shap']})"
        )

    return " | ".join(explanations)


@app.post("/explain_form", response_class=HTMLResponse)
def explain_form(
    edad: int = Form(...),
    antiguedad: int = Form(...),
    incidencias: int = Form(...),
    uso_medio: str = Form(...),
    descuento: str = Form(...)
):
    data = {
        "edad": edad,
        "antiguedad": antiguedad,
        "incidencias": incidencias,
        "uso_medio": uso_medio,
        "descuento": descuento
    }

    df = pd.DataFrame([data])
    X_trans = prep.transform(df)

    prob = float(model.predict_proba(X_trans)[0, 1])
    shap_values = explainer.shap_values(X_trans)

    feature_names = get_feature_names(prep)
    shap_row = shap_values[0]

    df_shap = pd.DataFrame({
        "feature": feature_names,
        "shap": shap_row
    })

    df_shap["abs_shap"] = df_shap["shap"].abs()
    df_shap = df_shap.sort_values("abs_shap", ascending=False)

    top_positive = df_shap[df_shap["shap"] > 0].head(3)[["feature", "shap"]].round(4).to_dict(orient="records")
    top_negative = df_shap[df_shap["shap"] < 0].head(3)[["feature", "shap"]].round(4).to_dict(orient="records")

    explanation_text = natural_language_explanation(top_positive, top_negative)

    risk = "ALTO" if prob > 0.7 else "BAJO"

    return f"""
    <html>
    <body>
        <h2>Resultado de evaluación</h2>

        <p><b>Probabilidad de riesgo:</b> {round(prob,4)}</p>
        <p><b>Nivel de riesgo:</b> {risk}</p>

        <h3>Explicación del modelo</h3>
        <p>{explanation_text}</p>

        <h4>Factores que aumentan el riesgo</h4>
        <ul>
            {''.join([f"<li>{f['feature']}: +{f['shap']}</li>" for f in top_positive])}
        </ul>

        <h4>Factores que reducen el riesgo</h4>
        <ul>
            {''.join([f"<li>{f['feature']}: {f['shap']}</li>" for f in top_negative])}
        </ul>

        <br>
        <a href="/">Volver al formulario</a>
    </body>
    </html>
    """

