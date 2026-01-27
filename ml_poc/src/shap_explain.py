import shap
import joblib
import pandas as pd

pipeline = joblib.load("model/pipeline_v1.joblib")
df = pd.read_csv("data/inference.csv")

X_trans = pipeline.named_steps["prep"].transform(df)
model = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_trans)

shap.summary_plot(shap_values, X_trans)
