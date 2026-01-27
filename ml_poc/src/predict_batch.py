import pandas as pd
import joblib

pipeline = joblib.load("model/pipeline_v1.joblib")

df = pd.read_csv("data/inference.csv")

df["probability"] = pipeline.predict_proba(df)[:, 1]
df["risk"] = df["probability"].apply(lambda x: "high" if x > 0.7 else "low")

df.to_csv("data/predictions.csv", index=False)
