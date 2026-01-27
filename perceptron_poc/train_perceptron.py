
# train_perceptron.py
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Cargar dataset Iris desde Hugging Face
ds = load_dataset("scikit-learn/iris")

# Columnas según Hugging Face
feature_columns = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

# Construcción de X (matriz) e y (vector)
X = np.array([[row[col] for col in feature_columns] for row in ds["train"]], dtype=float)

# Species viene como string ("Iris-setosa", etc.) → convertir a índice numérico
species = sorted(set(ds["train"]["Species"]))
species_to_int = {name: idx for idx, name in enumerate(species)}
y = np.array([species_to_int[s] for s in ds["train"]["Species"]], dtype=int)



# 2. Partición train/test (25% test), estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Pipeline: estandarización + Perceptron
clf = make_pipeline(
    StandardScaler(),
    Perceptron(max_iter=1000, eta0=1.0, random_state=42),
)

# 4. Entrenamiento
clf.fit(X_train, y_train)

# 5. Evaluación
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, digits=4))

# 6. Guardar el modelo con skops
try:
    import joblib as jl
    jl.dump(clf, "model.joblib")
    print("Modelo guardado en 'model.joblib'.")
except Exception as e:
    print("No se pudo guardar con joblib:", e)
