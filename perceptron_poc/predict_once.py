# predict_once.py
import numpy as np
import skops.io as sio

# Cargar el modelo entrenado
clf = sio.load("models.skops")

# Ejemplo: una flor con (sepal_length, sepal_width, petal_length, petal_width)
x = np.array([[5.1, 3.5, 1.4, 0.2]])  # valores típicos de setosa
pred = clf.predict(x)[0]

mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
print("Predicción:", mapping.get(int(pred), str(pred)))
