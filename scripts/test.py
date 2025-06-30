import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import load_model

# === Параметры ===
dataset_type = "traffic"
model_name = "default"
project_root = os.path.dirname(os.path.dirname(__file__))

# === Пути ===
model_path = os.path.join(project_root, "outputs", "models", dataset_type, f"{dataset_type}_{model_name}_lstm.keras")
data_dir = os.path.join(project_root, "data", dataset_type, "processed", "prepared_dataset")

# === Загрузка данных ===
X_test = np.load(os.path.join(data_dir, "X_test.npy"))
y_test = np.load(os.path.join(data_dir, "y_test.npy"))

# === Загрузка модели ===
model = load_model(model_path)

# === Предсказание ===
y_pred = model.predict(X_test).flatten()

# === Метрики ===
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Evaluation results for model: {dataset_type}_{model_name}_lstm.keras")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² : {r2:.4f}")
