import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

# Configurar paths absolutos
BASE_DIR = r"/Users/andresfelipe/Documents/Prog2/Programacion-2/Programacion-2/Challenge1"
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")

def load_latest_model():
    """Cargar el último modelo entrenado desde MLflow"""
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Breast_Cancer_Wisconsin")
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri), run_id

def preprocess_input(data, scaler_path):
    """Preprocesar datos de entrada usando el scaler guardado"""
    scaler = joblib.load(scaler_path)
    return scaler.transform(data)

if __name__ == "__main__":
    # Configurar MLflow
    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_URI}")
    
    # Cargar modelo y scaler
    model, run_id = load_latest_model()
    scaler_path = os.path.join(BASE_DIR, "scripts", "scaler.save")
    
    # Datos de ejemplo para predicción (deberías reemplazar esto con datos reales)
    example_data = {
        'radius_mean': [17.99], 'texture_mean': [10.38], 'perimeter_mean': [122.8],
        'area_mean': [1001], 'smoothness_mean': [0.1184], 'compactness_mean': [0.2776],
        'concavity_mean': [0.3001], 'concave points_mean': [0.1471], 'symmetry_mean': [0.2419],
        'fractal_dimension_mean': [0.07871], 'radius_se': [1.095], 'texture_se': [0.9053],
        'perimeter_se': [8.589], 'area_se': [153.4], 'smoothness_se': [0.006399],
        'compactness_se': [0.04904], 'concavity_se': [0.05373], 'concave points_se': [0.01587],
        'symmetry_se': [0.03003], 'fractal_dimension_se': [0.006193], 'radius_worst': [25.38],
        'texture_worst': [17.33], 'perimeter_worst': [184.6], 'area_worst': [2019],
        'smoothness_worst': [0.1622], 'compactness_worst': [0.6656], 'concavity_worst': [0.7119],
        'concave points_worst': [0.2654], 'symmetry_worst': [0.4601], 'fractal_dimension_worst': [0.1189]
    }
    input_df = pd.DataFrame(example_data)
    
    # Preprocesar datos
    X_scaled = preprocess_input(input_df, scaler_path)
    
    # Hacer predicción
    prediction = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Registrar predicción
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("latest_prediction", int(prediction[0]))
        mlflow.log_metric("probability_benign", probabilities[0][0])
        mlflow.log_metric("probability_malignant", probabilities[0][1])
    
    # Mostrar resultados
    diagnosis = "Benigno (0)" if prediction[0] == 0 else "Maligno (1)"
    print(f"\nResultado de la predicción:")
    print(f"Diagnóstico: {diagnosis}")
    print(f"Probabilidad Benigno: {probabilities[0][0]:.4f}")
    print(f"Probabilidad Maligno: {probabilities[0][1]:.4f}")