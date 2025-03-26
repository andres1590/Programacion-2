import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


# Configurar paths absolutos
BASE_DIR = r"/Users/andresfelipe/Documents/Prog2/Programacion-2/Programacion-2/Challenge1"
DATA_PATH = os.path.join(BASE_DIR, "data", "breast-cancer-wisconsin.data.csv")
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")

def load_data(file_path):
    """Cargar y preparar los datos"""
    df = pd.read_csv(file_path)
    df = df.drop(columns=["id", "Unnamed: 32"])
    df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])
    return df

def preprocess_data(df):
    """Preprocesamiento de datos"""
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y), scaler

def train_model(X_train, y_train):
    """Entrenar el modelo"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Configurar MLflow
    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment("Breast_Cancer_Wisconsin")
    
    # Cargar y preparar datos
    df = load_data(DATA_PATH)
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(df)
    
    with mlflow.start_run(run_name="Training_Run"):
        # Entrenar modelo
        model = train_model(X_train, y_train)
        
        # Registrar parámetros
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        
        # Guardar el scaler como artefacto
        import joblib
        scaler_path = os.path.join(BASE_DIR, "scripts", "scaler.save")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, "scaler")
        
        # Registrar modelo
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Modelo entrenado y registrado en MLflow. Run ID: {mlflow.active_run().info.run_id}")


      
        # Despues de ejecutar el Activar el server mlflow

        # mlflow server --backend-store-uri file:///C:/Users/osval/OneDrive/Documents/3.%20Projectos%20Visual%20Studio/Programacion2/Challenges/Challenge1/mlruns --host 127.0.0.1 --port 5000