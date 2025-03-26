import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib

# Configurar paths absolutos
BASE_DIR = r"/Users/andresfelipe/Documents/Prog2/Programacion-2/Programacion-2/Challenge1"
DATA_PATH = os.path.join(BASE_DIR, "data", "breast-cancer-wisconsin.data.csv")
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")

def evaluate_model(model, X_test, y_test, run_id):
    """Evaluar el modelo y registrar métricas"""
    with mlflow.start_run(run_id=run_id):
        # Hacer predicciones
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Guardar y registrar gráficos
        plot_roc_curve(fpr, tpr, roc_auc)
        plot_confusion_matrix(y_test, y_pred)
        
        return accuracy, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc):
    """Graficar y guardar curva ROC"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("roc_curve.png")
    plt.close()
    mlflow.log_artifact("roc_curve.png")

def plot_confusion_matrix(y_test, y_pred):
    """Graficar y guardar matriz de confusión"""
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benigno', 'Maligno'], 
                yticklabels=['Benigno', 'Maligno'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

if __name__ == "__main__":
    # Configurar MLflow
    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment("Breast_Cancer_Wisconsin")
    
    # Obtener el último run_id de entrenamiento
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Breast_Cancer_Wisconsin")
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    run_id = runs[0].info.run_id
    
    # Cargar datos
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id", "Unnamed: 32"])
    df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])
    
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    
    # Cargar scaler
    scaler_path = os.path.join(BASE_DIR, "scripts", "scaler.save")
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    
    # Dividir datos (usando mismo random_state para consistencia)
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Cargar modelo desde MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Evaluar modelo
    accuracy, roc_auc = evaluate_model(model, X_test, y_test, run_id)
    
    print(f"Evaluación completada - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")