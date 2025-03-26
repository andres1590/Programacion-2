# mlops_pipeline.py
import os
import mlflow as mlf
import mlflow.sklearn
from mlflow.models import infer_signature
from datetime import datetime

# Import your modules
from module_load_data import load_breast_cancer_data
from module_preprocessing import preprocess_module
from module_training import train_model_module
from module_evaluation import evaluate_model_module

# Set MLflow tracking URI
mlf.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlf.set_experiment("Breast Cancer Prediction Pipeline")

def main():
    # --- Main Pipeline Run ---
    with mlf.start_run(run_name="MLOps Pipeline"):
        mlf.log_param("pipeline_start_time", datetime.now().isoformat())

        # --- Load Data ---
        with mlf.start_run(run_name="Data Loading", nested=True):
            file_path = r"/Users/andresfelipe/Documents/Prog2/Programacion-2/Programacion-2/Challenge1/data/breast-cancer-wisconsin.data.csv"
            mlf.log_param("data_path", file_path)
            df = load_breast_cancer_data(file_path)
            if df is None:
                mlf.log_param("data_load_status", "Failed")
                return
            mlf.log_param("data_load_status", "Success")

        # --- Preprocess Data ---
        with mlf.start_run(run_name="Data Preprocessing", nested=True):
            X, y = preprocess_module(df)
            if y is None:
                mlf.log_param("preprocessing_status", "Failed - Target variable not found")
                print("Target variable not found. Exiting.")
                return
            mlf.log_param("preprocessing_status", "Success")
            mlf.log_param("feature_count", X.shape[1] if hasattr(X, 'shape') else None)
            mlf.log_param("sample_count", X.shape[0] if hasattr(X, 'shape') else None)

        # --- Model Training ---
        with mlf.start_run(run_name="LogisticRegression Training", nested=True): # Removed f-string
            model_name = "LogisticRegression"  # Define model_name here within the scope
            mlf.log_param("model_name", model_name)
            test_size = 0.2
            random_state = 42
            mlf.log_param("test_size", test_size)
            mlf.log_param("random_state", random_state)

            # Train the model
            model, X_test, y_test = train_model_module(
                X, y, model_name=model_name, test_size=test_size, random_state=random_state
            )

            # Infer model signature
            signature = infer_signature(X_train=X, model_input=X)
            mlf.sklearn.log_model(model, f"{model_name}_model", signature=signature)
            print(f"MLflow: Logged model signature.")
            mlf.log_param("model_logged", True)

        # --- Model Evaluation ---
        with mlf.start_run(run_name="LogisticRegression Evaluation", nested=True): # Removed f-string
            model_name = "LogisticRegression" # Define model_name here within the scope
            mlf.log_param("evaluated_model_name", model_name)
            # Evaluate the model (Metrics and artifacts are logged in the evaluation module)
            evaluate_model_module(model, X_test, y_test, model_name=model_name)
            mlf.log_param("evaluation_completed", True)

        mlf.log_param("pipeline_end_time", datetime.now().isoformat())
        print("\nMLOps Pipeline Completed Successfully.")

if _name_ == "_main_":
    main()