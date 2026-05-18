import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os
from pathlib import Path


# Set MLflow tracking URI (local by default)
# mlflow.set_tracking_uri("http://127/0.0.1:5000")
# mlflow.set_experiment("diabetes-prediction")

# Use local file-based tracking (no server needed)
mlflow.set_experiment("diabetes-prediction")

def load_data():
    data_path = Path("data/raw/diabetes.csv")
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    # Very basic preprocessing (you can improve this later)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y

def main():
    # Start MLflow run
    with mlflow.start_run(run_name="random-forest-baseline"):
        # Log parameters
        n_estimators = 100
        max_depth = 10
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Load and Split data
        df = load_data()
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # predict and Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log  model
        mlflow.sklearn.log_model(model, "model", registered_model_name="diabetes-random-forest")
        
        # Save model locally too
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/diabetes_model.pkl")
        
        print(f"✅ Model trained! Accuracy: {accuracy:.4f}")
        print("✅ Model registered in MLflow Registry as 'diabetes-random-forest'")
if __name__ == "__main__":
    main()
        