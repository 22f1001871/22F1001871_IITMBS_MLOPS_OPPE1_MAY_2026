import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# 1. Configure MLflow Tracking
# -----------------------------
mlflow.set_tracking_uri("http://34.132.103.154:5000/")  # <-- change only this
mlflow.set_experiment("stock_features_experiment")

# -----------------------------
# 2. Load Data
# -----------------------------
# Example: load your preprocessed CSVs
v0 = pd.read_csv("data/v0/train_v0.csv")
v1 = pd.read_csv("data/v1/train_v1.csv")

# Merge for iteration 2
v01 = pd.concat([v0, v1], ignore_index=True)

# -----------------------------
# 3. Utility: Train/Evaluate
# -----------------------------
def train_and_evaluate(df, run_name="iteration"):
    X = df[["open", "high", "low", "close", "volume", "rolling_avg_10", "volume_sum_10"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=run_name):
        # Simple Logistic Regression
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log metrics and model
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"{run_name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

# -----------------------------
# 4. Iteration 1 & 2
# -----------------------------
train_and_evaluate(v0, run_name="iteration_1_v0")
train_and_evaluate(v01, run_name="iteration_2_v0_v1")

# -----------------------------
# 5. Hyperparameter Sweep
# -----------------------------
param_grid = [
    {"C": 0.01, "penalty": "l2"},
    {"C": 0.1, "penalty": "l2"},
    {"C": 1.0, "penalty": "l2"},
    {"C": 10.0, "penalty": "l2"},
]

best_acc = 0
best_run_id = None

for params in param_grid:
    with mlflow.start_run(run_name=f"tuning_C={params['C']}"):
        model = LogisticRegression(max_iter=500, **params)
        X = v01[["open", "high", "low", "close", "volume", "rolling_avg_10", "volume_sum_10"]]
        y = v01["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        if acc > best_acc:
            best_acc = acc
            best_run_id = mlflow.active_run().info.run_id

print(f"Best run ID: {best_run_id}, Accuracy: {best_acc:.4f}")

# -----------------------------
# 6. Register Best Model
# -----------------------------
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, "stock_logistic_model")
