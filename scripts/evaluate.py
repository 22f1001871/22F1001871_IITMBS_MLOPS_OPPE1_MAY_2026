import argparse, mlflow, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--report", required=True)
args = parser.parse_args()

# Load model and data
model = mlflow.pyfunc.load_model(args.model)
df = pd.read_csv(args.data)

X = df[["open","high","low","close","volume","rolling_avg_10","volume_sum_10"]]
y = df["target"]

preds = model.predict(X)
acc = accuracy_score(y, preds)
f1 = f1_score(y, preds)

# Sanity checks
assert df["open"].notnull().all(), "Open has nulls!"
assert df["volume"].ge(0).all(), "Volume must be non-negative!"

# Plot predictions vs actual
plt.figure(figsize=(8,4))
plt.plot(y.values[:100], label="Actual")
plt.plot(preds[:100], label="Predicted")
plt.legend()
plt.savefig("pred_vs_actual.png")

# Write CML report
with open(args.report, "w") as f:
    f.write("# CI Evaluation Report\n\n")
    f.write(f"**Accuracy:** {acc:.4f}\n\n")
    f.write(f"**F1 Score:** {f1:.4f}\n\n")
    f.write("![Predictions vs Actual](pred_vs_actual.png)\n")
