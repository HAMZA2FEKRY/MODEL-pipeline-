
import mlflow
import sys
import os

THRESHOLD = 0.999

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

print(f"Accuracy: {accuracy}")

if accuracy < THRESHOLD:
    print(f" FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
    sys.exit(1)
else:
    print(f" PASSED: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}")