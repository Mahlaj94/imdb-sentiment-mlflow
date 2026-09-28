"""Select the highest-F1 run from the local comparison experiment."""
import mlflow
from mlflow.tracking import MlflowClient


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name("imdb_sentiment")
    if experiment is None:
        raise RuntimeError("Run train2.py before selecting a model.")
    runs = MlflowClient().search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.f1_score >= 0",
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No scored runs found. Run train2.py first.")
    run = runs[0]
    print(f"Model: {run.data.params.get('model', 'unknown')}")
    print(f"F1: {run.data.metrics['f1_score']:.4f}")
    print(f"MODEL_URI=runs:/{run.info.run_id}/sentiment_model")


if __name__ == "__main__":
    main()
