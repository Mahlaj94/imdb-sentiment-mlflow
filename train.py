"""Train a logistic-regression baseline and log a complete text pipeline."""
import mlflow
import mlflow.sklearn
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

EXPERIMENT = "imdb_sentiment"


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT)
    dataset = load_dataset("imdb")
    train = dataset["train"].shuffle(seed=42).select(range(5000))
    test = dataset["test"].shuffle(seed=42).select(range(1000))
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    with mlflow.start_run(run_name="LogisticRegression-baseline"):
        model.fit(train["text"], train["label"])
        predictions = model.predict(test["text"])
        mlflow.log_params({"model": "LogisticRegression", "sample_train": 5000, "sample_test": 1000, "seed": 42})
        mlflow.log_metrics({"accuracy": accuracy_score(test["label"], predictions), "f1_score": f1_score(test["label"], predictions)})
        mlflow.sklearn.log_model(model, artifact_path="sentiment_model", input_example=["A wonderful movie."])
        print("Baseline logged to MLflow.")


if __name__ == "__main__":
    main()
