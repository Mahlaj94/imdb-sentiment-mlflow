"""Compare three text classifiers on a reproducible IMDb sample."""
import mlflow
import mlflow.sklearn
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

EXPERIMENT = "imdb_sentiment"
SEED = 42


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT)
    dataset = load_dataset("imdb")
    train = dataset["train"].shuffle(seed=SEED).select(range(5000))
    test = dataset["test"].shuffle(seed=SEED).select(range(1000))
    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)),
        ("LinearSVM", LinearSVC(C=1.0, random_state=SEED)),
    ]
    for name, classifier in models:
        pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=10000)), ("classifier", classifier)])
        with mlflow.start_run(run_name=name):
            pipeline.fit(train["text"], train["label"])
            predictions = pipeline.predict(test["text"])
            metrics = {
                "accuracy": accuracy_score(test["label"], predictions),
                "f1_score": f1_score(test["label"], predictions),
            }
            mlflow.log_params({"model": name, "sample_train": 5000, "sample_test": 1000, "seed": SEED})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="sentiment_model", input_example=["A wonderful movie."])
            print(f"{name}: accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
