import os
import json
import yaml
import cv2
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)

from preprocessing import (
    load_manifest,
    handle_missing_values,
    encode_labels,
)


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"FAILED: {image_path}")
        return None

    image = cv2.resize(image, (64, 64))
    return image.flatten()


def build_dataset(df):
    X = []
    y = []

    for _, row in df.iterrows():
        print(row["label"])
        image_path = (
            f"data/raw/AMDNet23_Dataset/train/"
            f"{row['label']}/{row['image_name']}"
        )

        features = extract_features(image_path)

        if features is not None:
            X.append(features)
            y.append(str(row["label"]).lower())

    return X, y


def main():

    config = load_config()

    df = load_manifest()
    df = handle_missing_values(df)

    print(df.head())
    # Build image features first
    X, labels = build_dataset(df)
    print(f"Images loaded successfully: {len(X)}")
    print(f"Labels loaded successfully: {len(labels)}")

    print(f"Loaded {len(X)} images")
    
    if len(X) == 0:
        raise ValueError(
            "No images were loaded. Check image paths."
        )

    # Then encode labels
    label_mapping = {
        "amd": 0,
        "cataract": 1
    }

    y = [label_mapping[label] for label in labels]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=config["random_state"]
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            average="weighted"
        )
        recall = recall_score(
            y_test,
            predictions,
            average="weighted"
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall)
        }

        metrics_path = "metrics/scores.json"

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        model_path = "models/model.joblib"
        joblib.dump(model, model_path)

        mlflow.log_param(
            "n_estimators",
            100
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)

        print(metrics)


if __name__ == "__main__":
    main()
