import mlflow


def main():

    runs = mlflow.search_runs()

    if runs.empty:
        print("No MLflow runs found.")
        return

    best_run = runs.sort_values(
        by="metrics.accuracy",
        ascending=False
    ).iloc[0]

    print("\nBest Run:")
    print(f"Run ID: {best_run['run_id']}")
    print(
        f"Accuracy: "
        f"{best_run['metrics.accuracy']}"
    )

    if "metrics.precision" in best_run:
        print(
            f"Precision: "
            f"{best_run['metrics.precision']}"
        )

    if "metrics.recall" in best_run:
        print(
            f"Recall: "
            f"{best_run['metrics.recall']}"
        )


if __name__ == "__main__":
    main()