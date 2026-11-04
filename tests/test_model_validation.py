import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

def test_model_prediction_shape_and_type():
    """Verify that the model produces predictions of the correct type and shape."""
    X_sample = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_sample = np.array([0, 1, 0, 1])

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_sample, y_sample)

    preds = model.predict(X_sample)
    assert isinstance(preds, np.ndarray), "Model predictions must be a NumPy array."
    assert preds.shape == (X_sample.shape[0],), "Prediction shape does not match expected row count."

def test_model_performance_threshold():
    """Verify that a baseline model meets a minimum performance threshold on a known set."""
    X_sample = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 1], [4, 3]])
    y_sample = np.array([0, 1, 0, 1, 0, 1])

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_sample, y_sample)

    accuracy = model.score(X_sample, y_sample)
    min_threshold = 0.50 # Adjust baseline threshold as needed for the model
    assert accuracy >= min_threshold, f"Model accuracy ({accuracy}) is below the minimum threshold ({min_threshold})"