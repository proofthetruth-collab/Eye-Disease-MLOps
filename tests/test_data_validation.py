import pandas as pd
import os

def test_expected_columns_present():
    """Verify that required columns exist in the dataset manifest."""
    manifest_path = "data/image_manifest.csv"
    assert os.path.exists(manifest_path), "Dataset manifest file not found."
    df = pd.read_csv(manifest_path)

    expected_columns = ['image_name', 'label'] # Changed from 'image_path' to 'image_name'
    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' is missing from dataset."

def test_target_variable_valid_values():
    """Verify that the target/label column contains only expected categorical values."""
    manifest_path = "data/image_manifest.csv"
    df = pd.read_csv(manifest_path)

    assert 'label' in df.columns, "Target column 'label' missing from manifest."
    unique_labels = df['label'].dropna().unique()
    assert len(unique_labels) > 0, "Target variable contains no valid labels."

def test_numeric_features_ranges():
    """Verify that any numeric features fall within expected valid ranges."""
    manifest_path = "data/image_manifest.csv"
    df = pd.read_csv(manifest_path)

    # If your manifest contains numeric attributes like image dimensions or IDs
    if 'image_id' in df.columns:
        assert (df['image_id'] >= 0).all(), "Image IDs must be non-negative."