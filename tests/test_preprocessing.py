import pytest
import pandas as pd
from src.preprocessing import handle_missing_values, encode_labels, load_manifest

def test_handle_missing_values_drops_nans():
    """Verify that rows with NaN values in essential columns are dropped."""
    data = {
        "image_name": ["img1.jpg", None, "img3.jpg"],
        "label": ["amd", "cataract", None]
    }
    df = pd.DataFrame(data)
    df_clean = handle_missing_values(df)

    # Should only keep the row where both values are present
    assert len(df_clean) == 1
    assert df_clean.iloc[0]["image_name"] == "img1.jpg"

def test_handle_missing_values_raises_error_on_empty():
    """Verify that an empty DataFrame raises a ValueError."""
    df = pd.DataFrame(columns=["image_name", "label"])
    with pytest.raises(ValueError):
        handle_missing_values(df)

def test_encode_labels_correctly_maps():
    """Verify that categorical labels are mapped to expected integers"""
    data = {
        "image_name": ["img1.jpg", "img2.jpg"],
        "label": ["amd", "cataract"]
    }
    df = pd.DataFrame(data)
    df_encoded = encode_labels(df)

    # Check that labels are correctly mapped
    assert df_encoded.iloc[0]["label"] == 0  # amd -> 0
    assert df_encoded.iloc[1]["label"] == 1  # cataract -> 1

def test_data_manifest_loads_successfully():
    """Verify that the actual project dataset manifest loads correctly and has the required columns."""
    df = load_manifest()
    assert not df.empty, "Manifest file is empty."
    assert "image_name" in df.columns, "Manifest file is missing 'image_name' column."
    assert "label" in df.columns, "Manifest file is missing 'label' column."