import pandas as pd
import yaml
from pathlib import Path

def load_config(config_path="configs/config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) 

def load_manifest(manifest_path=None):
    """Loads the image manifest CSV file."""
    if manifest_path is None:
        config = load_config()
        manifest_path = config.get("data_manifest", "data/image_manifest.csv")
    
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    return pd.read_csv(manifest_path)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values in the DataFrame without modifying the original dataframe.
    Raises ValueError if input is not a dataframe or is empty."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Create a copy to ensure the original dataframe is not modified
    df_clean = df.copy()

    # Drop rows where essential columns are missing
    df_clean = df_clean.dropna(subset=["image_name", "label"])
    return df_clean

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical labels ("amd", "cataract") into numerical values.
    Creates a copy to avoid modifying the original dataframe."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column.")

    # Create a copy to ensure the original dataframe is not modified
    df_encoded = df.copy()

    # Mapping labels to integers (e.g., amd: 0, cataract: 1)
    label_mapping = {"amd": 0, "cataract": 1}

    # Check for unexpected labels 
    invalid_labels = set(df_encoded["label"].unique()) - set(label_mapping.keys())
    if invalid_labels:
        raise ValueError(f"Unexpected labels found: {invalid_labels}")
    
    df_encoded["label"] = df_encoded["label"].map(label_mapping)
    return df_encoded

if __name__ == "__main__":
    # Test loading and preprocessing pipeline
    config = load_config()
    print("Configuration loaded successfully:", config)

    df_clean = load_manifest()
    print("Manifest loaded successfully. Number of records:", len(df_clean))

    df_clean = handle_missing_values(df_clean)
    print("Missing values handled. Number of records after cleaning:", len(df_clean))

    df_processed = encode_labels(df_clean)
    print("Preprocessing completed successfully! Number of records after encoding:", len(df_processed))
    print(df_processed.head())