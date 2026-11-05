# Eye Disease MLOps

A small Python project for preparing an eye-image classification manifest, validating and preprocessing its labels, running basic tests, and generating a data-drift report. The repository includes configuration for a future classification workflow, but it does **not** currently implement or train an image-classification model.

> **Use limitation:** This repository is a development/demo project. Its code and tests are not a clinically validated diagnostic system and must not be used to make medical decisions.

## What the project does today

- Tracks the raw image dataset and generated manifest with [DVC](https://dvc.org/).
- Builds a CSV manifest from the `amd` and `cataract` training-image folders.
- Loads the manifest and provides helpers to remove incomplete rows and encode labels.
- Runs unit/data checks with `pytest`.
- Compares two portions of the manifest with Evidently and saves an HTML drift report.
- Runs tests in GitHub Actions on pushes and pull requests to `main`.

The configured CNN settings and model/metrics output paths are placeholders: there is no training or inference entry point in the current source.

## Repository layout

```text
.
├── .github/workflows/ci.yml  # GitHub Actions test workflow
├── configs/config.yaml       # Dataset and placeholder model settings
├── data/
│   ├── raw.dvc               # DVC pointer for the raw image directory
│   └── image_manifest.csv.dvc
├── src/
│   ├── generate_data.py      # Creates data/image_manifest.csv
│   ├── monitor_drift.py      # Creates reports/drift_report.html
│   └── preprocessing.py      # Manifest loading and preprocessing helpers
├── tests/                    # Data, preprocessing, and baseline model tests
├── requirements.txt
└── pytest.ini
```

Generated outputs such as `reports/` and trained model files are ignored by Git. The actual dataset is not stored as ordinary source files in Git; it is described by DVC pointer files.

## Requirements

- Python 3.11 (the version used by CI)
- Git
- DVC with its S3 support
- AWS credentials with access to the project's configured DVC remote, to pull the dataset
- Evidently, if running drift monitoring 

## Setup

From the repository root in your terminal (PowerShell or Git Bash):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements include `dvc[s3]`, which installs DVC and its S3 integration. If PowerShell blocks virtual-environment activation, use the Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Get the data and create the manifest

Configure AWS credentials for the DVC remote using your normal AWS credential provider (for example, an AWS profile or environment variables). Do not commit credentials to the repository. Then, from the repository root:

```powershell
dvc pull
python src/generate_data.py
```

`dvc pull` retrieves the tracked data from the S3 remote configured in `.dvc/config`. It requires valid credentials and remote access. The manifest generator looks for:

```text
data/raw/AMDNet23_Dataset/train/amd/
data/raw/AMDNet23_Dataset/train/cataract/
```

It includes files whose names end in lowercase `.jpg`, and writes `data/image_manifest.csv` with two columns:

| Column | Meaning |
|---|---|
| `image_name` | Image filename (not a full path) |
| `label` | `amd` or `cataract` |

If either class folder is absent, the generator prints a warning and skips it. If both are missing or contain no matching files, it still writes a manifest containing only the CSV header; tests that require a non-empty manifest will then fail.

## Preprocessing

The helpers in `src/preprocessing.py` can be imported from Python:

```python
from src.preprocessing import encode_labels, handle_missing_values, load_manifest

manifest = load_manifest()
clean_manifest = handle_missing_values(manifest)
encoded_manifest = encode_labels(clean_manifest)
```

- `load_manifest()` reads the path in `configs/config.yaml` (`data/image_manifest.csv` by default) and raises `FileNotFoundError` if it is missing.
- `handle_missing_values()` returns a copy with rows missing `image_name` or `label` removed. It rejects non-DataFrame and empty inputs.
- `encode_labels()` returns a copy with `amd` mapped to `0` and `cataract` mapped to `1`; it raises `ValueError` for missing `label` columns or unexpected labels.

To run the preprocessing module's small end-to-end check:

```powershell
python src/preprocessing.py
```

## Drift monitoring

Install Evidently in the active environment and run the monitor from the repository root:

```powershell
python -m pip install evidently
python src/monitor_drift.py
```

The script loads `data/image_manifest.csv`, randomly selects 70% of its rows as reference data (seed `42`), and treats the remaining rows as current data. It writes `reports/drift_report.html` and checks whether the share of drifted columns exceeds the hard-coded threshold of `0.5`; if so, it exits with status `1`.

This is a demonstration rather than production monitoring: the split is random, not time-based, and the report analyzes manifest columns (`image_name` and `label`), not image pixels or a live production feed. The script also currently prints a note and continues if it cannot parse the drift metrics, so a successful process exit does not necessarily mean the threshold check ran.

## Configuration

`configs/config.yaml` currently contains:

| Setting | Default | Current use |
|---|---|---|
| `data_manifest` | `data/image_manifest.csv` | Used by `load_manifest()` |
| `image_dir` | `data/raw/AMDNet23 Dataset/train` | Declared path; not used by current source |
| `test_size` | `0.2` | Placeholder; not used by current source |
| `random_state` | `42` | Declared setting; drift script currently hard-codes its own seed |
| `model_type`, `batch_size`, `epochs`, `learning_rate` | See config | Placeholder training settings |
| `model_output_path`, `metrics_output_path` | `models/best_model.pt`, `metrics/scores.json` | Placeholder output paths |

The configured `image_dir` uses a space in `AMDNet23 Dataset`, while the manifest generator uses `AMDNet23_Dataset` with an underscore. The generator's hard-coded path is the one that must match the directory present under `data/raw/`.

## Tests

After retrieving the data and generating the manifest, run:

```powershell
pytest
```

Or run the CI command explicitly:

```powershell
pytest tests/
```

The tests cover required manifest columns and labels, preprocessing behavior, and a small synthetic scikit-learn Random Forest prediction/performance check. The Random Forest test is not a test of a trained eye-image model. Tests that load the actual manifest require the dataset to have been pulled and the manifest to exist.

## Continuous integration

`.github/workflows/ci.yml` runs on pushes and pull requests targeting `main`. It uses Python 3.11, installs dependencies, authenticates to AWS from repository secrets, pulls DVC data, generates the manifest, and runs `pytest tests/`. The `train` job depends on the `test` job, but currently repeats the same setup and tests; it does not train a model.

The workflow expects these GitHub Actions secrets to be configured for its AWS credentials step:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Known gaps

- No image loader, model-training pipeline, inference API/CLI, or deployment process is implemented.
- CNN and training/output settings in the YAML are not yet consumed by the application code.
- Evidently is required by the drift script but is absent from `requirements.txt`.
- Drift monitoring compares a random split of a metadata manifest rather than incoming production data.
- The manifest contains filenames rather than paths, so downstream image loading would need to resolve filenames against the dataset directory.
