"""
pipeline.py (version 2)
---------------------

This module implements a reproducible machine‑learning pipeline with several
improvements over the first release.  It demonstrates how to load and
encrypt a dataset, perform feature engineering, train and evaluate a
classifier using cross‑validation, and persist all resulting artefacts
alongside a detailed log.  The use of cross‑validation reduces
variance in performance estimates and is recommended in MLOps guides for
reliable model selection【633013889571114†L178-L196】.  All file system
paths are derived from the project root, and directories are created on
demand.  Execution progress is logged to a rotating file as well as to
standard output.

Running this script produces:

* `data_v2/raw/data.csv` – the raw Iris dataset
* `data_v2/encrypted/data.enc` – an encrypted copy of the raw data
* `data_v2/raw/decoded_data.csv` – the decrypted dataset
* `data_v2/models/iris_rf.joblib` – the trained Random Forest classifier
* `data_v2/models/scaler.joblib` – the fitted StandardScaler
* `data_v2/models/metrics.json` – a JSON report of evaluation metrics
* `logs/pipeline.log` – a detailed log of the pipeline execution

To execute the pipeline run:

```bash
python pipeline/pipeline.py
```

"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def setup_logging(log_dir: Path) -> None:
    """Configure logging to output to both the console and a rotating file.

    Args:
        log_dir: Directory where the log file will be created.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"
    handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), handler],
    )


def generate_key(key_path: Path) -> bytes:
    """Generate a symmetric encryption key and save it to a file.

    Args:
        key_path: Path to store the key.

    Returns:
        The generated key.
    """
    key = Fernet.generate_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    logging.info("Generated new encryption key at %s", key_path)
    return key


def load_key(key_path: Path) -> bytes:
    """Load an encryption key from the given path, generating one if it does not exist."""
    if not key_path.exists():
        return generate_key(key_path)
    key = key_path.read_bytes()
    logging.info("Loaded existing encryption key from %s", key_path)
    return key


def encrypt_file(source_path: Path, dest_path: Path, key: bytes) -> None:
    """Encrypt a file using the provided key.

    Args:
        source_path: Path to the plaintext file.
        dest_path: Path where the encrypted file will be written.
        key: Symmetric encryption key.
    """
    fernet = Fernet(key)
    data = source_path.read_bytes()
    encrypted = fernet.encrypt(data)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(encrypted)
    logging.info("Encrypted %s -> %s", source_path.name, dest_path.name)


def decrypt_file(source_path: Path, dest_path: Path, key: bytes) -> None:
    """Decrypt a file using the provided key.

    Args:
        source_path: Path to the encrypted file.
        dest_path: Path where the decrypted file will be written.
        key: Symmetric encryption key.
    """
    fernet = Fernet(key)
    encrypted_data = source_path.read_bytes()
    decrypted = fernet.decrypt(encrypted_data)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(decrypted)
    logging.info("Decrypted %s -> %s", source_path.name, dest_path.name)


def prepare_dataset(raw_csv_path: Path) -> pd.DataFrame:
    """Load or create the Iris dataset and save it to a CSV.

    Args:
        raw_csv_path: Path to write the raw CSV.

    Returns:
        A pandas DataFrame containing the dataset.
    """
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target.rename("target")], axis=1)
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_csv_path, index=False)
    logging.info("Saved raw dataset to %s", raw_csv_path)
    return df


def feature_engineering(df: pd.DataFrame):
    """Perform basic feature scaling and split features/target.

    Args:
        df: DataFrame containing features and target.

    Returns:
        Tuple of training/testing arrays and the fitted scaler.
    """
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def cross_validate_model(X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """Perform stratified K‑fold cross‑validation and return statistics.

    Args:
        X: Feature matrix.
        y: Target vector.
        cv: Number of cross‑validation folds.

    Returns:
        A dictionary containing the mean and standard deviation of accuracy across folds.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    logging.info(
        "Cross‑validation accuracy: %.4f ± %.4f (n=%d)", mean_score, std_score, cv
    )
    return {"cv_mean_accuracy": mean_score, "cv_std_accuracy": std_score}


def train_model(X_train: np.ndarray, y_train: pd.Series) -> RandomForestClassifier:
    """Train a RandomForestClassifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        The trained classifier.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    logging.info("Trained RandomForestClassifier with %d trees", clf.n_estimators)
    return clf


def evaluate_model(clf: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """Evaluate the model and return multiple metrics.

    Args:
        clf: Trained classifier.
        X_test: Test features.
        y_test: Test target labels.

    Returns:
        Dictionary with accuracy and classification report.
    """
    preds = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True)
    logging.info("Evaluation accuracy: %.4f", accuracy)
    return {"accuracy": accuracy, "classification_report": report}


def main() -> None:
   
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data_v2"
    raw_csv = data_dir / "raw" / "data.csv"
    encrypted_file = data_dir / "encrypted" / "data.enc"
    decrypted_csv = data_dir / "raw" / "decoded_data.csv"
    key_path = data_dir / "encrypted" / "key.key"
    model_path = data_dir / "models" / "iris_rf.joblib"
    scaler_path = data_dir / "models" / "scaler.joblib"
    metrics_path = data_dir / "models" / "metrics.json"
    log_dir = project_root / "logs"

    setup_logging(log_dir)
    logging.info("Starting ML pipeline version 2")

    
    df = prepare_dataset(raw_csv)

    
    key = load_key(key_path)
    encrypt_file(raw_csv, encrypted_file, key)
    decrypt_file(encrypted_file, decrypted_csv, key)

    
    df_decoded = pd.read_csv(decrypted_csv)

   
    X_train, X_test, y_train, y_test, scaler = feature_engineering(df_decoded)

    
    full_X = df_decoded.drop(columns=["target"]).values
    full_y = df_decoded["target"].values
    cv_results = cross_validate_model(full_X, full_y, cv=5)

   
    clf = train_model(X_train, y_train)

    
    eval_results = evaluate_model(clf, X_test, y_test)

   
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    logging.info("Saved model to %s", model_path)
    logging.info("Saved scaler to %s", scaler_path)

    
    metrics = {
        "cv_results": cv_results,
        "test_accuracy": eval_results["accuracy"],
        "classification_report": eval_results["classification_report"],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Saved evaluation metrics to %s", metrics_path)

    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
