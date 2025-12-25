import os
import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from data_processing import load_and_preprocess_data


def evaluate_model(data_path, model_path, preprocessor_path, threshold=0.5):
    """
    Evaluate the trained model on test data using a specific threshold.

    Args:
        data_path (str): Path to the raw CSV data.
        model_path (str): Path to the trained Logistic Regression model.
        preprocessor_path (str): Path to the fitted preprocessor.
        threshold (float): Probability threshold for classifying fraud.

    Returns:
        tuple: (ROC-AUC score, confusion matrix, classification report)
    """
    print(f"[INFO] Starting model evaluation with threshold={threshold}...")

    # 1. Load processed data
    X, preprocessor, y = load_and_preprocess_data(data_path)

    # 2. Load trained model
    model = joblib.load(model_path)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # 5. Apply threshold
    y_pred = (y_proba >= threshold).astype(int)

    # 6. Metrics
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n========== MODEL EVALUATION ==========")
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return auc, cm, report


def threshold_tuning(data_path, model_path, thresholds=[0.3, 0.4, 0.5]):
    """
    Test multiple thresholds and print precision/recall/f1 for the fraud class.

    Args:
        data_path (str): Path to raw CSV data.
        model_path (str): Path to trained model.
        thresholds (list): List of probability thresholds to test.
    """
    # Load data and model
    X, preprocessor, y = load_and_preprocess_data(data_path)
    model = joblib.load(model_path)

    # Split test/train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Predict probabilities once
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n===== THRESHOLD TUNING RESULTS =====\n")

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']

        print(f"Threshold = {t}")
        print("Confusion Matrix:")
        print(cm)
        print(
            f"Precision (fraud): {precision:.2f}, "
            f"Recall (fraud): {recall:.2f}, "
            f"F1-score: {f1:.2f}\n"
        )


# -------------------------
# Local execution
# -------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH = os.path.join(BASE_DIR, "data", "sample_data.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_model.joblib")
    PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

    # Evaluate with default threshold 0.5
    evaluate_model(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        threshold=0.5
    )

    # Run threshold tuning (0.3, 0.4, 0.5)
    threshold_tuning(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        thresholds=[0.3, 0.4, 0.5]
    )
