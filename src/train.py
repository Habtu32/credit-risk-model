import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from data_processing import load_and_preprocess_data


def train_model(file_path, model_path, preprocessor_path):
    """
    Trains a Logistic Regression credit risk model and saves
    both the trained model and fitted preprocessor.
    """

    print("[INFO] Starting model training process...")

    # 1. Load & preprocess
    print(f"[INFO] Loading data from {file_path}")
    X_processed, preprocessor, y = load_and_preprocess_data(file_path)

    print(f"[INFO] Feature matrix shape: {X_processed.shape}")
    print(f"[INFO] Target shape: {y.shape}")

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Train model
    model = LogisticRegression(
        random_state=42,
        solver="liblinear"
    )

    model.fit(X_train, y_train)
    print("[INFO] Model training completed.")

    # 4. Save artifacts
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"[INFO] Model saved to: {model_path}")
    print(f"[INFO] Preprocessor saved to: {preprocessor_path}")
    print("[INFO] Training pipeline finished successfully.")


# -------------------------
# Local execution
# -------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH = os.path.join(BASE_DIR, "data", "sample_data.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_model.joblib")
    PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

    train_model(
        file_path=DATA_PATH,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH
    )
