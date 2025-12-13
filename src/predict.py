import joblib
import pandas as pd

def load_model(model_path: str):
    return joblib.load(model_path)

def predict(model, data: pd.DataFrame):
    return model.predict(data)

if __name__ == "__main__":
    # Example usage
    pass
