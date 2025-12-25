import pandas as pd
import numpy as np
import joblib
import os


def predict_risk(new_raw_data_df, model_path, preprocessor_path):
    """
    Loads saved model and preprocessor, preprocesses new raw customer data,
    and generates risk probability scores.
    """

    print("[INFO] Starting risk prediction process...")

    # 1. Load preprocessor and model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at: {preprocessor_path}")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    # 2. Copy input data
    df_new = new_raw_data_df.copy()

    # 3. Feature engineering (MUST match training)
    df_new['TransactionStartTime'] = pd.to_datetime(df_new['TransactionStartTime'])

    # Aggregate to customer level
    df_customer_aggregated = (
        df_new
        .groupby("CustomerId")
        .agg(
            transaction_count=("TransactionId", "count"),
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            max_amount=("Amount", "max"),
            min_amount=("Amount", "min")
        )
        .reset_index()
    )

    # Log transformations
    df_customer_aggregated['log_transaction_count'] = np.log1p(
        df_customer_aggregated['transaction_count']
    )

    df_customer_aggregated['log_total_amount'] = np.sign(
        df_customer_aggregated['total_amount']
    ) * np.log1p(np.abs(df_customer_aggregated['total_amount']))

    df_customer_aggregated['log_avg_amount'] = np.sign(
        df_customer_aggregated['avg_amount']
    ) * np.log1p(np.abs(df_customer_aggregated['avg_amount']))

    # Recency feature
    reference_date = df_new['TransactionStartTime'].max()
    customer_recency = (
        df_new.groupby("CustomerId")['TransactionStartTime']
        .max()
        .reset_index()
    )

    customer_recency['recency_days'] = (
        reference_date - customer_recency['TransactionStartTime']
    ).dt.days

    # Final customer dataset
    customer_df = customer_recency[['CustomerId', 'recency_days']].merge(
        df_customer_aggregated[
            [
                'CustomerId',
                'max_amount',
                'min_amount',
                'log_transaction_count',
                'log_total_amount',
                'log_avg_amount'
            ]
        ],
        on='CustomerId',
        how='left'
    )

    # Prepare model input
    X_new = customer_df.drop(columns=['CustomerId'])

    # 4. Preprocess
    X_new_processed = preprocessor.transform(X_new)

    # 5. Predict probabilities
    y_pred_proba = model.predict_proba(X_new_processed)[:, 1]

    # 6. Output
    results_df = pd.DataFrame({
        'CustomerId': customer_df['CustomerId'],
        'Predicted_Risk_Probability': y_pred_proba
    })

    print("[INFO] Prediction completed successfully.")
    return results_df


# -------------------------
# Local testing df = pd.read_csv('../data/raw/data.csv')
# -------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH = os.path.join(BASE_DIR, "data", "../data/raw/data.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_model.joblib")
    PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

    df_sample = pd.read_csv(DATA_PATH)

    results = predict_risk(
        df_sample.head(100),
        MODEL_PATH,
        PREPROCESSOR_PATH
    )

    print(results.head())
