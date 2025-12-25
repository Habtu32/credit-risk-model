import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_and_preprocess_data(file_path):
    """
    Loads raw transaction data, performs feature engineering,
    and fits a preprocessing pipeline.
    """
  
  # 1. Load dataset
    df = pd.read_csv('../data/raw/data.csv')

    # 2. Datetime conversion
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # 3. Aggregate to customer level
    df_customer = (
        df
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

    # 4. Log transformations
    df_customer['log_transaction_count'] = np.log1p(
        df_customer['transaction_count']
    )

    df_customer['log_total_amount'] = np.sign(
        df_customer['total_amount']
    ) * np.log1p(np.abs(df_customer['total_amount']))

    df_customer['log_avg_amount'] = np.sign(
        df_customer['avg_amount']
    ) * np.log1p(np.abs(df_customer['avg_amount']))

    # 5. Recency feature
    reference_date = df['TransactionStartTime'].max()

    customer_recency = (
        df.groupby('CustomerId')['TransactionStartTime']
        .max()
        .reset_index()
    )

    customer_recency['recency_days'] = (
        reference_date - customer_recency['TransactionStartTime']
    ).dt.days

    # 6. Final customer feature table
    customer_df = customer_recency[['CustomerId', 'recency_days']].merge(
        df_customer[
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

    # 7. Target variable (fraud at customer level)
    fraud_label = (
        df.groupby('CustomerId')['FraudResult']
        .max()
        .reset_index()
    )

    customer_df = customer_df.merge(
        fraud_label,
        on='CustomerId',
        how='left'
    )

    # 8. Split features and target
    X = customer_df.drop(columns=['CustomerId', 'FraudResult'])
    y = customer_df['FraudResult']

    # 9. Preprocessing pipeline
    numeric_features = X.columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features)
        ]
    )

    # 10. Fit preprocessor
    X_processed = preprocessor.fit_transform(X)

    return X_processed, preprocessor, y


# -------------------------
# Local test
# -------------------------
if __name__ == '__main__':
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'sample_data.csv')

    X_processed, preprocessor, y = load_and_preprocess_data(DATA_PATH)

    print("X shape:", X_processed.shape)
    print("y shape:", y.shape)
