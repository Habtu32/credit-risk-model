import pytest
import pandas as pd
from src.data_processing import preprocess_data

def test_preprocess_data():
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': [4, 5, 6]
    })
    
    clean_df = preprocess_data(df)
    
    assert len(clean_df) == 2
    assert clean_df.isnull().sum().sum() == 0
