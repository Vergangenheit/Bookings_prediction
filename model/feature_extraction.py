from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from preprocess.etl1 import merge, clean_sessions, split_and_pivot

session_list_cleaned = clean_sessions()
split_and_pivot()
sessions_merged = merge('ActBook')


def extract_feature_arrays(dataset: pd.DataFrame) -> (np.array, np.array, np.array, np.array):
    X = np.array(dataset.iloc[:, 2:])
    y = np.array(dataset['has_booking'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val
