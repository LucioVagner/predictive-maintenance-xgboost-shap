"""
Data preprocessing module for Predictive Maintenance.
Handles loading, cleaning, feature selection, and transformation.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

CATEGORICAL_FEATURES: List[str] = ['Type']

NUMERIC_FEATURES: List[str] = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN: str = 'Machine failure'

DROP_COLUMNS: List[str] = [
    'UDI',
    'Product ID',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
]

FEATURE_NAMES_SAFE: List[str] = [
    'Type_H',
    'Type_L',
    'Type_M',
    'Air_temperature_K',
    'Process_temperature_K',
    'Rotational_speed_rpm',
    'Torque_Nm',
    'Tool_wear_min'
]


def load_data(filepath: str) -> pd.DataFrame:
    """Loads dataset from CSV file."""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by removing identification and specific failure indicators
    to prepare for binary failure classification.
    """
    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=existing_drop_cols)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separates predictor features from the machine failure target."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Performs stratified train-test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    """Creates a ColumnTransformer to encode categorical features and pass numerics."""
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
