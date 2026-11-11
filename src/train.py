"""
Model training and evaluation module for Predictive Maintenance using XGBoost.
"""

from typing import Dict, Any
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs: Any
) -> XGBClassifier:
    """Instantiates and trains an XGBoost classifier."""
    model = XGBClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """Evaluates the model on test data and returns classification metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def save_model(model: Any, filepath: str) -> None:
    """Serializes model to disk using joblib."""
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """Loads serialized model from disk."""
    return joblib.load(filepath)
