"""
Model interpretability module using SHAP (SHapley Additive exPlanations).
"""

from typing import List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier


def compute_shap_values(
    model: XGBClassifier,
    X_encoded: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> shap.Explanation:
    """
    Computes SHAP values using TreeExplainer for an XGBoost model.
    """
    explainer = shap.TreeExplainer(model)
    if feature_names is not None:
        X_df = pd.DataFrame(X_encoded, columns=feature_names)
        shap_values = explainer(X_df)
    else:
        shap_values = explainer(X_encoded)
    return shap_values


def save_beeswarm_plot(
    shap_values: shap.Explanation,
    output_path: str,
    max_display: int = 8
) -> None:
    """
    Generates and saves a SHAP beeswarm plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
