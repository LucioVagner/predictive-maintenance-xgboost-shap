"""
End-to-end execution pipeline for Predictive Maintenance.
Loads data, trains XGBoost, evaluates metrics, and generates SHAP explanations.
"""

import os
from src.preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    split_train_test,
    build_preprocessor,
    FEATURE_NAMES_SAFE
)
from src.train import (
    train_xgboost,
    evaluate_model,
    save_model
)
from src.explain import (
    compute_shap_values,
    save_beeswarm_plot
)


def run_pipeline(
    raw_data_path: str = "data/raw/ai4i2020.csv",
    processed_data_path: str = "data/processed/machine-failure.csv",
    model_output_path: str = "models/xgboost_model.joblib",
    plot_output_path: str = "reports/figures/shap_beeswarm.png"
) -> None:
    """Executes the complete predictive maintenance pipeline."""
    print(f"[1/5] Loading raw data from: {raw_data_path}")
    raw_df = load_data(raw_data_path)

    print(f"[2/5] Cleaning data and saving to: {processed_data_path}")
    cleaned_df = clean_data(raw_df)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    cleaned_df.to_csv(processed_data_path, index=False)

    print("[3/5] Preprocessing and splitting data...")
    X, y = split_features_target(cleaned_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = build_preprocessor()
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    print("[4/5] Training XGBoost Classifier...")
    model = train_xgboost(X_train_encoded, y_train)

    metrics = evaluate_model(model, X_test_encoded, y_test)
    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    save_model(model, model_output_path)
    print(f"Model saved to: {model_output_path}")

    print("[5/5] Generating SHAP explainability plot...")
    shap_values = compute_shap_values(
        model,
        X_test_encoded,
        feature_names=FEATURE_NAMES_SAFE
    )
    save_beeswarm_plot(shap_values, plot_output_path)
    print(f"SHAP beeswarm plot saved to: {plot_output_path}")
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
