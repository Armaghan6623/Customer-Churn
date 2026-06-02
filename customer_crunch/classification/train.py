"""Model training logic (XGBoost) — dataset-agnostic via DatasetConfig."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from classification.dataset_config import DatasetConfig, KAGGLE_BANK_CHURN


def train_customer_churn_model(
    data_path: str,
    save_dir: str,
    config: DatasetConfig = KAGGLE_BANK_CHURN,
) -> str:
    """Train a churn pipeline for *any* dataset described by `config`.

    Parameters
    ----------
    data_path:
        Path to the CSV file.
    save_dir:
        Directory where the serialised pipeline will be written.
    config:
        DatasetConfig that describes the dataset schema.  Defaults to
        the Kaggle bank-churn preset so existing callers are unaffected.

    Returns
    -------
    str
        Absolute path to the saved model artifact.
    """
    print(f"📥 Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Drop metadata / ID columns defined in config
    X = df.drop(columns=config.drop_cols + [config.target_col], errors="ignore")
    y = df[config.target_col]

    # Restrict to the columns the config declares (handles extra columns gracefully)
    known_cols = [c for c in config.all_feature_cols if c in X.columns]
    X = X[known_cols]

    # Resolve actual numeric / categorical lists against what's in the data
    numeric_features = [c for c in config.numeric_features if c in X.columns]
    categorical_features = [c for c in config.categorical_features if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Handle class imbalance
    num_retained = (y == 0).sum()
    num_churned = (y == 1).sum()
    scale_weight = num_retained / num_churned if num_churned > 0 else 1.0
    print(f"⚖️  Class imbalance ratio (scale_pos_weight): {scale_weight:.2f}")

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                scale_pos_weight=scale_weight,
                random_state=42,
                eval_metric="logloss",
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚀 Training churn classification model (XGBoost)...")
    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, predictions))

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, config.model_filename)
    joblib.dump({"pipeline": model_pipeline, "config": config}, model_path)
    print(f"✅ Pipeline saved to: {model_path}")
    return model_path


if __name__ == "__main__":
    raw_data_path = "data/raw/Churn_Modelling kaggel.csv"
    output_directory = "saved_models"

    if os.path.exists(raw_data_path):
        train_customer_churn_model(raw_data_path, output_directory)
    else:
        print(f"❌ Data file not found at: {raw_data_path}")
