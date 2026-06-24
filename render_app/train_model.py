from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DATA_URL = "https://raw.githubusercontent.com/JoadanMatias10/videogames/main/Video_Games_Sales_as_at_22_Dec_2016.csv"
LOCAL_DATA_PATH = REPO_DIR / "Video_Games_Sales_as_at_22_Dec_2016.csv"
MODEL_PATH = BASE_DIR / "model_pipeline.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

TARGET = "Global_Sales"
FEATURES = ["Platform", "Genre", "NA_Sales", "EU_Sales", "Critic_Score", "Critic_Count"]
NUMERIC_FEATURES = ["NA_Sales", "EU_Sales", "Critic_Score", "Critic_Count"]
CATEGORICAL_FEATURES = ["Platform", "Genre"]


def load_dataset():
    try:
        data = pd.read_csv(DATA_URL, encoding="latin1", na_values=["N/A", ""], keep_default_na=True)
    except Exception:
        data = pd.read_csv(LOCAL_DATA_PATH, encoding="latin1", na_values=["N/A", ""], keep_default_na=True)

    for column in NUMERIC_FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.drop_duplicates().dropna(subset=[TARGET]).copy()


def build_pipeline():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=30,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def main():
    dataset = load_dataset()
    X = dataset[FEATURES].copy()
    y = dataset[TARGET].copy()

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "dataset": "Video Games Sales as at 22 Dec 2016",
        "target": TARGET,
        "target_unit": "millones de copias vendidas",
        "scenario": "Caracteristicas seleccionadas",
        "model": "Ridge",
        "features": FEATURES,
        "uses_pca": False,
        "render_url": "https://ps4-global-sales-predictor.onrender.com/",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": {
            feature: sorted(dataset[feature].dropna().astype(str).unique().tolist())
            for feature in CATEGORICAL_FEATURES
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Modelo guardado en {MODEL_PATH}")
    print(f"Metadatos guardados en {METADATA_PATH}")


if __name__ == "__main__":
    main()
