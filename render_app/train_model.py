from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import FeatureSelector


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "PS4_GamesSales.csv"
MODEL_PATH = BASE_DIR / "model_pipeline.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"
RANDOM_STATE = 42


def load_dataset():
    dataset_raw = pd.read_csv(
        DATA_PATH,
        encoding="latin1",
        na_values=["N/A", ""],
        keep_default_na=True,
    )
    return dataset_raw.rename(columns={"Global": "Global_Sales"}).copy()


def build_preprocessor():
    numeric_features = ["Year", "North America", "Europe", "Japan", "Rest of World"]
    categorical_features = ["Genre", "Publisher"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_model_definitions():
    return {
        "Regresion Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Red Neuronal MLP": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=600,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }


def select_features(preprocessor, X_train, y_train, n_selected_features=6):
    prepared_preprocessor = clone(preprocessor).set_output(transform="pandas")
    X_train_prepared = prepared_preprocessor.fit_transform(X_train, y_train)

    corr_with_target = X_train_prepared.apply(lambda col: col.corr(y_train)).fillna(0)
    correlation_table = pd.DataFrame(
        {
            "Caracteristica": corr_with_target.index,
            "Correlacion con y": corr_with_target.values,
            "Correlacion absoluta": corr_with_target.abs().values,
        }
    ).sort_values("Correlacion absoluta", ascending=False)

    rfe = RFE(
        estimator=LinearRegression(),
        n_features_to_select=n_selected_features,
    )
    rfe.fit(X_train_prepared, y_train)
    rfe_table = pd.DataFrame(
        {
            "Caracteristica": X_train_prepared.columns,
            "Ranking RFE": rfe.ranking_,
            "Seleccionada por RFE": rfe.support_,
        }
    )

    tree_model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    tree_model.fit(X_train_prepared, y_train)
    tree_importance_table = pd.DataFrame(
        {
            "Caracteristica": X_train_prepared.columns,
            "Importancia en arboles": tree_model.feature_importances_,
        }
    ).sort_values("Importancia en arboles", ascending=False)

    selection_comparison = (
        correlation_table[["Caracteristica", "Correlacion con y", "Correlacion absoluta"]]
        .merge(rfe_table, on="Caracteristica")
        .merge(tree_importance_table, on="Caracteristica")
    )
    selection_comparison["Top correlacion"] = (
        selection_comparison["Correlacion absoluta"].rank(ascending=False, method="min") <= 15
    )
    selection_comparison["Top arboles"] = (
        selection_comparison["Importancia en arboles"].rank(ascending=False, method="min") <= 15
    )
    selection_comparison["Puntaje conjunto"] = (
        selection_comparison["Top correlacion"].astype(int)
        + selection_comparison["Seleccionada por RFE"].astype(int)
        + selection_comparison["Top arboles"].astype(int)
    )
    selection_comparison = selection_comparison.sort_values(
        ["Puntaje conjunto", "Importancia en arboles", "Correlacion absoluta"],
        ascending=[False, False, False],
    )

    return selection_comparison.head(n_selected_features)["Caracteristica"].tolist()


def get_pca_component_count(preprocessor, X_train, y_train):
    pca_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("pca", PCA(random_state=RANDOM_STATE)),
        ]
    )
    pca_pipeline.fit(X_train, y_train)
    cumulative_variance = np.cumsum(
        pca_pipeline.named_steps["pca"].explained_variance_ratio_
    )
    return int(np.argmax(cumulative_variance >= 0.95) + 1)


def build_selected_pipeline(preprocessor, selected_features, model):
    selected_preprocessor = clone(preprocessor).set_output(transform="pandas")
    return Pipeline(
        steps=[
            ("preprocessor", selected_preprocessor),
            ("feature_selector", FeatureSelector(selected_features)),
            ("model", model),
        ]
    )


def build_pca_pipeline(preprocessor, n_components, model):
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )


def main():
    dataset = load_dataset()

    target = "Global_Sales"
    removed_columns = ["Game"]
    feature_cols = [col for col in dataset.columns if col not in removed_columns + [target]]

    X = dataset[feature_cols].copy()
    y = dataset[target].copy()

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    selected_features = select_features(preprocessor, X_train, y_train)
    n_components_95 = get_pca_component_count(preprocessor, X_train, y_train)
    model_definitions = get_model_definitions()

    cv_strategy = KFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE)
    experiment_rows = []
    experiment_builders = {
        "Caracteristicas seleccionadas": lambda model: build_selected_pipeline(
            preprocessor, selected_features, model
        ),
        "PCA": lambda model: build_pca_pipeline(preprocessor, n_components_95, model),
    }

    for scenario_name, builder in experiment_builders.items():
        for model_name, model in model_definitions.items():
            pipeline = builder(clone(model))
            cv_scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring={
                    "R2": "r2",
                    "MAE": "neg_mean_absolute_error",
                },
                n_jobs=1,
            )
            experiment_rows.append(
                {
                    "Escenario": scenario_name,
                    "Modelo": model_name,
                    "R2 medio": float(cv_scores["test_R2"].mean()),
                    "MAE medio": float((-cv_scores["test_MAE"]).mean()),
                    "Desv est R2": float(cv_scores["test_R2"].std(ddof=1)),
                }
            )

    summary = pd.DataFrame(experiment_rows).sort_values(
        ["R2 medio", "MAE medio", "Desv est R2"],
        ascending=[False, True, True],
    )
    best = summary.iloc[0]

    if best["Escenario"] == "Caracteristicas seleccionadas":
        final_pipeline = build_selected_pipeline(
            preprocessor,
            selected_features,
            clone(model_definitions[best["Modelo"]]),
        )
    else:
        final_pipeline = build_pca_pipeline(
            preprocessor,
            n_components_95,
            clone(model_definitions[best["Modelo"]]),
        )

    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH)

    genres = sorted(dataset["Genre"].dropna().astype(str).unique().tolist())
    publishers = sorted(dataset["Publisher"].fillna("Unknown").astype(str).unique().tolist())
    if "Unknown" not in publishers:
        publishers.insert(0, "Unknown")

    metadata = {
        "dataset": "PS4_GamesSales.csv",
        "target": target,
        "target_unit": "millones de copias vendidas",
        "input_features": feature_cols,
        "removed_columns": removed_columns,
        "regional_sales_features": [
            "North America",
            "Europe",
            "Japan",
            "Rest of World",
        ],
        "best_scenario": str(best["Escenario"]),
        "best_model": str(best["Modelo"]),
        "selected_features": selected_features,
        "pca_components": int(n_components_95),
        "cv_summary": summary.to_dict(orient="records"),
        "genres": genres,
        "publishers": publishers,
        "year_min": None if pd.isna(dataset["Year"].min()) else int(dataset["Year"].min()),
        "year_max": None if pd.isna(dataset["Year"].max()) else int(dataset["Year"].max()),
        "sales_min": float(dataset[["North America", "Europe", "Japan", "Rest of World"]].min().min()),
        "sales_max": float(dataset[["North America", "Europe", "Japan", "Rest of World"]].max().max()),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Metadata guardada en: {METADATA_PATH}")
    print(f"Mejor experimento: {best['Escenario']} - {best['Modelo']}")


if __name__ == "__main__":
    main()
