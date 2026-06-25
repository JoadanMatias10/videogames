from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_pipeline.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

app = Flask(__name__)

model_pipeline = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

TARGET_UNIT = metadata.get("target_unit", "millones de copias vendidas")

NUMERIC_FIELDS = {
    "Year_of_Release": {
        "label": "Ano de lanzamiento",
        "help": "Ejemplo: 2016. Si se deja vacio, el pipeline imputara el valor.",
        "min": 1970,
        "max": 2030,
        "step": 1,
    },
    "NA_Sales": {
        "label": "Ventas en Norteamerica",
        "help": "Millones de copias vendidas en Norteamerica.",
        "min": 0,
        "max": 50,
        "step": 0.01,
    },
    "EU_Sales": {
        "label": "Ventas en Europa",
        "help": "Millones de copias vendidas en Europa.",
        "min": 0,
        "max": 50,
        "step": 0.01,
    },
    "JP_Sales": {
        "label": "Ventas en Japon",
        "help": "Millones de copias vendidas en Japon.",
        "min": 0,
        "max": 50,
        "step": 0.01,
    },
    "Other_Sales": {
        "label": "Ventas en otras regiones",
        "help": "Millones de copias vendidas en otras regiones.",
        "min": 0,
        "max": 50,
        "step": 0.01,
    },
    "Critic_Score": {
        "label": "Puntaje de critica",
        "help": "Valor de 0 a 100.",
        "min": 0,
        "max": 100,
        "step": 1,
    },
    "Critic_Count": {
        "label": "Cantidad de criticas",
        "help": "Numero de resenas de critica.",
        "min": 0,
        "max": 200,
        "step": 1,
    },
    "User_Score": {
        "label": "Puntaje de usuarios",
        "help": "Valor de 0 a 10.",
        "min": 0,
        "max": 10,
        "step": 0.1,
    },
    "User_Count": {
        "label": "Cantidad de usuarios",
        "help": "Numero de resenas de usuarios.",
        "min": 0,
        "max": 12000,
        "step": 1,
    },
}

CATEGORICAL_FIELDS = {
    "Platform": {
        "label": "Plataforma",
        "help": "Consola o plataforma del videojuego.",
        "required": True,
        "input_type": "select",
    },
    "Genre": {
        "label": "Genero",
        "help": "Genero principal del videojuego.",
        "required": True,
        "input_type": "select",
    },
    "Publisher": {
        "label": "Publicador",
        "help": "Empresa publicadora. Se puede escribir una nueva.",
        "required": False,
        "input_type": "text",
    },
    "Developer": {
        "label": "Desarrollador",
        "help": "Empresa desarrolladora. Se puede dejar vacio.",
        "required": False,
        "input_type": "text",
    },
    "Rating": {
        "label": "Clasificacion",
        "help": "Clasificacion de contenido.",
        "required": False,
        "input_type": "select",
    },
}


def empty_form_data():
    data = {}
    for field in metadata["features"]:
        if field in CATEGORICAL_FIELDS:
            categories = metadata["categories"].get(field, [])
            data[field] = categories[0] if categories else ""
        else:
            data[field] = ""
    return data


def parse_numeric(form, field, config):
    raw_value = form.get(field, "").strip()
    if not raw_value:
        return np.nan, None

    try:
        value = float(raw_value)
    except ValueError:
        return np.nan, f"{config['label']} debe ser numerico."

    if value < config["min"] or value > config["max"]:
        return np.nan, (
            f"{config['label']} debe estar entre "
            f"{config['min']} y {config['max']}."
        )

    return value, None


def parse_categorical(form, field, config):
    value = form.get(field, "").strip()
    if not value:
        if config["required"]:
            return "", f"Selecciona {config['label'].lower()}."
        return np.nan, None
    return value, None


def validate_form(form):
    errors = []
    row = {}

    for field in metadata["features"]:
        if field in NUMERIC_FIELDS:
            value, error = parse_numeric(form, field, NUMERIC_FIELDS[field])
        else:
            value, error = parse_categorical(form, field, CATEGORICAL_FIELDS[field])

        if error:
            errors.append(error)
        row[field] = value

    return row, errors


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = []
    form_data = empty_form_data()

    if request.method == "POST":
        form_data = {field: request.form.get(field, "").strip() for field in metadata["features"]}
        row, errors = validate_form(request.form)

        if not errors:
            input_data = pd.DataFrame([row], columns=metadata["features"])
            predicted_value = float(model_pipeline.predict(input_data)[0])
            prediction = max(predicted_value, 0.0)

    return render_template(
        "index.html",
        prediction=prediction,
        errors=errors,
        form_data=form_data,
        metadata=metadata,
        target_unit=TARGET_UNIT,
        numeric_fields=NUMERIC_FIELDS,
        categorical_fields=CATEGORICAL_FIELDS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
