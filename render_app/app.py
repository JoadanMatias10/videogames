from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from model_utils import FeatureSelector  # noqa: F401 required for joblib loading


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_pipeline.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

app = Flask(__name__)

model_pipeline = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

REGIONAL_FIELDS = {
    "North America": "Norteamerica",
    "Europe": "Europa",
    "Japan": "Japon",
    "Rest of World": "Resto del mundo",
}


def parse_sales_value(form, field, label):
    raw_value = form.get(field, "").strip()
    if not raw_value:
        return None, f"Ingresa las ventas de {label}."

    try:
        value = float(raw_value)
    except ValueError:
        return None, f"Las ventas de {label} deben ser numericas."

    if value < 0 or value > 50:
        return None, f"Las ventas de {label} deben estar entre 0 y 50 millones."

    return value, None


def validate_form(form):
    errors = []

    year_raw = form.get("Year", "").strip()
    genre = form.get("Genre", "").strip()
    publisher = form.get("Publisher", "").strip()

    if year_raw:
        try:
            year = float(year_raw)
        except ValueError:
            errors.append("El año debe ser numerico.")
            year = np.nan
        else:
            if year < 2013 or year > 2030:
                errors.append("El año debe estar entre 2013 y 2030.")
    else:
        year = np.nan

    if not genre:
        errors.append("Selecciona un genero.")
    elif genre not in metadata["genres"]:
        errors.append("El genero seleccionado no es valido.")

    if not publisher:
        errors.append("Selecciona un publicador.")
    elif publisher not in metadata["publishers"]:
        errors.append("El publicador seleccionado no es valido.")

    regional_values = {}
    for field, label in REGIONAL_FIELDS.items():
        value, error = parse_sales_value(form, field, label)
        if error:
            errors.append(error)
        else:
            regional_values[field] = value

    data = {
        "Year": year,
        "Genre": genre,
        "Publisher": publisher,
        **regional_values,
    }
    return data, errors


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = []
    form_data = {
        "Year": "",
        "Genre": metadata["genres"][0] if metadata["genres"] else "",
        "Publisher": metadata["publishers"][0] if metadata["publishers"] else "",
        "North America": "",
        "Europe": "",
        "Japan": "",
        "Rest of World": "",
    }

    if request.method == "POST":
        form_data = {
            "Year": request.form.get("Year", "").strip(),
            "Genre": request.form.get("Genre", "").strip(),
            "Publisher": request.form.get("Publisher", "").strip(),
            "North America": request.form.get("North America", "").strip(),
            "Europe": request.form.get("Europe", "").strip(),
            "Japan": request.form.get("Japan", "").strip(),
            "Rest of World": request.form.get("Rest of World", "").strip(),
        }
        row, errors = validate_form(request.form)

        if not errors:
            input_data = pd.DataFrame([row])
            predicted_value = float(model_pipeline.predict(input_data)[0])
            prediction = max(predicted_value, 0.0)

    return render_template(
        "index.html",
        prediction=prediction,
        errors=errors,
        form_data=form_data,
        metadata=metadata,
        regional_fields=REGIONAL_FIELDS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
