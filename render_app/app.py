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

    data = {
        "Year": year,
        "Genre": genre,
        "Publisher": publisher,
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
    }

    if request.method == "POST":
        form_data = {
            "Year": request.form.get("Year", "").strip(),
            "Genre": request.form.get("Genre", "").strip(),
            "Publisher": request.form.get("Publisher", "").strip(),
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
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
