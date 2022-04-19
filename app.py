from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import (
    create_prediction_model,
    model_predictions,
    dataframe_summary,
    execution_time,
    missing_values_list,
    outdated_packages_list,
)
import json
import os

from scoring import score_model

app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])

prediction_model = create_prediction_model()


@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    data = request.get_json()
    name = data["file_name"]
    try:
        df = pd.read_csv(f"{name}")
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    prediction = model_predictions(prediction_model, df)
    return jsonify({"prediction": str(prediction)}), 200


@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring_stats():
    score = score_model()
    return jsonify({"score": score}), 200


@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summary_stats():
    summary = dataframe_summary()
    return jsonify({"summary_stats": summary}), 200


@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def stats():
    times = execution_time()
    missing_values = missing_values_list()
    outdated_packages = outdated_packages_list()
    return (
        jsonify(
            {
                "execution_time": str(times),
                "missing_values": str(missing_values),
                "outdated_packages": str(outdated_packages.decode("utf8").strip()),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
