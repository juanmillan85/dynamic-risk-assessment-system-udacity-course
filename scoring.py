import pandas as pd
import logging
import pickle
import os
from sklearn import metrics
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as f:
    config = json.load(f)

test_data_path = os.path.join(config["test_data_path"])
test_file = config["test_file"]
output_model_path = os.path.join(config["output_model_path"])
output_model = config["output_model"]
numeric_cols = config["numeric_cols"]
label = config["label"]


def score_model():
    """
    This function should take a trained model, load test data, and calculate an
    F1 score for the model relative to the test data and it should write the result
     to the latestscore.txt file
    """
    with open(f"{output_model_path}/{output_model}", "rb") as file:
        model = pickle.load(file)
    test_data = pd.read_csv(f"{test_data_path}/{test_file}")
    X = test_data.loc[:, numeric_cols].values.reshape(-1, len(numeric_cols))
    y = test_data[label].values.reshape(-1, 1).ravel()
    logging.info("Model predicting and scoring with F1")
    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)

    logging.info(f"F1 score: {f1_score}")
    with open(f"{output_model_path}/latestscore.txt", "w") as file:
        file.write(f"{f1_score}")

    return f1_score


if __name__ == "__main__":
    score_model()
