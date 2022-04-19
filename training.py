import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
import json

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
output_file = config["output_file"]
model_path = os.path.join(config["output_model_path"])
output_model = config["output_model"]
label = config["label"]
numeric_cols = config["numeric_cols"]


def pre_process_data(file_path):
    """
    Read cvs file and format the dataset into X and y
    """
    training_data = pd.read_csv(file_path)
    X = training_data.loc[:, numeric_cols].values.reshape(-1, len(numeric_cols))
    y = training_data[label].values.reshape(-1, 1).ravel()
    return X, y


def train_model():
    """
    Train the model based on Logistic Regression algorithm and saved
    """
    logging.info("Define Logistic Regression Model")
    logit = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=1000,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.000001,
        verbose=1,
        warm_start=False,
    )

    logging.info("Pre Process Data")
    X, y = pre_process_data(f"{dataset_csv_path}/{output_file}")

    logging.info("Training")
    model = logit.fit(X, y)

    logging.info(f"Save the model to file {output_model}")
    pickle.dump(model, open(f"{model_path}/{output_model}", "wb"))


if __name__ == "__main__":
    train_model()
