import pickle
import json
import os
import logging
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
test_data_path = os.path.join(config["test_data_path"])
test_file = config["test_file"]
output_model_path = os.path.join(config["output_model_path"])
output_model = config["output_model"]
numeric_cols = config["numeric_cols"]
label = config["label"]


def score_model(is_production=False):
    """
    Calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """
    if is_production:
        logger.info("Production")
        with open(f"{prod_deployment_path}/{output_model}", "rb") as file:
            model = pickle.load(file)
    else:
        logger.info("Dev")
        with open(f"{output_model_path}/{output_model}", "rb") as file:
            model = pickle.load(file)

    test_data = pd.read_csv(f"{test_data_path}/{test_file}")
    X = test_data.loc[:, numeric_cols].values.reshape(-1, 3)
    y = test_data[label].values.reshape(-1, 1)
    y_pred = model.predict(X)

    logging.info("Calculate Confusion Matrix")
    confusion_matrix = metrics.confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        square=True,
        cmap="Reds",
    )
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    logging.info("Saved confusion matrix")
    plt.savefig(f"{output_model_path}/confusionmatrix.png")


if __name__ == "__main__":
    score_model()
