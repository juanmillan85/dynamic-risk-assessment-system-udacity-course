import pandas as pd
import timeit
import subprocess
import os
import json
import pickle
import logging

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
test_data_path = os.path.join(config["test_data_path"])
test_file = config["test_file"]
output_model = config["output_model"]
output_file = config["output_file"]
numeric_cols = config["numeric_cols"]
label = config["label"]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


def model_predictions(model, dataframe):
    X = dataframe.loc[:, numeric_cols].values.reshape(-1, len(numeric_cols))
    logging.info("Model predicting")
    y_pred = model.predict(X)
    return y_pred


def create_prediction_model():
    with open(f"{prod_deployment_path}/{output_model}", "rb") as file:
        model = pickle.load(file)
    return model


def dataframe_summary():

    dataset = pd.read_csv(f"{dataset_csv_path}/{output_file}")

    list_means = list(dataset.mean(numeric_only=True))

    list_medians = list(dataset.median(numeric_only=True))

    list_stds = list(dataset.std(numeric_only=True))
    logging.info("Summary")

    return [list_means, list_medians, list_stds]


def missing_values_list():
    dataset = pd.read_csv(f"{dataset_csv_path}/{output_file}")

    missing_values = list(dataset.isna().sum())

    missing_values_percentage = [
        missing_values[i] / len(dataset.index) for i in range(len(missing_values))
    ]
    logging.info("Missing Values")

    return missing_values_percentage


def ingestion_timing():
    start_time = timeit.default_timer()
    os.system("python3 ingestion.py")
    timing = timeit.default_timer() - start_time
    logging.info(f"Ingestion Time: {timing}")
    return timing


def training_timing():
    start_time = timeit.default_timer()
    os.system("python3 training.py")
    timing = timeit.default_timer() - start_time
    logging.info(f"Training Time: {timing}")

    return timing


def execution_time():
    time_record = []
    time_record.append(ingestion_timing())
    time_record.append(training_timing())
    return time_record


def outdated_packages_list():
    outdated = subprocess.check_output(["pip", "list", "--outdated"])
    with open("outdated.txt", "wb") as f:
        f.write(outdated)
    return outdated


def main():
    prediction_model = create_prediction_model()
    test_data = pd.read_csv(f"{test_data_path}/{test_file}")
    model_predictions(prediction_model, test_data)
    dataframe_summary()
    missing_values_list()
    execution_time()
    outdated_packages_list()


if __name__ == "__main__":
    main()
