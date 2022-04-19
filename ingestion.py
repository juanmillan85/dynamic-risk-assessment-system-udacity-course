import os
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as file:
    config = json.load(file)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
output_file = config["output_file"]
test_data_path = config["test_data_path"]
test_file = config["test_file"]
ingest_file = config["ingest_file"]


def merge_multiple_dataframe():
    """
    Merge all datasets, and store them in one csv file
    """
    file_names = os.listdir(os.getcwd() + f"/{input_folder_path}/")
    ingested_files = []
    datasets = [filename for filename in file_names if filename.endswith(".csv")]
    logging.info(f"Combined all datasets")

    with open(f"{output_folder_path}/{ingest_file}", "w") as file:
        combined = pd.DataFrame()
        for dataset in datasets:
            data = pd.read_csv(f"{input_folder_path}/{dataset}")
            combined = pd.concat([combined, data]).reset_index(drop=True)
            ingested_files.append(dataset)

        file.write(str(ingested_files))

    logging.info(f"Save combined datasets to {output_folder_path} folder")
    result = combined.drop_duplicates()
    train, test = train_test_split(result, test_size=0.2, random_state=42, shuffle=True)
    train.to_csv(f"{output_folder_path}/{output_file}", index=False)
    test.to_csv(f"{test_data_path}/{test_file}", index=False)


if __name__ == "__main__":
    merge_multiple_dataframe()
