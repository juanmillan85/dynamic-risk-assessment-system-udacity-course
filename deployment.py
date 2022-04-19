import os
import json
import shutil

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
output_model_path = os.path.join(config["output_model_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
output_model = config["output_model"]


def store_model_into_pickle():
    """
    Copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file
    into the deployment directory
    """
    latest_pickle_file = os.path.join(output_model_path, output_model)
    new_pickle_file_path = os.path.join(prod_deployment_path, output_model)
    shutil.copy2(latest_pickle_file, new_pickle_file_path)

    score_file_path = os.path.join(output_model_path, "latestscore.txt")
    new_score_file_path = os.path.join(prod_deployment_path, "latestscore.txt")
    shutil.copy2(score_file_path, new_score_file_path)

    ingest_file_path = os.path.join(dataset_csv_path, "ingestedfiles.txt")
    new_ingest_file_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
    shutil.copy2(ingest_file_path, new_ingest_file_path)


def main():
    store_model_into_pickle()


if __name__ == "__main__":
    main()
