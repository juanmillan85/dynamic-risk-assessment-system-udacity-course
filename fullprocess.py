import ast
import os
import json
import logging

import scoring
import training
import deployment
import diagnostics
import reporting
import ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
output_file = config["output_file"]
prod_deployment_path = config["prod_deployment_path"]
test_data_path = os.path.join(config["test_data_path"])
test_file = config["test_file"]


with open(f"{prod_deployment_path}/ingestedfiles.txt", "r") as f:
    ingested_files = ast.literal_eval(f.read())

filenames = os.listdir(os.getcwd() + f"/{input_folder_path}/")

filenames = [
    filename
    for filename in filenames
    if filename.endswith(".csv") and filename not in ingested_files
]
if len(filenames) == 0:
    logger.info("No new data found. Exiting process.")
    exit()

logger.info("Ingesting new dataset")
ingestion.merge_multiple_dataframe()
logger.info("Data ingested.")

logger.info("Training")
training.train_model()
logger.info("Model trained.")

logger.info("Scoring")
current_model_score = scoring.score_model()
logger.info("Model scored")

with open(f"{prod_deployment_path}/latestscore.txt", "r") as f:
    previous_model_score = f.read()

logger.info(previous_model_score)
logger.info(current_model_score)
drift_check = float(current_model_score) > float(previous_model_score)
logger.info("Checking for model drift")
if not drift_check:
    logger.info("No model drift")
    logger.info("Exiting process")
    exit()

logger.info("Deploying")
deployment.main()
logger.info("Model deployed")

logger.info("Diagnostics")
diagnostics.main()
logger.info("Diagnostics completed")

logger.info("Reporting")
reporting.score_model(True)
logger.info("Reporting completed")
logger.info("Exiting Re-Deployment process")


