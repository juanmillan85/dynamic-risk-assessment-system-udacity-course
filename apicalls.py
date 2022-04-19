import requests
import json
import os

URL = "http://127.0.0.1:8000"
with open("config.json", "r") as f:
    config = json.load(f)
output_model_path = os.path.join(config["output_model_path"])
header = {"Content-Type": "application/json"}
data = {"file_name": "testdata/testdata.csv"}
response1 = requests.post(
    URL + "/prediction", headers=header, data=json.dumps(data)
).json()

response2 = requests.get(URL + "/scoring").json()
response3 = requests.get(URL + "/summarystats").json()
response4 = requests.get(URL + "/diagnostics").json()

with open(f"{output_model_path}/apireturns.txt", "w") as file:
    file.write("API Responses\n")
    file.write("Test data predictions\n")
    file.write(str(response1["prediction"]))
    file.write("\n------------------------------------------------\n")
    file.write("F1 score for production model\n")
    file.write(str(response2["score"]))
    file.write("\n------------------------------------------------\n")
    file.write("Summary stats\n")
    file.write("\n")
    for stats, stat_type in zip(response3["summary_stats"], ["Mean", "Median", "Std"]):
        file.write(f"{stat_type} {stats[0]} - {stats[1]} - {stats[2]} - {stats[3]}\n")
    file.write("\n------------------------------------------------\n")

    file.write("% of missing data per column\n")
    file.write(f"{response4['missing_values']}\n")
    file.write("\n------------------------------------------------\n")
    file.write("Execution times\n")
    file.write(f"{response4['execution_time']}\n")
    file.write("\n------------------------------------------------\n")
    file.write("List of outdated packages\n")
    file.write(response4["outdated_packages"])
